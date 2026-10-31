import logging, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from app.core.config import settings
from app.storage import database as db
from app.retrieval.ollama import status
from app.retrieval.store import collection, backfill_owner_metadata, backfill_text_search
from app.services.documents import list_docs, remove, reconcile
from app.ingestion.extractor import SUPPORTED
from app.services.job_queue import submit, start_queue, shutdown_queue
from app.rag.chat import answer
from app.cache.manager import stats as cache_stats

LOCAL_OWNER_ID = "local"
Path(settings.logs_path).mkdir(parents=True, exist_ok=True)
from logging.handlers import RotatingFileHandler
_handler = RotatingFileHandler(Path(settings.logs_path) / "application.log", maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
logging.basicConfig(level=logging.INFO, handlers=[_handler, logging.StreamHandler()], format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI(title="Offline Document Intelligence Chatbot", version="0.5.0-local")

class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=2000)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=8)

@app.on_event("startup")
def startup_recovery():
    db.initialize()
    # This build is single-user and loopback-only. Preserve all existing records,
    # but normalize old authenticated installs to one local owner so old documents
    # remain visible and searchable after login is removed.
    db.migrate_to_local_owner()
    try:
        backfill_owner_metadata(db.list_documents())
        backfill_text_search()
    except Exception:
        logger.exception("Could not migrate legacy ChromaDB owner metadata; retrieval diagnostics will show if no chunks match.")
    if settings.recovery_on_startup:
        try:
            events = reconcile()
            logger.info("Startup reconciliation complete: %s event(s)", len(events))
        except Exception:
            logger.exception("Startup reconciliation encountered an error; data was not cleared.")
    start_queue()

@app.on_event("shutdown")
def shutdown():
    shutdown_queue()

@app.get("/")
def root():
    return {"app": app.title, "docs": "/docs", "health": "/health", "mode": "local-single-user"}

@app.get("/health")
def health():
    try:
        c = collection()
        db_health = {"available": True, "count": c.count(), "path": str(settings.chroma_path)}
    except Exception as e:
        db_health = {"available": False, "error": str(e)}
    o = status()
    return {"status": "ok" if o.get("available") and db_health["available"] else "degraded",
            "ollama": o, "chromadb": db_health, "documents": len(list_docs()),
            "cache": cache_stats(),
            "sqlite_path": str(settings.sqlite_path), "data_path": str(settings.data_path), "uploads_path": str(settings.uploads_path),
            "extracted_path": str(settings.extracted_path), "jobs_path": str(settings.jobs_path),
            "logs_path": str(settings.logs_path), "limits": {"max_file_mb": settings.max_file_mb,
            "max_batch_mb": settings.max_batch_mb, "max_upload_files": settings.max_upload_files}}

@app.post("/documents/upload", status_code=202)
async def upload(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "Choose at least one file.")
    if len(files) > settings.max_upload_files:
        raise HTTPException(413, f"Too many files. Maximum is {settings.max_upload_files} per batch.")
    max_file = settings.max_file_mb * 1024 * 1024
    max_batch = settings.max_batch_mb * 1024 * 1024
    sizes, total = [], 0
    for f in files:
        try:
            f.file.seek(0, 2); size = f.file.tell(); f.file.seek(0)
        except Exception:
            size = -1
        filename = Path(f.filename or "upload").name
        if Path(filename).suffix.lower() not in SUPPORTED:
            raise HTTPException(415, f"{filename}: unsupported file type. Supported extensions: {', '.join(sorted(SUPPORTED))}")
        if size < 0 or size > max_file:
            raise HTTPException(413, f"{filename}: individual file limit is {settings.max_file_mb} MB.")
        total += size; sizes.append(size)
    if total > max_batch:
        raise HTTPException(413, f"Batch exceeds {settings.max_batch_mb} MB total.")

    results = []
    batch_started = time.perf_counter()
    Path(settings.jobs_path).mkdir(parents=True, exist_ok=True)
    try:
        for f, size in zip(files, sizes):
            filename = Path(f.filename or "upload").name
            job_id = str(uuid.uuid4())
            ext = Path(filename).suffix.lower()
            staged = Path(settings.jobs_path) / f"{job_id}{ext}.upload"
            copied = 0
            try:
                with staged.open("wb") as out:
                    while True:
                        block = await f.read(1024 * 1024)
                        if not block: break
                        copied += len(block)
                        if copied > max_file or copied > size:
                            raise HTTPException(413, f"{filename}: upload exceeded configured size limit.")
                        out.write(block)
                    out.flush()
                now = datetime.now(timezone.utc).isoformat()
                db.create_job({"job_id": job_id, "owner_id": LOCAL_OWNER_ID, "filename": filename,
                    "staged_path": str(staged), "size_bytes": copied, "status": "queued",
                    "stage": "Queued for extraction", "progress": 0,
                    "created_at": now, "updated_at": now, "document_id": None, "error": None})
                submit(job_id)
                results.append({"job_id": job_id, "filename": filename, "status": "queued", "size_bytes": copied})
            except Exception:
                staged.unlink(missing_ok=True)
                raise
            finally:
                await f.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Batch upload staging failed")
        raise HTTPException(500, f"Could not stage upload: {e}") from e
    return {"status": "accepted", "files": len(results), "total_bytes": total,
            "staging_seconds": round(time.perf_counter()-batch_started, 3),
            "message": "Files are saved durably and queued. Job status includes the current processing stage.",
            "results": results}

@app.get("/jobs")
def jobs():
    return {"jobs": db.list_jobs(LOCAL_OWNER_ID)}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = db.get_job(job_id, LOCAL_OWNER_ID)
    if not job: raise HTTPException(404, "Job not found.")
    return job

@app.post("/jobs/{job_id}/retry")
def retry_job(job_id: str):
    job = db.get_job(job_id, LOCAL_OWNER_ID)
    if not job: raise HTTPException(404, "Job not found.")
    if job["status"] != "failed": raise HTTPException(409, "Only failed jobs can be retried.")
    if not Path(job["staged_path"]).exists(): raise HTTPException(410, "Staged file is missing; upload it again.")
    db.update_job(job_id, status="queued", stage="Queued for retry", progress=0, updated_at=datetime.now(timezone.utc).isoformat(), error=None)
    submit(job_id)
    return {"job_id": job_id, "status": "queued"}

@app.get("/documents")
def documents():
    return {"documents": list_docs(LOCAL_OWNER_ID)}

@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    doc = db.get_document(document_id)
    if not doc or doc.get("owner_id", LOCAL_OWNER_ID) != LOCAL_OWNER_ID:
        raise HTTPException(404, "Document not found.")
    try:
        if not remove(document_id): raise HTTPException(404, "Document not found.")
        return {"status": "deleted", "document_id": document_id}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, f"Deletion incomplete and recorded for recovery: {e}")

@app.post("/admin/reconcile")
def admin_reconcile():
    try: return {"events": reconcile()}
    except Exception as e:
        logger.exception("Manual reconciliation failed")
        raise HTTPException(500, f"Reconciliation failed: {e}")

@app.get("/admin/recovery-events")
def get_recovery_events():
    return {"events": db.recovery_events()}

@app.post("/chat")
def chat(req: ChatRequest):
    started = time.perf_counter()
    result = answer(req.question, owner_id=LOCAL_OWNER_ID, history=[item.model_dump() for item in req.history])
    result["response_seconds"] = round(time.perf_counter()-started, 3)
    return result
