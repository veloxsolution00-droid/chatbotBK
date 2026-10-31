"""Persistent, bounded background ingestion queue. Staged originals remain on disk until successful."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import logging
from app.core.config import settings
from app.storage import database as db
from app.services.documents import ingest

logger = logging.getLogger(__name__)
_executor = None

def now(): return datetime.now(timezone.utc).isoformat()

def _run(job_id):
    job = db.get_job(job_id)
    if not job or job["status"] == "complete": return
    path = Path(job["staged_path"])
    db.update_job(job_id, status="processing", stage="Starting", progress=1, updated_at=now(), error=None)
    try:
        # At most one staged file is loaded for each worker; keep worker count low on 8 GB RAM.
        content = path.read_bytes()
        def progress(stage, percent, message):
            db.update_job(job_id, status="processing", stage=f"{stage}: {message}", progress=max(1, min(99, int(percent))), updated_at=now())
        result = ingest(job["filename"], content, owner_id="local", progress_callback=progress)
        doc_id = result.get("document_id")
        doc = db.get_document(doc_id) if doc_id else None
        if not doc or doc.get("processing_status") != "indexed" or doc.get("indexing_status") != "complete":
            detail = (doc or {}).get("processing_error") or "Ingestion returned without a fully indexed document."
            raise RuntimeError(detail)
        db.update_job(job_id, status="complete", stage="Searchable", progress=100, updated_at=now(), document_id=doc_id, error=None)
        path.unlink(missing_ok=True)
    except Exception as exc:
        logger.exception("Ingestion job failed: %s", job_id)
        db.update_job(job_id, status="failed", stage="Failed", progress=100, updated_at=now(), error=str(exc))
        # Staged file intentionally retained for diagnosis/retry; never silently discard it.

def start_queue():
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=max(1, min(settings.ingestion_workers, 2)), thread_name_prefix="doc-ingest")
    for job in db.recoverable_jobs():
        _executor.submit(_run, job["job_id"])

def submit(job_id):
    if _executor is None: start_queue()
    _executor.submit(_run, job_id)

def shutdown_queue():
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=False)
        _executor = None
