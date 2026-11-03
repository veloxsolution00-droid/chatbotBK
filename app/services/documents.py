from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json, uuid, logging, time, os, tempfile
from app.core.config import settings
from app.storage import database as db
from app.storage.files import atomic_write
from app.ingestion.extractor import iter_extract, SUPPORTED
from app.ingestion.classify import detect_document_type
from app.retrieval.chunker import chunk_text, digest
from app.retrieval.store import add, delete, document_chunk_count

logger = logging.getLogger(__name__)

def now():
    return datetime.now(timezone.utc).isoformat()

def _atomic_write(path: Path, content: bytes):
    atomic_write(path, content)

def _artifact_path(document_id):
    return Path(settings.extracted_path) / f"{document_id}.json"

def list_docs(owner_id=None):
    return db.list_documents(owner_id)

def _process_existing(doc, progress_callback=None):
    """Idempotently extract and index a durable original; never remove the original."""
    docid = doc["document_id"]
    original = Path(doc["original_path"])
    if not original.exists():
        db.update_document(docid, updated_at=now(), processing_status="recovery_required",
                           extraction_status="missing_original", indexing_status="not_indexed",
                           processing_error=f"Original file is missing: {original}")
        db.log_recovery_event("error", "missing_original", str(original), docid)
        return db.get_document(docid)

    db.update_document(docid, updated_at=now(), processing_status="processing",
                       extraction_status="processing", indexing_status="pending", processing_error=None)
    total_started = time.perf_counter()
    try:
        extraction_started = time.perf_counter()
        if progress_callback:
            progress_callback("Extracting text", 10, "Reading document text and source locations")
        artifact_path = _artifact_path(docid)
        chunking_started = time.perf_counter()
        source_metadata = []
        profile_sample = []
        profile_chars = 0
        extracted_chars = block_count = expected_chunks = 0
        indexed_started = False
        Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
        # Keep the existing JSON artifact contract but write its blocks one at a
        # time to a temporary file. The final replacement is atomic, so a crash
        # never leaves a partial extraction artifact visible.
        temp_handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                                  dir=str(Path(artifact_path).parent), suffix=".tmp")
        temp_path = Path(temp_handle.name)
        if progress_callback:
            progress_callback("Splitting into chunks", 35, "Splitting extracted text into searchable chunks")
        try:
            temp_handle.write(json.dumps({"document_id": docid, "filename": doc["filename"],
                                           "content_hash": doc["content_hash"], "extracted_at": now()}, ensure_ascii=False)[:-1])
            temp_handle.write(',"blocks":[\n')
            for loc, text in iter_extract(original):
                if block_count:
                    temp_handle.write(',\n')
                temp_handle.write(json.dumps({"source_location": loc, "text": text}, ensure_ascii=False))
                block_count += 1
                extracted_chars += len(text)
                if profile_chars < 60000:
                    profile_sample.append(text[:60000 - profile_chars])
                    profile_chars += len(text)
                source_metadata.append({"source_location": loc})
                records = [{
                    "id": digest(f"{docid}:{loc}:{i}:{chunk}"), "text": chunk,
                    "metadata": {"document_id": docid, "owner_id": doc.get("owner_id", "local"),
                                 "filename": doc["filename"], "source_location": loc, "chunk_index": i}
                } for i, chunk in enumerate(chunk_text(text, settings.chunk_size, settings.chunk_overlap))]
                if not records:
                    continue
                if not indexed_started:
                    # Replace only this document's prior vectors once new, readable
                    # content has been found. Deterministic IDs make retries safe.
                    delete(docid)
                    indexed_started = True
                expected_chunks += len(records)
                add(records, progress_callback=progress_callback)
            temp_handle.write('\n]}')
            temp_handle.flush()
            os.fsync(temp_handle.fileno())
            temp_handle.close()
            if not expected_chunks:
                raise ValueError("No searchable text chunks were produced from the uploaded file.")
            os.replace(temp_path, artifact_path)
        except Exception:
            temp_handle.close()
            temp_path.unlink(missing_ok=True)
            raise

        extraction_seconds = time.perf_counter() - extraction_started
        chunking_seconds = time.perf_counter() - chunking_started
        logger.info("Extraction/index staging complete: id=%s filename=%s blocks=%d chars=%d chunks=%d seconds=%.2f", docid, doc["filename"], block_count, extracted_chars, expected_chunks, extraction_seconds)
        if progress_callback:
            progress_callback("Saving extracted text", 95, f"Extracted {extracted_chars} characters from {block_count} source block(s)")
        actual_count = document_chunk_count(docid)
        if actual_count < expected_chunks:
            raise RuntimeError(f"ChromaDB has {actual_count} chunks; expected {expected_chunks}.")
        document_profile = detect_document_type("\n".join(profile_sample), doc["filename"])
        logger.info("Document type detected: id=%s type=%s confidence=%s", docid,
                    document_profile["type"], document_profile["confidence"])
        db.update_document(
            docid, updated_at=now(), processing_status="indexed",
            extraction_status="complete", indexing_status="complete",
            chunk_count=actual_count,
            source_metadata={"locations": source_metadata,
                             "document_type": document_profile["type"],
                             "type_confidence": document_profile["confidence"],
                             "type_signals": document_profile["signals"]},
            extracted_path=str(artifact_path), processing_error=None, last_reconciled_at=now()
        )
        logger.info("Document indexed: id=%s filename=%s chunks=%s extraction_seconds=%.2f chunking_seconds=%.2f total_seconds=%.2f", docid, doc["filename"], actual_count, extraction_seconds, chunking_seconds, time.perf_counter()-total_started)
        if progress_callback:
            progress_callback("Complete", 100, f"Document searchable: {actual_count} chunks indexed")
        return db.get_document(docid)
    except Exception as exc:
        logger.exception("Document processing failed: %s", docid)
        db.update_document(docid, updated_at=now(), processing_status="recovery_required",
                           extraction_status="failed", indexing_status="incomplete",
                           processing_error=str(exc))
        db.log_recovery_event("error", "ingestion_failed", str(exc), docid)
        raise

def ingest(filename, content, owner_id="local", progress_callback=None):
    db.initialize()
    if len(content) > settings.max_file_mb * 1024 * 1024:
        raise ValueError(f"File exceeds {settings.max_file_mb} MB.")
    name = Path(filename).name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file extension: {ext}")
    content_hash = sha256(content).hexdigest()

    existing = db.get_by_hash(content_hash, owner_id)
    if existing:
        if existing["processing_status"] == "indexed":
            # Confirm the original still exists before declaring a duplicate healthy.
            if Path(existing["original_path"]).exists():
                return {**existing, "status": "duplicate"}
        # Incomplete document: reuse its stable ID and durable original when available.
        if Path(existing["original_path"]).exists():
            return _process_existing(existing, progress_callback=progress_callback)
        # If original is absent, preserve the registry and save a recovery copy from upload.
        _atomic_write(Path(settings.uploads_path) / existing["stored_filename"], content)
        db.update_document(existing["document_id"], original_path=str(Path(settings.uploads_path) / existing["stored_filename"]),
                           updated_at=now(), processing_status="recovery_required")
        return _process_existing(db.get_document(existing["document_id"]), progress_callback=progress_callback)

    docid = str(uuid.uuid4())
    stored_name = f"{docid}{ext}"
    original_path = Path(settings.uploads_path) / stored_name
    _atomic_write(original_path, content)
    doc = {
        "document_id": docid, "owner_id": owner_id, "filename": name, "stored_filename": stored_name,
        "original_path": str(original_path), "content_hash": content_hash, "file_format": ext,
        "uploaded_at": now(), "updated_at": now(), "processing_status": "pending",
        "extraction_status": "pending", "indexing_status": "pending", "chunk_count": 0,
        "source_metadata": {}, "extracted_path": str(_artifact_path(docid)), "processing_error": None,
        "last_reconciled_at": None
    }
    # Register after the durable original exists. A crash between these steps is
    # repaired by startup reconciliation scanning uploads/ for unregistered files.
    db.save_document(doc)
    return _process_existing(doc, progress_callback=progress_callback)

def _complete_delete(operation):
    opid, docid = operation["operation_id"], operation["document_id"]
    doc = db.get_document(docid)
    try:
        # Explicit deletion intent is persisted before any artifact is removed.
        delete(docid)
        if doc:
            for p in (Path(doc.get("extracted_path") or _artifact_path(docid)),
                      Path(settings.uploads_path) / doc.get("stored_filename", "")):
                if p.is_file():
                    p.unlink()
        db.remove_document_record(docid)
        db.update_deletion_operation(opid, "complete", None)
        db.log_recovery_event("info", "deletion_complete", f"Deletion operation {opid}", docid)
        return True
    except Exception as exc:
        db.update_deletion_operation(opid, "failed", str(exc))
        db.log_recovery_event("error", "deletion_incomplete", str(exc), docid)
        logger.exception("Deletion incomplete for document %s", docid)
        raise

def remove(docid):
    db.initialize()
    doc = db.get_document(docid)
    if not doc:
        return False
    opid = str(uuid.uuid4())
    db.create_deletion_operation(opid, docid)
    _complete_delete({"operation_id": opid, "document_id": docid})
    return True

def reconcile():
    """Conservative startup recovery. It never deletes files or ChromaDB chunks automatically."""
    db.initialize()
    events = []
    known = {d["document_id"]: d for d in db.list_documents()}

    # Recover an original saved before a crash interrupted registry creation.
    for path in Path(settings.uploads_path).glob("*"):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        if any(d["stored_filename"] == path.name for d in known.values()):
            continue
        try:
            raw = path.read_bytes()
            ext = path.suffix.lower()
            if ext not in SUPPORTED:
                db.log_recovery_event("warning", "unregistered_upload_unsupported",
                                      f"Preserved unregistered file {path.name}; unsupported format.")
                events.append(f"Unsupported unregistered file preserved: {path.name}")
                continue
            docid = path.stem
            if not docid or db.get_document(docid):
                docid = str(uuid.uuid4())
                new_path = Path(settings.uploads_path) / f"{docid}{ext}"
                _atomic_write(new_path, raw)
                path = new_path
            now_value = now()
            doc = {
                "document_id": docid, "owner_id": "local", "filename": path.name, "stored_filename": path.name,
                "original_path": str(path), "content_hash": sha256(raw).hexdigest(),
                "file_format": ext, "uploaded_at": now_value, "updated_at": now_value,
                "processing_status": "recovery_required", "extraction_status": "pending",
                "indexing_status": "pending", "chunk_count": 0, "source_metadata": {},
                "extracted_path": str(_artifact_path(docid)),
                "processing_error": "Recovered original file missing from registry.",
                "last_reconciled_at": now_value
            }
            db.save_document(doc)
            known[docid] = doc
            db.log_recovery_event("warning", "unregistered_upload_recovered",
                                  f"Registered preserved original {path.name}", docid)
            events.append(f"Registered untracked original: {path.name}")
        except Exception as exc:
            db.log_recovery_event("error", "upload_reconciliation_failed", str(exc))
            events.append(f"Could not reconcile upload {path.name}: {exc}")

    # Mark missing originals and recover incomplete documents; healthy indexed docs are untouched.
    for doc in db.list_documents():
        original = Path(doc["original_path"])
        if not original.exists():
            db.update_document(doc["document_id"], updated_at=now(),
                               processing_status="recovery_required",
                               processing_error=f"Original file missing: {original}")
            db.log_recovery_event("error", "registry_file_missing", str(original), doc["document_id"])
            events.append(f"Original missing for {doc['filename']}; registry preserved.")
            continue
        if doc["processing_status"] != "indexed" or doc["indexing_status"] != "complete":
            try:
                _process_existing(doc)
                events.append(f"Recovered incomplete document: {doc['filename']}")
            except Exception as exc:
                events.append(f"Recovery still needed for {doc['filename']}: {exc}")

    # Resume only deletions that the user explicitly requested earlier.
    for op in db.pending_deletions():
        try:
            _complete_delete(op)
            events.append(f"Completed previously requested deletion for {op['document_id']}")
        except Exception as exc:
            events.append(f"Deletion still incomplete for {op['document_id']}: {exc}")

    # Report orphaned chunks but never silently remove them.
    try:
        from app.retrieval.store import list_chunk_document_ids
        registry_ids = {d["document_id"] for d in db.list_documents()}
        for orphan_id in sorted(list_chunk_document_ids() - registry_ids):
            db.log_recovery_event("warning", "orphaned_chroma_chunks",
                                  "Chunks preserved; manual recovery required.", orphan_id)
            events.append(f"Orphaned ChromaDB chunks preserved for ID {orphan_id}")
    except Exception as exc:
        db.log_recovery_event("error", "chroma_reconciliation_failed", str(exc))
        events.append(f"ChromaDB reconciliation could not complete: {exc}")

    return events
