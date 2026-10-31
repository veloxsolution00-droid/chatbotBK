from datetime import datetime, timezone

def test_local_owner_migration(monkeypatch, tmp_path):
    from tests.test_persistence import configure_test_storage
    db = configure_test_storage(monkeypatch, tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    doc = {"document_id":"old-doc", "owner_id":"old-user", "filename":"x.txt", "stored_filename":"x.txt",
           "original_path":str(tmp_path/"x.txt"), "content_hash":"abc", "file_format":".txt",
           "uploaded_at":now, "updated_at":now, "processing_status":"indexed", "extraction_status":"complete",
           "indexing_status":"complete", "chunk_count":1, "source_metadata":{}, "extracted_path":None,
           "processing_error":None, "last_reconciled_at":now}
    db.save_document(doc)
    db.create_job({"job_id":"job-old", "owner_id":"old-user", "filename":"x.txt", "staged_path":"/tmp/x.txt",
                   "size_bytes":1, "status":"queued", "created_at":now, "updated_at":now})
    db.migrate_to_local_owner()
    assert db.get_document("old-doc")["owner_id"] == "local"
    assert db.get_job("job-old", "local")["owner_id"] == "local"
    with db.connect() as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "users" not in tables and "sessions" not in tables

def test_job_progress_fields_persist(monkeypatch, tmp_path):
    from tests.test_persistence import configure_test_storage
    db = configure_test_storage(monkeypatch, tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    db.create_job({"job_id":"job-progress", "owner_id":"local", "filename":"x.txt", "staged_path":"/tmp/x.txt",
                   "size_bytes":1, "status":"queued", "stage":"Queued", "progress":0, "created_at":now, "updated_at":now})
    db.update_job("job-progress", status="processing", stage="Embedding chunks", progress=65, updated_at=now)
    job = db.get_job("job-progress", "local")
    assert job["stage"] == "Embedding chunks" and job["progress"] == 65
