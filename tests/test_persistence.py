import hashlib
import json
from pathlib import Path
import pytest

def configure_test_storage(monkeypatch, tmp_path):
    from app.core import config
    from app.storage import database as db
    monkeypatch.setattr(config.settings, "data_path", tmp_path / "data")
    monkeypatch.setattr(config.settings, "uploads_path", tmp_path / "data" / "uploads")
    monkeypatch.setattr(config.settings, "extracted_path", tmp_path / "data" / "extracted")
    monkeypatch.setattr(config.settings, "chroma_path", tmp_path / "data" / "chroma_db")
    monkeypatch.setattr(config.settings, "metadata_path", tmp_path / "data" / "metadata")
    monkeypatch.setattr(config.settings, "logs_path", tmp_path / "data" / "logs")
    monkeypatch.setattr(config.settings, "sqlite_path", tmp_path / "data" / "metadata" / "documents.sqlite3")
    db.initialize()
    return db

def test_sqlite_registry_survives_reconnect(monkeypatch, tmp_path):
    db = configure_test_storage(monkeypatch, tmp_path)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    path = tmp_path / "data" / "uploads" / "doc.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("permanent", encoding="utf-8")
    doc = {"document_id":"doc1","filename":"doc.txt","stored_filename":"doc.txt",
           "original_path":str(path),"content_hash":hashlib.sha256(b"permanent").hexdigest(),
           "file_format":".txt","uploaded_at":now,"updated_at":now,
           "processing_status":"pending","extraction_status":"pending","indexing_status":"pending",
           "chunk_count":0,"source_metadata":{},"extracted_path":None,"processing_error":None,
           "last_reconciled_at":None}
    db.save_document(doc)
    # A fresh SQLite connection sees the committed record, simulating a process restart.
    assert db.get_document("doc1")["filename"] == "doc.txt"
    assert Path(db.get_document("doc1")["original_path"]).read_text() == "permanent"

def test_initialization_does_not_clear_registry(monkeypatch, tmp_path):
    db = configure_test_storage(monkeypatch, tmp_path)
    db.log_recovery_event("info", "test", "persist")
    before = db.recovery_events()
    db.initialize()
    assert db.recovery_events()[0]["details"] == before[0]["details"]

def test_atomic_artifact_write_and_extract_persistence(monkeypatch, tmp_path):
    configure_test_storage(monkeypatch, tmp_path)
    from app.storage.files import atomic_write
    artifact = tmp_path / "data" / "extracted" / "doc.json"
    payload = json.dumps({"text":"saved extracted content"}).encode()
    atomic_write(artifact, payload)
    # Reading from disk, not session state, proves artifact persistence.
    assert json.loads(artifact.read_text(encoding="utf-8"))["text"] == "saved extracted content"
    assert not list(artifact.parent.glob("*.tmp"))

def test_duplicate_hash_lookup(monkeypatch, tmp_path):
    db = configure_test_storage(monkeypatch, tmp_path)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    path = tmp_path / "data" / "uploads" / "x.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("same", encoding="utf-8")
    h = hashlib.sha256(b"same").hexdigest()
    doc = {"document_id":"x","filename":"x.txt","stored_filename":"x.txt",
           "original_path":str(path),"content_hash":h,"file_format":".txt",
           "uploaded_at":now,"updated_at":now,"processing_status":"indexed",
           "extraction_status":"complete","indexing_status":"complete","chunk_count":1,
           "source_metadata":{},"extracted_path":None,"processing_error":None,"last_reconciled_at":now}
    db.save_document(doc)
    assert db.get_by_hash(h)["document_id"] == "x"

def test_explicit_delete_operation_is_recoverable(monkeypatch, tmp_path):
    db = configure_test_storage(monkeypatch, tmp_path)
    db.create_deletion_operation("op1", "doc1")
    assert db.pending_deletions()[0]["operation_id"] == "op1"
    db.update_deletion_operation("op1", "failed", "simulated failure")
    assert db.pending_deletions()[0]["error"] == "simulated failure"


def test_registry_missing_original_is_preserved_and_flagged(monkeypatch, tmp_path):
    db = configure_test_storage(monkeypatch, tmp_path)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    doc = {"document_id":"missing1","filename":"gone.txt","stored_filename":"missing1.txt",
           "original_path":str(tmp_path / "not-there.txt"),"content_hash":"abc",
           "file_format":".txt","uploaded_at":now,"updated_at":now,
           "processing_status":"indexed","extraction_status":"complete","indexing_status":"complete",
           "chunk_count":1,"source_metadata":{},"extracted_path":None,"processing_error":None,
           "last_reconciled_at":now}
    db.save_document(doc)
    # Stub vector-store imports: this branch must preserve a missing-file registry
    # entry even if the optional vector database dependency is unavailable.
    import sys, types
    fake_store = types.ModuleType("app.retrieval.store")
    fake_store.add = lambda records: None
    fake_store.delete = lambda document_id: None
    fake_store.document_chunk_count = lambda document_id: 0
    fake_store.list_chunk_document_ids = lambda: set()
    monkeypatch.setitem(sys.modules, "app.retrieval.store", fake_store)
    from app.services.documents import reconcile
    events = reconcile()
    assert db.get_document("missing1")["processing_status"] == "recovery_required"
    assert any("Original missing" in e for e in events)

def test_deletion_journal_survives_database_reinitialization(monkeypatch, tmp_path):
    db = configure_test_storage(monkeypatch, tmp_path)
    db.create_deletion_operation("delete-op", "doc-abc")
    db.initialize()
    assert db.pending_deletions()[0]["operation_id"] == "delete-op"

def test_recovery_events_are_durable(monkeypatch, tmp_path):
    db = configure_test_storage(monkeypatch, tmp_path)
    db.log_recovery_event("warning", "orphaned_chunks", "preserved", "doc-x")
    db.initialize()
    events = db.recovery_events()
    assert events[0]["event_type"] == "orphaned_chunks"
    assert events[0]["document_id"] == "doc-x"

def test_atomic_write_replaces_file_without_temp_remnants(monkeypatch, tmp_path):
    configure_test_storage(monkeypatch, tmp_path)
    from app.storage.files import atomic_write
    target = tmp_path / "data" / "uploads" / "original.bin"
    atomic_write(target, b"first")
    atomic_write(target, b"second")
    assert target.read_bytes() == b"second"
    assert list(target.parent.glob("*.tmp")) == []

def test_same_filename_different_hashes_are_distinct_registry_entries(monkeypatch, tmp_path):
    db = configure_test_storage(monkeypatch, tmp_path)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    for did, payload in [("v1", b"old"), ("v2", b"new")]:
        path = tmp_path / "data" / "uploads" / f"{did}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        doc = {"document_id":did,"filename":"report.txt","stored_filename":path.name,
               "original_path":str(path),"content_hash":hashlib.sha256(payload).hexdigest(),
               "file_format":".txt","uploaded_at":now,"updated_at":now,
               "processing_status":"pending","extraction_status":"pending","indexing_status":"pending",
               "chunk_count":0,"source_metadata":{},"extracted_path":None,"processing_error":None,
               "last_reconciled_at":None}
        db.save_document(doc)
    assert len([d for d in db.list_documents() if d["filename"] == "report.txt"]) == 2

def test_storage_paths_are_separate(monkeypatch, tmp_path):
    configure_test_storage(monkeypatch, tmp_path)
    from app.core.config import settings
    assert settings.uploads_path != settings.extracted_path
    assert settings.extracted_path != settings.chroma_path
    assert settings.logs_path != settings.metadata_path

def test_fts_chunk_index_returns_exact_term_matches(monkeypatch, tmp_path):
    db = configure_test_storage(monkeypatch, tmp_path)
    record = {
        "id": "chunk-1", "text": "Invoice INV-2026-0042 is due on 15 March.",
        "metadata": {"document_id": "doc-1", "owner_id": "local", "filename": "invoice.txt", "source_location": "page 1"},
    }
    db.index_chunks([record])
    matches = db.search_indexed_chunks('"INV-2026-0042"', "local")
    assert matches[0]["chunk_id"] == "chunk-1"
    db.remove_indexed_chunks("doc-1")
    assert not db.search_indexed_chunks('"INV-2026-0042"', "local")

def test_chromadb_restart_persistence_integration():
    chromadb = pytest.importorskip("chromadb")
    import tempfile
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
        path = str(Path(temp) / "chroma")
        client = chromadb.PersistentClient(path=path)
        collection = client.get_or_create_collection("persist_test")
        collection.upsert(ids=["id1"], documents=["persistent"], embeddings=[[0.1, 0.2, 0.3]])
        del collection
        del client
        client2 = chromadb.PersistentClient(path=path)
        collection2 = client2.get_or_create_collection("persist_test")
        assert collection2.get(ids=["id1"])["documents"] == ["persistent"]
        del collection2
        del client2


def test_fts_index_exposes_document_vocabulary_for_spelling_recovery(monkeypatch, tmp_path):
    db = configure_test_storage(monkeypatch, tmp_path)
    db.index_chunks([{"id": "vocab-1", "text": "Contractor authorization requires manager approval.",
                      "metadata": {"document_id": "doc-vocab", "owner_id": "local", "filename": "policy.txt", "source_location": "page 1"}}])
    vocabulary = db.indexed_vocabulary()
    assert "contractor" in vocabulary
    assert "authorization" in vocabulary
