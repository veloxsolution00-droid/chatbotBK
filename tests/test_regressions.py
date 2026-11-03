"""Regression tests for issues found during the principal-engineer code review."""
import sys
import types


def test_multi_answer_citations_remap_comma_separated_references(monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    import app.rag.chat as chat

    assert chat._remap_citations("Fact A [1]. Fact B [1, 2].", {1: 3, 2: 4}) == \
        "Fact A [3]. Fact B [3, 4]."
    # Unknown local ranks are preserved rather than silently dropped.
    assert chat._remap_citations("Odd [7].", {1: 3}) == "Odd [7]."


def test_lexical_candidate_limit_is_respected_as_a_real_cap(monkeypatch):
    from app.retrieval import store
    from app.core.config import settings

    monkeypatch.setattr(settings, "lexical_candidate_limit", 7)
    monkeypatch.setattr(settings, "top_k", 5)
    monkeypatch.setattr(settings, "retrieval_candidate_multiplier", 10)
    monkeypatch.setattr(store, "embed_query", lambda question: [0.1, 0.2])

    class FakeCollection:
        def count(self):
            return 100

        def query(self, **kwargs):
            return {
                "ids": [["c1"]],
                "documents": [["policy text"]],
                "metadatas": [[{"document_id": "d1", "owner_id": "local",
                                "filename": "a.txt", "source_location": "page 1"}]],
                "distances": [[0.2]],
            }

    monkeypatch.setattr(store, "collection", lambda: FakeCollection())
    captured = {}

    def fake_fts(query, owner_id, limit):
        captured["limit"] = limit
        return []

    monkeypatch.setattr(store.db, "search_indexed_chunks", fake_fts)
    hits = store.search("policy approval", owner_id="local")
    assert hits and hits[0]["id"] == "c1"
    assert captured["limit"] == 7


def test_fts_owner_is_migrated_with_the_rest_of_the_install(monkeypatch, tmp_path):
    from app.storage import database as db
    from tests.test_persistence import configure_test_storage

    configure_test_storage(monkeypatch, tmp_path)
    db.index_chunks([{
        "id": "chunk-old", "text": "Invoice INV-2026-0042 is due.",
        "metadata": {"document_id": "doc-old", "owner_id": "old-user",
                     "filename": "invoice.txt", "source_location": "page 1"},
    }])
    assert db.search_indexed_chunks('"invoice"', "local") == []
    db.migrate_to_local_owner()
    matches = db.search_indexed_chunks('"invoice"', "local")
    assert matches and matches[0]["chunk_id"] == "chunk-old"


def test_job_claim_prevents_double_processing_and_recovers_after_restart(monkeypatch, tmp_path):
    from tests.test_persistence import configure_test_storage
    from datetime import datetime, timezone

    db = configure_test_storage(monkeypatch, tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    db.create_job({"job_id": "job-claim", "owner_id": "local", "filename": "x.txt",
                   "staged_path": str(tmp_path / "x.txt"), "size_bytes": 1, "status": "queued",
                   "stage": "Queued", "progress": 0, "created_at": now, "updated_at": now})
    assert db.claim_job("job-claim") is True
    assert db.get_job("job-claim")["status"] == "processing"
    assert db.claim_job("job-claim") is False
    db.reset_processing_jobs()
    assert db.get_job("job-claim")["status"] == "queued"
    assert db.claim_job("job-claim") is True


def test_direct_calculation_works_without_any_indexed_documents(monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    import app.rag.chat as chat

    monkeypatch.setattr(chat, "list_docs", lambda owner_id=None: [])
    monkeypatch.setattr(chat, "generate",
                        lambda prompt: (_ for _ in ()).throw(AssertionError("model must not be called")))
    result = chat.answer("What is 10% of 250?")
    assert result["status"] == "success"
    assert result["calculation"]["result"] == 25
    assert result["retrieval"]["indexed_documents"] == 0


def test_multi_question_processing_errors_are_not_mislabelled_as_no_evidence(monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    import app.rag.chat as chat

    monkeypatch.setattr(chat, "list_docs", lambda owner_id=None: [
        {"document_id": "d1", "processing_status": "indexed", "indexing_status": "complete"}])
    monkeypatch.setattr(chat, "_answer_one", lambda question, indexed, owner_id, doc_types=None: {
        "question": question, "status": "processing_error", "answer": "Local generation failed.",
        "sources": [], "abstained": True})
    result = chat.answer("First question? Second question?")
    assert result["status"] == "processing_error"


def test_html_extraction_drops_markup_scripts_and_styles(tmp_path):
    from app.ingestion.extractor import extract

    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><style>.x{color:red}</style><script>alert('bad')</script></head>"
        "<body><h1>Policy</h1><p>Hello &amp; welcome</p></body></html>",
        encoding="utf-8",
    )
    text = extract(path)[0][1]
    assert "Hello & welcome" in text
    assert "alert" not in text
    assert "color:red" not in text
    assert "<p>" not in text


def test_csv_with_cp1252_encoding_is_readable(tmp_path):
    from app.ingestion.extractor import extract

    path = tmp_path / "contacts.csv"
    path.write_bytes("name,city\nJosé,Montréal\n".encode("cp1252"))
    text = extract(path)[0][1]
    assert "José" in text and "Montréal" in text


def test_direct_calculation_never_evaluates_year_ranges_or_identifiers():
    from app.rag.calculations import direct_calculation

    # These were previously read as subtraction (2024 - 2025 == -1).
    assert direct_calculation("What is 2024-2025?") is None
    assert direct_calculation("What is INV-2026-42?") is None
    assert direct_calculation("Calculate SKU-2026-11") is None
    # Genuine arithmetic still works, including spaced subtraction.
    assert direct_calculation("Calculate 12 * 4")["result"] == 48
    assert direct_calculation("What is 2026 - 42?")["result"] == 1984
    assert direct_calculation("What is 15% of 200?")["result"] == 30


def test_relative_path_overrides_are_anchored_to_the_project(tmp_path):
    from app.core.config import ROOT, Settings

    configured = Settings(data_path=tmp_path / "data", uploads_path="relative-uploads")
    assert configured.uploads_path == (ROOT / "relative-uploads").resolve()
    # Absolute overrides are still honored unchanged.
    absolute = tmp_path / "elsewhere"
    assert Settings(data_path=tmp_path / "data", uploads_path=absolute).uploads_path == absolute


def test_requeue_failed_job_is_atomic(monkeypatch, tmp_path):
    from datetime import datetime, timezone
    from tests.test_persistence import configure_test_storage

    db = configure_test_storage(monkeypatch, tmp_path)
    now = datetime.now(timezone.utc).isoformat()
    db.create_job({"job_id": "job-retry", "owner_id": "local", "filename": "x.txt",
                   "staged_path": str(tmp_path / "x.txt"), "size_bytes": 1, "status": "failed",
                   "stage": "Failed", "progress": 100, "created_at": now, "updated_at": now})
    assert db.requeue_failed_job("job-retry") is True
    assert db.get_job("job-retry")["status"] == "queued"
    # A second concurrent retry must not queue the same job again.
    assert db.requeue_failed_job("job-retry") is False


def test_json_with_utf8_bom_is_extracted(tmp_path):
    from app.ingestion.extractor import extract

    path = tmp_path / "data.json"
    path.write_bytes(b"\xef\xbb\xbf{\"name\": \"caf\xc3\xa9\"}")
    text = extract(path)[0][1]
    assert "café" in text


def test_question_splitter_keeps_trailing_statements_with_their_question():
    import app.rag.chat as chat

    assert chat.split_questions("What is the date? Who signed it?") == \
        ["What is the date", "Who signed it"]
    # A statement after a question mark is not a second question.
    assert chat.split_questions("What is the date? It was 2024.") == \
        ["What is the date? It was 2024."]
    # Explicit numbered lists are still separate questions.
    assert chat.split_questions("1. What is the date?\n2. Who signed it?") == \
        ["What is the date", "Who signed it"]
    # A second arithmetic question is kept even without an interrogative word.
    assert chat.split_questions("What is 10% of 250? And 20% of 100?") == \
        ["What is 10% of 250", "And 20% of 100"]


def test_health_degrades_instead_of_raising_when_metadata_is_unavailable(monkeypatch):
    import app.main as main

    def unavailable(owner_id=None):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(main, "list_docs", unavailable)
    monkeypatch.setattr(main, "collection", lambda: types.SimpleNamespace(count=lambda: 0))
    monkeypatch.setattr(main, "status", lambda: {"available": True})
    monkeypatch.setattr(main, "cache_stats", lambda: {"enabled": True, "entries": 0})

    payload = main.health()
    assert payload["status"] == "degraded"
    assert payload["documents"] is None
    assert payload["metadata"]["available"] is False
    assert "locked" in payload["metadata"]["error"]
