import sys
import types


def test_embedding_cache_round_trip_is_model_scoped(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    from app.core.config import settings
    from app.cache import manager

    monkeypatch.setattr(settings, "metadata_path", tmp_path)
    monkeypatch.setattr(settings, "cache_enabled", True)
    manager.put_embedding("what is revenue", "nomic-embed-text", [0.1, 0.2, 0.3])
    assert manager.get_embedding("what is revenue", "nomic-embed-text") == [0.1, 0.2, 0.3]
    assert manager.get_embedding("what is revenue", "different-model") is None


def test_semantic_retrieval_cache_reuses_paraphrase_but_respects_hard_anchors(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    from app.core.config import settings
    from app.cache import manager

    monkeypatch.setattr(settings, "metadata_path", tmp_path)
    monkeypatch.setattr(settings, "cache_enabled", True)
    corpus = "docset-v1"
    config = "retrieval-v1"
    payload = {"hits": [{"id": "chunk-1", "text": "Revenue declined because costs rose.",
                          "metadata": {"document_id": "d1", "filename": "report.pdf", "source_location": "page 3"},
                          "similarity": 0.91}], "retrieval": {"selected_chunks": 1}}
    manager.put_result("retrieval", "Why did revenue decline?", "owner-a", corpus, config,
                       payload, query_vector=[1.0, 0.0], question_type="general")

    hit = manager.lookup_result("retrieval", "What caused revenue to fall?", "owner-a", corpus, config,
                                 query_vector=[0.999, 0.02], question_type="general", semantic_threshold=0.94)
    assert hit is not None
    assert hit["match_kind"] == "semantic"
    assert hit["payload"]["hits"][0]["id"] == "chunk-1"

    # Number/date constraints must not collide, even when vectors are identical.
    manager.put_result("retrieval", "Why did revenue decline in 2024?", "owner-a", corpus, config,
                       payload, query_vector=[1.0, 0.0], question_type="general")
    blocked = manager.lookup_result("retrieval", "Why did revenue decline in 2025?", "owner-a", corpus, config,
                                    query_vector=[1.0, 0.0], question_type="general", semantic_threshold=0.94)
    assert blocked is None


def test_answer_cache_only_stores_successful_cited_results(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    from app.core.config import settings
    from app.cache import manager

    monkeypatch.setattr(settings, "metadata_path", tmp_path)
    monkeypatch.setattr(settings, "cache_enabled", True)
    scope = ("owner-a", "docs-v1", "answer-v1")
    bad = {"status": "insufficient_evidence", "abstained": True, "sources": []}
    manager.put_result("answer", "question", *scope, bad, query_vector=[1.0], question_type="general")
    assert manager.lookup_result("answer", "question", *scope, question_type="general") is None

    good = {"status": "success", "abstained": False, "sources": [{"filename": "policy.pdf"}], "answer": "Approval is required [1]."}
    manager.put_result("answer", "question", *scope, good, query_vector=[1.0], question_type="general")
    hit = manager.lookup_result("answer", "question", *scope, question_type="general")
    assert hit is not None
    assert hit["payload"]["answer"] == "Approval is required [1]."


def test_chat_exact_answer_cache_skips_search_and_generation(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    from app.core.config import settings
    import app.rag.chat as chat

    monkeypatch.setattr(settings, "metadata_path", tmp_path)
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "verify_answers", False)
    monkeypatch.setattr(settings, "min_similarity", 0.25)
    monkeypatch.setattr(settings, "top_k", 5)
    monkeypatch.setattr(chat, "list_docs", lambda owner_id: [{
        "document_id": "d1", "owner_id": owner_id, "content_hash": "hash1", "updated_at": "2026-10-31",
        "chunk_count": 1, "processing_status": "indexed", "indexing_status": "complete"
    }])
    calls = {"search": 0, "generate": 0}
    def fake_search(question, owner_id):
        calls["search"] += 1
        return [{"id": "c1", "text": "The policy requires manager approval.",
                 "metadata": {"document_id": "d1", "owner_id": owner_id, "filename": "policy.pdf", "source_location": "page 1"},
                 "similarity": 0.9}]
    def fake_generate(prompt):
        calls["generate"] += 1
        return "The policy requires manager approval [1]."
    monkeypatch.setattr(chat, "search", fake_search)
    monkeypatch.setattr(chat, "embed_query", lambda question: [1.0, 0.0])
    monkeypatch.setattr(chat, "generate", fake_generate)
    monkeypatch.setattr(chat, "indexed_vocabulary", lambda: [])

    first = chat._answer_one("What does the policy require?", {"d1"}, "owner-a")
    second = chat._answer_one("What does the policy require?", {"d1"}, "owner-a")
    assert first["status"] == "success"
    assert second["status"] == "success"
    assert second["answer_cache"]["hit"] is True
    assert calls["search"] > 0
    assert calls["generate"] == 1


def test_cache_is_scoped_to_owner_and_document_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    from app.core.config import settings
    from app.cache import manager

    monkeypatch.setattr(settings, "metadata_path", tmp_path)
    monkeypatch.setattr(settings, "cache_enabled", True)
    payload = {"hits": [{"id": "private-chunk", "metadata": {"document_id": "d1"}}]}
    manager.put_result("retrieval", "question", "owner-a", "docs-v1", "retrieval-v1", payload,
                       query_vector=[1.0, 0.0], question_type="general")
    assert manager.lookup_result("retrieval", "question", "owner-b", "docs-v1", "retrieval-v1") is None
    assert manager.lookup_result("retrieval", "question", "owner-a", "docs-v2", "retrieval-v1") is None


def test_semantic_answer_cache_is_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    from app.core.config import settings
    import app.rag.chat as chat

    monkeypatch.setattr(settings, "metadata_path", tmp_path)
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "semantic_answer_cache_enabled", False)
    monkeypatch.setattr(settings, "verify_answers", False)
    monkeypatch.setattr(settings, "min_similarity", 0.25)
    monkeypatch.setattr(settings, "top_k", 5)
    monkeypatch.setattr(chat, "list_docs", lambda owner_id: [{
        "document_id": "d1", "owner_id": owner_id, "content_hash": "hash1", "updated_at": "2026-10-31",
        "chunk_count": 1, "processing_status": "indexed", "indexing_status": "complete"
    }])
    calls = {"search": 0, "generate": 0}
    def fake_search(question, owner_id):
        calls["search"] += 1
        return [{"id": "c1", "text": "Revenue declined because operating costs rose.",
                 "metadata": {"document_id": "d1", "owner_id": owner_id, "filename": "report.pdf", "source_location": "page 1"},
                 "similarity": 0.9}]
    def fake_generate(prompt):
        calls["generate"] += 1
        return "Revenue declined because operating costs rose [1]."
    monkeypatch.setattr(chat, "search", fake_search)
    monkeypatch.setattr(chat, "embed_query", lambda question: [1.0, 0.0])
    monkeypatch.setattr(chat, "generate", fake_generate)
    monkeypatch.setattr(chat, "indexed_vocabulary", lambda: [])

    first = chat._answer_one("Why did revenue decline?", {"d1"}, "owner-a")
    second = chat._answer_one("What caused revenue to fall?", {"d1"}, "owner-a")
    assert first["status"] == "success"
    assert second["status"] == "success"
    assert second.get("answer_cache", {}).get("match") != "semantic"
    # Semantic retrieval may be reused, but a fresh answer is generated for the new wording.
    assert calls["generate"] == 2
