def test_long_compound_question_is_decomposed_into_requirements():
    from app.question.query_planner import plan_question
    question = ("Explain the complete hiring and onboarding process described in the documents. "
                "Identify who approves a new position, what documents HR must collect, which background checks are required, "
                "how long each stage takes, and what happens if a candidate fails verification. "
                "Compare the process across departments and identify any contradictions between the documents. "
                "Finally, explain which steps are mandatory and which are optional, citing the source document and page for every conclusion.")
    plan = plan_question(question, max_parts=10)
    assert plan["is_long"] is True
    assert plan["decomposed"] is True
    assert 2 <= len(plan["subquestions"]) <= 10
    assert any("Compare" in q or "compare" in q for q in plan["subquestions"])


def test_simple_question_is_not_needlessly_decomposed():
    from app.question.query_planner import plan_question
    plan = plan_question("Who approved the policy?")
    assert plan["subquestions"] == ["Who approved the policy?"]
    assert not plan["decomposed"]


def test_rank_candidates_combines_dense_and_keyword_signals():
    from app.retrieval.pipeline import rank_candidates
    candidates = [
        {"id": "dense", "text": "The policy explains approvals", "similarity": .8,
         "semantic_similarity": .9, "lexical_similarity": .1, "metadata": {}},
        {"id": "keyword", "text": "Approval deadline is 10 days", "similarity": .7,
         "semantic_similarity": .5, "lexical_similarity": .9, "metadata": {}},
    ]
    ranked = rank_candidates("approval deadline", candidates)
    assert {h["id"] for h in ranked} == {"dense", "keyword"}
    assert all("rerank_score" in h for h in ranked)


def test_evidence_selector_reports_coverage_and_missing_terms():
    from app.retrieval.evidence_selector import select_evidence
    selected = select_evidence("manager approves travel budget", [{"id": "1", "text": "The manager approves travel", "metadata": {}}])
    assert selected["evidence_found"]
    assert "budget" in selected["missing_terms"]


def test_answer_coverage_metric():
    from app.evaluation.answer_metrics import answer_coverage
    assert answer_coverage(4, 3) == .75
    assert answer_coverage(0, 0) == 1.0


def test_chat_runs_each_decomposed_requirement(monkeypatch):
    import sys, types
    monkeypatch.setitem(sys.modules, "chromadb", types.ModuleType("chromadb"))
    import app.rag.chat as chat
    # This test exercises decomposition capacity, so it must not depend on the
    # deployment's MAX_QUESTION_PARTS cap.
    monkeypatch.setattr(chat.settings, "max_question_parts", 10)
    monkeypatch.setattr(chat, "list_docs", lambda owner_id: [
        {"document_id": "d1", "processing_status": "indexed", "indexing_status": "complete"}
    ])
    seen = []
    def fake_answer_one(question, indexed, owner_id):
        seen.append(question)
        return {"question": question, "status": "success", "answer": f"Evidence found for {question} [1].",
                "sources": [{"rank": 1, "filename": "policy.pdf", "source_location": "page 1", "supporting_text": question}],
                "abstained": False}
    monkeypatch.setattr(chat, "_answer_one", fake_answer_one)
    question = ("Explain the complete hiring and onboarding process described in the documents. "
                "Identify who approves a new position, what documents HR must collect, which background checks are required, "
                "how long each stage takes, and what happens if a candidate fails verification. "
                "Compare the process across departments and identify any contradictions between the documents. "
                "Finally, explain which steps are mandatory and which are optional, citing the source document and page for every conclusion.")
    result = chat.answer(question)
    assert len(seen) >= 8
    assert result["coverage"]["total_requirements"] == len(seen)
    assert result["coverage"]["coverage_ratio"] == 1.0
    assert result["question_plan"]["decomposed"]
