import pytest
from app.rag.calculations import evaluate_expression, direct_calculation
from app.rag.citations import validate_citation_references
from app.rag.evaluation import reciprocal_rank, recall_at_k, summarize_case_results
from app.rag.planner import classify_question, query_variants

@pytest.mark.parametrize(("expr", "expected"), [("2 + 3 * 4", 14), ("(100 - 25) / 3", 25), ("2 ** 3", 8)])
def test_safe_arithmetic(expr, expected):
    assert evaluate_expression(expr) == expected

def test_arithmetic_rejects_code_and_division_by_zero():
    with pytest.raises(ValueError): evaluate_expression("__import__('os').system('whoami')")
    with pytest.raises(ValueError, match="zero"): evaluate_expression("1/0")

def test_direct_calculation_only_uses_explicit_user_values():
    assert direct_calculation("Calculate 12 * 4")['result'] == 48
    assert direct_calculation("What is 15% of 200?")['result'] == 30
    assert direct_calculation("How much budget is allowed?") is None

def test_planner_classifies_scenario_comparison_and_followup():
    assert classify_question("Suppose a contractor needs access") == "scenario"
    assert classify_question("Compare policy A and policy B") == "comparison"
    assert classify_question("What about contractors?") == "followup"
    assert classify_question("What is 15% of 200?") == "calculation"

def test_query_variants_are_conservative_and_bounded():
    variants = query_variants("What does the document say about manager approval?")
    assert len(variants) <= 3
    assert variants[0].startswith("What does")
    assert any("manager" in v.lower() for v in variants)

def test_citation_validator_accepts_valid_and_rejects_unknown_refs():
    assert validate_citation_references("Approval is required [1].", 1)["valid"]
    result = validate_citation_references("Approval is required [3].", 1)
    assert not result["valid"] and result["invalid_reference_numbers"] == [3]
    assert not validate_citation_references("Uncited answer", 1)["valid"]

def test_evaluation_metrics():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert recall_at_k(["a", "b"], {"b", "c"}, 2) == 0.5
    assert summarize_case_results([{"passed": True}, {"passed": False}, {}]) == {
        "total_cases": 3, "scored_cases": 2, "passed": 1, "pass_rate": 0.5, "unscored_cases": 1}

def test_conflict_detector_flags_different_values_for_overlapping_claims():
    from app.rag.conflicts import detect_potential_conflicts
    hits = [
        {"text": "Contractors must receive approval within 5 days.", "metadata": {"filename": "policy-a.pdf", "source_location": "page 2"}},
        {"text": "Contractors must receive approval within 10 days.", "metadata": {"filename": "policy-b.pdf", "source_location": "page 4"}},
    ]
    findings = detect_potential_conflicts(hits)
    assert findings and "different numeric values" in findings[0]["signals"]

def test_vocabulary_spelling_correction_preserves_original_query_route():
    from app.rag.planner import correct_query_from_vocabulary
    assert correct_query_from_vocabulary("contractor authoriztion", ["contractor", "authorization"]) == "contractor authorization"
    assert correct_query_from_vocabulary("invoice INV-2026-0042", ["invoice", "approval"]) == "invoice INV-2026-0042"

def test_evidence_calculation_plan_only_uses_numbers_present_in_question_or_evidence():
    import json
    from app.rag.calculations import apply_evidence_calculation_plan
    valid = apply_evidence_calculation_plan(json.dumps({"expression": "50 * 3", "source_refs": [1], "unit": "dollars"}),
                                            "Unit cost is 50 dollars. Quantity is 3.", "Calculate total cost", 1)
    assert valid and valid["result"] == 150
    invalid = apply_evidence_calculation_plan(json.dumps({"expression": "50 * 300", "source_refs": [1]}),
                                              "Unit cost is 50 dollars. Quantity is 3.", "Calculate total cost", 1)
    assert invalid is None

def test_offline_configuration_rejects_non_loopback_ollama_endpoints():
    from app.core.config import validate_loopback_url
    assert validate_loopback_url("http://127.0.0.1:11434")
    assert validate_loopback_url("http://localhost:11434")
    assert validate_loopback_url("http://[::1]:11434")
    assert not validate_loopback_url("https://127.0.0.1:11434")
    assert not validate_loopback_url("http://example.com:11434")


def test_chunker_prefers_paragraph_boundaries_and_respects_size():
    from app.retrieval.chunker import chunk_text
    text = "# Policy\nApproval is required for all contractors.\n\nRetention lasts seven years.\n\nExceptions require written approval."
    chunks = chunk_text(text, chunk_size=70, overlap=10)
    assert chunks
    assert all(0 < len(chunk) <= 70 for chunk in chunks)
    assert any("Approval is required" in chunk for chunk in chunks)
    assert any("Exceptions require written approval" in chunk for chunk in chunks)


def test_chunker_handles_very_long_single_paragraph_with_progress():
    from app.retrieval.chunker import chunk_text
    text = ("sentence about document evidence. " * 300)
    chunks = chunk_text(text, chunk_size=180, overlap=40)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 180 for chunk in chunks)


def test_vocabulary_spelling_correction_fixes_multiple_typos_in_one_query():
    from app.rag.planner import correct_query_from_vocabulary
    corrected = correct_query_from_vocabulary(
        "contractor authoriztion proces needs aproval",
        ["contractor", "authorization", "process", "approval", "invoice"],
    )
    assert corrected == "contractor authorization process needs approval"


def test_vocabulary_spelling_correction_preserves_names_ids_and_acronyms():
    from app.rag.planner import correct_query_from_vocabulary
    question = "What does ACME need for INV-2026-0042 and Jon Smyth?"
    corrected = correct_query_from_vocabulary(
        question, ["document", "invoice", "approval", "smith", "authorization"]
    )
    assert "ACME" in corrected
    assert "INV-2026-0042" in corrected
    assert "Jon Smyth" in corrected


def test_vocabulary_spelling_correction_does_not_change_words_already_in_vocabulary():
    from app.rag.planner import correct_query_from_vocabulary
    question = "process approval authorization"
    assert correct_query_from_vocabulary(question, ["process", "approval", "authorization"]) == question
