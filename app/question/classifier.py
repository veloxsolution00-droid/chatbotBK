"""Deterministic intent classification adapters for the local RAG pipeline."""
from app.rag.planner import classify_question as _classify


def classify_question(question: str) -> dict:
    """Return a stable structured intent label; this is routing, not evidence."""
    intent = _classify(question or "")
    words = len((question or "").split())
    return {
        "intent": intent,
        "is_long": words >= 45,
        "word_count": words,
        "requires_multi_evidence": intent in {"comparison", "scenario", "summary"} or words >= 45,
    }
