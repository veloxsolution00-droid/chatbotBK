"""Offline evaluation helpers for retrieval and answer regression datasets."""
from __future__ import annotations

def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
    return len(set(retrieved_ids[:max(0, k)]) & relevant_ids) / len(relevant_ids)

def summarize_case_results(results: list[dict]) -> dict:
    """Summarize manually or automatically scored cases; missing labels stay missing."""
    total = len(results)
    scored = [r for r in results if isinstance(r.get("passed"), bool)]
    return {"total_cases": total, "scored_cases": len(scored),
            "passed": sum(r["passed"] for r in scored),
            "pass_rate": (sum(r["passed"] for r in scored) / len(scored)) if scored else None,
            "unscored_cases": total - len(scored)}
