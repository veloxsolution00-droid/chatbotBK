"""Retrieval metrics for labeled query-to-evidence datasets."""
from app.rag.evaluation import recall_at_k, reciprocal_rank


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    items = retrieved_ids[:max(0, k)]
    return len(set(items) & relevant_ids) / len(items) if items else 0.0


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return float(bool(set(retrieved_ids[:max(0, k)]) & relevant_ids))

__all__ = ["recall_at_k", "reciprocal_rank", "precision_at_k", "hit_rate_at_k"]
