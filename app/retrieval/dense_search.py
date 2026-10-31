"""Dense-ranking adapter over the existing hybrid candidate retriever.

The underlying store currently retrieves a shared candidate pool; this adapter
orders that pool by semantic similarity without changing ChromaDB storage.
"""
from app.retrieval.store import search as _search


def rank_dense_candidates(candidates: list[dict]) -> list[dict]:
    hits = [dict(h) for h in candidates if h.get("semantic_similarity") is not None]
    hits.sort(key=lambda h: float(h.get("semantic_similarity") or 0), reverse=True)
    return hits


def search_dense(query: str, owner_id: str = "local", limit: int | None = None) -> list[dict]:
    hits = rank_dense_candidates(_search(query, owner_id=owner_id))
    return hits[:limit] if limit else hits
