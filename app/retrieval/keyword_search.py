"""Keyword-ranking adapter over FTS/exact-term scores from the current store."""
from app.retrieval.store import search as _search


def rank_keyword_candidates(candidates: list[dict]) -> list[dict]:
    hits = [dict(h) for h in candidates]
    hits.sort(key=lambda h: float(h.get("lexical_similarity") or 0), reverse=True)
    return hits


def search_keyword(query: str, owner_id: str = "local", limit: int | None = None) -> list[dict]:
    hits = rank_keyword_candidates(_search(query, owner_id=owner_id))
    return hits[:limit] if limit else hits
