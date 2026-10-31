"""In-memory retrieval orchestration that reuses the existing indexed store.

The store remains the single source of candidates and persistent indexes. This
pipeline applies dense/keyword views, reciprocal-rank fusion, reranking and
bounded evidence selection without re-embedding a query for each stage.
"""
from app.retrieval.dense_search import rank_dense_candidates
from app.retrieval.keyword_search import rank_keyword_candidates
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank


def rank_candidates(query: str, candidates: list[dict], limit: int | None = None) -> list[dict]:
    dense = rank_dense_candidates(candidates)
    keyword = rank_keyword_candidates(candidates)
    fused = reciprocal_rank_fusion([dense, keyword, candidates])
    return rerank(query, fused, limit=limit)
