"""Rank-based fusion that combines independent retrieval routes safely."""
from __future__ import annotations


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    fused: dict[str, dict] = {}
    for results in result_lists:
        for rank, hit in enumerate(results, 1):
            key = str(hit.get("id") or ((hit.get("metadata") or {}).get("document_id"), hit.get("text")))
            if key not in fused:
                fused[key] = {**hit, "fusion_score": 0.0, "fusion_ranks": []}
            fused[key]["fusion_score"] += 1.0 / (max(1, k) + rank)
            fused[key]["fusion_ranks"].append(rank)
            # Keep strongest component scores when duplicate chunk routes differ.
            for score_key in ("similarity", "semantic_similarity", "lexical_similarity"):
                if hit.get(score_key) is not None:
                    old = fused[key].get(score_key)
                    fused[key][score_key] = max(float(old or 0), float(hit[score_key]))
    return sorted(fused.values(), key=lambda h: h["fusion_score"], reverse=True)
