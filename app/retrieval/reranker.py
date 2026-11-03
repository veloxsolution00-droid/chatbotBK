"""Lightweight local reranker using retrieval relevance and query-term coverage."""
import re

_STOP = {"the", "and", "for", "with", "what", "when", "where", "which", "who", "does", "from", "that", "this", "are", "was", "how", "about", "into", "you"}

# Editorial front/back matter is frequently the closest lexical match for a
# "who is X" question while containing no answer. Down-weight it, but keep it
# available when nothing better exists (it is often the only source of titles).
_BOILERPLATE = re.compile(
    r"(about this edition|table of contents|^\s*contents\s*$|copyright|isbn|"
    r"transcription, correction|editorial commentary|\[UVAstudstaff\]|"
    r"^\s*glossary\s*$|^\s*index\s*$)",
    re.I | re.M,
)


def _boilerplate(text: str) -> bool:
    return bool(_BOILERPLATE.search(text or ""))


def rerank(query: str, hits: list[dict], limit: int | None = None) -> list[dict]:
    terms = {t.lower() for t in re.findall(r"[\w-]+", query) if len(t) > 2 and t.lower() not in _STOP}
    max_fusion = max((float(h.get("fusion_score") or 0) for h in hits), default=0.0)
    ranked = []
    for hit in hits:
        text_terms = {t.lower() for t in re.findall(r"[\w-]+", hit.get("text", ""))}
        coverage = len(terms & text_terms) / max(1, len(terms))
        # Keep the store's hybrid relevance as the dominant signal. RRF only
        # breaks ties/combines routes; it must not overwhelm a strong similarity.
        similarity = max(0.0, min(1.0, float(hit.get("similarity", 0) or 0)))
        fusion = float(hit.get("fusion_score") or 0) / max_fusion if max_fusion else 0.0
        item = dict(hit)
        item["query_term_coverage"] = round(coverage, 4)
        item["rerank_score"] = 0.65 * similarity + 0.20 * coverage + 0.15 * fusion
        if _boilerplate(hit.get("text", "")):
            item["rerank_score"] *= 0.72
            item["boilerplate"] = True
        ranked.append(item)
    ranked.sort(key=lambda h: h["rerank_score"], reverse=True)
    return ranked[:limit] if limit else ranked
