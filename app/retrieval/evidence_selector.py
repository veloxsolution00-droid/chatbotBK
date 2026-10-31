"""Evidence selection with per-requirement lexical coverage accounting."""
import re

_STOP = {"the", "and", "for", "with", "what", "when", "where", "which", "who", "does", "from", "that", "this", "are", "was", "how", "about", "into", "you", "document", "documents"}


def select_evidence(requirement: str, hits: list[dict], max_chunks: int = 5, max_chars: int = 9000) -> dict:
    terms = {t.lower() for t in re.findall(r"[\w-]+", requirement) if len(t) > 2 and t.lower() not in _STOP}
    selected, seen, chars = [], set(), 0
    for hit in hits:
        key = hit.get("id") or (hit.get("metadata", {}).get("document_id"), hit.get("text"))
        if key in seen:
            continue
        if selected and chars + len(hit.get("text", "")) > max_chars:
            continue
        selected.append(hit); seen.add(key); chars += len(hit.get("text", ""))
        if len(selected) >= max(1, max_chunks):
            break
    covered = set()
    for hit in selected:
        covered.update(t.lower() for t in re.findall(r"[\w-]+", hit.get("text", "")))
    missing = sorted(terms - covered)
    return {"hits": selected, "required_terms": sorted(terms), "covered_terms": sorted(terms & covered),
            "missing_terms": missing, "term_coverage": round(len(terms & covered) / max(1, len(terms)), 4),
            "evidence_found": bool(selected)}
