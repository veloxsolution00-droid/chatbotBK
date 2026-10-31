"""Conservative heuristic to flag evidence pairs that deserve conflict review.

This is a warning signal, not a truth detector. Semantic conflict decisions remain
with the grounded answer stage, which must cite both statements and avoid silently
choosing one source.
"""
from __future__ import annotations
import re

_STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "for", "in", "on", "and", "or", "with", "must", "may", "can", "should", "document", "policy"}
_NEGATION = {"not", "never", "no", "cannot", "can't", "prohibited", "forbidden", "disallowed", "without"}
_POSITIVE = {"must", "required", "allowed", "permitted", "approved", "shall", "can"}

def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z0-9-]*", (text or "").lower()) if len(t) > 2 and t not in _STOP}

def detect_potential_conflicts(hits: list[dict], max_pairs: int = 5) -> list[dict]:
    findings = []
    for i, left in enumerate(hits):
        lt = _terms(left.get("text", ""))
        if not lt: continue
        for right in hits[i + 1:]:
            rt = _terms(right.get("text", ""))
            shared = lt & rt
            denominator = max(1, min(len(lt), len(rt)))
            overlap = len(shared) / denominator
            if overlap < 0.55: continue
            ltext, rtext = left.get("text", ""), right.get("text", "")
            ln = bool(_NEGATION & _terms(ltext)); rn = bool(_NEGATION & _terms(rtext))
            lnums = set(re.findall(r"(?<!\w)\d+(?:\.\d+)?%?(?!\w)", ltext))
            rnums = set(re.findall(r"(?<!\w)\d+(?:\.\d+)?%?(?!\w)", rtext))
            reasons = []
            if ln != rn: reasons.append("opposite polarity/negation signal")
            if lnums and rnums and lnums != rnums: reasons.append("different numeric values")
            # Only report a candidate when there is an observable disagreement cue.
            if reasons:
                findings.append({"left_filename": left.get("metadata", {}).get("filename"),
                    "left_location": left.get("metadata", {}).get("source_location"),
                    "right_filename": right.get("metadata", {}).get("filename"),
                    "right_location": right.get("metadata", {}).get("source_location"),
                    "shared_terms": sorted(shared)[:12], "signals": reasons})
                if len(findings) >= max_pairs: return findings
    return findings
