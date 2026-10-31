"""Compatibility wrapper around the existing heuristic conflict detector."""
from app.rag.conflicts import detect_potential_conflicts


def inspect_conflicts(hits: list[dict]) -> dict:
    conflicts = detect_potential_conflicts(hits)
    return {"potential_conflicts": conflicts, "needs_review": bool(conflicts),
            "warning": "Heuristic conflict flags are advisory; passages must be compared directly." if conflicts else None}
