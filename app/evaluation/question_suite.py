"""Load and validate a labeled question suite for repeatable local evaluation."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def load_question_suite(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases", data) if isinstance(data, dict) else data
    if not isinstance(cases, list):
        raise ValueError("Question suite must be a JSON list or an object containing 'cases'.")
    required = {"id", "question"}
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not required.issubset(case):
            raise ValueError(f"Case {index} must be an object with id and question fields.")
    return cases


def score_question_case(case: dict, response: dict) -> dict:
    """Score observable structure only; semantic correctness needs human labels."""
    expected_parts = int(case.get("expected_parts", 1))
    coverage = response.get("coverage") or {}
    actual_parts = int(coverage.get("total_requirements", len(response.get("answers", [])) or 1))
    return {
        "id": case.get("id"),
        "expected_parts": expected_parts,
        "actual_parts": actual_parts,
        "part_count_match": actual_parts == expected_parts,
        "citation_validation_present": bool(response.get("citation_validation") or any(a.get("citation_validation") for a in response.get("answers", []))),
        "coverage": coverage.get("coverage_ratio"),
        "status": response.get("status"),
        "semantic_correctness": case.get("human_score"),
    }
