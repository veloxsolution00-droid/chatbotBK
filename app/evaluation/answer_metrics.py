"""Metrics for answer coverage, abstention and citation structure."""
from __future__ import annotations


def answer_coverage(expected_requirements: int, answered_requirements: int) -> float:
    if expected_requirements <= 0:
        return 1.0
    return max(0, min(expected_requirements, answered_requirements)) / expected_requirements


def citation_validity(valid_answers: int, answers_with_citations: int) -> float:
    if answers_with_citations <= 0:
        return 1.0 if valid_answers == 0 else 0.0
    return max(0, min(valid_answers, answers_with_citations)) / answers_with_citations


def abstention_accuracy(correct_abstentions: int, unanswerable_cases: int) -> float:
    if unanswerable_cases <= 0:
        return 1.0
    return max(0, min(correct_abstentions, unanswerable_cases)) / unanswerable_cases
