"""Minimal stable schema for internal answer/evidence status records."""
from typing import Any, TypedDict


class SourceRecord(TypedDict, total=False):
    rank: int
    filename: str | None
    source_location: str | None
    relevance_score: float
    supporting_text: str


class AnswerRecord(TypedDict, total=False):
    question: str
    status: str
    answer: str
    sources: list[SourceRecord]
    abstained: bool
    question_type: str
    retrieval: dict[str, Any]
