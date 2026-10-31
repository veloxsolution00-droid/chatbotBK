"""Public answer validation adapter; syntax validation is not entailment proof."""
from app.rag.citations import validate_citation_references


def validate_answer(answer: str, source_count: int, source_labels: list[str] | None = None) -> dict:
    return validate_citation_references(answer, source_count, source_labels)
