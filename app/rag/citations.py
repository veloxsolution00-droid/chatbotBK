"""Citation reference validation against the exact evidence supplied to the model."""
from __future__ import annotations
import re

def validate_citation_references(answer: str, source_count: int, source_labels: list[str] | None = None) -> dict:
    """Validate numeric evidence references [1], [2], etc.; report, do not guess."""
    # Accept both individual references ([1] [2]) and a comma-separated list
    # ([1, 2]). Small local models often vary this formatting despite the prompt.
    refs = []
    for match in re.finditer(r"\[\s*(\d{1,3}(?:\s*,\s*\d{1,3})*)\s*\]", answer or ""):
        refs.extend(int(value.strip()) for value in match.group(1).split(","))
    invalid = sorted({n for n in refs if n < 1 or n > source_count})
    valid = sorted({n for n in refs if 1 <= n <= source_count})
    labels = source_labels or []
    # Catch explicit filename-style citations that name a source not in the evidence.
    known = {str(x).lower() for x in labels if x}
    filename_cites = re.findall(r"(?:\[|\()([^\]\()\n]+\.(?:pdf|docx?|txt|md|xlsx?|pptx?|csv|html?))(?:,|\]|\))", answer or "", flags=re.I)
    unknown_labels = sorted({name.strip() for name in filename_cites if name.strip().lower() not in known})
    present = bool(refs or filename_cites)
    return {"valid": not invalid and not unknown_labels and present, "references_found": valid,
            "invalid_reference_numbers": invalid, "unknown_source_labels": unknown_labels,
            "citation_present": present}
