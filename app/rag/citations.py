"""Citation reference validation against the exact evidence supplied to the model."""
from __future__ import annotations
import re

_SENTENCE = re.compile(r"[^.!?]+[.!?]?")
_NUMERIC_REF = re.compile(r"\[\s*\d{1,3}(?:\s*,\s*\d{1,3})*\s*\]")
_ANY_BRACKET = re.compile(r"\[([^\]\n]+)\]")
_FILENAME = re.compile(r"\.(?:pdf|docx?|txt|md|xlsx?|pptx?|csv|html?)\b", re.I)
_BRACKET_SEGMENT = re.compile(r"\[[^\]\n]*\]")
_CITATION = re.compile(
    r"\[\s*\d{1,3}(?:\s*,\s*\d{1,3})*\s*\]"
    r"|\[[^\]\n]*\.(?:pdf|docx?|txt|md|xlsx?|pptx?|csv|html?)[^\]\n]*\]",
    re.I,
)

# Bracketed text that is formatting rather than a source claim.
_BRACKET_ALLOW = {"sic", "note", "table", "page", "figure", "excerpt", "emphasis added"}


def _sentences(text: str) -> list[str]:
    """Split into sentences without breaking inside a bracketed citation.

    A filename citation such as "[policy.pdf, page 2]" contains a full stop that
    is not a sentence boundary.
    """
    text = text or ""
    stashed: dict[str, str] = {}

    def stash(match):
        key = f"\x00{len(stashed)}\x00"
        stashed[key] = match.group(0)
        return key

    masked = _BRACKET_SEGMENT.sub(stash, text)
    parts = [part.strip() for part in _SENTENCE.findall(masked) if part.strip()]
    restored = []
    for part in parts:
        for key, original in stashed.items():
            part = part.replace(key, original)
        restored.append(part)
    return restored


def _content_words(sentence: str) -> int:
    """Count substantive words, ignoring bracketed citations and digits."""
    stripped = _BRACKET_SEGMENT.sub(" ", sentence or "")
    return len(re.findall(r"[A-Za-z]{2,}", stripped))


def validate_citation_references(answer: str, source_count: int, source_labels: list[str] | None = None) -> dict:
    """Validate numeric evidence references [1], [2], etc.; report, do not guess."""
    answer = answer or ""
    # Accept both individual references ([1] [2]) and a comma-separated list
    # ([1, 2]). Small local models often vary this formatting despite the prompt.
    refs = []
    for match in re.finditer(r"\[\s*(\d{1,3}(?:\s*,\s*\d{1,3})*)\s*\]", answer):
        refs.extend(int(value.strip()) for value in match.group(1).split(","))
    invalid = sorted({n for n in refs if n < 1 or n > source_count})
    valid = sorted({n for n in refs if 1 <= n <= source_count})
    labels = source_labels or []
    # Catch explicit filename-style citations that name a source not in the evidence.
    known = {str(x).lower() for x in labels if x}
    filename_cites = re.findall(
        r"(?:\[|\()([^\]\()\n]+\.(?:pdf|docx?|txt|md|xlsx?|pptx?|csv|html?))(?:,|\]|\))",
        answer, flags=re.I)
    unknown_labels = sorted({name.strip() for name in filename_cites if name.strip().lower() not in known})
    # Bracketed text that is neither a numeric reference nor a known filename is
    # a phantom citation: small models sometimes copy editorial markers from the
    # source (for example "[UVAstudstaff]") and present them as sources.
    phantom = []
    for match in _ANY_BRACKET.finditer(answer):
        content = match.group(1).strip()
        if not content or _NUMERIC_REF.fullmatch(match.group(0)):
            continue
        if content.lower() in _BRACKET_ALLOW or _FILENAME.search(content):
            continue
        phantom.append(content)
    phantom = sorted(set(phantom))
    # Report citation density: a long factual sentence with no reference at all
    # is a coverage gap even when the answer contains one valid citation.
    factual = [s for s in _sentences(answer) if _content_words(s) >= 3]
    uncited = [s[:120] for s in factual if not _CITATION.search(s)]
    coverage = round(1 - (len(uncited) / len(factual)), 3) if factual else 0.0
    present = bool(refs or filename_cites)
    return {
        "valid": not invalid and not unknown_labels and not phantom and present,
        "references_found": valid,
        "invalid_reference_numbers": invalid,
        "unknown_source_labels": unknown_labels,
        "phantom_labels": phantom,
        "uncited_sentences": uncited,
        "citation_present": present,
        "citation_coverage": coverage,
    }
