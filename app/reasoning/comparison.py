"""Prompt helper for evidence-balanced comparisons."""

def comparison_instructions() -> str:
    return ("Compare requested items using the same criteria. Cite evidence for each side, "
            "mark missing criteria as 'not found in the supplied documents', and do not treat "
            "missing information as proof that a feature or rule does not exist.")
