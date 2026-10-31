"""Prompt-building helper for grounded scenario analysis."""

def scenario_instructions() -> str:
    return ("For scenario questions, separate (1) explicit document facts with citations, "
            "(2) assumptions explicitly supplied by the user, (3) conditional conclusions, "
            "and (4) missing conditions or evidence. Never infer permission from silence.")
