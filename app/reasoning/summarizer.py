"""Prompt helper for summaries that preserve qualifications and source traceability."""

def summarization_instructions() -> str:
    return ("Summarize only the retrieved document evidence. Group related points, preserve "
            "exceptions, dates, thresholds, and qualifications, and cite each factual claim. "
            "State explicitly when a requested section is not supported by the retrieved evidence.")
