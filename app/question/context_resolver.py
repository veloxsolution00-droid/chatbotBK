"""Conservative follow-up resolution using only recent conversation context."""
import re


def clean_history(history, limit: int = 8, max_chars: int = 2000) -> list[dict]:
    cleaned = []
    if not isinstance(history, list):
        return cleaned
    for item in history[-limit:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            cleaned.append({"role": item["role"], "content": re.sub(r"\s+", " ", content).strip()[:max_chars]})
    return cleaned


def build_resolution_prompt(question: str, history: list[dict]) -> str:
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in clean_history(history))
    return ("Rewrite CURRENT QUESTION as a standalone document-search question using only conversation context. "
            "Do not answer, add facts, or assume unstated details. If context does not help, return the current question unchanged. "
            "Return only the rewritten question.\nCONVERSATION:\n" + transcript +
            "\nCURRENT QUESTION: " + question + "\nSTANDALONE QUESTION:")
