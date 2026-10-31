"""Bounded, testable decomposition for long multi-part questions.

The planner deliberately uses conservative deterministic boundaries rather than
asking an LLM to invent subquestions. The original question is retained in the
plan for traceability; each subquestion is retrieved independently downstream.
"""
from __future__ import annotations
import re
from app.question.classifier import classify_question

def _dedupe(parts: list[str]) -> list[str]:
    out, seen = [], set()
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip(" \t\r\n-:;,.")
        key = part.casefold()
        if len(part.split()) >= 3 and key not in seen:
            out.append(part)
            seen.add(key)
    return out


def _explicit_parts(question: str) -> list[str]:
    # Numbered/bulleted requirements and actual question boundaries are reliable.
    if re.search(r"\n\s*(?:[-*]|\d+[.)])\s+", question):
        return _dedupe(re.split(r"\n\s*(?:[-*]|\d+[.)])\s+", question))
    if question.count("?") >= 2:
        return _dedupe(re.split(r"\?+\s*", question))
    if ";" in question:
        return _dedupe(re.split(r";\s*", question))
    return []


def _long_compound_parts(question: str) -> list[str]:
    if len(question.split()) < 38:
        return []
    # Split into sentence-level tasks first; this avoids breaking a clause at
    # every occurrence of "who/what/how" and accidentally dropping its context.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", question) if s.strip()]
    if len(sentences) < 2:
        return []
    context = sentences[0].rstrip(" .!?;:")
    fragments: list[str] = [sentences[0].rstrip(" .!?;:")]
    for sentence in sentences[1:]:
        sentence = re.sub(r"^(?:finally|also|then|next),?\s+", "", sentence, flags=re.I)
        # Split parallel requirements in an explicit command sentence, e.g.
        # "Identify who..., what..., which..., how..., and what happens...".
        command_match = re.match(r"^(Identify|List|Explain|Describe|Specify|Determine|State)\s+", sentence, flags=re.I)
        if command_match:
            command = command_match.group(1)
            body = sentence[command_match.end():]
            clauses = re.split(r",\s*(?=(?:and\s+)?(?:what|which|who|when|where|how|whether)\b)", body, flags=re.I)
            clauses = [c.strip(" ,;.") for c in clauses if c.strip(" ,;.")]
            if len(clauses) > 1:
                # Compute the stripped clause first: a backslash inside an
                # f-string expression is a SyntaxError on Python < 3.12.
                stripped = [re.sub(r"^(?:and|or)\s+", "", clause, flags=re.I) for clause in clauses]
                fragments.extend(f"{command} {clause}" for clause in stripped)
                continue
        # Split separate coordinated actions, but not ordinary "and" phrases.
        coordinated = re.split(r"\s+and\s+(?=(?:identify|list|explain|compare|contrast|calculate|summari[sz]e|describe|determine|state)\b)", sentence, flags=re.I)
        if len(coordinated) > 1:
            fragments.extend(c.strip(" ,;.") for c in coordinated if c.strip(" ,;."))
        else:
            fragments.append(sentence.rstrip(" .!?;:"))
    fragments = _dedupe(fragments)
    if len(fragments) < 2:
        return []
    # Keep each requirement independently searchable while preserving a short
    # topic anchor. The full original is also returned in the plan for auditing.
    result = []
    for i, fragment in enumerate(fragments):
        if i == 0:
            result.append(fragment)
        else:
            result.append(f"Context: {context}. Requirement: {fragment}")
    return result


def plan_question(question: str, max_parts: int = 10) -> dict:
    original = re.sub(r"\s+", " ", (question or "")).strip()
    classification = classify_question(original)
    parts = _explicit_parts(question or "")
    if not parts:
        parts = _long_compound_parts(original)
    parts = _dedupe(parts)[:max(1, int(max_parts))]
    if not parts:
        parts = [original] if original else []
    return {
        "original_question": original,
        "intent": classification["intent"],
        "is_long": classification["is_long"],
        "word_count": classification["word_count"],
        "subquestions": parts,
        "decomposed": len(parts) > 1,
        "planned_parts": len(parts),
        "truncated_to_limit": len(_dedupe(_explicit_parts(question or "") or _long_compound_parts(original))) > max(1, int(max_parts)),
    }
