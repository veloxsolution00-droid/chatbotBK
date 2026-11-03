"""Deterministic question planning helpers for document-grounded workflows.

The planner routes work; it is not a source of facts. Keep it deterministic so
classification and retrieval fallbacks can be unit-tested without Ollama.
"""
from __future__ import annotations
import re
from difflib import get_close_matches

INTENT_TERMS = {
    "scenario": ("what if", "suppose", "scenario", "assume", "if i ", "if we ", "if they ", "would happen", "under the condition", "in the event"),
    "comparison": ("compare", "comparison", "difference between", "versus", " vs ", "which is better", "contrast"),
    "summary": ("summarize", "summarise", "summary", "overview", "key points", "main points", "tl;dr"),
    "calculation": ("calculate", "calculation", "how many", "how much", "total", "percentage", "average", "sum of", "multiply", "divide", "subtract"),
    "explanation": ("why", "how does", "how do", "explain", "reason", "mechanism"),
    "followup": ("what about", "and what about", "how about", "what then", "why is that", "how so", "it", "they", "them", "that"),
}

def classify_question(question: str) -> str:
    q = re.sub(r"\s+", " ", (question or "").lower()).strip()
    # Explicit arithmetic/percentage syntax is a strong deterministic signal.
    if re.search(r"(?i)\b\d+(?:\.\d+)?\s*%\s+of\s+\d+(?:\.\d+)?", q):
        return "calculation"
    if re.search(r"(?i)\b(calculate|compute|evaluate)\b", q) and re.search(r"\d.*[+*/%()-].*\d", q):
        return "calculation"
    # Prefer specific task types; 'how much' can be a factual question but usually asks a quantity.
    for intent in ("scenario", "comparison", "summary", "calculation", "explanation"):
        if any(term in q for term in INTENT_TERMS[intent]):
            return intent
    if len(q.split()) <= 7 and any(q.startswith(term) for term in ("what about", "and ", "how about", "why is that", "how so")):
        return "followup"
    return "factual"

def query_variants(question: str) -> list[str]:
    """Return conservative lexical variants; never invent entities or facts."""
    q = re.sub(r"\s+", " ", (question or "")).strip()
    variants = [q]
    stripped = re.sub(r"^(please\s+)?(tell me|explain|summarize|summarise|what does the document say about|according to the document|according to the documents|in the document|in the documents)\s+", "", q, flags=re.I).strip(" ?.:;,")
    if stripped and stripped.lower() != q.lower() and len(stripped) >= 3:
        variants.append(stripped)
    # A compact content-word query helps lexical retrieval when the original is verbose.
    terms = re.findall(r"[\w.-]+", q, flags=re.UNICODE)
    stop = {"what", "when", "where", "which", "who", "does", "do", "did", "is", "are", "was", "were", "the", "a", "an", "to", "of", "in", "on", "for", "with", "and", "or", "please", "tell", "me", "about", "according", "document", "documents"}
    content = [t for t in terms if t.lower() not in stop]
    compact = " ".join(content[:12])
    if len(content) >= 2 and compact.lower() not in {x.lower() for x in variants}:
        variants.append(compact)
    return variants[:3]

def task_guidance(intent: str) -> str:
    return {
        "scenario": ("Use a structured scenario analysis: list documented rules with citations; list assumptions supplied by the user; "
                     "apply only rules that actually cover the scenario; identify missing conditions, exceptions, and missing rules; "
                     "state the conclusion and its limits. Do not invent a policy or infer permission from silence. "
                     "If the documents do not cover this scenario, state that clearly."),
        "comparison": "Compare each requested item using the same criteria. Cite evidence for each side; explicitly mark criteria with missing evidence. If items are missing from documents, do not invent details.",
        "summary": "Synthesize the important points from all relevant evidence, grouped by theme. Preserve exceptions and cite sources. If no relevant info exists, state that clearly.",
        "calculation": "Prefer deterministic calculation when the question contains a directly evaluable expression. Otherwise identify inputs, formula, units, and assumptions; do not invent missing inputs.",
        "explanation": "Explain only mechanisms and reasons supported by evidence. Label inferences and distinguish them from explicit statements. If the evidence is insufficient to explain, say so.",
        "followup": "Resolve the short follow-up from the supplied conversation context, then answer only from retrieved document evidence. If the evidence doesn't answer it, abstain.",
        "factual": "Answer directly using only the evidence. If the information is not in the documents, you must state that you cannot find it.",
    }.get(intent, "Answer directly using only the evidence. If not found, say so.")


def correct_query_from_vocabulary(question: str, vocabulary: list[str], cutoff: float = 0.86) -> str:
    """Correct multiple likely typos using terms in the local document index.

    This creates an *additional retrieval query*; callers must also search the
    original question. To avoid corrupting names and identifiers, only alphabetic
    words are eligible; capitalised words need a stricter match and keep their
    capitalisation. Ambiguous fuzzy matches are left unchanged.
    """
    if not isinstance(question, str) or not question or not vocabulary:
        return question
    vocab = {term.lower() for term in vocabulary
             if isinstance(term, str) and term.isalpha() and len(term) >= 3}
    if not vocab:
        return question
    stop = {
        "what", "when", "where", "which", "who", "whom", "whose", "does", "do", "did",
        "have", "has", "had", "with", "from", "that", "this", "there", "their", "about",
        "would", "could", "should", "document", "documents", "please", "tell", "show",
        "your", "you", "into", "than", "then", "them", "they", "were", "will", "shall",
        "must", "can", "may", "might", "not", "and", "the", "for", "are", "was", "but",
        "all", "any", "how", "why", "who", "its", "our", "out", "use", "used", "using",
    }

    def replace(match):
        token = match.group(0)
        low = token.lower()
        # Never fuzz identifiers, numbers, acronyms or short/common words. Keep
        # punctuation and original token boundaries.
        if (len(low) < 5 or not token.isalpha()
                or low in stop or low in vocab):
            return token
        # Capitalised tokens are usually proper nouns. Correct them too, but only
        # on a stricter match, and preserve the original capitalisation.
        capitalised = token[0].isupper() and not token.isupper()
        threshold = max(cutoff, 0.88) if capitalised else cutoff
        matches = get_close_matches(low, vocab, n=2, cutoff=threshold)
        if not matches:
            return token
        # If the two best candidates are too close in spelling distance, avoid a
        # speculative replacement. This is deliberately conservative.
        if len(matches) > 1:
            from difflib import SequenceMatcher
            best = SequenceMatcher(None, low, matches[0]).ratio()
            runner_up = SequenceMatcher(None, low, matches[1]).ratio()
            if best - runner_up < 0.04:
                return token
        return matches[0].capitalize() if capitalised else matches[0]

    return re.sub(r"\b[\w]+\b", replace, question)
