# Local Document Chatbot: Improvements and Accuracy Guide

## What changed

- Retrieval normalizes common input typos without removing spaces from questions.
- Explicit multi-question messages are answered one question at a time.
- Recent user/assistant turns can be passed to the backend; an Ollama rewrite step attempts to turn follow-up questions into standalone retrieval queries.
- Lightweight question classification selects specific grounding guidance for scenarios, comparisons, summaries, explanations, calculations, and factual questions. Scenario guidance separates document facts, user assumptions, and derived conclusions.
- Chunks use bounded overlap and natural sentence/paragraph boundaries, avoiding pathological work on large files.
- CSV and Excel extraction is streamed in row blocks; document processing runs in a durable background queue.
- Images and scanned PDF pages use local Tesseract OCR. Blank or unreadable uploads report a clear no-readable-text failure instead of being indexed.
- ChromaDB semantic search is supplemented by SQLite FTS5 exact-term search. This improves retrieval of names, IDs, dates, and values.
- Every response includes source filename and source location. A configurable verifier asks the local model to reject drafts unsupported by retrieved evidence.

## Accuracy settings

Use these defaults first:

```env
TOP_K=5
RETRIEVAL_CANDIDATE_MULTIPLIER=10
HYBRID_LEXICAL_WEIGHT=0.20
LEXICAL_CANDIDATE_LIMIT=20
MIN_SIMILARITY=0.25
VERIFY_ANSWERS=true
```

Raise `TOP_K` cautiously for questions that require several sections. Lower `MIN_SIMILARITY` only after testing with real documents, because it increases the chance of weakly related evidence. Set `VERIFY_ANSWERS=false` only when faster responses are worth the reduced guardrail.

## How to measure accuracy

Create a small set of representative questions for each document, with expected facts and source locations. Include scenario tests with known assumptions and expected rule application. Include:

1. Direct factual questions.
2. Questions requiring multiple chunks.
3. Names, dates, IDs, and numbers.
4. Misspelled questions.
5. Questions with no answer in any uploaded document.
6. Multi-question messages.
7. Blank images and scanned PDFs.
8. Follow-ups that rely on earlier turns (for example, “What about contractors?”).
9. Hypothetical scenarios with both sufficient and missing policy conditions.
10. Conflicting statements across documents and questions with no answer in any document.

For each result, record whether the answer is correct, whether its citation supports it, whether an unsupported question abstains, and the response time. Tune the settings above against those results instead of relying on subjective impressions.

## Operating boundary

This remains a single-user, local-only application. Keep both services bound to `127.0.0.1`. It does not use authentication by design.
