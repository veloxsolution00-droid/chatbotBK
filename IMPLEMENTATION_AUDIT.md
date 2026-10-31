# Implementation Audit — Offline Document Chatbot

## Included engineering work

- Existing local-only FastAPI + Streamlit + Ollama + ChromaDB + SQLite architecture retained.
- Chat API accepts a bounded recent history; UI sends prior user/assistant turns for follow-up resolution.
- Question planner routes factual, scenario, comparison, summary, explanation, calculation, and follow-up intents.
- Retrieval retains dense semantic + FTS5 keyword matching; it searches original and normalized query forms, adds conservative query variants after weak retrieval, and can correct likely typos toward terms found in indexed document vocabulary.
- Evidence selection filters to indexed documents and uses multiple chunks within a configured context limit.
- Answer prompts treat uploaded content as untrusted data, separate scenario facts/assumptions/conclusions, request citations, and instruct the model to report document conflicts instead of silently selecting one.
- Optional local-model answer verification retained.
- Citation validation checks numeric reference bounds, explicit filename references, and missing citations; invalid answers are withheld.
- Potential conflict heuristics identify likely differing numbers or polarity signals for model review.
- Explicit arithmetic uses a restricted AST evaluator. For evidence-based calculations, the local model proposes a formula and source references; source indices and numeric operands are checked before deterministic evaluation.
- Runtime Ollama endpoint is constrained to plain HTTP loopback hosts.
- Added structure-aware paragraph/heading/sentence chunking with bounded overlap and regression tests. This is heuristic chunking, not embedding-based semantic segmentation.
- Added retrieval metrics helpers, a local API evaluation runner, sample evaluation cases, regression tests, and updated documentation.
- Added `FEATURE_COVERAGE.md`, a requirement-by-requirement matrix that explicitly marks partial, blocked, and not-run items.
- Added a bounded SQLite cache for query embeddings, exact/semantic retrieval results, and successful cited answers; cache scope includes owner, corpus fingerprint, and relevant model/config fingerprints.
- Existing ingestion, persistence, recovery, and data folders retained. No startup data-clearing behavior was introduced.

## Verification

- `python -m compileall -q app tests evaluation`: passed.
- `python -m pytest -q -rs`: **67 passed, 1 skipped**.
- Chat API request schema smoke test passed with a lightweight ChromaDB import stub.

## Limits requiring real-machine validation

- This environment does not have the ChromaDB package installed, so live ChromaDB operations were not exercised here.
- No live Ollama model call, real document upload/indexing, OCR validation, or performance test was run.
- The single skipped test remains skipped; inspect its condition in the test suite before deployment.
- Structure-aware chunking is not embedding-based semantic segmentation; ranking is not a calibrated confidence model.
- Scenario reasoning is not a formal policy engine. LLM-based formula selection can still be wrong even when numeric operands and arithmetic syntax validate.
- Citation checks prove reference bounds, not semantic entailment. Model verification is not independent proof.
- Conflict detection is heuristic and can both over-flag and miss contradictions.
- The evaluation case file is a template; it must be populated with ground truth from your own documents.
- Only one UI implementation was present in the supplied archive. A true merge of two distinct UIs requires the other UI source as well.
