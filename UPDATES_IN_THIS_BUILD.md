# Update Notes — Offline Document Chatbot

## Implemented in this build

- Added bounded recent conversation history to `POST /chat` and passed recent user/assistant turns from Streamlit.
- Added local-model follow-up rewriting with a safe fallback to the original question.
- Added deterministic question routing for factual, scenario, comparison, summary, explanation, calculation, and follow-up categories.
- Added conservative query variants and a second retrieval pass when the initial pass returns no evidence above the configured threshold. The original user spelling is also searched when common-typo normalization changes the query.
- Improved vocabulary spelling correction to fix multiple likely typos in one query; preserves acronyms, title-case names, and numeric/hyphenated identifiers, and avoids ambiguous fuzzy matches. The corrected query is searched even if the initial query returned some hits; the original query remains a separate route.
- Preserved hybrid ChromaDB semantic retrieval plus SQLite FTS5 exact-term retrieval; candidate retrieval defaults are `TOP_K=5` and `RETRIEVAL_CANDIDATE_MULTIPLIER=10`.
- Added a safe arithmetic evaluator for explicit arithmetic/percentage expressions using a restricted AST (no Python `eval`). For document-derived calculations, the model proposes a formula/source references; the code checks source indices and verifies numeric operands appear in the question/evidence before evaluating the expression. Formula selection remains a semantic risk.
- Added a configuration guard that rejects non-loopback or HTTPS Ollama endpoints, enforcing the local-only runtime configuration.
- Added numeric/filename citation-reference validation; generated answers with missing or invalid citations are withheld. Source passages remain available in the API/UI.
- Added a conservative potential-conflict detector for retrieved passages with similar terms and differing numbers or polarity signals. This is only a warning signal; the model must inspect the passages and cite both sides.
- Kept grounded instructions for scenario facts, user assumptions, derived conclusions, missing conditions, comparisons, summaries, calculations, and explanations.
- Added offline evaluation metric helpers and an API evaluation runner with configurable case checks.
- Updated the Streamlit diagnostics to show retrieval passes, citation validation warnings, and potential evidence conflicts.
- Reworked chunking to prefer paragraph/heading/sentence boundaries, preserve bounded overlap, and enforce a hard chunk-size cap; added long-paragraph and structure tests.
- Removed the unreferenced `queue_ingest` and `process_document` duplicate ingestion entry points from `app/services/documents.py`; the active durable job queue uses `ingest` directly. See `DEAD_CODE_AUDIT.md`.
- Added `FEATURE_COVERAGE.md` to make implemented, partial, blocked, and not-run requirements explicit.
- Updated README, `.env.example`, and regression tests.
- Added persistent, bounded query-embedding, exact/semantic retrieval, and exact/strict-semantic answer caching; `/health` reports cache entry counts and hit/miss metrics.
- Preserved existing storage and ingestion architecture; no code path was added to clear the user's persistent data at startup.

## Validation performed in this environment

- Python compilation: passed.
- `python -m pytest -q -rs`: **87 passed, 0 skipped** (Python 3.11.9 and the project `.venv`).
- Live runtime checks: FastAPI lifespan startup, `/health`, `/documents`, `/jobs`, cached `/chat`, and real `uvicorn` + Streamlit loopback launch all succeeded against the existing 5-document / 1,785-chunk corpus.
- Deterministic tests cover arithmetic safety, question classification, query variants, multi-typo correction, identifier/name preservation, corrected retrieval despite initial hits, citation validation, conflict signals, retrieval fallback, and existing persistence/ingestion behavior.

## Not claimed as solved

- This is not a formal policy/rule engine and cannot guarantee correct scenario reasoning for every possible case.
- The arithmetic engine safely evaluates expressions. Identifying the right figures/formula from arbitrary documents remains an evidence-selection and interpretation problem; the code validates numeric operands but cannot prove the formula is conceptually correct.
- The conflict detector is heuristic and may flag unrelated numbers; it may also miss semantic contradictions.
- The LLM verifier is another model judgment, not an independent proof of truth.
- The evaluation runner only measures the checks that you define in your own case file; it is not a pre-filled benchmark of your private documents.
- Live Ollama generation/embedding, ChromaDB indexing/retrieval, FastAPI startup, and a loopback Streamlit launch were exercised here. OCR, firewall-level network isolation, and latency/memory profiling were not exercised with representative custom documents and should be validated on the target Windows machine.
- The second UI was not included as a separate artifact in the current `RND` archive, so a true merge of two distinct UI implementations cannot be completed from this archive alone.


## Local caching added

- Added a dedicated SQLite cache at `data/metadata/rag_cache.sqlite3`; this is separate from the permanent document metadata database and ChromaDB index.
- Added query-embedding caching for repeated question embeddings. Document/chunk embeddings are deliberately not put into this query cache.
- Added exact retrieval caching and conservative semantic retrieval reuse. Semantic hits require a high embedding similarity, matching question workflow, matching hard anchors (numbers/IDs/acronyms), matching owner, unchanged indexed-document fingerprint, and unchanged retrieval configuration.
- Added exact answer caching and stricter semantic answer reuse. Only successful, cited, non-abstained document-grounded answers are stored. Answer cache scope includes generation model, temperature, verifier setting, retrieval configuration, and prompt version.
- Cache TTL and entry count are bounded. Cached evidence is filtered against the current indexed owner scope before use.
- Cache never deletes or replaces uploaded documents, extracted text, ChromaDB collections, or the primary metadata database. Clearing `rag_cache.sqlite3` is safe but only clears performance cache data.
- Configuration: `CACHE_ENABLED`, `CACHE_TTL_SECONDS`, `CACHE_MAX_ENTRIES`, `SEMANTIC_RETRIEVAL_THRESHOLD`, `SEMANTIC_ANSWER_THRESHOLD`.
- Limit: semantic matching is a high-confidence heuristic, not a proof of equivalent intent. Semantic answer reuse uses a stricter threshold than retrieval reuse; dates/IDs/acronyms must match. Benchmark answer correctness before lowering thresholds.


## Cache correctness safeguard

Semantic answer reuse is now disabled by default (`SEMANTIC_ANSWER_CACHE_ENABLED=false`). A semantic retrieval-cache hit may reuse candidate chunks, but the application generates a fresh answer for the current question. Exact normalized-question answer hits remain available when corpus/config fingerprints match. This is intentionally conservative: embedding similarity is not proof that two questions request the same fact. Enabling semantic answer reuse should be done only after evaluating a labeled set of paraphrase and hard-negative question pairs.

## Follow-up and citation recovery fixes

- Standalone questions are no longer sent to the follow-up resolver just because history exists. Only elliptical/context-dependent forms such as "What about his role?" are resolved using conversation history.
- Query normalization removes an accidental comma inserted between letters (for example, `bra,m` becomes `bram`) before retrieval. Vocabulary-based correction now starts from the normalized query and uses a slightly more permissive cutoff; the original query is still searched.
- If the first generated answer is an abstention or fails citation-reference validation, the app retries once using the same retrieved evidence and strict numeric citation bounds.
- If citation validation still fails, the status is `citation_validation_failed`. These are no longer mislabeled as a genuine lack of evidence.
- The generation prompt forbids added analysis, interpretation, motives, symbolism, or conclusions, and forbids copying a term from the question that the evidence never uses. The model otherwise padded correct answers with invented commentary and echoed question vocabulary (for example answering "Transylvania" where the evidence says "Carpathian mountains").
- The answer verifier is a reporting guardrail, not a gate. Its verdict is dominated by wording: a draft that only reuses a term supplied by the question is flagged, so a failed verdict marks the answer `verification: {"checked": true, "passed": false}` and the UI warns, instead of failing the request with `answer_validation_failed`.
- The answer verifier prompt is anchored by concrete examples rather than an abstract definition of "supported". With the strict wording the local model rejected grounded, correctly cited drafts (a false negative).
- Generation calls send `keep_alive` so the model stays resident across the closely spaced generate/retry/verify calls of one request. On CPU-only hosts Ollama otherwise unloads the model between calls and each reload is paid as latency.
- `TIMEOUT` is a per-Ollama-call limit and defaults to 300 s. CPU-only inference of a grounded answer can exceed 90 s, so the previous 90 s default aborted valid generations.
- The answer-cache prompt fingerprint was bumped so cached results from the previous answer-generation rules are not reused.
- Added regression tests for standalone-question preservation, elliptical follow-ups, punctuation typo normalization, and citation recovery.

