# Question and retrieval pipeline implementation

This build adds the requested modular layout while preserving the existing FastAPI, Streamlit, ingestion, storage, and public chat API. Integration changes are limited to `app/rag/chat.py` (to invoke planning, retrieval ranking, evidence selection, and coverage reporting) and `app/core/config.py` (the maximum planned parts default is now 10). Existing storage formats and data directories are unchanged.

## Added modules

- `app/question/classifier.py`: intent and long-question classification.
- `app/question/context_resolver.py`: bounded conversation-history cleaning and follow-up prompt construction.
- `app/question/query_planner.py`: conservative decomposition of explicit and long compound questions, retaining the original question and reporting truncation.
- `app/retrieval/dense_search.py`, `keyword_search.py`, `fusion.py`, `reranker.py`, `evidence_selector.py`, `pipeline.py`: ranking adapters, reciprocal-rank fusion, lightweight reranking, bounded context selection, and query-term coverage.
- `app/reasoning/scenario_analyzer.py`, `comparison.py`, `calculator.py`, `summarizer.py`, `conflict_resolver.py`, `answer_planner.py`: focused reasoning instructions/adapters and per-requirement coverage tracking.
- `app/rag/prompts.py`, `answer_schema.py`, `answer_validator.py`: prompt composition, response record typing, and citation validation adapter.
- `app/evaluation/question_suite.py`, `retrieval_metrics.py`, `answer_metrics.py`: labeled question-suite loading and retrieval/answer coverage metrics.

## Long-question execution

When the incoming question is long and contains recognizable separate requirements, the planner creates up to 10 subquestions. Each subquestion independently calls the existing answer/retrieval path, so one weak requirement does not prevent retrieval for the others. The response includes `question_plan`, per-question `answers`, and `coverage` with answered/unanswered requirement counts. If the configured maximum is exceeded, `truncated_to_limit` is set; increase `max_question_parts` if your questions routinely contain more than 10 requirements.

## Honest implementation boundaries

- The dense and keyword modules are adapters over the current store's shared candidate results; they do not create separate persistent indexes. The existing store already combines Chroma semantic results and SQLite FTS keyword hits.
- The reranker is deterministic lexical/score reranking, not a trained cross-encoder.
- Term coverage is a diagnostic, not proof that a passage entails the answer. It can mark synonymous wording as missing or unrelated word overlap as covered.
- Intent classification and decomposition are heuristic; unusual prose may not split perfectly. Inspect `question_plan` and validate with real labeled documents.
- Citation validation checks reference syntax/range, not semantic entailment.
- The evaluation helpers do not fabricate correctness labels; semantic answer correctness still requires ground-truth labels or human review.


## Caching

The request pipeline checks a validated exact answer cache first, then an exact retrieval cache. On a miss it obtains a locally cached query embedding and may check semantic answer/retrieval caches. Semantic cache entries are scoped by owner, current indexed-document fingerprint, relevant model/config fingerprints, workflow type, high cosine similarity, and compatible numeric/ID/acronym anchors. Retrieval cache stores selected evidence chunks and retrieval diagnostics; answer cache stores only successful answers with sources. Cache files live in `data/metadata/rag_cache.sqlite3`, independently of durable document/index data. Cache metrics are available through `app.cache.manager.stats()` for local diagnostics.
