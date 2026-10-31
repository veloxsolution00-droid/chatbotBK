# Dead / Leftover Code Audit

## Actions taken

- Removed `app.services.documents.queue_ingest`: no call sites existed in the repository. It duplicated the active ingestion registration path and did not submit a durable background job itself.
- Removed `app.services.documents.process_document`: no call sites existed in the repository. It was a second wrapper around the same internal processing routine.
- Removed the now-unused `Lock` import and `_INGEST_SLOT` lock that existed only for `process_document`.
- Kept `_atomic_write`, `_process_existing`, and `ingest`: these are used by the active ingestion/recovery flow.

## Static scan findings and false positives reviewed

A repository-wide AST scan found no obvious unused imports in `app/`. It flagged database functions such as `db.get_document` as unreferenced because they are called as attributes (`db.get_document`) rather than bare names; these are used by the application and were retained. It also flagged `app.rag.evaluation` helpers, which are intentionally used by tests. `app.ingestion.extractor.extract` is retained as a convenience wrapper and is used by tests.

## Files intentionally retained

- `dracula-bram-stoker.pdf` and `testing.xlsx` are not referenced by the application code. They appear to be sample/user-provided artifacts, so they were **not deleted** without confirmation. If they are not intended as demo or evaluation data, they can be removed manually after backup.
- `benchmark_chatbot.py` is a standalone command-line benchmark utility; it is not imported by the app and is intentionally retained as an optional tool.
- Legacy `owner_id` database fields and migration functions remain because they preserve compatibility with older databases after authentication was removed. They do not imply that a login flow is active.

## Limits of this audit

This is a repository-level static reference audit, not proof that no runtime/plugin/dynamic-import path exists outside the supplied repository. Before deleting more code, verify external consumers and run the full suite.


## Added modular question pipeline

The following modules are now called by the live chat path or intentionally exposed as evaluation/type helpers: question planning/classification, follow-up context prompt construction, dense/keyword candidate ranking adapters, reciprocal-rank fusion, reranking, evidence selection, reasoning prompt helpers, answer planning/coverage, and citation validation. `app/rag/answer_schema.py` is a TypedDict contract for callers and is not imported at runtime. `app/evaluation/*` are offline evaluation utilities, intentionally not executed on every chat request. The dense/keyword adapter functions can also be called independently, but the main path reuses a single candidate retrieval pass to avoid duplicate embedding calls.

The integration modified only `app/rag/chat.py` and the `max_question_parts` default in `app/core/config.py`; the API route shapes, UI, ingestion pipeline, database schema, and persistent data layout were not changed for this feature.
