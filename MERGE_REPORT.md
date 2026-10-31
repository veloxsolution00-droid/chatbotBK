# Merge report

Merged the two supplied archives with the authentication/job-queue branch as the base and the performance-focused ingestion/Ollama implementations from the optimized-local branch.

## Combined
- FastAPI local API and Streamlit UI
- SQLite-backed users and sessions
- Owner-scoped document listing, jobs, and chat retrieval
- Persistent background ingestion jobs and retry endpoint
- Persistent SQLite/ChromaDB storage and recovery/reconciliation paths
- Batched Ollama embeddings where supported, with legacy API fallback
- Bounded ingestion worker and serialized local Ollama requests
- Benchmark script and persistence/multi-tenant tests

## Deliberately not copied
- `.env` secrets/configuration from the optimized archive
- SQLite databases, ChromaDB runtime files, uploaded PDFs/DOCX files, extracted user data, logs, and Python cache files
- Runtime data is user-owned and should not be overwritten by a source-code merge. Back up your existing `data/` folder before replacing any local project code.

## Verification performed here
- Python compileall: passed
- pytest: 18 passed, 1 skipped
- The skipped test requires an integration dependency/environment not available in this run.
- No Ollama generation-speed benchmark was run here; timings must be measured on the target PC.
