# Feature Coverage Matrix — Offline Document Chatbot

This matrix is intentionally explicit: **implemented** means code exists and has a relevant local test or a direct code path; **partial** means the behavior is heuristic or needs live validation; **blocked** means a required input/dependency is absent. Do not interpret a green unit-test suite as proof of end-to-end RAG correctness.

| Requirement | Status | Evidence / validation |
|---|---|---|
| Local-only app endpoints (loopback) | Implemented by launch configuration | README run commands bind API/UI to `127.0.0.1`; Ollama endpoint rejects non-loopback URLs. Verify Windows firewall and actual running processes locally. |
| Ollama local generation and embeddings | Implemented, runtime not validated here | Config defaults to `qwen2.5:3b` and `nomic-embed-text`; requires models installed and Ollama reachable. |
| Durable original uploads and extracted text | Implemented | Durable data folders, atomic writes, database job/document state, recovery/reconciliation; no startup cache-clearing path added. |
| Multi-format extraction | Implemented with format-specific limitations | PDF, DOCX, TXT/MD/log/HTML/RTF/ODT, image OCR, XLSX/XLSM/ODS/CSV, PPTX, JSON/XML/YAML/EML. Legacy XLS/PPT conversion is not native. |
| Scanned PDF/image OCR | Partial | Tesseract path/config and OCR fallback exist. Must install/configure Tesseract and test representative scans; OCR quality is not guaranteed. |
| Empty/no-text upload handling | Implemented | Extractor raises `NoReadableTextError` when no readable text is found. |
| Page/slide/sheet/row source labels | Implemented | Extractor yields labelled blocks and stores source locations with chunk metadata. Layout fidelity varies by file format. |
| Structure-aware chunking | Implemented as heuristic | Chunks prefer paragraphs/headings/sentences and enforce size bounds/overlap; this is **not** embedding-based semantic chunking. Regression tests added. |
| Dense semantic retrieval | Implemented, live ChromaDB untested here | ChromaDB persistent collection + Ollama embeddings. |
| Exact-term/hybrid retrieval | Implemented | SQLite FTS5 keyword search plus semantic candidates and lexical scoring; exact values/IDs can be found by FTS. Ranking weights need corpus evaluation. |
| Document-vocabulary typo correction | Partial | Corrects multiple lowercase alphabetic misspellings per query against local FTS vocabulary, skips identifiers/acronyms/title-case words and ambiguous candidates, preserves the original query route, and searches corrected variants even when the first pass has hits. Fuzzy matching can still be wrong; names/technical tokens and out-of-vocabulary typos remain imperfect. |
| Long-question decomposition and coverage | Implemented with heuristic limits | Long compound prompts are decomposed into up to 10 independently retrieved requirements; output includes a plan, per-requirement answers, and answered/unanswered coverage. Unusual grammar may still split imperfectly; `truncated_to_limit` exposes overflow. |
| Dense/keyword fusion and reranking | Partial | Added ranking adapters, reciprocal-rank fusion, score/term-coverage reranking, and evidence selection over the existing shared candidate pool. Dense and keyword adapters are not separate persistent indexes; reranker is not a trained cross-encoder. |
| Paraphrase/alternate-query retry | Partial | Conservative query variants and a second pass are implemented; not a general semantic query planner. |
| Multiple evidence chunks | Implemented | Top-k/candidate multiplier and bounded context selection; defaults are configurable. |
| Confidence / abstention | Partial | Similarity threshold, evidence filtering, citation checks, and optional LLM verifier exist. Scores are not calibrated probabilities and do not prove entailment. |
| Multi-document answers and comparisons | Partial | Multiple evidence chunks/documents are available to the answer prompt; no formal cross-document reasoning guarantee. Validate against ground-truth cases. |
| Multi-question in one message | Implemented | Explicit question splitting and combined response path with source remapping. Test edge cases on your own prompts. |
| Follow-up questions | Partial | UI sends bounded history and model-assisted follow-up rewrite; ambiguous history resolution may fail. |
| Scenario / what-if analysis | Partial | Planner and prompt structure separate documented rules, user assumptions, conclusions, and missing conditions; not a formal rules engine. |
| Deterministic arithmetic | Partial | Restricted AST arithmetic for direct expressions; document-derived formulas are model-selected, operands/source references checked, then arithmetic evaluated deterministically. Formula correctness is not guaranteed. |
| Citation display and reference validation | Partial | Numeric reference range and explicit filename checks; syntax/reference validity does not prove that a citation supports the claim. |
| Contradiction detection | Partial | Heuristic detects likely number/polarity disagreements and warns the answer model; can false-positive or miss semantic conflicts. |
| Answer verification | Partial | Optional second local-model pass; a verifier is not independent proof. |
| Duplicate uploads | Implemented | SHA-256 duplicate detection in document workflow. |
| Background ingestion and job status | Implemented | Job queue, progress endpoints, retry path, recovery hooks. |
| Persistent ChromaDB + SQLite FTS | Implemented, live integration not validated here | Existing collections are not deleted/recreated at startup; FTS backfill support exists. |
| Health/status diagnostics | Implemented | `/health` includes Ollama, ChromaDB, storage paths, configured limits, and local cache counts/hit/miss metrics. |
| Upload limits / supported-extension checks | Implemented | Per-file, batch-size, file-count and extension validation. |
| Evaluation runner | Implemented, requires user's gold cases | API evaluation harness checks expected answer terms, abstention, sources, question type, and history. Example cases are placeholders. |
| Unit/regression tests | Passed in build environment | `67 passed, 1 skipped`; skipped test needs ChromaDB package. See `IMPLEMENTATION_AUDIT.md`. |
| Real Ollama + ChromaDB end-to-end test | Not run in this environment | Requires target Windows machine, installed dependencies, and local models. |
| Network-disabled/offline proof | Not run in this environment | Must disconnect/disable network after setup and test upload, retrieval, and chat. |
| 200 MB workload / latency / memory benchmark | Not run in this environment | Use `performance.md` and record upload, extraction, embedding/indexing, query latency, RAM, and disk usage on target hardware. |
| Merge two distinct UI implementations | Blocked: second UI absent from supplied archive | Only one UI implementation was available in the supplied `RND` archive. Provide the other UI source to perform an actual merge; this cannot be inferred or recreated faithfully from a description. |
| Multi-user/network deployment | Out of scope for current local single-user app | No authentication layer by design; do not expose this build to LAN or internet. |

## Minimum release gate on the target Windows machine

1. Install `requirements.txt`, Ollama, the configured models, and Tesseract if OCR is needed.
2. Run `python -m pytest -q -rs`.
3. Start the API and UI bound to `127.0.0.1` and confirm `/health` reports Ollama and ChromaDB available.
4. Upload at least one born-digital PDF, scanned PDF, image, DOCX, spreadsheet, and PPTX; verify job completion and source locations.
5. Ask answerable, unanswerable, single- and multi-typo, proper-name/ID, paraphrased, multi-document, conflicting-source, calculation, multi-question, and follow-up questions. Check every cited passage manually.
6. Review `DEAD_CODE_AUDIT.md` before removing test assets or compatibility code.
7. Run `evaluation/run_api_evaluation.py` with ground-truth cases from your own documents.
8. Repeat key tests with the network disconnected. Record timing/RAM/disk for the largest intended workload.
9. Back up `data/` before replacing an existing installation. Never test on the only copy of persistent data.

## Local caching (added in the cache build)

| Capability | Status | Notes |
|---|---|---|
| Exact query embedding cache | Implemented | SQLite-backed; keyed by normalized query and embedding model. |
| Exact retrieval cache | Implemented | Stores selected chunks and retrieval diagnostics; scoped to owner, document fingerprint, and retrieval settings. |
| Semantic retrieval cache | Implemented | High cosine threshold, same question workflow, and matching numeric/ID/acronym anchors. |
| Exact answer cache | Implemented | Only successful, cited, non-abstained answers; answer model/prompt/verifier settings are part of the cache scope. |
| Semantic answer cache | Implemented conservatively | Stricter similarity threshold than retrieval cache; still a heuristic, not a semantic equivalence proof. |
| Invalidation | Implemented through versioned keys and TTL | Document/index changes alter the corpus fingerprint; model/retrieval configuration changes alter cache keys. Expired entries and entries beyond the configured cap are pruned. |
| Cache observability | Implemented | `/health` includes local cache entry counts and hit/miss counters. |
| Live speed/correctness benchmark | Not run | Must be measured on the target Windows machine with Ollama and ChromaDB installed and real documents indexed. |
