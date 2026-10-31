# Ingestion and retrieval performance notes

- Ingestion logs now separately report extraction, chunking, embedding/storage, and total elapsed seconds.
- Vector indexing uses configurable bounded batches (`EMBEDDING_BATCH_SIZE`, default 16), up from 8. On 8 GB RAM, keep the value modest; if Ollama times out or RAM pressure rises, set it to 8.
- The chat retrieval setting defaults to `TOP_K=5` and `RETRIEVAL_CANDIDATE_MULTIPLIER=10`. ChromaDB retrieves a wider candidate set before status and similarity filtering, then the RAG layer ranks eligible chunks and passes up to five into the generation prompt, subject to `MAX_CONTEXT_CHARS`. The defaults are starting points, not a claim that five chunks is optimal for every corpus.
- `relevance_score` is a hybrid semantic/lexical ranking score, not a calibrated probability or guaranteed factual confidence. Tune `MIN_SIMILARITY` against representative questions.
- These changes add measurements and reduce per-request overhead; actual speed gains must be benchmarked on the target PC.

- When initial retrieval produces no eligible hits, the chat layer tries conservative query variants and a bounded FTS5 vocabulary-based spelling correction. This can add embedding calls; the original user spelling is retained as a search route.
- Citation checks validate numeric source-reference bounds and explicit filename references. They cannot prove that a cited passage semantically entails every claim; the optional local-model verifier is a second opinion, not a proof.
- Potential contradiction detection is heuristic (overlapping terms plus numeric/polarity signals). Inspect flagged passages; both false positives and missed contradictions are possible.
- `OLLAMA_BASE_URL` is required to be plain HTTP on a loopback host (`localhost`, `127.0.0.1`, or `::1`) in this build.
