# Local RAG evaluation

1. Start Ollama, FastAPI, and Streamlit (the API suite only needs FastAPI).
2. Upload and index a small, known set of test documents.
3. Copy `cases.example.json` to `my_cases.json` and replace the placeholders with claims and source names that are true for those documents.
4. Run `python evaluation/run_api_evaluation.py evaluation/my_cases.json`.
5. Inspect the JSON report. A passing substring/source check is a regression signal, not a semantic correctness proof.

Recommended test matrix:

- Direct fact with exact source location.
- Paraphrase and common spelling mistake.
- Follow-up question requiring prior context.
- Question requiring evidence from two or more documents.
- Scenario with all rules stated.
- Scenario with a missing rule (should state limits or abstain).
- Two documents with conflicting values (should cite both and state the conflict).
- Explicit arithmetic and a document-based calculation.
- Unsupported question (should abstain).
- OCR/image-only file and a malformed/empty file.

Track retrieval Recall@k / MRR separately from answer quality. A useful answer-quality rubric scores factual support, citation correctness, completeness, conflict handling, and abstention separately. Never use only model self-verification as your accuracy metric.
