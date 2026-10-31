# 📄 Offline Document Intelligence Chatbot

An AI-powered, local-only document chatbot. Upload documents and ask questions grounded in their contents. Built using **FastAPI, Streamlit, ChromaDB, SQLite, and Ollama**, it supports offline inference and persistent local storage.

## 🚀 Features

- **Offline AI:** Run language models locally using Ollama.
- **Multi-Format Support:** Process PDFs, Word documents, spreadsheets, images, presentations, and more.
- **Grounded RAG Chat:** Generate answers from relevant document content; unrelated questions return a clear no-relevant-information response.
- **Hybrid Retrieval & Verification:** Combine semantic and exact-term matching, then verify that each draft answer is supported by the retrieved evidence.
- **Visible Citations:** Show the filenames and source locations used for every answer.
- **Persistent Storage:** Preserve uploaded files, extracted text, metadata, and embeddings across restarts.
- **Multi-Question Support:** Answer each explicitly separated question in one message.
- **Follow-Up Context:** Send a bounded window of recent chat messages so elliptical questions can be rewritten as standalone retrieval queries.
- **Scenario Reasoning:** For hypothetical questions, distinguish document facts, user-supplied assumptions, and conclusions derived from documented rules; flag missing conditions rather than inventing them.
- **Question Planning:** Classify factual, scenario, comparison, summary, explanation, calculation, and follow-up questions; retry weak retrieval with conservative query variants and document-vocabulary spelling correction while retaining the original query route. Chunking prefers paragraph, heading, and sentence boundaries but is heuristic rather than embedding-based semantic segmentation.
- **Citation Checks:** Require a citation in the generated answer and reject invalid numeric references or explicit references to unknown filenames. The UI also lists supporting passages.
- **Conflict Warnings:** Flag likely numeric/polarity disagreements between retrieved passages for model review. These are heuristics, not proof that statements conflict.
- **Safe Arithmetic:** Evaluate explicit user-supplied arithmetic and percentage expressions using a restricted AST evaluator, never `eval`. For document-derived calculations, validate numeric operands and source references before deterministic evaluation; formula selection can still be wrong.
- **Loopback Guard:** Reject non-loopback Ollama endpoints to help enforce local-only runtime operation.
- **Evaluation Harness:** Run local API regression cases for answer content, abstention, source filenames, question type, and follow-up context.
- **Background Processing:** Process documents asynchronously using an ingestion queue.
- **OCR Support:** Extract text from scanned documents and images using Tesseract.
- **Duplicate Detection:** Identify duplicate uploads using SHA-256 hashing.
- **Recovery Mechanism:** Recover interrupted ingestion operations.
- **Explicit Deletion:** Documents are not automatically deleted as temporary cache.

## 🏗️ Architecture

```text
Streamlit UI
     |
     v
FastAPI Backend
     |
     +---- Document Ingestion
     |          |
     |          +---- Text Extraction / OCR
     |          +---- Chunking
     |          +---- Ollama Embeddings
     |          +---- ChromaDB Indexing
     |
     +---- RAG Chat
                |
                +---- Follow-up Context Resolution
                +---- Question Planner / Conservative Query Variants
                +---- Hybrid Retrieval (ChromaDB + SQLite FTS5)
                +---- Evidence Selection / Conflict Heuristics
                +---- Grounded Answer Generation (Ollama)
                +---- Answer Verification + Citation Reference Validation
                +---- Abstention when evidence/citations are insufficient

Deterministic calculation module:
- Explicit arithmetic and percentage questions use a restricted AST evaluator.
- Calculations that require identifying figures from documents still depend on correct evidence selection and model interpretation.

Persistent Storage:
├── data/uploads/     - Original documents
├── data/extracted/   - Extracted text
├── data/chroma_db/   - Vector embeddings
├── data/metadata/    - SQLite database
└── data/logs/        - Application logs
```

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| FastAPI | Backend API |
| Streamlit | User interface |
| ChromaDB | Vector storage |
| SQLite | Metadata and job tracking |
| Ollama | Local LLM and embeddings |
| Tesseract OCR | Scanned document extraction |

## ⚙️ Installation

**Prerequisites:** Python 3.11+, Ollama, and Tesseract OCR if scanned-document processing is required.

### 1. Clone the Repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd offline_document_chatbot
```

### 2. Create a Virtual Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Install Local AI Models

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### 4. Configure Environment Variables

```powershell
Copy-Item .env.example .env
```

Configure the settings in `.env`. Never commit this file to GitHub. The application is deliberately local-only and has no login or network-facing authentication layer.

### 5. Start the Backend

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 6. Start the Frontend

Open another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

**Access the application:** http://127.0.0.1:8501

**API documentation:** http://127.0.0.1:8000/docs

## 🔐 Data Persistence & Security

- Uploaded documents and indexed data are stored permanently until explicitly deleted.
- This build is single-user and intended for one local computer.
- Keep services bound to `127.0.0.1`; do not expose them to a network without adding proper authentication and authorization.
- Runtime data is stored in the configurable `data/` directory.
- Back up the complete data directory regularly.
- Keep services bound to `127.0.0.1` for local-only access.
- Verify authentication and user isolation before network deployment.

**Important:** Persistent storage does not replace backups. RAG reduces unsupported answers but cannot guarantee that every response is correct.

Follow-up resolution may use an additional local model call when chat history is supplied. Answer verification performs one additional local model call for supported answers. Set `VERIFY_ANSWERS=false` in `.env` if faster responses are more important than this extra guardrail; citation reference validation still runs. Conservative retrieval retries can also add embedding calls when the first pass returns no evidence.

## Automated tests and evaluation

Run unit/regression tests:

```powershell
python -m pytest -q
```

For API-level evaluation, upload/index the documents used by your test cases, copy `evaluation/cases.example.json` to a new JSON file, replace the placeholders with facts that are true for those documents, and run:

```powershell
python evaluation/run_api_evaluation.py evaluation/my_cases.json
```

See `evaluation/README.md` for the recommended test matrix and `FEATURE_COVERAGE.md` for a requirement-by-requirement status matrix, known limitations, and the target-machine release gate. A passing test is a regression signal, not a guarantee of semantic correctness. The live Ollama integration, actual document retrieval quality, and performance must be measured on the target Windows machine.

Exact-term matching uses a local SQLite FTS5 index. It is especially useful for dates, names, invoice numbers, IDs, and values in tabular documents; it remains fully offline.

## 📂 Supported Formats

PDF, DOCX, TXT, Markdown, logs, HTML, RTF, ODT, JPG, JPEG, PNG, TIFF, BMP, WEBP, XLSX, XLSM, CSV, ODS, PPTX, JSON, XML, YAML, and EML.

OCR requires Tesseract. Legacy XLS and PPT files may require conversion.

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check application status |
| POST | `/documents/upload` | Upload documents |
| GET | `/documents` | List documents |
| DELETE | `/documents/{document_id}` | Delete a document |
| POST | `/chat` | Ask document-based questions |
| POST | `/admin/reconcile` | Recover incomplete operations |
| GET | `/admin/recovery-events` | View recovery events |

Refer to the running API documentation for actual request schemas and available routes.

## 📌 Key Objectives

- Keep AI processing local.
- Preserve documents and embeddings across restarts.
- Abstain instead of guessing when uploaded documents do not support an answer.
- Improve ingestion performance through background processing.
- Generate answers grounded in uploaded documents.


### Local RAG cache

The application uses a bounded, local SQLite cache at `data/metadata/rag_cache.sqlite3` for query embeddings, retrieved evidence, and successful cited answers. Cache data is separate from the persistent documents and ChromaDB index. Exact answer hits can skip retrieval and Ollama generation; semantic retrieval/answer reuse uses conservative similarity thresholds and is scoped to the owner, indexed-document fingerprint, and relevant configuration. Cache hit metrics are included in `/health`.

Settings: `CACHE_ENABLED=true`, `CACHE_TTL_SECONDS=604800`, `CACHE_MAX_ENTRIES=2000`, `SEMANTIC_RETRIEVAL_THRESHOLD=0.94`, `SEMANTIC_ANSWER_THRESHOLD=0.975`. Disable caching with `CACHE_ENABLED=false` if debugging retrieval.
#   c h a t b o t B K  
 