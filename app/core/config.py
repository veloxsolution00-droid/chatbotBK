from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from urllib.parse import urlparse
import ipaddress

def validate_loopback_url(url: str) -> bool:
    """Return True only for an HTTP endpoint bound to a loopback host."""
    try:
        parsed = urlparse(url)
        if parsed.scheme != "http" or not parsed.hostname:
            return False
        host = parsed.hostname.lower()
        if host == "localhost":
            return True
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False

ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    ollama_base_url: str = "http://127.0.0.1:11434"
    generation_model: str = "qwen2.5:3b"
    embedding_model: str = "nomic-embed-text"

    # All durable application data lives below data/ by default.
    data_path: Path = ROOT / "data"
    uploads_path: Path | None = None
    extracted_path: Path | None = None
    chroma_path: Path | None = None
    metadata_path: Path | None = None
    logs_path: Path | None = None
    jobs_path: Path | None = None
    sqlite_path: Path | None = None

    collection_name: str = "document_chunks"
    chunk_size: int = 1800
    chunk_overlap: int = 300
    top_k: int = 5
    retrieval_candidate_multiplier: int = 10
    hybrid_lexical_weight: float = 0.20
    lexical_candidate_limit: int = 20
    embedding_batch_size: int = 16
    min_similarity: float = 0.25
    max_file_mb: int = 100
    max_batch_mb: int = 500
    max_upload_files: int = 50
    ingestion_workers: int = 1
    inference_concurrency: int = 1
    temperature: float = 0.0
    # Per-Ollama-call limit. Must exceed the slowest single generation on the
    # host; CPU-only inference of a grounded answer routinely passes 90 s.
    timeout: int = 300
    tesseract_cmd: str = ""
    ocr_enabled: bool = True
    max_pdf_pages: int = 5000
    max_image_pixels: int = 40_000_000
    extraction_block_rows: int = 1000
    max_question_parts: int = 10
    max_context_chars: int = 9000
    verify_answers: bool = True
    # Decode caps in tokens. A small local model keeps generating until it emits
    # an end token, and that decode time is the dominant cost on CPU.
    answer_max_tokens: int = 220
    verify_max_tokens: int = 8
    recovery_on_startup: bool = True

    # Local, bounded cache. Cache data is disposable and stored separately from
    # document metadata/indexes; disabling it never affects persistent documents.
    cache_enabled: bool = True
    cache_ttl_seconds: int = 604800
    cache_max_entries: int = 2000
    semantic_retrieval_threshold: float = 0.94
    # Semantic answer reuse is disabled by default: similar embeddings do not prove equivalent intent.
    semantic_answer_cache_enabled: bool = False
    semantic_answer_threshold: float = 0.975

    # Resolve configuration relative to the application, not whichever folder
    # happened to be current when Uvicorn was started.  This makes the launcher
    # scripts, tests, and manual starts behave consistently.
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    def model_post_init(self, __context):
        if not validate_loopback_url(self.ollama_base_url):
            raise ValueError("OLLAMA_BASE_URL must use plain HTTP on localhost/127.0.0.1/::1 for offline-only operation.")
        configured_data_path = Path(self.data_path).expanduser()
        # Keep a relative DATA_PATH relative to the application directory,
        # rather than to an arbitrary terminal working directory.
        self.data_path = (configured_data_path if configured_data_path.is_absolute()
                          else ROOT / configured_data_path).resolve()
        defaults = {
            "uploads_path": self.data_path / "uploads",
            "extracted_path": self.data_path / "extracted",
            "chroma_path": self.data_path / "chroma_db",
            "metadata_path": self.data_path / "metadata",
            "logs_path": self.data_path / "logs",
            "jobs_path": self.data_path / "jobs",
            "sqlite_path": self.data_path / "metadata" / "documents.sqlite3",
        }
        for key, default in defaults.items():
            value = getattr(self, key)
            if value:
                path = Path(value).expanduser()
                # A relative override is documented as project-relative. Resolving
                # it against the process CWD made launcher scripts and manual
                # starts write to different folders.
                setattr(self, key, path if path.is_absolute() else (ROOT / path).resolve())
            else:
                setattr(self, key, default.resolve())

settings = Settings()
