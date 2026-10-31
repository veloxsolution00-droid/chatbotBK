import sqlite3
from contextlib import contextmanager
from pathlib import Path
from app.core.config import settings

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL DEFAULT 'local',
    filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    original_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    file_format TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    processing_status TEXT NOT NULL,
    extraction_status TEXT NOT NULL,
    indexing_status TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    source_metadata_json TEXT NOT NULL DEFAULT '{}',
    extracted_path TEXT,
    processing_error TEXT,
    last_reconciled_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    staged_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    document_id TEXT,
    error TEXT,
    stage TEXT NOT NULL DEFAULT 'Queued',
    progress INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_owner_created ON ingestion_jobs(owner_id, created_at DESC);
CREATE TABLE IF NOT EXISTS deletion_operations (
    operation_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS recovery_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    severity TEXT NOT NULL,
    document_id TEXT,
    event_type TEXT NOT NULL,
    details TEXT NOT NULL
);
"""

def ensure_directories():
    for path in (settings.data_path, settings.uploads_path, settings.extracted_path,
                 settings.chroma_path, settings.metadata_path, settings.logs_path, settings.jobs_path):
        Path(path).mkdir(parents=True, exist_ok=True)
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)

def connect():
    ensure_directories()
    conn = sqlite3.connect(str(settings.sqlite_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def initialize():
    with connect() as conn:
        conn.executescript(SCHEMA)
        # FTS5 is bundled with standard SQLite builds. Keep the application
        # usable on an unusual build without it; vector retrieval remains the
        # safe fallback in that case.
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(
                    chunk_id UNINDEXED, document_id UNINDEXED, owner_id UNINDEXED,
                    filename UNINDEXED, source_location UNINDEXED, text
                )
            """)
        except sqlite3.OperationalError as exc:
            if "fts5" not in str(exc).lower():
                raise
        # Safe, additive migration for databases created by earlier versions.
        columns = {r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "owner_id" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'local'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash_owner ON documents(owner_id, content_hash)")
        job_columns = {r[1] for r in conn.execute("PRAGMA table_info(ingestion_jobs)").fetchall()}
        if "stage" not in job_columns:
            conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN stage TEXT NOT NULL DEFAULT 'Queued'")
        if "progress" not in job_columns:
            conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN progress INTEGER NOT NULL DEFAULT 0")

@contextmanager
def transaction():
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def row_to_document(row):
    if row is None:
        return None
    d = dict(row)
    import json
    d["source_metadata"] = json.loads(d.pop("source_metadata_json") or "{}")
    return d

def get_document(document_id):
    with connect() as conn:
        return row_to_document(conn.execute(
            "SELECT * FROM documents WHERE document_id=?", (document_id,)
        ).fetchone())

def get_by_hash(content_hash, owner_id="local"):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE content_hash=? AND owner_id=? ORDER BY uploaded_at LIMIT 1",
            (content_hash, owner_id)
        ).fetchone()
        return row_to_document(row)

def list_documents(owner_id=None):
    with connect() as conn:
        if owner_id is None:
            rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
        else:
            rows = conn.execute("SELECT * FROM documents WHERE owner_id=? ORDER BY uploaded_at DESC", (owner_id,)).fetchall()
        return [row_to_document(r) for r in rows]

def save_document(doc):
    import json
    with transaction() as conn:
        conn.execute("""
        INSERT INTO documents (
          document_id,owner_id,filename,stored_filename,original_path,content_hash,file_format,
          uploaded_at,updated_at,processing_status,extraction_status,indexing_status,
          chunk_count,source_metadata_json,extracted_path,processing_error,last_reconciled_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(document_id) DO UPDATE SET
          owner_id=excluded.owner_id, filename=excluded.filename, stored_filename=excluded.stored_filename,
          original_path=excluded.original_path, content_hash=excluded.content_hash,
          file_format=excluded.file_format, uploaded_at=excluded.uploaded_at,
          updated_at=excluded.updated_at, processing_status=excluded.processing_status,
          extraction_status=excluded.extraction_status, indexing_status=excluded.indexing_status,
          chunk_count=excluded.chunk_count, source_metadata_json=excluded.source_metadata_json,
          extracted_path=excluded.extracted_path, processing_error=excluded.processing_error,
          last_reconciled_at=excluded.last_reconciled_at
        """, (
            doc["document_id"], doc.get("owner_id", "local"), doc["filename"], doc["stored_filename"], doc["original_path"],
            doc["content_hash"], doc["file_format"], doc["uploaded_at"], doc["updated_at"],
            doc["processing_status"], doc["extraction_status"], doc["indexing_status"],
            doc.get("chunk_count", 0), json.dumps(doc.get("source_metadata", {})),
            doc.get("extracted_path"), doc.get("processing_error"), doc.get("last_reconciled_at")
        ))

def update_document(document_id, **fields):
    import json
    allowed = {"updated_at","processing_status","extraction_status","indexing_status",
               "chunk_count","source_metadata","extracted_path","processing_error","last_reconciled_at"}
    values = []
    assignments = []
    for key, value in fields.items():
        if key not in allowed:
            raise ValueError(f"Unsupported document field: {key}")
        column = "source_metadata_json" if key == "source_metadata" else key
        if key == "source_metadata":
            value = json.dumps(value)
        assignments.append(f"{column}=?")
        values.append(value)
    if not assignments:
        return
    values.append(document_id)
    with transaction() as conn:
        conn.execute(f"UPDATE documents SET {', '.join(assignments)} WHERE document_id=?", values)

def remove_document_record(document_id):
    with transaction() as conn:
        conn.execute("DELETE FROM documents WHERE document_id=?", (document_id,))

def log_recovery_event(severity, event_type, details, document_id=None):
    from datetime import datetime, timezone
    with transaction() as conn:
        conn.execute(
            "INSERT INTO recovery_events(created_at,severity,document_id,event_type,details) VALUES(?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), severity, document_id, event_type, str(details))
        )

def create_deletion_operation(operation_id, document_id):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with transaction() as conn:
        conn.execute("INSERT INTO deletion_operations VALUES(?,?,?,?,?,?)",
                     (operation_id, document_id, "pending", now, now, None))

def update_deletion_operation(operation_id, status, error=None):
    from datetime import datetime, timezone
    with transaction() as conn:
        conn.execute("UPDATE deletion_operations SET status=?,updated_at=?,error=? WHERE operation_id=?",
                     (status, datetime.now(timezone.utc).isoformat(), error, operation_id))

def pending_deletions():
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM deletion_operations WHERE status!='complete' ORDER BY created_at"
        ).fetchall()]

def recovery_events(limit=100):
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM recovery_events ORDER BY event_id DESC LIMIT ?", (limit,)
        ).fetchall()]


def create_job(job):
    with transaction() as conn:
        conn.execute("INSERT INTO ingestion_jobs(job_id,owner_id,filename,staged_path,size_bytes,status,created_at,updated_at,document_id,error,stage,progress) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                     (job["job_id"], job.get("owner_id", "local"), job["filename"], job["staged_path"], job["size_bytes"], job["status"], job["created_at"], job["updated_at"], job.get("document_id"), job.get("error"), job.get("stage", "Queued"), job.get("progress", 0)))

def update_job(job_id, **fields):
    allowed = {"status", "updated_at", "document_id", "error", "stage", "progress"}
    if not fields or not set(fields).issubset(allowed):
        raise ValueError("Unsupported job fields")
    with transaction() as conn:
        conn.execute("UPDATE ingestion_jobs SET " + ",".join(f"{k}=?" for k in fields) + " WHERE job_id=?", (*fields.values(), job_id))

def get_job(job_id, owner_id=None):
    with connect() as conn:
        if owner_id is None:
            row = conn.execute("SELECT * FROM ingestion_jobs WHERE job_id=?", (job_id,)).fetchone()
        else:
            row = conn.execute("SELECT * FROM ingestion_jobs WHERE job_id=? AND owner_id=?", (job_id, owner_id)).fetchone()
        return dict(row) if row else None

def list_jobs(owner_id="local"):
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM ingestion_jobs WHERE owner_id=? ORDER BY created_at DESC LIMIT 200", (owner_id,)).fetchall()]

def migrate_to_local_owner():
    """Convert this local install to one owner and remove obsolete auth tables."""
    with transaction() as conn:
        conn.execute("UPDATE documents SET owner_id='local' WHERE owner_id IS NULL OR owner_id!='local'")
        conn.execute("UPDATE ingestion_jobs SET owner_id='local' WHERE owner_id IS NULL OR owner_id!='local'")
        # Login was explicitly removed from this local-only build. Drop its old
        # credential/session tables; document, job, extraction, and vector data remain.
        conn.execute("DROP TABLE IF EXISTS sessions")
        conn.execute("DROP TABLE IF EXISTS users")

def recoverable_jobs():
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM ingestion_jobs WHERE status IN ('queued','processing') ORDER BY created_at").fetchall()]


def index_chunks(records):
    """Mirror searchable chunk text in SQLite FTS5 for exact-term retrieval."""
    if not records:
        return
    try:
        with transaction() as conn:
            for record in records:
                metadata = record.get("metadata", {})
                conn.execute("DELETE FROM chunk_search WHERE chunk_id=?", (record["id"],))
                conn.execute(
                    "INSERT INTO chunk_search(chunk_id,document_id,owner_id,filename,source_location,text) VALUES(?,?,?,?,?,?)",
                    (record["id"], metadata.get("document_id"), metadata.get("owner_id", "local"),
                     metadata.get("filename"), metadata.get("source_location"), record["text"]),
                )
    except sqlite3.OperationalError as exc:
        if "chunk_search" not in str(exc) and "fts5" not in str(exc).lower():
            raise


def remove_indexed_chunks(document_id):
    try:
        with transaction() as conn:
            conn.execute("DELETE FROM chunk_search WHERE document_id=?", (document_id,))
    except sqlite3.OperationalError as exc:
        if "chunk_search" not in str(exc) and "fts5" not in str(exc).lower():
            raise


def search_indexed_chunks(query, owner_id="local", limit=20):
    """Return exact keyword matches without loading all document text into RAM."""
    if not query:
        return []
    try:
        with connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id,document_id,filename,source_location,text,bm25(chunk_search) AS rank "
                "FROM chunk_search WHERE chunk_search MATCH ? AND owner_id=? ORDER BY rank LIMIT ?",
                (query, owner_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.OperationalError as exc:
        if "chunk_search" in str(exc) or "fts5" in str(exc).lower() or "syntax error" in str(exc).lower():
            return []
        raise


def indexed_chunk_count():
    try:
        with connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunk_search").fetchone()[0])
    except sqlite3.OperationalError as exc:
        if "chunk_search" in str(exc) or "fts5" in str(exc).lower():
            return 0
        raise


def indexed_vocabulary(limit=20000):
    """Return a bounded vocabulary from the persistent FTS index for typo recovery."""
    try:
        with connect() as conn:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.chunk_vocab USING fts5vocab(main, chunk_search, 'row')")
            rows = conn.execute(
                "SELECT term FROM temp.chunk_vocab WHERE length(term) >= 3 ORDER BY doc DESC, term LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
            return [str(row[0]) for row in rows]
    except sqlite3.OperationalError as exc:
        if "chunk_search" in str(exc) or "fts5" in str(exc).lower():
            return []
        raise
