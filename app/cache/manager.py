"""Bounded, local-only SQLite cache with conservative semantic matching.

Cached RAG results are scoped by owner, indexed-document fingerprint and a
retrieval/answer configuration fingerprint. The cache is an optimization only;
ChromaDB and the original document store remain the source of truth.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
import logging
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)
_LOCK = RLock()
_METRICS = {"embedding_hits": 0, "embedding_misses": 0, "retrieval_hits": 0,
            "retrieval_misses": 0, "answer_hits": 0, "answer_misses": 0,
            "semantic_hits": 0, "writes": 0, "invalidations": 0}
_SCHEMA_VERSION = "rag-cache-v1"
_SCHEMA_READY_PATH = None


def normalize_query(text: str) -> str:
    text = str(text or "").casefold().strip()
    text = re.sub(r"[^\w\s%./:-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _digest(*parts: Any) -> str:
    raw = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _db_path() -> Path:
    # Dedicated cache database: it can be deleted without touching documents,
    # extracted text, the metadata database, or the ChromaDB index.
    return Path(settings.metadata_path) / "rag_cache.sqlite3"


def _connect() -> sqlite3.Connection:
    global _SCHEMA_READY_PATH
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    if _SCHEMA_READY_PATH != str(path):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_type TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                owner_id TEXT NOT NULL DEFAULT '',
                normalized_query TEXT NOT NULL DEFAULT '',
                query_vector_json TEXT,
                corpus_fingerprint TEXT NOT NULL DEFAULT '',
                config_fingerprint TEXT NOT NULL DEFAULT '',
                question_type TEXT NOT NULL DEFAULT '',
                anchors_json TEXT NOT NULL DEFAULT '[]',
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_access REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(cache_type, cache_key)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_scope ON cache_entries(cache_type, owner_id, corpus_fingerprint, config_fingerprint, last_access)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expiry ON cache_entries(created_at)")
        conn.commit()
        _SCHEMA_READY_PATH = str(path)
    return conn


@contextmanager
def _session():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _enabled() -> bool:
    return bool(getattr(settings, "cache_enabled", True))


def _ttl() -> int:
    return max(60, int(getattr(settings, "cache_ttl_seconds", 604800)))


def _max_entries() -> int:
    return max(100, int(getattr(settings, "cache_max_entries", 2000)))


def _clean_expired(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM cache_entries WHERE created_at < ?", (time.time() - _ttl(),))
    excess = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0] - _max_entries()
    if excess > 0:
        conn.execute("DELETE FROM cache_entries WHERE rowid IN (SELECT rowid FROM cache_entries ORDER BY last_access ASC LIMIT ?)", (excess,))


def _anchors(text: str) -> set[str]:
    """Capture numbers, dates, IDs and acronyms that paraphrases must preserve."""
    found = set(re.findall(r"\b(?:\d{2,4}(?:[-/.]\d{1,2}){0,2}|[A-Z]{2,}[A-Z0-9_-]*|[A-Za-z]*\d+[A-Za-z0-9_-]*)\b", str(text)))
    return {x.casefold() for x in found}


def _compatible_anchors(query: str, anchors_json: str) -> bool:
    try:
        prior = set(json.loads(anchors_json or "[]"))
    except (ValueError, TypeError):
        return False
    current = _anchors(query)
    # If either query carries a hard constraint, both must carry exactly the
    # same constraints. This prevents 2024/2025 or ID A12/B12 cache collisions.
    return current == prior


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    return dot / (na * nb) if na and nb else -1.0


def get_embedding(text: str, model: str) -> list[float] | None:
    if not _enabled():
        return None
    key = _digest(_SCHEMA_VERSION, "embedding", model, text)
    with _LOCK, _session() as conn:
        row = conn.execute("SELECT payload_json, created_at FROM cache_entries WHERE cache_type='embedding' AND cache_key=?", (key,)).fetchone()
        if not row or row["created_at"] < time.time() - _ttl():
            _METRICS["embedding_misses"] += 1
            return None
        conn.execute("UPDATE cache_entries SET last_access=?, hit_count=hit_count+1 WHERE cache_type='embedding' AND cache_key=?", (time.time(), key))
        _METRICS["embedding_hits"] += 1
        try:
            return json.loads(row["payload_json"])
        except (ValueError, TypeError):
            return None


def put_embedding(text: str, model: str, vector: list[float]) -> None:
    if not _enabled() or not vector:
        return
    key = _digest(_SCHEMA_VERSION, "embedding", model, text)
    now = time.time()
    with _LOCK, _session() as conn:
        conn.execute("INSERT OR REPLACE INTO cache_entries(cache_type,cache_key,normalized_query,payload_json,created_at,last_access) VALUES('embedding',?,?,?,?,?)",
                     (key, text, json.dumps(vector, separators=(",", ":")), now, now))
        _clean_expired(conn)
        conn.commit()
        _METRICS["writes"] += 1


def lookup_result(cache_type: str, query: str, owner_id: str, corpus_fingerprint: str,
                  config_fingerprint: str, query_vector: list[float] | None = None,
                  question_type: str = "", semantic_threshold: float = 1.1) -> dict | None:
    """Return an exact or conservative semantic result cache entry."""
    if not _enabled() or cache_type not in {"retrieval", "answer"}:
        return None
    normalized = normalize_query(query)
    key = _digest(_SCHEMA_VERSION, cache_type, owner_id, normalized, corpus_fingerprint, config_fingerprint)
    with _LOCK, _session() as conn:
        row = conn.execute("""SELECT * FROM cache_entries WHERE cache_type=? AND cache_key=?
            AND owner_id=? AND corpus_fingerprint=? AND config_fingerprint=? AND created_at>=?""",
            (cache_type, key, str(owner_id), corpus_fingerprint, config_fingerprint, time.time() - _ttl())).fetchone()
        match_kind, similarity = "exact", 1.0
        if row is None and query_vector and semantic_threshold <= 1.0:
            rows = conn.execute("""SELECT * FROM cache_entries WHERE cache_type=? AND owner_id=?
                AND corpus_fingerprint=? AND config_fingerprint=? AND question_type=? AND created_at>=?
                AND query_vector_json IS NOT NULL ORDER BY last_access DESC LIMIT 300""",
                (cache_type, str(owner_id), corpus_fingerprint, config_fingerprint, question_type, time.time() - _ttl())).fetchall()
            best = None
            best_score = -1.0
            current_anchors = _anchors(query)
            for candidate in rows:
                try:
                    prior_vector = json.loads(candidate["query_vector_json"])
                except (ValueError, TypeError):
                    continue
                score = _cosine(query_vector, prior_vector)
                if score >= semantic_threshold and score > best_score and _compatible_anchors(query, candidate["anchors_json"]):
                    best, best_score = candidate, score
            row = best
            if row is not None:
                match_kind, similarity = "semantic", best_score
        if row is None:
            _METRICS[f"{cache_type}_misses"] = _METRICS.get(f"{cache_type}_misses", 0) + 1
            return None
        conn.execute("UPDATE cache_entries SET last_access=?, hit_count=hit_count+1 WHERE cache_type=? AND cache_key=?",
                     (time.time(), cache_type, row["cache_key"]))
        try:
            payload = json.loads(row["payload_json"])
        except (ValueError, TypeError):
            return None
        _METRICS[f"{cache_type}_hits"] = _METRICS.get(f"{cache_type}_hits", 0) + 1
        if match_kind == "semantic":
            _METRICS["semantic_hits"] += 1
        return {"payload": payload, "match_kind": match_kind,
                "similarity": round(float(similarity), 5), "cached_query": row["normalized_query"]}


def put_result(cache_type: str, query: str, owner_id: str, corpus_fingerprint: str,
               config_fingerprint: str, payload: dict, query_vector: list[float] | None = None,
               question_type: str = "") -> None:
    if not _enabled() or cache_type not in {"retrieval", "answer"}:
        return
    if not isinstance(payload, dict):
        return
    if cache_type == "answer" and (payload.get("status") != "success" or payload.get("abstained") or not payload.get("sources")):
        return
    normalized = normalize_query(query)
    key = _digest(_SCHEMA_VERSION, cache_type, owner_id, normalized, corpus_fingerprint, config_fingerprint)
    now = time.time()
    anchors = sorted(_anchors(query))
    with _LOCK, _session() as conn:
        conn.execute("""INSERT OR REPLACE INTO cache_entries
            (cache_type,cache_key,owner_id,normalized_query,query_vector_json,corpus_fingerprint,
             config_fingerprint,question_type,anchors_json,payload_json,created_at,last_access,hit_count)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,0)""",
            (cache_type, key, str(owner_id), normalized,
             json.dumps(query_vector, separators=(",", ":")) if query_vector else None,
             corpus_fingerprint, config_fingerprint, question_type, json.dumps(anchors),
             json.dumps(payload, ensure_ascii=False, separators=(",", ":")), now, now))
        _clean_expired(conn)
        conn.commit()
        _METRICS["writes"] += 1


def corpus_fingerprint(documents: list[dict], indexed_ids: set[str]) -> str:
    material = []
    for doc in documents or []:
        if doc.get("document_id") not in indexed_ids:
            continue
        material.append((doc.get("document_id"), doc.get("content_hash"), doc.get("updated_at"), doc.get("chunk_count"), doc.get("indexing_status")))
    # For tests or integrations passing only indexed IDs, remain deterministic.
    if not material:
        material = [(doc_id,) for doc_id in sorted(indexed_ids)]
    return _digest(_SCHEMA_VERSION, json.dumps(sorted(material), sort_keys=True, default=str))


def config_fingerprint(kind: str, values: dict) -> str:
    return _digest(_SCHEMA_VERSION, kind, json.dumps(values, sort_keys=True, default=str))


def stats() -> dict:
    result = dict(_METRICS)
    if not _enabled():
        result.update({"enabled": False, "entries": 0})
        return result
    try:
        with _LOCK, _session() as conn:
            rows = conn.execute("SELECT cache_type, COUNT(*) AS n FROM cache_entries GROUP BY cache_type").fetchall()
        result["entries_by_type"] = {row["cache_type"]: row["n"] for row in rows}
        result["entries"] = sum(result["entries_by_type"].values())
    except Exception:
        logger.exception("Unable to read cache statistics")
        result["entries"] = None
    result["enabled"] = True
    result["database"] = str(_db_path())
    return result


def clear_result_caches(owner_id: str | None = None) -> int:
    """Clear RAG answer/retrieval entries only; leave query embeddings intact."""
    if not _enabled():
        return 0
    with _LOCK, _session() as conn:
        if owner_id is None:
            cursor = conn.execute("DELETE FROM cache_entries WHERE cache_type IN ('retrieval','answer')")
        else:
            cursor = conn.execute("DELETE FROM cache_entries WHERE cache_type IN ('retrieval','answer') AND owner_id=?", (str(owner_id),))
        count = max(0, cursor.rowcount)
        conn.commit()
        _METRICS["invalidations"] += count
        return count
