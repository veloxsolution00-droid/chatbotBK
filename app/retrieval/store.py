from pathlib import Path
import logging
import time
import re
import chromadb
from app.core.config import settings
from app.storage import database as db
from app.retrieval.ollama import embed, embed_query

_client = None
_collection = None
logger = logging.getLogger(__name__)

_LEXICAL_STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "with",
}

def _query_terms(question):
    return {term for term in re.findall(r"[\w-]+", question.lower(), flags=re.UNICODE)
            if len(term) > 1 and term not in _LEXICAL_STOP_WORDS}

def _lexical_similarity(question_terms, text):
    """A small exact-term boost for names, dates, IDs, and table values.

    ChromaDB still supplies the candidate set, so this remains bounded even for
    large local document collections.
    """
    if not question_terms:
        return 0.0
    document_terms = set(re.findall(r"[\w-]+", text.lower(), flags=re.UNICODE))
    return len(question_terms & document_terms) / len(question_terms)

def _fts_query(question_terms):
    # Quote individual terms so punctuation cannot become FTS query syntax.
    return " OR ".join(f'"{term.replace(chr(34), "")}"' for term in sorted(question_terms))

def collection():
    global _client, _collection
    if _collection is None:
        Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(settings.chroma_path))
        # Never delete or recreate an existing collection at startup.
        _collection = _client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def add(records, progress_callback=None):
    """Embed and persist chunks in bounded batches, logging stage timings."""
    c = collection()
    batch_size = max(1, int(settings.embedding_batch_size))
    total_embed_seconds = 0.0
    total_write_seconds = 0.0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        started = time.perf_counter()
        vectors = embed([x["text"] for x in batch])
        total_embed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        c.upsert(ids=[x["id"] for x in batch],
                 documents=[x["text"] for x in batch],
                 embeddings=vectors,
                 metadatas=[x["metadata"] for x in batch])
        db.index_chunks(batch)
        total_write_seconds += time.perf_counter() - started
        if progress_callback:
            completed = min(i + len(batch), len(records))
            progress_callback("Embedding and saving chunks", 55 + int(40 * completed / max(1, len(records))),
                              f"Embedded and saved {completed}/{len(records)} chunks")
    logger.info(
        "Index stages: chunks=%d batch_size=%d embedding_seconds=%.2f chroma_write_seconds=%.2f",
        len(records), batch_size, total_embed_seconds, total_write_seconds
    )

def search(question, owner_id="local"):
    c = collection()
    if not c.count():
        return []
    q_vector = embed_query(question)
    multiplier = max(1, int(settings.retrieval_candidate_multiplier))
    wanted = max(1, int(settings.top_k))
    # Do not fetch every owner chunk just to count it.  That made chat latency
    # grow linearly with a user's corpus. Chroma safely returns fewer filtered
    # matches when this owner has fewer than the requested candidates.
    candidate_count = min(max(wanted, wanted * multiplier), c.count())
    r = c.query(query_embeddings=[q_vector], n_results=candidate_count,
                where={"owner_id": owner_id}, include=["documents", "metadatas", "distances"])
    docs = (r.get("documents") or [[]])[0] or []
    metas = (r.get("metadatas") or [[]])[0] or []
    distances = (r.get("distances") or [[]])[0] or []
    ids = (r.get("ids") or [[]])[0] or []
    lexical_weight = max(0.0, min(1.0, float(settings.hybrid_lexical_weight)))
    question_terms = _query_terms(question)
    results = []
    seen_chunk_ids = set()
    for chunk_id, document, metadata, distance in zip(ids, docs, metas, distances):
        if not document or distance is None:
            continue
        semantic_similarity = 1 - float(distance)
        lexical_similarity = _lexical_similarity(question_terms, document)
        hybrid_similarity = ((1 - lexical_weight) * semantic_similarity
                             + lexical_weight * lexical_similarity)
        results.append({
            "id": chunk_id, "text": document, "metadata": metadata or {},
            "similarity": hybrid_similarity,
            "semantic_similarity": semantic_similarity,
            "lexical_similarity": lexical_similarity,
        })
        seen_chunk_ids.add(chunk_id)

    # FTS brings in exact hits that semantic retrieval may not rank highly,
    # particularly identifiers, dates, codes, and values from tables.
    lexical_limit = max(candidate_count, int(settings.lexical_candidate_limit))
    for fts_hit in db.search_indexed_chunks(_fts_query(question_terms), owner_id, lexical_limit):
        if fts_hit["chunk_id"] in seen_chunk_ids:
            continue
        lexical_similarity = _lexical_similarity(question_terms, fts_hit["text"])
        results.append({
            "id": fts_hit["chunk_id"], "text": fts_hit["text"],
            "metadata": {"document_id": fts_hit["document_id"], "owner_id": owner_id,
                         "filename": fts_hit["filename"], "source_location": fts_hit["source_location"]},
            # A complete exact-term match is eligible even when its embedding
            # rank was weak or outside ChromaDB's semantic candidate window.
            "similarity": lexical_similarity,
            "semantic_similarity": None, "lexical_similarity": lexical_similarity,
        })
    logger.info("Retrieval: owner=%s candidates=%d returned=%d", owner_id, candidate_count, len(results))
    return sorted(results, key=lambda item: item["similarity"], reverse=True)

def delete(document_id):
    # Explicit deletion/recovery only; never called on normal initialization.
    collection().delete(where={"document_id": document_id})
    db.remove_indexed_chunks(document_id)

def document_chunk_count(document_id):
    c = collection()
    return len(c.get(where={"document_id": document_id}, include=["metadatas"])["ids"])

def list_chunk_document_ids():
    c = collection()
    ids = set()
    total = c.count()
    batch_size = 500
    for offset in range(0, total, batch_size):
        result = c.get(limit=batch_size, offset=offset, include=["metadatas"])
        for meta in result.get("metadatas") or []:
            if meta and meta.get("document_id"):
                ids.add(meta["document_id"])
    return ids


def backfill_owner_metadata(documents):
    """Add ownership metadata to legacy chunks without deleting/recreating the collection."""
    c = collection()
    for doc in documents:
        docid = doc.get("document_id")
        owner_id = "local"
        if not docid:
            continue
        result = c.get(where={"document_id": docid}, include=["metadatas"])
        ids = result.get("ids", [])
        metas = result.get("metadatas", [])
        for start in range(0, len(ids), 500):
            batch_ids = ids[start:start+500]
            batch_metas = []
            for meta in metas[start:start+500]:
                m = dict(meta or {})
                m["owner_id"] = owner_id
                batch_metas.append(m)
            if batch_ids:
                c.update(ids=batch_ids, metadatas=batch_metas)


def backfill_text_search():
    """Populate FTS for pre-existing ChromaDB chunks after an application update."""
    c = collection()
    total = c.count()
    if not total or db.indexed_chunk_count() >= total:
        return
    batch_size = 500
    for offset in range(0, total, batch_size):
        result = c.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
        records = []
        for chunk_id, text, metadata in zip(result.get("ids") or [], result.get("documents") or [], result.get("metadatas") or []):
            if text and metadata:
                records.append({"id": chunk_id, "text": text, "metadata": metadata})
        db.index_chunks(records)
