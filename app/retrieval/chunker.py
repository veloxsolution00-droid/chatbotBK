"""Structure-aware, bounded-overlap text chunking for offline RAG."""
from __future__ import annotations
import hashlib
import re

_HEADING = re.compile(
    r"(?m)^\s*(?:#{1,6}\s+|\d+(?:\.\d+)*[.)]?\s+|[A-Z][A-Z0-9 /&()_-]{3,}:\s*$)"
)


def _split_long_block(block: str, chunk_size: int) -> list[str]:
    """Split a long paragraph at sentence/line/word boundaries when possible."""
    parts = []
    start = 0
    while start < len(block):
        end = min(start + chunk_size, len(block))
        if end < len(block):
            candidates = [
                block.rfind("\n", start, end),
                block.rfind(". ", start, end),
                block.rfind("? ", start, end),
                block.rfind("! ", start, end),
                block.rfind("; ", start, end),
                block.rfind(" ", start, end),
            ]
            boundary = max(candidates)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        piece = block[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= len(block):
            break
        start = end
    return parts


def chunk_text(text, chunk_size=1800, overlap=300):
    """Prefer paragraph/heading boundaries, then sentences, with bounded overlap.

    The extractor already yields page/slide/sheet/row-labelled blocks. This
    function preserves those source boundaries and avoids splitting normal
    paragraphs unless they exceed the configured chunk size. It is a
    structure-aware heuristic, not embedding-based semantic segmentation.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    overlap = max(0, min(int(overlap), chunk_size // 2))
    text = re.sub(r"\r\n?", "\n", str(text or ""))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    # Split on blank lines and heading starts while keeping headings attached
    # to the section that follows them.
    raw_blocks = re.split(r"\n\s*\n", text)
    blocks: list[str] = []
    for raw in raw_blocks:
        lines = raw.splitlines()
        current = []
        for line in lines:
            if _HEADING.match(line) and current:
                blocks.append("\n".join(current).strip())
                current = []
            current.append(line)
        if current:
            block = "\n".join(current).strip()
            if block:
                blocks.append(block)

    atomic: list[str] = []
    for block in blocks:
        if len(block) <= chunk_size:
            atomic.append(block)
        else:
            atomic.extend(_split_long_block(block, chunk_size))

    result: list[str] = []
    current = ""
    for block in atomic:
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            result.append(current.strip())
            if overlap:
                # Carry a tail of the previous chunk into the next one, but
                # preserve the forward-progress guarantee for long blocks.
                tail = current[-overlap:].strip()
                current = f"{tail}\n\n{block}" if tail else block
            else:
                current = block
            if len(current) > chunk_size:
                pieces = _split_long_block(current, chunk_size)
                result.extend(pieces[:-1])
                current = pieces[-1] if pieces else ""
        else:
            pieces = _split_long_block(block, chunk_size)
            result.extend(pieces[:-1])
            current = pieces[-1] if pieces else ""
    if current.strip():
        result.append(current.strip())

    # Guard the invariant: every chunk is non-empty and bounded.
    return [part for part in result if part and len(part) <= chunk_size]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
