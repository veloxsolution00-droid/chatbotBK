from app.retrieval.chunker import chunk_text,digest
from app.ingestion.extractor import extract
import pytest
def test_chunks():
    assert len(chunk_text("hello. "*100,100,10))>1
def test_chunker_still_makes_progress_when_overlap_is_too_large():
    chunks = chunk_text("abcdefghij" * 20, 20, 20)
    assert 1 < len(chunks) < 30
def test_chunker_rejects_invalid_size():
    with pytest.raises(ValueError): chunk_text("text", 0, 0)
def test_digest():
    assert digest("a")==digest("a") and digest("a")!=digest("b")
def test_txt(tmp_path):
    p=tmp_path/"a.txt"; p.write_text("sample",encoding="utf-8")
    assert extract(p)[0][1]=="sample"
def test_markdown_is_extracted_as_text(tmp_path):
    p=tmp_path/"notes.md"; p.write_text("# Notes\nImportant detail",encoding="utf-8")
    assert "Important detail" in extract(p)[0][1]
def test_empty_text_reports_no_readable_content(tmp_path):
    from app.ingestion.extractor import NoReadableTextError
    p=tmp_path/"empty.txt"; p.write_text(" \n\t",encoding="utf-8")
    with pytest.raises(NoReadableTextError, match="No readable text"):
        extract(p)
def test_unsupported(tmp_path):
    p=tmp_path/"a.exe"; p.write_text("x")
    with pytest.raises(ValueError): extract(p)
