"""Ingestion tests for document typing and structure-preserving extraction."""
import pytest

from app.ingestion.classify import detect_document_type
from app.ingestion.extractor import _clean_cell, _looks_tabular, _table_blocks, iter_extract


def test_detects_invoice_text():
    text = ("Invoice No: 2041\nBill To: Acme Ltd\nSubtotal 100.00\n"
            "VAT 20.00\nTotal Due 120.00\nDue Date: 2026-01-31")
    result = detect_document_type(text, "invoice-2041.pdf")
    assert result["type"] == "invoice"
    assert result["confidence"] >= 0.7


def test_detects_legal_contract():
    text = ("AGREEMENT\nWHEREAS the parties agree as follows.\n"
            "1. Termination. Either party may terminate this agreement with 30 days notice.\n"
            "Governing Law: England. Liability is limited.")
    result = detect_document_type(text, "service-contract.pdf")
    assert result["type"] == "legal"


def test_detects_question_bank():
    text = ("Which systems are in scope? How are alerts triaged? "
            "Who owns the SIEM? What is the retention period? How often is it reviewed?")
    result = detect_document_type(text, "siem.docx")
    assert result["type"] == "question_bank"
    assert result["signals"]["question_ratio"] >= 0.6


def test_spreadsheet_extension_wins():
    assert detect_document_type("anything", "ledger.xlsx")["type"] == "spreadsheet"


def test_detects_prose_book():
    text = ("It was a fine day and the sun shone brightly over the hills. " * 40)
    result = detect_document_type(text, "novel.txt")
    assert result["type"] == "prose"


def test_table_blocks_repeat_header_and_batch_rows():
    rows = [["Description", "Qty", "Amount"],
            ["Consulting", "10", "1500"],
            ["Support", "2", "200"],
            ["Training", "1", "300"]]
    blocks = list(_table_blocks(rows, "Table 1", rows_per_block=2))
    assert len(blocks) == 2
    assert all("Columns: Description | Qty | Amount" in body for _, body in blocks)
    assert "Row 1: Consulting | 10 | 1500" in blocks[0][1]
    assert "Row 3: Training | 1 | 300" in blocks[1][1]


def test_table_blocks_drop_empty_columns():
    rows = [["Description", "", "Amount"], ["Service", "", "100"]]
    blocks = list(_table_blocks(rows, "Table 1", rows_per_block=5))
    assert "Columns: Description | Amount" in blocks[0][1]


def test_clean_cell_joins_detached_punctuation():
    assert _clean_cell("1500\n.") == "1500."
    assert _clean_cell("two\nlines") == "two lines"


def test_looks_tabular_separates_prose():
    assert _looks_tabular("A    B    C\n1    2    3\n4    5    6")
    assert not _looks_tabular(
        "This is an ordinary paragraph with normal spacing. It continues for a while. "
        "Another sentence follows here without column alignment.")


def test_docx_preserves_paragraph_table_paragraph_order(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "invoice.docx"
    document = docx.Document()
    document.add_paragraph("Invoice No: 7")
    document.add_paragraph("Bill To: Acme")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Item"
    table.cell(0, 1).text = "Amount"
    table.cell(1, 0).text = "Service"
    table.cell(1, 1).text = "100"
    document.add_paragraph("Governing Law: England")
    document.save(path)

    blocks = list(iter_extract(path))
    locations = [loc for loc, _ in blocks]
    assert locations == ["document", "Table 1", "document part 2"]
    assert "Invoice No: 7" in blocks[0][1]
    assert "Columns: Item | Amount" in blocks[1][1]
    assert "Governing Law" in blocks[2][1]


def test_eml_extracts_supported_attachment(tmp_path):
    email_message = pytest.importorskip("email.message")
    path = tmp_path / "message.eml"
    message = email_message.EmailMessage()
    message["Subject"] = "Invoice 42"
    message["From"] = "sender@example.com"
    message.set_content("Please find the invoice attached.")
    message.add_attachment(b"Invoice No: 42\nTotal Due: 120.00\n",
                           maintype="text", subtype="plain", filename="invoice.txt")
    path.write_bytes(message.as_bytes())

    blocks = list(iter_extract(path))
    attachment_blocks = [text for loc, text in blocks if loc.startswith("email attachment:")]
    assert attachment_blocks
    assert "Total Due: 120.00" in attachment_blocks[0]
