from pathlib import Path
import json, csv, shutil, re, os, tempfile, xml.etree.ElementTree as ET
from html.parser import HTMLParser
SUPPORTED={".pdf",".docx",".txt",".md",".log",".html",".htm",".rtf",".odt",".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp",
           ".xlsx",".xlsm",".ods",".csv",".pptx",".json",".xml",".yaml",".yml",".eml"}

# A page with fewer characters than this is probably a scan (or a page whose
# text layer only covers a header), so OCR is worth attempting even if some text
# was found. Previously OCR ran only when the page text was completely empty.
_MIN_TEXT_LAYER_CHARS = 80
_TABLE_NAME_HINT = re.compile(r"invoice|receipt|statement|bill|quote|purchase|order|ledger|table|report", re.I)

class NoReadableTextError(ValueError):
    """The file was processed successfully but contains no indexable text."""


class _VisibleTextParser(HTMLParser):
    """Extract human-visible text while dropping scripts/styles and tags."""

    _BLOCK_TAGS = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")
        elif tag in {"td", "th"}:
            self._parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self._parts.append(data)

    def text(self):
        return re.sub(r"\n{3,}", "\n\n", "".join(self._parts)).strip()


def _html_to_text(markup):
    parser = _VisibleTextParser()
    try:
        parser.feed(str(markup or ""))
        parser.close()
        return parser.text()
    except Exception:
        # Malformed HTML should not make an otherwise readable file fail; fall
        # back to the raw text rather than discarding user content.
        return re.sub(r"\s+", " ", str(markup or "")).strip()


def _text_encoding(path):
    """Pick a practical encoding for delimited/text files without failing on one byte."""
    sample = Path(path).open("rb").read(65536)
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "utf-8"


def _get_tesseract():
    from app.core.config import settings
    if not settings.ocr_enabled:
        return None
    cmd = settings.tesseract_cmd
    if cmd and Path(cmd).is_file():
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = cmd
        return pytesseract
    if shutil.which("tesseract"):
        import pytesseract
        return pytesseract
    return None

def _row_text(row):
    return " | ".join("" if value is None else str(value) for value in row)


def _looks_tabular(text):
    """Cheap pre-check so table probing is skipped on ordinary prose pages."""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    def tabular_line(line):
        if line.count(" | ") >= 2:
            return True
        # Count column gaps without consuming the following word, otherwise two
        # adjacent matches overlap and a three-column row scores as one.
        runs = len(re.findall(r"\S\s{2,}(?=\S)", line))
        if runs >= 2:
            return True
        # Short aligned rows ("Total    120.00") still count; long prose lines
        # with a double space after a full stop do not.
        return runs >= 1 and len(line.split()) <= 8

    spaced = sum(1 for line in lines if tabular_line(line))
    return spaced / len(lines) >= 0.25


def _clean_cell(value):
    """Collapse a cell's internal line breaks without inserting stray spaces."""
    parts = [part.strip() for part in str(value or "").replace("\r", "").splitlines() if part.strip()]
    cleaned = ""
    for part in parts:
        if cleaned and re.fullmatch(r"[.,;:%)]+", part):
            cleaned += part
        elif cleaned:
            cleaned += " " + part
        else:
            cleaned = part
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _table_blocks(rows, label, rows_per_block):
    """Yield table rows with the header repeated, preserving column structure."""
    cleaned = [[_clean_cell(cell) for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return
    # Bordered-table detection can invent empty columns where a text span sits
    # next to a rule. Drop columns that are empty in every row; no data is lost.
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    keep = [index for index in range(width) if any(row[index] for row in cleaned)]
    if keep:
        cleaned = [[row[index] for index in keep] for row in cleaned]
    header, body = cleaned[0], cleaned[1:]
    if not body:
        yield (label, f"{label}\n{_row_text(header)}")
        return
    batch_size = max(1, int(rows_per_block or 1))
    for start in range(0, len(body), batch_size):
        batch = body[start:start + batch_size]
        lines = [f"Row {start + index + 1}: {_row_text(row)}"
                 for index, row in enumerate(batch)]
        yield (label, f"{label}\nColumns: {_row_text(header)}\n" + "\n".join(lines))


def _text_outside_tables(page, table_rects):
    """Read page text but skip blocks that belong to an extracted table.

    Without this the table text would be indexed twice: once flattened by
    `get_text()` and once as structured rows. The flattened copy is exactly the
    view that loses column meaning, so it is the copy to drop.
    """
    if not table_rects:
        return page.get_text().strip()
    try:
        import pymupdf
        Rect = pymupdf.Rect
    except ImportError:
        from fitz import Rect
    parts = []
    for block in page.get_text("blocks"):
        if len(block) < 7 or block[6] != 0:
            continue
        rect = Rect(block[0], block[1], block[2], block[3])
        if any(rect.intersects(table_rect) for table_rect in table_rects):
            continue
        text = str(block[4]).strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _ocr_image(pytess, image):
    """OCR one image with block segmentation, which handles tables far better."""
    from PIL import ImageOps
    try:
        return pytess.image_to_string(ImageOps.grayscale(image), config="--psm 6").strip()
    except Exception:
        return ""


def _ocr_pdf_page(pytess, page):
    from PIL import Image
    from io import BytesIO
    try:
        pixmap = page.get_pixmap(dpi=300)
        return _ocr_image(pytess, Image.open(BytesIO(pixmap.tobytes("png"))))
    except Exception:
        return ""


def _csv_blocks(path, rows_per_block):
    with open(path, encoding=_text_encoding(path), errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return
        batch, start = [], 2
        for row_number, row in enumerate(reader, 2):
            batch.append(f"Row {row_number}: {_row_text(row)}")
            if len(batch) >= rows_per_block:
                yield (f"CSV rows {start}-{row_number}", "Columns: " + _row_text(header) + "\n" + "\n".join(batch))
                batch, start = [], row_number + 1
        if batch:
            yield (f"CSV rows {start}-{start + len(batch) - 1}", "Columns: " + _row_text(header) + "\n" + "\n".join(batch))


def _excel_blocks(path, ext, rows_per_block):
    # read_only mode avoids loading a whole workbook into RAM.  ODS remains on
    # pandas because openpyxl does not support it.
    if ext == ".ods":
        import pandas as pd
        workbook = pd.ExcelFile(path, engine="odf")
        for sheet in workbook.sheet_names:
            frame = pd.read_excel(path, sheet_name=sheet, dtype=str, engine="odf").fillna("")
            header = list(map(str, frame.columns))
            for start in range(0, len(frame), rows_per_block):
                part = frame.iloc[start:start + rows_per_block]
                lines = [f"Row {idx + 2}: " + " | ".join(f"{column}={row[column]}" for column in frame.columns) for idx, row in part.iterrows()]
                if lines:
                    yield (f"sheet {sheet} rows {start + 2}-{start + len(lines) + 1}", "Sheet: " + sheet + "\nColumns: " + " | ".join(header) + "\n" + "\n".join(lines))
        return
    from openpyxl import load_workbook
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        for worksheet in workbook.worksheets:
            rows = worksheet.iter_rows(values_only=True)
            header = next(rows, None)
            if header is None:
                continue
            batch, start = [], 2
            for row_number, row in enumerate(rows, 2):
                batch.append(f"Row {row_number}: {_row_text(row)}")
                if len(batch) >= rows_per_block:
                    yield (f"sheet {worksheet.title} rows {start}-{row_number}", "Sheet: " + worksheet.title + "\nColumns: " + _row_text(header) + "\n" + "\n".join(batch))
                    batch, start = [], row_number + 1
            if batch:
                yield (f"sheet {worksheet.title} rows {start}-{start + len(batch) - 1}", "Sheet: " + worksheet.title + "\nColumns: " + _row_text(header) + "\n" + "\n".join(batch))
    finally:
        workbook.close()


def iter_extract(path):
    """Yield source-labelled text blocks without loading tabular files as one string."""
    from app.core.config import settings
    ext=Path(path).suffix.lower(); yielded = False
    if ext not in SUPPORTED: raise ValueError(f"Unsupported extension: {ext}")
    if ext==".pdf":
        try:
            import pymupdf
        except ImportError:  # PyMuPDF 1.24.0-1.24.2 only ships the legacy module name
            import fitz as pymupdf
        with pymupdf.open(path) as pdf:
            if pdf.page_count > settings.max_pdf_pages:
                raise ValueError(f"PDF has {pdf.page_count} pages, above the configured MAX_PDF_PAGES limit of {settings.max_pdf_pages}.")
            probe_tables = bool(_TABLE_NAME_HINT.search(Path(path).name))
            pytess = None
            for i,p in enumerate(pdf,1):
                raw_text = p.get_text().strip()
                # Bordered tables often have almost no plain text between cells;
                # column-aligned invoices show repeated whitespace runs instead.
                tables = []
                if probe_tables or _looks_tabular(raw_text):
                    try:
                        tables = list(p.find_tables().tables)
                    except Exception:
                        tables = []
                table_rects = []
                for table in tables:
                    try:
                        table_rects.append(pymupdf.Rect(table.bbox))
                    except Exception:
                        continue
                text = _text_outside_tables(p, table_rects).strip()
                if len(raw_text) < _MIN_TEXT_LAYER_CHARS:
                    # A short text layer usually means a scan or a header-only
                    # page. OCR the rendered page and keep it when it is richer.
                    if pytess is None:
                        pytess = _get_tesseract()
                    if pytess:
                        ocr_text = _ocr_pdf_page(pytess, p)
                        if len(ocr_text) > len(text):
                            text = ocr_text
                if text:
                    yielded = True; yield (f"page {i}", text)
                for table_number, table in enumerate(tables, 1):
                    try:
                        rows = table.extract()
                    except Exception:
                        continue
                    for loc, body in _table_blocks(
                            rows, f"Table {table_number} — page {i}",
                            settings.extraction_block_rows):
                        yielded = True; yield (loc, body)
    elif ext==".docx":
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        document = Document(path)
        # Walk the body in document order so a table stays next to the heading
        # that introduces it. The old version appended every table at the end.
        buffer, part, table_number = [], 1, 0
        for child in document.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                text = Paragraph(child, document).text.strip()
                if text:
                    buffer.append(text)
            elif tag == "tbl":
                if buffer:
                    loc = "document" if part == 1 else f"document part {part}"
                    yielded = True; yield (loc, "\n".join(buffer))
                    buffer, part = [], part + 1
                table_number += 1
                try:
                    rows = [[cell.text.strip() for cell in row.cells]
                            for row in Table(child, document).rows]
                except Exception:
                    continue
                for loc, body in _table_blocks(rows, f"Table {table_number}",
                                               settings.extraction_block_rows):
                    yielded = True; yield (loc, body)
        if buffer:
            loc = "document" if part == 1 else f"document part {part}"
            yielded = True; yield (loc, "\n".join(buffer))
    elif ext in {".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp"}:
        pytess = _get_tesseract()
        if not pytess:
            raise RuntimeError(
                f"Tesseract OCR is not installed or configured. Cannot extract text from '{Path(path).name}'. "
                f"Install Tesseract OCR or set TESSERACT_CMD in .env to the tesseract.exe path."
            )
        from PIL import Image, ImageSequence
        with Image.open(path) as image:
            # Multi-page TIFFs are common for scanned invoices. The old code read
            # only the first frame, silently dropping every later page.
            frame_count = getattr(image, "n_frames", 1)
            if frame_count > settings.max_pdf_pages:
                raise ValueError(f"Image has {frame_count} frames, above the configured MAX_PDF_PAGES limit of {settings.max_pdf_pages}.")
            for index, frame in enumerate(ImageSequence.Iterator(image), 1):
                if frame.width * frame.height > settings.max_image_pixels:
                    raise ValueError(f"Image frame has {frame.width * frame.height:,} pixels, above the configured MAX_IMAGE_PIXELS limit of {settings.max_image_pixels:,}.")
                text = _ocr_image(pytess, frame)
                if text:
                    yielded = True
                    yield (f"image frame {index}" if frame_count > 1 else "image", text)
    elif ext in {".txt", ".md", ".log", ".html", ".htm", ".yaml", ".yml", ".rtf"}:
        text=Path(path).read_text(encoding=_text_encoding(path),errors="replace")
        if ext in {".html", ".htm"}:
            text=_html_to_text(text)
        elif ext==".rtf":
            from striprtf.striprtf import rtf_to_text
            text=rtf_to_text(text)
        if text.strip(): yielded = True; yield ("document",text)
    elif ext in {".xlsx",".xlsm",".ods"}:
        for block in _excel_blocks(path, ext, settings.extraction_block_rows):
            yielded = True; yield block
    elif ext==".csv":
        for block in _csv_blocks(path, settings.extraction_block_rows):
            yielded = True; yield block
    elif ext==".pptx":
        from pptx import Presentation
        for i,s in enumerate(Presentation(path).slides,1):
            text="\n".join(sh.text for sh in s.shapes if hasattr(sh,"text") and sh.text.strip())
            if text: yielded = True; yield (f"slide {i}",text)
    elif ext==".json":
        text = Path(path).read_text(encoding=_text_encoding(path), errors="replace").lstrip("\ufeff")
        text = json.dumps(json.loads(text),ensure_ascii=False,indent=2)
        if text.strip(): yielded = True; yield ("JSON", text)
    elif ext==".xml":
        text = ET.tostring(ET.parse(path).getroot(),encoding="unicode")
        if text.strip(): yielded = True; yield ("XML", text)
    elif ext==".eml":
        from email import policy
        from email.parser import BytesParser
        m=BytesParser(policy=policy.default).parsebytes(Path(path).read_bytes()); body=m.get_body(preferencelist=("plain","html"))
        content = body.get_content() if body else ""
        if body is not None and body.get_content_type() == "text/html":
            content = _html_to_text(content)
        header = (f"Subject: {m.get('subject','')}\nFrom: {m.get('from','')}\n"
                  f"To: {m.get('to','')}\nDate: {m.get('date','')}\n")
        text = header + content
        if text.strip(): yielded = True; yield ("email", text)
        # Invoices and contracts usually arrive as attachments, not as the email
        # body. Extract supported attachments under the same size limits.
        for attachment in m.iter_attachments():
            filename = attachment.get_filename() or "attachment"
            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED or suffix == ".eml":
                continue
            try:
                data = attachment.get_payload(decode=True) or b""
            except Exception:
                continue
            if not data or len(data) > settings.max_file_mb * 1024 * 1024:
                continue
            temp = None
            try:
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp.write(data); temp.close()
                for loc, block in iter_extract(temp.name):
                    yielded = True
                    yield (f"email attachment: {filename} / {loc}", block)
            except Exception:
                # A single unreadable attachment must not discard the email body.
                continue
            finally:
                if temp is not None:
                    try:
                        os.unlink(temp.name)
                    except OSError:
                        pass
    elif ext==".odt":
        import zipfile
        with zipfile.ZipFile(path) as z: xml=z.read("content.xml")
        tree=ET.fromstring(xml); ns={"t":"urn:oasis:names:tc:opendocument:xmlns:text:1.0"}
        text = "\n".join("".join(p.itertext()) for p in tree.findall(".//t:p",ns))
        if text.strip(): yielded = True; yield ("ODT", text)
    if not yielded:
        raise NoReadableTextError("No readable text was found in the uploaded file. It may be empty, image-only with OCR disabled, or OCR may have failed.")


def extract(path):
    """Compatibility wrapper used by callers that need all blocks at once."""
    return list(iter_extract(path))
