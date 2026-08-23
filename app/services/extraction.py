"""Extract plain text from supported study documents without shared filesystem access."""
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation

MAX_CHARS = 60_000


class UnsupportedDocumentError(Exception):
    pass


def extract_text(data: bytes, mime_type: str, filename: str = "document") -> str:
    suffix = Path(filename).suffix.lower()

    if mime_type == "application/pdf" or suffix == ".pdf":
        text = _extract_pdf(data)
    elif (
        mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or suffix == ".docx"
    ):
        text = _extract_docx(data)
    elif mime_type == "text/plain" or suffix == ".txt":
        text = _extract_txt(data)
    elif (
        mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        or suffix == ".pptx"
    ):
        text = _extract_pptx(data)
    else:
        raise UnsupportedDocumentError(f"Unsupported document type: {mime_type}")

    text = text.strip()
    if not text:
        raise ValueError("No extractable text was found in this document.")
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n\n[Document truncated for length]"
    return text


def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx(data: bytes) -> str:
    doc = DocxDocument(BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _extract_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def _extract_pptx(data: bytes) -> str:
    prs = Presentation(BytesIO(data))
    parts: list[str] = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text.strip()
                if text:
                    parts.append(text)
    return "\n".join(parts)
