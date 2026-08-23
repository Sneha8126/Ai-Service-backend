"""
Extract text from study documents.

Supports:
- PDF
- Scanned PDF
- Handwritten PDF
- DOCX
- PPTX
- TXT
- JPG / JPEG
- PNG
- WEBP
- GIF
- BMP

For scanned/handwritten pages and images, OpenAI Vision is used
as an OCR fallback.
"""

from io import BytesIO
from pathlib import Path
import base64
import os

from pypdf import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
from openai import OpenAI

# PyMuPDF is used to render scanned PDF pages as images
import fitz


MAX_CHARS = 60_000

# Vision model for OCR
OCR_MODEL = os.getenv("OCR_MODEL", "gpt-4o-mini")


class UnsupportedDocumentError(Exception):
    pass


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    return OpenAI(api_key=api_key)


def extract_text(
    data: bytes,
    mime_type: str,
    filename: str = "document",
) -> str:

    suffix = Path(filename).suffix.lower()
    mime_type = (mime_type or "").lower().split(";")[0].strip()

    # ---------------------------------------------------------
    # PDF
    # ---------------------------------------------------------
    if mime_type == "application/pdf" or suffix == ".pdf":
        text = _extract_pdf(data)

        # If PDF has no usable text, use OCR
        if not text.strip():
            text = _ocr_pdf(data)

    # ---------------------------------------------------------
    # DOCX
    # ---------------------------------------------------------
    elif (
        mime_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or suffix == ".docx"
    ):
        text = _extract_docx(data)

    # ---------------------------------------------------------
    # PPTX
    # ---------------------------------------------------------
    elif (
        mime_type
        == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        or suffix == ".pptx"
    ):
        text = _extract_pptx(data)

    # ---------------------------------------------------------
    # TXT
    # ---------------------------------------------------------
    elif mime_type == "text/plain" or suffix == ".txt":
        text = _extract_txt(data)

    # ---------------------------------------------------------
    # IMAGES
    # ---------------------------------------------------------
    elif (
        mime_type.startswith("image/")
        or suffix in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
            ".bmp",
        }
    ):
        text = _ocr_image(data, mime_type, suffix)

    else:
        raise UnsupportedDocumentError(
            f"Unsupported document type: {mime_type or suffix}"
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "No readable text was found in this document. "
            "Try uploading a clearer PDF or image."
        )

    if len(text) > MAX_CHARS:
        text = (
            text[:MAX_CHARS]
            + "\n\n[Document truncated for length]"
        )

    return text


# ============================================================
# NORMAL PDF TEXT EXTRACTION
# ============================================================

def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))

    parts = []

    for page in reader.pages:
        page_text = page.extract_text() or ""

        if page_text.strip():
            parts.append(page_text)

    return "\n\n".join(parts)


# ============================================================
# OCR SCANNED / HANDWRITTEN PDF
# ============================================================

def _ocr_pdf(data: bytes) -> str:
    """
    Render each PDF page as an image and send it to
    OpenAI Vision for OCR.

    This handles:
    - scanned PDFs
    - handwritten PDFs
    - photos saved as PDFs
    """

    pdf = fitz.open(stream=data, filetype="pdf")

    parts = []

    try:
        for page_number, page in enumerate(pdf):

            # Render page at good resolution
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False,
            )

            image_bytes = pix.tobytes("png")

            page_text = _ocr_image_bytes(
                image_bytes,
                "image/png",
                page_number + 1,
            )

            if page_text.strip():
                parts.append(
                    f"[Page {page_number + 1}]\n{page_text}"
                )

            # Avoid processing an extremely large PDF
            if sum(len(x) for x in parts) >= MAX_CHARS:
                break

    finally:
        pdf.close()

    return "\n\n".join(parts)


# ============================================================
# IMAGE OCR
# ============================================================

def _ocr_image(
    data: bytes,
    mime_type: str,
    suffix: str = "",
) -> str:

    if not mime_type.startswith("image/"):
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }.get(suffix, "image/png")

    return _ocr_image_bytes(
        data,
        mime_type,
        1,
    )


def _ocr_image_bytes(
    image_bytes: bytes,
    mime_type: str,
    page_number: int = 1,
) -> str:

    client = _get_openai_client()

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=OCR_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a highly accurate OCR system for QuizNest. "
                    "Extract all readable educational content from the "
                    "provided image. This may contain printed text, "
                    "handwritten text, mathematical expressions, "
                    "questions, options, headings and notes. "
                    "Preserve the original meaning and structure. "
                    "Do not summarize, explain or invent missing text. "
                    "If some text is unclear, make the best possible "
                    "transcription and mark genuinely unreadable portions "
                    "as [unclear]."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all text from this image. "
                            "Include handwritten and printed text. "
                            "Return only the extracted/transcribed text."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{mime_type};base64,"
                                f"{encoded_image}"
                            )
                        },
                    },
                ],
            },
        ],
    )

    return response.choices[0].message.content or ""


# ============================================================
# DOCX
# ============================================================

def _extract_docx(data: bytes) -> str:
    doc = DocxDocument(BytesIO(data))

    parts = []

    # Normal paragraphs
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        if text:
            parts.append(text)

    # Tables
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip()
                for cell in row.cells
            )

            if row_text.strip():
                parts.append(row_text)

    return "\n".join(parts)


# ============================================================
# TXT
# ============================================================

def _extract_txt(data: bytes) -> str:
    return data.decode(
        "utf-8",
        errors="ignore",
    )


# ============================================================
# PPTX
# ============================================================

def _extract_pptx(data: bytes) -> str:
    prs = Presentation(BytesIO(data))

    parts = []

    for slide_number, slide in enumerate(prs.slides, start=1):

        slide_parts = []

        for shape in slide.shapes:

            if shape.has_text_frame:
                text = shape.text.strip()

                if text:
                    slide_parts.append(text)

        if slide_parts:
            parts.append(
                f"[Slide {slide_number}]\n"
                + "\n".join(slide_parts)
            )

    return "\n\n".join(parts)