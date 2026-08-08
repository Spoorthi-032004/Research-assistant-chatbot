"""
PDF text extraction using PyMuPDF (fitz).
"""
from typing import List

import fitz  # PyMuPDF

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class InvalidPDFError(Exception):
    pass


def validate_pdf(file_path: str) -> None:
    """Raises InvalidPDFError if the file is not a readable PDF."""
    try:
        doc = fitz.open(file_path)
        if doc.page_count == 0:
            raise InvalidPDFError("PDF has no pages")
        doc.close()
    except InvalidPDFError:
        raise
    except Exception as exc:
        raise InvalidPDFError(f"File is not a valid PDF: {exc}") from exc


def extract_pages(file_path: str) -> List[str]:
    """Extract text per page, in order. Index 0 == page 1."""
    pages = []
    with fitz.open(file_path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return pages


def extract_text(file_path: str) -> str:
    """Extract full text content from a PDF, page by page, in order."""
    text_parts = extract_pages(file_path)
    full_text = "\n".join(text_parts)
    logger.info("Extracted %d characters from %s", len(full_text), file_path)
    return full_text


def extract_title_guess(file_path: str) -> str:
    """Best-effort title extraction: PDF metadata, else first non-empty line."""
    with fitz.open(file_path) as doc:
        meta_title = (doc.metadata or {}).get("title", "").strip()
        if meta_title:
            return meta_title
        if doc.page_count > 0:
            first_page_text = doc[0].get_text("text")
            for line in first_page_text.splitlines():
                stripped = line.strip()
                if len(stripped) > 8:
                    return stripped
    return "Untitled Paper"
