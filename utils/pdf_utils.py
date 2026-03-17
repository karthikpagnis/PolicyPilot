"""PDF utility helpers for page rendering and metadata extraction."""

import base64
import fitz   # PyMuPDF

DEFAULT_RENDER_DPI = 220


def pdf_pages_to_images(pdf_bytes: bytes, dpi: int = DEFAULT_RENDER_DPI) -> dict:
    """
    Converts every page of a PDF into a base64-encoded PNG image.
    Returns: { "page_1": "base64string...", "page_2": "...", ... }
    Higher DPI helps OCR for scanned or low-quality documents.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = {}

    for i, page in enumerate(doc):
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        png_bytes = pix.tobytes("png")
        b64 = base64.standard_b64encode(png_bytes).decode("utf-8")
        pages[f"page_{i + 1}"] = b64

    doc.close()
    return pages


def get_specific_page_images(
    pdf_bytes: bytes,
    page_numbers: list,
    dpi: int = DEFAULT_RENDER_DPI,
) -> dict:
    """
    Returns base64 PNG images for ONLY the requested page numbers (1-indexed).
    Extraction agents use this so they only see their relevant pages.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = {}
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for page_num in page_numbers:
        idx = page_num - 1
        if idx < 0 or idx >= len(doc):
            print(f"   Warning: page {page_num} does not exist (total pages: {len(doc)})")
            continue

        pix = doc[idx].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        png_bytes = pix.tobytes("png")
        b64 = base64.standard_b64encode(png_bytes).decode("utf-8")
        pages[f"page_{page_num}"] = b64

    doc.close()
    return pages


def get_page_count(pdf_bytes: bytes) -> int:
    """Returns the total number of pages in a PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


def is_likely_plain_pdf(
    pdf_bytes: bytes,
    sample_pages: int = 3,
    min_text_chars: int = 40,
) -> bool:
    """Heuristic: True when PDF appears to contain machine-readable text."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return False

    total_chars = 0
    try:
        pages_to_check = min(len(doc), max(1, sample_pages))
        for idx in range(pages_to_check):
            total_chars += len(doc[idx].get_text("text").strip())
            if total_chars >= min_text_chars:
                return True
        return False
    finally:
        doc.close()


def is_scanned_pdf(
    pdf_bytes: bytes,
    sample_pages: int = 3,
) -> bool:
    """Check if PDF is primarily scanned/image-based (has images on pages)."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return False

    try:
        pages_to_check = min(len(doc), max(1, sample_pages))
        for idx in range(pages_to_check):
            page = doc[idx]
            images = page.get_images()
            if len(images) > 0:
                return True
        return False
    finally:
        doc.close()
