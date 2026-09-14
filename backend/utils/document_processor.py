"""
backend/utils/document_processor.py
====================================
Unified document ingestion pipeline.

Responsibilities
----------------
1. Detect file type from path extension.
2. Native-text PDFs  → extract per-page text with pypdf.
3. Scanned PDF pages → render page to image (pdf2image / Poppler) then OCR.
4. Image files       → OCR via ocr_engine.
5. Split text into overlapping semantic chunks.
6. Attach rich metadata: filename, file_type, page, ocr_method, source.
7. Return list[Document] ready for ChromaDB.

Graceful degradation
--------------------
* If pdf2image / Poppler is not installed, scanned pages are skipped with a
  warning rather than crashing.  Native-text pages are still processed.
* If Tesseract is not installed, OCR pages are skipped with a warning.
"""

import io
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.utils.ocr_engine import extract_text_from_image, is_page_scanned

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

# ---------------------------------------------------------------------------
# Optional heavy imports – fail gracefully
# ---------------------------------------------------------------------------
try:
    from pypdf import PdfReader

    _PYPDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYPDF_AVAILABLE = False
    logger.warning("pypdf not installed – PDF text extraction disabled.")

_PDF2IMAGE_AVAILABLE = False
_convert_from_path = None
try:
    from pdf2image import convert_from_path as _convert_from_path  # type: ignore

    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    logger.warning(
        "pdf2image not installed (or Poppler missing). "
        "Scanned PDF pages will be skipped. "
        "Install pdf2image + Poppler to enable scanned-PDF OCR."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=SEPARATORS,
    )


def _subject_from_filename(filename: str) -> str:
    """Derive a subject tag from the filename (stem, lower-cased)."""
    return Path(filename).stem.lower().replace(" ", "_")


def _make_documents(
    text: str,
    base_metadata: dict,
) -> List[Document]:
    """Split *text* and wrap each chunk in a Document with *base_metadata*."""
    if not text or not text.strip():
        return []
    splitter = _get_splitter()
    chunks = splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        meta = {**base_metadata, "chunk_index": i}
        docs.append(Document(page_content=chunk, metadata=meta))
    return docs


def _poppler_path() -> Optional[str]:
    """Return POPPLER_PATH env var if set, else None (use system PATH)."""
    path = os.getenv("POPPLER_PATH", "").strip()
    return path if path else None


# ---------------------------------------------------------------------------
# Per-format extractors
# ---------------------------------------------------------------------------

def _process_pdf(file_path: str) -> List[Document]:
    """
    Extract text from a PDF.

    * Native pages  → use pypdf.
    * Scanned pages → render via pdf2image then OCR.
    """
    if not _PYPDF_AVAILABLE:
        logger.error("pypdf unavailable – cannot process PDF: %s", file_path)
        return []

    filename = Path(file_path).name
    subject = _subject_from_filename(filename)
    upload_ts = datetime.now(timezone.utc).isoformat()
    all_docs: List[Document] = []

    try:
        reader = PdfReader(file_path)
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", file_path, exc)
        return []

    for page_num, page in enumerate(reader.pages, start=1):
        t_page = time.perf_counter()
        native_text = ""
        try:
            native_text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("pypdf text extraction failed on page %d: %s", page_num, exc)

        base_meta = {
            "source": file_path,
            "filename": filename,
            "file_type": "pdf",
            "page": page_num,
            "subject": subject,
            "upload_timestamp": upload_ts,
        }

        if is_page_scanned(native_text):
            # ----------------------------------------------------------------
            # Scanned page – try OCR
            # ----------------------------------------------------------------
            if not _PDF2IMAGE_AVAILABLE:
                logger.warning(
                    "Page %d of '%s' appears scanned but pdf2image/Poppler is "
                    "not available – skipping OCR for this page.",
                    page_num,
                    filename,
                )
                continue

            try:
                poppler = _poppler_path()
                kwargs: dict[str, object] = dict(first_page=page_num, last_page=page_num, dpi=200)
                if poppler:
                    kwargs["poppler_path"] = poppler

                if _convert_from_path is None:  # pragma: no cover
                    continue
                images = _convert_from_path(file_path, **kwargs)
                if not images:
                    logger.warning("pdf2image returned no images for page %d", page_num)
                    continue

                img_bytes = io.BytesIO()
                images[0].save(img_bytes, format="PNG")
                img_bytes.seek(0)

                ocr_result = extract_text_from_image(img_bytes.read())
                ocr_text = (ocr_result.get("text") or "").strip()
                ocr_method = ocr_result.get("method", "ocr")

                if ocr_text:
                    page_ms = (time.perf_counter() - t_page) * 1000
                    logger.info(
                        "OCR page %d of '%s' → %d chars (method=%s, %.0f ms)",
                        page_num, filename, len(ocr_text), ocr_method, page_ms,
                    )
                    docs = _make_documents(
                        ocr_text,
                        {**base_meta, "ocr_method": ocr_method},
                    )
                    all_docs.extend(docs)
                else:
                    logger.warning("OCR returned no text for page %d of '%s'", page_num, filename)

            except Exception as exc:
                logger.error(
                    "OCR failed for page %d of '%s': %s", page_num, filename, exc
                )
        else:
            # ----------------------------------------------------------------
            # Native-text page
            # ----------------------------------------------------------------
            text = native_text.strip()
            if not text:
                continue
            page_ms = (time.perf_counter() - t_page) * 1000
            logger.debug("Native text page %d of '%s' → %d chars (%.0f ms)", page_num, filename, len(text), page_ms)
            docs = _make_documents(text, {**base_meta, "ocr_method": "native"})
            all_docs.extend(docs)

    logger.info("PDF '%s' → %d document chunks", filename, len(all_docs))
    return all_docs


def _process_image(file_path: str) -> List[Document]:
    """Extract text from an image file via OCR."""
    filename = Path(file_path).name
    subject = _subject_from_filename(filename)
    upload_ts = datetime.now(timezone.utc).isoformat()
    try:
        with open(file_path, "rb") as f:
            image_bytes = f.read()
    except Exception as exc:
        logger.error("Cannot read image file '%s': %s", file_path, exc)
        return []

    ocr_result = extract_text_from_image(image_bytes)
    if ocr_result.get("error"):
        logger.error("OCR error for '%s': %s", filename, ocr_result["error"])
        return []

    text = (ocr_result.get("text") or "").strip()
    if not text:
        logger.warning("OCR returned no text for image '%s'", filename)
        return []

    ocr_method = ocr_result.get("method", "ocr")
    logger.info("Image '%s' → %d chars (method=%s)", filename, len(text), ocr_method)

    ext = Path(file_path).suffix.lower().lstrip(".")
    base_meta = {
        "source": file_path,
        "filename": filename,
        "file_type": ext,
        "page": 1,
        "ocr_method": ocr_method,
        "subject": subject,
        "upload_timestamp": upload_ts,
    }
    docs = _make_documents(text, base_meta)
    logger.info("Image '%s' → %d document chunks", filename, len(docs))
    return docs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_file(file_path: str) -> List[Document]:
    """
    Process a single file and return a list of Document chunks.

    Supports: .pdf, .png, .jpg, .jpeg

    Returns an empty list if the file type is unsupported or processing fails.
    """
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning("Unsupported file type '%s' for '%s' – skipping.", ext, file_path)
        return []

    if ext == ".pdf":
        return _process_pdf(file_path)
    else:
        return _process_image(file_path)
