"""
tests/test_ingestion.py
========================
Unit tests for the document ingestion pipeline.

Run:
    python -m pytest tests/test_ingestion.py -v

Note: Tests that touch ChromaDB require the embedding model to be available.
      Lightweight tests (metadata, chunking) run without external dependencies.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from backend.utils.document_processor import (
    _make_documents,
    _subject_from_filename,
    process_file,
    SUPPORTED_EXTENSIONS,
)
from backend.utils.indexer import _calculate_chunk_ids


# ---------------------------------------------------------------------------
# _subject_from_filename
# ---------------------------------------------------------------------------

def test_subject_from_filename_pdf():
    assert _subject_from_filename("Physics_Notes.pdf") == "physics_notes"


def test_subject_from_filename_image():
    assert _subject_from_filename("exam question 3.png") == "exam_question_3"


def test_subject_from_filename_simple():
    assert _subject_from_filename("notes.pdf") == "notes"


# ---------------------------------------------------------------------------
# _make_documents
# ---------------------------------------------------------------------------

def test_make_documents_basic():
    """Verify _make_documents creates chunks with correct metadata."""
    text = "Hello world. " * 200  # Enough text to produce multiple chunks
    meta = {"source": "test.pdf", "filename": "test.pdf", "page": 1}
    docs = _make_documents(text, meta)

    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)

    # Check metadata propagation
    for doc in docs:
        assert doc.metadata["source"] == "test.pdf"
        assert doc.metadata["page"] == 1
        assert "chunk_index" in doc.metadata


def test_make_documents_empty():
    """Empty text should produce no documents."""
    assert _make_documents("", {"source": "x"}) == []
    assert _make_documents("   ", {"source": "x"}) == []
    assert _make_documents(None, {"source": "x"}) == []


# ---------------------------------------------------------------------------
# _calculate_chunk_ids
# ---------------------------------------------------------------------------

def test_chunk_id_format():
    """Chunk IDs should follow source:page:chunk_index format."""
    chunks = [
        Document(page_content="a", metadata={"source": "file.pdf", "page": 3, "chunk_index": 0}),
        Document(page_content="b", metadata={"source": "file.pdf", "page": 3, "chunk_index": 1}),
    ]
    result = _calculate_chunk_ids(chunks)
    assert result[0].metadata["id"] == "file.pdf:3:0"
    assert result[1].metadata["id"] == "file.pdf:3:1"


def test_chunk_id_uniqueness():
    """Different page/chunk combinations should produce unique IDs."""
    chunks = [
        Document(page_content="a", metadata={"source": "f.pdf", "page": 1, "chunk_index": 0}),
        Document(page_content="b", metadata={"source": "f.pdf", "page": 1, "chunk_index": 1}),
        Document(page_content="c", metadata={"source": "f.pdf", "page": 2, "chunk_index": 0}),
    ]
    result = _calculate_chunk_ids(chunks)
    ids = [c.metadata["id"] for c in result]
    assert len(ids) == len(set(ids)), "Chunk IDs should be unique"


# ---------------------------------------------------------------------------
# process_file – extension validation
# ---------------------------------------------------------------------------

def test_process_file_unsupported_extension():
    """Unsupported file types should return empty list."""
    docs = process_file("document.docx")
    assert docs == []


def test_supported_extensions_set():
    """Verify supported extensions match expected types."""
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".png" in SUPPORTED_EXTENSIONS
    assert ".jpg" in SUPPORTED_EXTENSIONS
    assert ".jpeg" in SUPPORTED_EXTENSIONS
    assert ".docx" not in SUPPORTED_EXTENSIONS
