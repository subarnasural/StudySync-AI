"""
backend/utils/indexer.py
=========================
ChromaDB service layer.

Provides a clean interface for indexing documents and managing the vector store.
All direct ChromaDB calls are centralised here so the rest of the codebase
stays free of persistence details.
"""

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.llm_manager import get_embedding_function
from backend.utils.document_processor import process_file, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_PATH = os.path.abspath(os.getenv("CHROMA_PATH", "chroma"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_chroma() -> Chroma:
    """Return a ChromaDB client (re-uses embedding singleton)."""
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )


def _calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Assign a stable, deterministic ID to each chunk.

    Format: ``<source>:<page>:<chunk_index>``

    The ID is stored in ``chunk.metadata["id"]`` so ChromaDB can
    deduplicate on re-ingestion.
    """
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        chunk_index = chunk.metadata.get("chunk_index", 0)
        chunk.metadata["id"] = f"{source}:{page}:{chunk_index}"
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_file(file_path: str) -> Dict:
    """
    Process *file_path* and add its chunks to ChromaDB.

    Returns a dict::

        {
            "filename":       str,
            "chunks_added":   int,
            "chunks_skipped": int,   # already existed in DB
            "total_chunks":   int,
        }
    """
    filename = Path(file_path).name
    logger.info("Indexing file: %s", filename)

    # 1. Extract + split
    chunks = process_file(file_path)
    if not chunks:
        logger.warning("No chunks produced for '%s' – nothing to index.", filename)
        return {
            "filename": filename,
            "chunks_added": 0,
            "chunks_skipped": 0,
            "total_chunks": 0,
        }

    # 2. Assign IDs
    chunks = _calculate_chunk_ids(chunks)

    # 3. Deduplication
    db = _get_chroma()
    existing = db.get(include=[])
    existing_ids = set(existing["ids"])

    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]
    skipped = len(chunks) - len(new_chunks)

    # 4. Stamp indexed_at on new chunks
    indexed_at = datetime.now(timezone.utc).isoformat()
    for c in new_chunks:
        c.metadata["indexed_at"] = indexed_at

    # 5. Insert
    if new_chunks:
        new_ids = [c.metadata["id"] for c in new_chunks]
        db.add_documents(new_chunks, ids=new_ids)

        # Invalidate BM25 cache so hybrid retriever rebuilds on next query
        try:
            from backend.agents.hybrid_retriever import invalidate_bm25_cache
            invalidate_bm25_cache()
        except ImportError:
            pass  # Agent modules not installed yet

        logger.info(
            "Indexed '%s': %d added, %d skipped (duplicates).",
            filename, len(new_chunks), skipped,
        )
    else:
        logger.info("All chunks for '%s' already in DB – nothing added.", filename)

    return {
        "filename": filename,
        "chunks_added": len(new_chunks),
        "chunks_skipped": skipped,
        "total_chunks": len(chunks),
    }


def index_directory(directory: str) -> List[Dict]:
    """
    Index all supported files in *directory*.

    Returns a list of per-file result dicts (same shape as ``index_file``).
    """
    results = []
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning("Directory does not exist: %s", directory)
        return results

    files = [
        str(p)
        for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.warning("No supported files found in '%s'.", directory)
        return results

    logger.info("Indexing %d file(s) from '%s'", len(files), directory)
    for file_path in files:
        result = index_file(file_path)
        results.append(result)

    total_added = sum(r["chunks_added"] for r in results)
    logger.info("Directory indexing complete: %d total chunks added.", total_added)
    return results


def clear_index() -> None:
    """Delete the entire ChromaDB store from disk."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logger.info("ChromaDB cleared at '%s'.", CHROMA_PATH)
    else:
        logger.info("ChromaDB path '%s' does not exist – nothing to clear.", CHROMA_PATH)


def get_index_stats() -> Dict:
    """Return basic stats about the current ChromaDB contents."""
    try:
        db = _get_chroma()
        result = db.get(include=[])
        count = len(result["ids"])
        return {"total_chunks": count, "chroma_path": CHROMA_PATH}
    except Exception as exc:
        logger.error("Could not read ChromaDB stats: %s", exc)
        return {"total_chunks": -1, "chroma_path": CHROMA_PATH, "error": str(exc)}


def get_indexed_files() -> List[Dict]:
    """
    Return a deduplicated list of files that have been indexed.

    Each entry::

        {
            "filename":  str,
            "source":    str,
            "file_type": str,
            "pages":     int,    # number of distinct pages indexed
            "chunks":    int,    # total chunks for that file
        }
    """
    try:
        db = _get_chroma()
        result = db.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])
    except Exception as exc:
        logger.error("Could not list indexed files: %s", exc)
        return []

    file_map: Dict[str, Dict] = {}
    for meta in metadatas:
        fname = meta.get("filename", "unknown")
        if fname not in file_map:
            file_map[fname] = {
                "filename": fname,
                "source": meta.get("source", ""),
                "file_type": meta.get("file_type", ""),
                "pages": set(),
                "chunks": 0,
            }
        file_map[fname]["pages"].add(meta.get("page", 0))
        file_map[fname]["chunks"] += 1

    return [
        {
            "filename": v["filename"],
            "source": v["source"],
            "file_type": v["file_type"],
            "pages": len(v["pages"]),
            "chunks": v["chunks"],
        }
        for v in file_map.values()
    ]
