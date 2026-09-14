"""
scripts/populate_database.py
=============================
CLI tool to (re-)index all supported files in the upload directory.

Usage
-----
    python -m scripts.populate_database            # incremental index
    python -m scripts.populate_database --reset    # wipe DB then re-index
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from backend.utils.indexer import clear_index, index_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_PATH = os.path.join("uploaded_data", "files")


def main(reset_db: bool = False) -> None:
    # Allow --reset from CLI even when called as function
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Populate ChromaDB from uploaded files.")
        parser.add_argument("--reset", action="store_true", help="Wipe the database before indexing.")
        args, _ = parser.parse_known_args()
        reset_db = reset_db or args.reset

    if reset_db:
        logger.info("--reset flag set: clearing existing ChromaDB …")
        clear_index()

    if not os.path.exists(DATA_PATH):
        logger.warning("Data directory '%s' does not exist. Nothing to index.", DATA_PATH)
        return

    logger.info("Starting indexing from '%s' …", os.path.abspath(DATA_PATH))
    results = index_directory(DATA_PATH)

    if not results:
        logger.info("No files were processed.")
        return

    total_added   = sum(r["chunks_added"]   for r in results)
    total_skipped = sum(r["chunks_skipped"] for r in results)

    logger.info("=" * 50)
    logger.info("Indexing complete.")
    logger.info("  Files processed : %d", len(results))
    logger.info("  Chunks added    : %d", total_added)
    logger.info("  Chunks skipped  : %d (already in DB)", total_skipped)
    logger.info("=" * 50)

    for r in results:
        logger.info(
            "  %-40s  added=%d  skipped=%d",
            r["filename"], r["chunks_added"], r["chunks_skipped"],
        )


if __name__ == "__main__":
    main()
