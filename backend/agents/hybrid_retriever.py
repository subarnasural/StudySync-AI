"""
backend/agents/hybrid_retriever.py
===================================
Hybrid retrieval engine combining vector similarity search (ChromaDB)
with BM25 keyword matching, fused via Reciprocal Rank Fusion (RRF).

Why hybrid?
-----------
Vector search captures semantic meaning but can miss exact technical
terms (e.g., "backpropagation", "ReLU"). BM25 excels at exact keyword
matching. By fusing both, we get significantly better recall for
academic/technical content.

Usage
-----
    from backend.agents.hybrid_retriever import get_hybrid_retriever

    retriever = get_hybrid_retriever()
    results = retriever.retrieve("Explain the chain rule in backpropagation")
    # returns: list[tuple[Document, float]]  (same format as ChromaDB)
"""

import logging
import re
import threading
from typing import Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from backend.llm_manager import get_embedding_function

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RRF_K = 60          # Standard RRF constant (controls rank sensitivity)
BM25_WEIGHT = 0.4   # Weight for BM25 in final fusion (0.0–1.0)
VECTOR_WEIGHT = 0.6  # Weight for vector search in final fusion


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_retriever_singleton: Optional["HybridRetriever"] = None
_retriever_lock = threading.Lock()
_bm25_valid = False  # Flag to track if BM25 index needs rebuild


def invalidate_bm25_cache() -> None:
    """Signal that the BM25 index should be rebuilt on next query.

    Call this after adding new documents to ChromaDB.
    """
    global _bm25_valid
    _bm25_valid = False
    logger.info("BM25 index cache invalidated — will rebuild on next query.")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines ChromaDB vector search with BM25 keyword search using
    Reciprocal Rank Fusion (RRF).

    The BM25 index is built lazily from all ChromaDB documents on first
    query, then cached until ``invalidate_bm25_cache()`` is called.
    """

    def __init__(self, chroma_db: Chroma, top_k: int = 5) -> None:
        self.db = chroma_db
        self.top_k = top_k

        # BM25 state
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_docs: List[Document] = []
        self._corpus_tokens: List[List[str]] = []
        self._build_lock = threading.Lock()

    # -----------------------------------------------------------------
    # BM25 index management
    # -----------------------------------------------------------------

    def _build_bm25_index(self) -> None:
        """Pull all documents from ChromaDB, tokenize, and build BM25."""
        global _bm25_valid

        with self._build_lock:
            if _bm25_valid and self._bm25 is not None:
                return  # Another thread already rebuilt

            logger.info("Building BM25 index from ChromaDB...")
            try:
                result = self.db.get(include=["documents", "metadatas"])
                documents = result.get("documents", [])
                metadatas = result.get("metadatas", [])
                ids = result.get("ids", [])

                if not documents:
                    logger.warning("ChromaDB is empty — BM25 index will be empty.")
                    self._bm25 = None
                    self._corpus_docs = []
                    self._corpus_tokens = []
                    _bm25_valid = True
                    return

                # Build Document objects and tokenized corpus
                self._corpus_docs = []
                self._corpus_tokens = []
                for i, (doc_text, meta) in enumerate(zip(documents, metadatas)):
                    if not doc_text:
                        continue
                    doc = Document(
                        page_content=doc_text,
                        metadata=meta if meta else {},
                    )
                    self._corpus_docs.append(doc)
                    self._corpus_tokens.append(_tokenize(doc_text))

                if self._corpus_tokens:
                    self._bm25 = BM25Okapi(self._corpus_tokens)
                    logger.info(
                        "BM25 index built: %d documents indexed.",
                        len(self._corpus_docs),
                    )
                else:
                    self._bm25 = None
                    logger.warning("No tokenizable documents found for BM25.")

                _bm25_valid = True

            except Exception as exc:
                logger.error("Failed to build BM25 index: %s", exc)
                self._bm25 = None
                _bm25_valid = True  # Don't retry endlessly on errors

    # -----------------------------------------------------------------
    # Search methods
    # -----------------------------------------------------------------

    def _vector_search(
        self, query: str, k: int
    ) -> List[Tuple[Document, float]]:
        """Standard ChromaDB vector similarity search."""
        try:
            return self.db.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)
            return []

    def _bm25_search(
        self, query: str, k: int
    ) -> List[Tuple[Document, float]]:
        """BM25 keyword search over the cached corpus."""
        if self._bm25 is None or not self._corpus_docs:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        try:
            scores = self._bm25.get_scores(query_tokens)

            # Get top-k indices sorted by score descending
            scored_indices = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )[:k]

            results = []
            for idx, score in scored_indices:
                if score > 0:
                    results.append((self._corpus_docs[idx], float(score)))

            return results
        except Exception as exc:
            logger.warning("BM25 search failed: %s", exc)
            return []

    # -----------------------------------------------------------------
    # Reciprocal Rank Fusion
    # -----------------------------------------------------------------

    def _rrf_fuse(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        Fuse two ranked lists using weighted Reciprocal Rank Fusion.

        RRF score for document d:
            score(d) = w_vec * 1/(k + rank_vec(d)) + w_bm25 * 1/(k + rank_bm25(d))
        """
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        def _doc_key(doc: Document) -> str:
            """Generate a unique key for deduplication."""
            return doc.metadata.get("id", "") or hash(doc.page_content[:200])

        # Score vector results
        for rank, (doc, _distance) in enumerate(vector_results):
            key = _doc_key(doc)
            rrf_score = VECTOR_WEIGHT * (1.0 / (RRF_K + rank + 1))
            doc_scores[key] = doc_scores.get(key, 0) + rrf_score
            doc_map[key] = doc

        # Score BM25 results
        for rank, (doc, _bm25_score) in enumerate(bm25_results):
            key = _doc_key(doc)
            rrf_score = BM25_WEIGHT * (1.0 / (RRF_K + rank + 1))
            doc_scores[key] = doc_scores.get(key, 0) + rrf_score
            doc_map[key] = doc

        # Sort by fused score descending
        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        return [(doc_map[key], score) for key, score in sorted_items]

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid retrieval: vector + BM25 → RRF fusion.

        Parameters
        ----------
        query : str
            The search query.
        top_k : int, optional
            Number of results to return. Defaults to self.top_k.

        Returns
        -------
        list[tuple[Document, float]]
            Ranked results with RRF fusion scores.
        """
        k = top_k or self.top_k

        # Ensure BM25 index is built
        global _bm25_valid
        if not _bm25_valid or self._bm25 is None:
            self._build_bm25_index()

        # Fetch more candidates from each source for better fusion
        fetch_k = k * 3

        vector_results = self._vector_search(query, k=fetch_k)
        bm25_results = self._bm25_search(query, k=fetch_k)

        logger.info(
            "Hybrid retrieval: %d vector + %d BM25 candidates for query '%s'",
            len(vector_results),
            len(bm25_results),
            query[:60],
        )

        # If one source failed, use the other directly
        if not vector_results and not bm25_results:
            return []
        if not bm25_results:
            return vector_results[:k]
        if not vector_results:
            return [(doc, score) for doc, score in bm25_results[:k]]

        # Fuse results
        fused = self._rrf_fuse(vector_results, bm25_results)
        return fused[:k]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

def get_hybrid_retriever(chroma_db: Chroma = None, top_k: int = 5) -> HybridRetriever:
    """Return the shared HybridRetriever singleton.

    If *chroma_db* is ``None``, a default instance is created using
    the standard CHROMA_PATH and embedding function.
    """
    global _retriever_singleton

    with _retriever_lock:
        if _retriever_singleton is None:
            if chroma_db is None:
                import os
                CHROMA_PATH = os.path.abspath(os.getenv("CHROMA_PATH", "chroma"))
                chroma_db = Chroma(
                    persist_directory=CHROMA_PATH,
                    embedding_function=get_embedding_function(),
                )
            _retriever_singleton = HybridRetriever(chroma_db, top_k=top_k)
        return _retriever_singleton
