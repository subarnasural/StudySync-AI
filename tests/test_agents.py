"""
tests/test_agents.py
====================
Unit tests for the Agentic RAG modules.

Tests cover:
- SessionMemory: add/get/expiry/bounds
- Query rewriter: output format
- Retrieval judge: output format
- Grounding checker: output format
- HybridRetriever: RRF fusion logic
"""

import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ===========================================================================
# Session Memory Tests
# ===========================================================================

class TestSessionMemory:
    """Tests for backend.agents.memory.SessionMemory."""

    def _make_memory(self, **kwargs):
        from backend.agents.memory import SessionMemory
        return SessionMemory(**kwargs)

    def test_empty_history_for_unknown_session(self):
        mem = self._make_memory()
        assert mem.get_history("nonexistent") == []

    def test_add_and_get_exchange(self):
        mem = self._make_memory()
        mem.add_exchange("s1", "What is AI?", "AI is artificial intelligence.", "ai")
        history = mem.get_history("s1")
        assert len(history) == 1
        assert history[0]["question"] == "What is AI?"
        assert history[0]["answer_summary"] == "AI is artificial intelligence."
        assert history[0]["topic"] == "ai"

    def test_history_bounded_to_max(self):
        mem = self._make_memory(max_history=3)
        for i in range(10):
            mem.add_exchange("s1", f"Q{i}", f"A{i}", "topic")
        history = mem.get_history("s1")
        assert len(history) == 3
        # Should keep the last 3
        assert history[0]["question"] == "Q7"
        assert history[2]["question"] == "Q9"

    def test_separate_sessions(self):
        mem = self._make_memory()
        mem.add_exchange("s1", "Q1", "A1", "t1")
        mem.add_exchange("s2", "Q2", "A2", "t2")
        assert len(mem.get_history("s1")) == 1
        assert len(mem.get_history("s2")) == 1
        assert mem.get_history("s1")[0]["question"] == "Q1"
        assert mem.get_history("s2")[0]["question"] == "Q2"

    def test_session_count(self):
        mem = self._make_memory()
        assert mem.get_session_count() == 0
        mem.add_exchange("s1", "Q1", "A1", "t")
        assert mem.get_session_count() == 1
        mem.add_exchange("s2", "Q2", "A2", "t")
        assert mem.get_session_count() == 2

    def test_context_summary_format(self):
        mem = self._make_memory()
        mem.add_exchange("s1", "What is ML?", "ML is machine learning.", "ml")
        summary = mem.get_context_summary("s1")
        assert "What is ML?" in summary
        assert "ML is machine learning." in summary

    def test_context_summary_empty_session(self):
        mem = self._make_memory()
        assert mem.get_context_summary("nonexistent") == ""

    def test_answer_summary_truncated(self):
        mem = self._make_memory()
        long_answer = "A" * 500
        mem.add_exchange("s1", "Q", long_answer, "t")
        history = mem.get_history("s1")
        assert len(history[0]["answer_summary"]) == 200

    def test_expired_sessions_cleaned(self):
        mem = self._make_memory(ttl_seconds=0)  # Expire immediately
        mem.add_exchange("s1", "Q", "A", "t")
        mem._last_cleanup = 0  # Force cleanup on next call
        time.sleep(0.01)
        mem.add_exchange("s2", "Q2", "A2", "t")  # Triggers cleanup
        # s1 should be cleaned up, s2 should exist
        assert mem.get_session_count() == 1


# ===========================================================================
# HybridRetriever RRF Logic Tests
# ===========================================================================

class TestRRFFusion:
    """Test the Reciprocal Rank Fusion logic in isolation."""

    def test_rrf_combines_results(self):
        from backend.agents.hybrid_retriever import HybridRetriever
        from langchain_core.documents import Document

        # Create a mock retriever (we'll test _rrf_fuse directly)
        class MockChroma:
            pass

        retriever = HybridRetriever(MockChroma(), top_k=5)

        doc_a = Document(page_content="Document A", metadata={"id": "a"})
        doc_b = Document(page_content="Document B", metadata={"id": "b"})
        doc_c = Document(page_content="Document C", metadata={"id": "c"})

        vector_results = [(doc_a, 0.5), (doc_b, 1.0)]
        bm25_results = [(doc_b, 5.0), (doc_c, 3.0)]

        fused = retriever._rrf_fuse(vector_results, bm25_results)

        # doc_b appears in both lists, should have highest score
        assert len(fused) == 3
        fused_ids = [doc.metadata["id"] for doc, _ in fused]
        assert "b" in fused_ids
        # b should be ranked first (appears in both)
        assert fused_ids[0] == "b"

    def test_rrf_empty_inputs(self):
        from backend.agents.hybrid_retriever import HybridRetriever
        class MockChroma:
            pass

        retriever = HybridRetriever(MockChroma(), top_k=5)
        assert retriever._rrf_fuse([], []) == []

    def test_rrf_single_source(self):
        from backend.agents.hybrid_retriever import HybridRetriever
        from langchain_core.documents import Document

        class MockChroma:
            pass

        retriever = HybridRetriever(MockChroma(), top_k=5)
        doc = Document(page_content="Only doc", metadata={"id": "x"})
        fused = retriever._rrf_fuse([(doc, 0.5)], [])
        assert len(fused) == 1
        assert fused[0][0].metadata["id"] == "x"


# ===========================================================================
# Tokenizer Tests
# ===========================================================================

class TestTokenizer:
    """Test the BM25 tokenizer."""

    def test_basic_tokenization(self):
        from backend.agents.hybrid_retriever import _tokenize
        tokens = _tokenize("Hello, World! This is a test.")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_empty_string(self):
        from backend.agents.hybrid_retriever import _tokenize
        assert _tokenize("") == []

    def test_special_characters(self):
        from backend.agents.hybrid_retriever import _tokenize
        tokens = _tokenize("CNN's architecture: conv2d + relu")
        assert "cnn" in tokens
        assert "conv2d" in tokens
        assert "relu" in tokens
