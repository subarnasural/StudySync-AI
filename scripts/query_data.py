"""
scripts/query_data.py
======================
Agentic RAG query pipeline.

Pipeline stages:
    1. Session Memory  → resolve follow-up references
    2. Query Rewriter  → expand / decompose the question
    3. Hybrid Retriever → vector + BM25 → RRF fusion
    4. Retrieval Judge  → evaluate quality, retry if needed
    5. Prompt Builder   → mode-aware prompt construction
    6. LLM Generation   → Gemini generates the answer
    7. Grounding Check  → verify answer against context
    8. Analytics        → record question for dashboard

Backward compatibility
----------------------
The return dict keeps all original fields (answer, sources, source_details,
chunks_used, processing_time_ms, context_text) and adds new optional
agentic metadata fields.
"""

import argparse
import logging
import os
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from backend.llm_manager import get_embedding_function, get_fallback_llm
from backend.utils.prompt_modes import get_mode_prompt
from backend.analytics.progress_tracker import record_question

# Agent modules
from backend.agents.hybrid_retriever import get_hybrid_retriever
from backend.agents.query_rewriter import rewrite_query
from backend.agents.retrieval_judge import judge_retrieval
from backend.agents.memory import get_memory
from backend.agents.grounding_checker import check_grounding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_PATH = os.path.abspath(os.getenv("CHROMA_PATH", "chroma"))
DEFAULT_TOP_K     = int(os.getenv("RAG_TOP_K",            "5"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))
MAX_CHUNK_CHARS   = int(os.getenv("RAG_MAX_CHUNK_CHARS",   "1200"))

RELEVANCE_DISTANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "2.0"))

# Agentic RAG settings
MAX_RETRIEVAL_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "2"))
RETRIEVAL_JUDGE_THRESHOLD = int(os.getenv("RAG_JUDGE_THRESHOLD", "4"))

NO_CONTEXT_REPLY = (
    "I'm sorry, I could not find relevant information in the uploaded course "
    "materials to answer your question. Please make sure the relevant document "
    "has been uploaded and indexed, then try again."
)

PROMPT_TEMPLATE = """
You are an AI Teaching Assistant helping students understand academic material.

Answer ONLY using the provided context below. Do NOT use any prior knowledge.

RULES:
- Base your answer STRICTLY on the context provided.
- Provide a detailed, comprehensive, and clear explanation.
- Use proper markdown formatting (headings, bullet points, **bold** for key terms).
- For mathematical formulas, variables, subscripts, coordinates, or symbols, ALWAYS use standard LaTeX format:
  * Wrap block or display equations in double dollar signs (e.g., $$h_t = f(x_t)$$).
  * Wrap inline variables, mathematical expressions, or symbols in single dollar signs (e.g., $x$ or $y$).
  * Do NOT omit any variables, subscripts, coordinates, or indices from mathematical formulas present in the context.
  * Explain every formula in plain English after showing it.
- Ensure all special tokens (like start-of-sequence, end-of-sequence, or other boundary tokens) are properly enclosed in backticks or code blocks to avoid getting stripped (e.g., `<sos>` or `<eos>`).
- Do NOT add unnecessary caveats, disclaimers, or "not enough information" messages — just answer from what the context contains.
- Do NOT add excessive blank lines or padding in your response.

Context:
{context}

---

Question:
{question}

Answer:
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_response_text(response) -> str:
    """Convert provider-specific response payloads into plain text."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text).strip())
        joined = " ".join([p for p in parts if p]).strip()
        if joined:
            return joined

    return str(content).strip()


def _fix_unicode_symbols(text: str) -> str:
    """Replace stray Unicode math symbols with proper LaTeX equivalents and clean OCR token artifacts."""
    replacements = {
        "ŷ{y}": r"\hat{y}",
        "Σ": r"\sum",
        "λ": r"\lambda",
        "σ": r"\sigma",
        "μ": r"\mu",
        "α": r"\alpha",
        "β": r"\beta",
        "π": r"\pi",
        "∞": r"\infty",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Normalize NLP special tokens / OCR boundary artifacts to backticked code for safe frontend rendering
    token_replacements = {
        ".<START.>": "`<START>`",
        ".<STOP.>": "`<STOP>`",
        ".<END.>": "`<END>`",
        ".<BOS.>": "`<BOS>`",
        ".<EOS.>": "`<EOS>`",
        ".<PAD.>": "`<PAD>`",
        ".<UNK.>": "`<UNK>`",
        "<START>": "`<START>`",
        "<STOP>": "`<STOP>`",
        "<END>": "`<END>`",
        "<BOS>": "`<BOS>`",
        "<EOS>": "`<EOS>`",
    }
    for bad, good in token_replacements.items():
        text = text.replace(bad, good)

    return text


def _build_source_label(metadata: dict) -> str:
    """Build a human-readable source string from chunk metadata."""
    filename = metadata.get("filename") or metadata.get("source", "unknown")
    page = metadata.get("page")
    if page:
        return f"{filename} (page {page})"
    return str(filename)


def _assemble_context(
    results: List[Tuple[Document, float]],
) -> Tuple[List[str], List[str], List[dict], str]:
    """
    Filter retrieval results by relevance and assemble context within budget.

    Returns (snippets, source_labels, source_details, context_text).
    """
    selected_snippets = []
    selected_sources = []
    source_details = []
    current_len = 0

    for doc, score in results:
        snippet = (doc.page_content or "")[:MAX_CHUNK_CHARS].strip()
        if not snippet:
            continue

        projected = current_len + len(snippet) + 8
        if projected > MAX_CONTEXT_CHARS and selected_snippets:
            break

        selected_snippets.append(snippet)
        selected_sources.append(_build_source_label(doc.metadata))
        source_details.append({
            "filename": doc.metadata.get("filename", "unknown"),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("id", ""),
            "subject": doc.metadata.get("subject", ""),
            "distance": round(score, 4),
        })
        current_len = projected

    context_text = "\n\n".join(selected_snippets)
    return selected_snippets, selected_sources, source_details, context_text


# ---------------------------------------------------------------------------
# Public API — Agentic RAG Pipeline
# ---------------------------------------------------------------------------

def query_rag(
    query_text: str,
    mode: str = "default",
    language: str = "english",
    session_id: str = None,
    fast_mode: Optional[bool] = None,
) -> dict:
    """
    Run the RAG pipeline for *query_text*.
    In fast_mode (default), executes direct hybrid retrieval and 1 LLM generation call,
    dropping response time to ~1.5-2.5s and preserving API quota.
    """
    t_start = time.perf_counter()

    if fast_mode is None:
        fast_mode = os.getenv("RAG_FAST_MODE", "true").strip().lower() in ("true", "1", "yes")

    # -----------------------------------------------------------------
    # Stage 1: Session Memory — get conversation history for context
    # -----------------------------------------------------------------
    history = []
    if session_id:
        try:
            history = get_memory().get_history(session_id)
        except Exception as exc:
            logger.warning("Memory lookup failed: %s", exc)

    # -----------------------------------------------------------------
    # Stage 2: Query Rewriting (bypassed in fast mode for 1-call speed)
    # -----------------------------------------------------------------
    if not fast_mode:
        try:
            rewritten_queries = rewrite_query(query_text, history=history)
        except Exception as exc:
            logger.warning("Query rewriter failed: %s", exc)
            rewritten_queries = [query_text]
        logger.info("Agentic RAG: rewritten queries = %s", rewritten_queries)
    else:
        rewritten_queries = [query_text]
        logger.info("Fast Direct RAG: direct query execution")

    # -----------------------------------------------------------------
    # Stage 3: Hybrid Retrieval — vector + BM25 → RRF fusion
    # -----------------------------------------------------------------
    retriever = get_hybrid_retriever(top_k=DEFAULT_TOP_K)

    all_results: List[Tuple[Document, float]] = []
    seen_ids: set = set()

    for rq in rewritten_queries:
        try:
            for doc, score in retriever.retrieve(rq, top_k=DEFAULT_TOP_K):
                doc_id = doc.metadata.get("id", "") or str(hash(doc.page_content[:200]))
                if doc_id not in seen_ids:
                    all_results.append((doc, score))
                    seen_ids.add(doc_id)
        except Exception as exc:
            logger.warning("Retrieval failed for query '%s': %s", rq[:60], exc)

    # Sort by score descending (RRF scores — higher is better)
    all_results.sort(key=lambda x: x[1], reverse=True)
    results = all_results[:DEFAULT_TOP_K]

    logger.info(
        "Agentic RAG: %d unique chunks retrieved from %d queries",
        len(results),
        len(rewritten_queries),
    )

    # -----------------------------------------------------------------
    # Stage 4: Context Assembly + Retrieval Judge + Retry
    # -----------------------------------------------------------------
    selected_snippets, selected_sources, source_details, context_text = (
        _assemble_context(results)
    )

    retrieval_attempts = 1
    retrieval_score = 5  # default optimistic

    if selected_snippets and not fast_mode:
        try:
            judgment = judge_retrieval(query_text, context_text)
            retrieval_score = judgment.get("score", 5)

            # Retry loop if quality is low
            for attempt in range(MAX_RETRIEVAL_RETRIES):
                if retrieval_score >= RETRIEVAL_JUDGE_THRESHOLD:
                    break

                suggestion = judgment.get("suggestion")
                if not suggestion:
                    break

                logger.info(
                    "Agentic RAG: retrieval retry %d (score=%d, suggestion='%s')",
                    attempt + 1,
                    retrieval_score,
                    suggestion[:60],
                )

                # Re-retrieve with the judge's suggested query
                try:
                    retry_results = retriever.retrieve(suggestion, top_k=DEFAULT_TOP_K)
                    for doc, score in retry_results:
                        doc_id = doc.metadata.get("id", "") or str(hash(doc.page_content[:200]))
                        if doc_id not in seen_ids:
                            all_results.append((doc, score))
                            seen_ids.add(doc_id)

                    all_results.sort(key=lambda x: x[1], reverse=True)
                    results = all_results[:DEFAULT_TOP_K]

                    # Reassemble context
                    selected_snippets, selected_sources, source_details, context_text = (
                        _assemble_context(results)
                    )

                    retrieval_attempts += 1

                    # Re-judge
                    judgment = judge_retrieval(query_text, context_text)
                    retrieval_score = judgment.get("score", 5)

                except Exception as exc:
                    logger.warning("Retry retrieval failed: %s", exc)
                    break

        except Exception as exc:
            logger.warning("Retrieval judge failed: %s", exc)

    # -----------------------------------------------------------------
    # No-context guard
    # -----------------------------------------------------------------
    if not selected_snippets:
        elapsed = (time.perf_counter() - t_start) * 1000
        return {
            "answer": NO_CONTEXT_REPLY,
            "sources": [],
            "source_details": [],
            "chunks_used": 0,
            "processing_time_ms": round(elapsed, 1),
            "context_text": "",
            # Agentic metadata
            "rewritten_queries": rewritten_queries,
            "retrieval_method": "hybrid_bm25_vector",
            "retrieval_attempts": retrieval_attempts,
            "retrieval_score": retrieval_score,
            "grounding_score": None,
            "grounding_verified": None,
        }

    # -----------------------------------------------------------------
    # Stage 5: Build prompt + call LLM
    # -----------------------------------------------------------------
    if mode and mode.lower() != "default" or language.lower() != "english":
        prompt = get_mode_prompt(mode, context_text, query_text, language)
    else:
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context_text,
            question=query_text,
        )

    llm = get_fallback_llm()
    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        elapsed = (time.perf_counter() - t_start) * 1000
        return {
            "answer": f"The AI model could not generate a response. Error: {exc}",
            "sources": [],
            "source_details": source_details,
            "chunks_used": len(selected_snippets),
            "processing_time_ms": round(elapsed, 1),
            "context_text": context_text,
            "rewritten_queries": rewritten_queries,
            "retrieval_method": "hybrid_bm25_vector",
            "retrieval_attempts": retrieval_attempts,
            "retrieval_score": retrieval_score,
            "grounding_score": None,
            "grounding_verified": None,
        }

    response_text = _normalize_response_text(response)
    response_text = _fix_unicode_symbols(response_text)

    # -----------------------------------------------------------------
    # Stage 7: Grounding Verification
    # -----------------------------------------------------------------
    grounding_score = 1.0
    grounding_verified = True

    if not fast_mode:
        try:
            grounding = check_grounding(context_text, response_text)
            grounding_score = grounding.get("confidence", 1.0)
            grounding_verified = grounding.get("grounded", True)

            if not grounding_verified:
                corrected = grounding.get("corrected_answer")
                if corrected and corrected.strip():
                    logger.warning(
                        "Grounding check failed — using corrected answer for query: %s",
                        query_text[:80],
                    )
                    response_text = _fix_unicode_symbols(corrected)
                else:
                    # Add a warning notice to the response
                    response_text += (
                        "\n\n> ⚠️ *Note: Some parts of this answer may extend "
                        "beyond the uploaded source material.*"
                    )
        except Exception as exc:
            logger.warning("Grounding checker failed: %s", exc)

    # -----------------------------------------------------------------
    # Deduplicate sources
    # -----------------------------------------------------------------
    seen = set()
    unique_sources = []
    for s in selected_sources:
        if s not in seen:
            unique_sources.append(s)
            seen.add(s)

    # -----------------------------------------------------------------
    # Stage 8: Record Analytics + Save to Memory
    # -----------------------------------------------------------------
    primary_topic = source_details[0]["subject"] if source_details else "unknown"
    record_question(query_text, primary_topic, unique_sources, mode or "default")

    # Save exchange to session memory
    if session_id:
        try:
            get_memory().add_exchange(
                session_id,
                query_text,
                response_text[:200],
                primary_topic,
            )
        except Exception as exc:
            logger.warning("Failed to save to memory: %s", exc)

    elapsed = (time.perf_counter() - t_start) * 1000
    return {
        # Original fields (backward compatible)
        "answer": response_text,
        "sources": unique_sources,
        "source_details": source_details,
        "chunks_used": len(selected_snippets),
        "processing_time_ms": round(elapsed, 1),
        "context_text": context_text,
        # Agentic RAG metadata (new)
        "rewritten_queries": rewritten_queries,
        "retrieval_method": "hybrid_bm25_vector",
        "retrieval_attempts": retrieval_attempts,
        "retrieval_score": retrieval_score,
        "grounding_score": round(grounding_score, 2),
        "grounding_verified": grounding_verified,
    }


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG query CLI")
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--mode", type=str, default="default", help="Explanation mode.")
    parser.add_argument("--lang", type=str, default="english", help="Language (english/hindi).")
    args = parser.parse_args()

    result = query_rag(args.query_text, mode=args.mode, language=args.lang)
    print("\n=== Answer ===")
    print(result["answer"])
    print("\n=== Sources ===")
    for s in result["sources"]:
        print(" -", s)
    print("\n=== Agentic Metadata ===")
    print(f"  Rewritten queries: {result.get('rewritten_queries')}")
    print(f"  Retrieval method:  {result.get('retrieval_method')}")
    print(f"  Retrieval attempts: {result.get('retrieval_attempts')}")
    print(f"  Retrieval score:   {result.get('retrieval_score')}")
    print(f"  Grounding score:   {result.get('grounding_score')}")
    print(f"  Grounding verified: {result.get('grounding_verified')}")


if __name__ == "__main__":
    main()
