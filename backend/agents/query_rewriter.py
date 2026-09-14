"""
backend/agents/query_rewriter.py
=================================
LLM-powered query rewriter that transforms raw student questions into
optimized retrieval queries.

Capabilities
------------
1. **Expand** vague queries with synonyms and technical terms.
2. **Decompose** multi-part questions into separate search queries.
3. **Resolve** follow-up references using conversation history
   (e.g., "What about its complexity?" → "What is the time complexity of merge sort?").

Graceful degradation
--------------------
On any LLM error, silently falls back to the original question.
"""

import json
import logging
import re
from typing import List, Optional

from backend.llm_manager import get_fallback_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

REWRITE_PROMPT = """You are a query optimizer for an academic study material search engine.

Given the student's question (and optionally their recent conversation history), produce 1-3 search queries optimized for retrieving relevant passages from course textbooks and lecture notes.

Rules:
- Expand abbreviations and add relevant synonyms or technical terms
- If the question has multiple parts or compares topics, decompose into separate focused queries
- If conversation history is provided and the question references prior context (e.g., "it", "this", "that algorithm"), resolve the reference using the history
- Keep each query concise (under 100 characters)
- Output ONLY a valid JSON array of strings, nothing else

Student's question: {question}
{history_section}
Optimized queries (JSON array):"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_query(
    question: str,
    history: Optional[List[dict]] = None,
) -> List[str]:
    """
    Rewrite a student question into 1-3 optimized retrieval queries.

    Parameters
    ----------
    question : str
        The raw student question.
    history : list[dict], optional
        Recent conversation history. Each dict should have
        ``question`` and ``answer_summary`` keys.

    Returns
    -------
    list[str]
        1-3 optimized query strings. On any failure, returns ``[question]``.
    """
    if not question or not question.strip():
        return [question]

    # Build history section
    history_section = ""
    if history:
        history_lines = []
        for h in history[-3:]:  # Last 3 exchanges max
            q = h.get("question", "")
            a = h.get("answer_summary", "")
            if q:
                history_lines.append(f"  Student: {q}")
            if a:
                history_lines.append(f"  Assistant: {a[:150]}...")
        if history_lines:
            history_section = (
                "Recent conversation history:\n" + "\n".join(history_lines)
            )

    prompt = REWRITE_PROMPT.format(
        question=question,
        history_section=history_section,
    )

    try:
        llm = get_fallback_llm()
        response = llm.invoke(prompt)

        # Extract text content
        content = getattr(response, "content", response)
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
            response_text = " ".join(parts)
        elif isinstance(content, str):
            response_text = content
        else:
            response_text = str(content)

        # Parse JSON array from response
        json_match = re.search(r"\[.*?\]", response_text, re.DOTALL)
        if json_match:
            queries = json.loads(json_match.group())
            # Validate: must be a list of non-empty strings
            queries = [
                q.strip()
                for q in queries
                if isinstance(q, str) and q.strip()
            ]
            if queries:
                logger.info(
                    "Query rewriter: '%s' → %s",
                    question[:60],
                    queries,
                )
                return queries[:3]  # Cap at 3 queries

        logger.warning(
            "Query rewriter could not parse response: %s",
            response_text[:200],
        )
        return [question]

    except Exception as exc:
        logger.warning("Query rewriter failed (using original): %s", exc)
        return [question]
