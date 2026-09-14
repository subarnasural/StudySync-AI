"""
backend/agents/retrieval_judge.py
==================================
LLM-powered retrieval quality evaluator.

After retrieval, this module asks the LLM to judge whether the retrieved
context is sufficient to answer the student's question. If the score is
low, it suggests a better search query for retry.

This eliminates the dead-end "I couldn't find relevant information"
responses by giving the retrieval pipeline a second (or third) chance.
"""

import json
import logging
import re
from typing import Optional

from backend.llm_manager import get_fallback_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Maximum characters of context sent to the judge (cost control)
JUDGE_CONTEXT_PREVIEW_CHARS = 800

# Minimum score to accept retrieval without retry
ACCEPTABLE_SCORE = 4

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are a retrieval quality evaluator for an educational Q&A system.

A student asked a question and the system retrieved context from their study materials.
Evaluate how well the retrieved context can answer the question.

Question: {question}

Retrieved context (preview):
{context_preview}

Rate from 1 to 5:
1 = Completely irrelevant — context has nothing to do with the question
2 = Marginally relevant — mentions the topic but lacks specific information needed
3 = Partially relevant — covers some aspects but missing key information
4 = Mostly relevant — has enough information to provide a good answer
5 = Highly relevant — contains all the information needed for a comprehensive answer

If rating < 4, suggest ONE alternative search query that might retrieve better results.
The suggestion should be a specific, focused query targeting the missing information.

Output ONLY valid JSON:
{{"score": <int 1-5>, "reason": "<brief explanation>", "suggestion": "<alternative query or null>"}}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def judge_retrieval(
    question: str,
    context: str,
) -> dict:
    """
    Evaluate whether retrieved context is sufficient to answer the question.

    Parameters
    ----------
    question : str
        The student's question.
    context : str
        The assembled context text from retrieval.

    Returns
    -------
    dict
        ``{"score": int, "reason": str, "suggestion": str|None}``

        On any failure, returns ``{"score": 5, "reason": "judge_unavailable", "suggestion": None}``
        (optimistic fallback — don't block the pipeline).
    """
    fallback = {"score": 5, "reason": "judge_unavailable", "suggestion": None}

    if not question or not context:
        return fallback

    # Truncate context for cost control
    context_preview = context[:JUDGE_CONTEXT_PREVIEW_CHARS]
    if len(context) > JUDGE_CONTEXT_PREVIEW_CHARS:
        context_preview += "\n... [truncated]"

    prompt = JUDGE_PROMPT.format(
        question=question,
        context_preview=context_preview,
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

        # Parse JSON from response
        json_match = re.search(r"\{.*?\}", response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())

            # Validate and normalize
            score = int(result.get("score", 5))
            score = max(1, min(5, score))  # Clamp to 1-5
            reason = str(result.get("reason", ""))
            suggestion = result.get("suggestion")

            if suggestion and not isinstance(suggestion, str):
                suggestion = None
            if suggestion and suggestion.lower() in ("null", "none", ""):
                suggestion = None

            verdict = {
                "score": score,
                "reason": reason,
                "suggestion": suggestion,
            }

            logger.info(
                "Retrieval judge: score=%d, reason='%s', suggestion=%s",
                score,
                reason[:80],
                repr(suggestion[:60]) if suggestion else "None",
            )
            return verdict

        logger.warning("Retrieval judge could not parse response: %s", response_text[:200])
        return fallback

    except Exception as exc:
        logger.warning("Retrieval judge failed (using optimistic fallback): %s", exc)
        return fallback
