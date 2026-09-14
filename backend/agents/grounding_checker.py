"""
backend/agents/grounding_checker.py
====================================
Post-generation grounding verification for the AI Teaching Assistant.

After the LLM generates an answer, this module verifies that every
factual claim is supported by the retrieved context. This is critical
for an educational platform where incorrect answers can mislead students.

If the answer contains unsupported claims, this module either:
- Removes the unsupported parts and returns a corrected answer, or
- Flags the answer with a warning for the student.
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
# Maximum context chars sent to the grounding checker
MAX_CONTEXT_CHARS = 2000

# Maximum answer chars to check
MAX_ANSWER_CHARS = 3000

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

GROUNDING_PROMPT = """You are a rigorous fact-checker for an educational AI tutoring system.

Your job: verify that EVERY factual claim in the generated answer is supported by the source context.

Source Context (from student's study material):
{context}

Generated Answer:
{answer}

Instructions:
1. Check each factual claim, definition, formula, and example in the answer against the context.
2. A claim is "grounded" if it is directly stated in or logically derivable from the context.
3. General knowledge framing (e.g., "This is important because...") is acceptable.
4. Formulas, specific numbers, definitions, and technical claims MUST be in the context.

Output ONLY valid JSON:
{{
  "grounded": true/false,
  "confidence": 0.0 to 1.0,
  "unsupported_claims": ["list of specific claims not supported by context"],
  "corrected_answer": "the answer with unsupported claims removed or marked with [unverified]"
}}

If the answer is fully grounded, set "grounded": true, "unsupported_claims": [], and "corrected_answer": null."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_grounding(
    context: str,
    answer: str,
) -> dict:
    """
    Verify that a generated answer is grounded in the retrieved context.

    Parameters
    ----------
    context : str
        The retrieved context text that the answer should be based on.
    answer : str
        The LLM-generated answer to verify.

    Returns
    -------
    dict
        ``{"grounded": bool, "confidence": float, "unsupported_claims": list, "corrected_answer": str|None}``

        On any failure, returns an optimistic fallback
        ``{"grounded": True, "confidence": 1.0, ...}`` to avoid blocking.
    """
    fallback = {
        "grounded": True,
        "confidence": 1.0,
        "unsupported_claims": [],
        "corrected_answer": None,
    }

    if not context or not answer:
        return fallback

    # Truncate for cost control
    context_trimmed = context[:MAX_CONTEXT_CHARS]
    if len(context) > MAX_CONTEXT_CHARS:
        context_trimmed += "\n... [truncated]"

    answer_trimmed = answer[:MAX_ANSWER_CHARS]
    if len(answer) > MAX_ANSWER_CHARS:
        answer_trimmed += "\n... [truncated]"

    prompt = GROUNDING_PROMPT.format(
        context=context_trimmed,
        answer=answer_trimmed,
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
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())

            grounded = bool(result.get("grounded", True))
            confidence = float(result.get("confidence", 1.0))
            confidence = max(0.0, min(1.0, confidence))  # Clamp

            unsupported = result.get("unsupported_claims", [])
            if not isinstance(unsupported, list):
                unsupported = []

            corrected = result.get("corrected_answer")
            if corrected and not isinstance(corrected, str):
                corrected = None
            if corrected and corrected.lower() in ("null", "none"):
                corrected = None

            verdict = {
                "grounded": grounded,
                "confidence": round(confidence, 2),
                "unsupported_claims": unsupported,
                "corrected_answer": corrected,
            }

            if not grounded:
                logger.warning(
                    "Grounding check FAILED (confidence=%.2f): %d unsupported claims",
                    confidence,
                    len(unsupported),
                )
            else:
                logger.info(
                    "Grounding check PASSED (confidence=%.2f)",
                    confidence,
                )

            return verdict

        logger.warning(
            "Grounding checker could not parse response: %s",
            response_text[:200],
        )
        return fallback

    except Exception as exc:
        logger.warning("Grounding checker failed (using optimistic fallback): %s", exc)
        return fallback
