"""
backend/utils/quiz_generator.py
==============================
Gemini-powered quiz generation for the AI Teaching Assistant.
"""

import logging
import json
import re
from typing import Dict, List, Any

from backend.llm_manager import get_fallback_llm

logger = logging.getLogger(__name__)

QUIZ_PROMPT_TEMPLATE = """
You are an expert Educational Content Generator.

Generate a high-quality quiz based ONLY on the provided context below.
Language: {language}

Context:
{context}

---

Topic: {topic}
Quiz Type: {quiz_type}
Number of Questions: {num_questions}

Instructions:
1. Generate exactly {num_questions} questions of type '{quiz_type}'.
2. For MCQ: Provide 4 options, the correct answer, and a brief explanation.
3. For True/False: Provide the question, correct answer (True or False), and a brief explanation.
4. For Short Answer: Provide the question and the key points expected in the answer.
5. All content (questions, options, explanations) should be in {language}.
6. IMPORTANT: For ALL mathematical expressions, variables, and formulas use proper LaTeX notation wrapped in $..$ for inline math. For example, write $q_i^T H q_j = 0$ NOT q_i^THq_j=0. Use $$...$$ for display/block formulas.
7. Format the entire response as a VALID JSON object with the following structure:
   {{
     "quiz": [
       {{
         "question": "...",
         "type": "{quiz_type}",
         "options": ["...", "...", "...", "..."], // Only for MCQ
         "answer": "...",
         "explanation": "..."
       }}
     ]
   }}
8. Ensure all questions are grounded in the context provided.
9. Avoid hallucinating information not present in the context.

JSON Output:
"""

def generate_quiz(context: str, topic: str, quiz_type: str = "mcq", num_questions: int = 3, language: str = "english") -> Dict[str, Any]:
    """
    Generate a quiz using Gemini based on the provided context.
    """
    if not context or context.strip() == "":
        return {"error": "No context available to generate quiz.", "quiz": []}

    prompt = QUIZ_PROMPT_TEMPLATE.format(
        context=context,
        topic=topic,
        quiz_type=quiz_type,
        num_questions=num_questions,
        language=language
    )

    llm = get_fallback_llm()
    try:
        response = llm.invoke(prompt)
        raw_content = getattr(response, "content", None)

        # Normalize content: it may be a string or a list of content blocks
        if isinstance(raw_content, list):
            parts = []
            for block in raw_content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
            response_text = " ".join(parts)
        elif isinstance(raw_content, str):
            response_text = raw_content
        else:
            response_text = str(response)

        # Extract JSON from response
        json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
        if json_match:
            quiz_data = json.loads(json_match.group(1))
            return quiz_data
        else:
            logger.error(f"Failed to extract JSON from Gemini response: {response_text}")
            return {"error": "Failed to parse quiz data.", "quiz": []}
            
    except Exception as e:
        logger.error(f"Quiz generation failed: {e}")
        return {"error": str(e), "quiz": []}
