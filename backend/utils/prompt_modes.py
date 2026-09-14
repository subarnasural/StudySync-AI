"""
backend/utils/prompt_modes.py
============================
Specialized prompt templates for different explanation modes and languages.
"""

from langchain_core.prompts import ChatPromptTemplate

MODES = {
    "beginner": {
        "description": "Simple language, analogies, and basic concepts.",
        "instructions": "Use simple language, real-world analogies, and avoid technical jargon. Explain as if to a 10-year-old."
    },
    "exam": {
        "description": "Structured, concise, and focused on key points for university exams.",
        "instructions": "Provide a structured, academic answer suitable for a university-level exam (5-10 marks). Use headings, bullet points, and highlight key terms."
    },
    "technical": {
        "description": "In-depth analysis with advanced terminology and formulas.",
        "instructions": "Provide a deep, advanced technical explanation. Include rich architectural details, advanced terminology, and mathematically complete LaTeX formulas. Ensure every formula's mathematical structure and variables are precisely transcribed."
    }
}

BILINGUAL_INSTRUCTIONS = {
    "hindi": "Please provide the answer in Hindi. Ensure the technical terms are preserved in English in parentheses if necessary. Ground the answer strictly in the provided context.",
    "english": "Please provide the answer in English. Ground the answer strictly in the provided context."
}

def get_mode_prompt(mode: str, context: str, question: str, language: str = "english") -> str:
    """
    Construct a prompt based on the selected mode and language.
    """
    mode_info = MODES.get(mode.lower(), MODES["beginner"])
    lang_instruction = BILINGUAL_INSTRUCTIONS.get(language.lower(), BILINGUAL_INSTRUCTIONS["english"])
    
    template = f"""
You are an expert Educational Assistant helping students understand academic material.

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

{mode_info['instructions']}
{lang_instruction}

Context:
{{context}}

---

Question: {{question}}
Answer:
"""
    return ChatPromptTemplate.from_template(template).format(
        context=context,
        question=question
    )
