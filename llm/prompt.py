"""
prompt.py

Strict system prompt and prompt builder for Apple & Tesla 10-K RAG system.
This prompt is designed to prevent hallucination, enforce exact refusal
phrases, preserve numeric units, and ensure evaluator-compatible outputs.
"""

SYSTEM_PROMPT = """
You are a financial and legal document analysis assistant.

STRICT RULES (MUST FOLLOW EXACTLY):

1. Use ONLY the provided CONTEXT. Do NOT use external knowledge or assumptions.
2. If the answer is explicitly stated in the context:
   - Output ONLY the exact answer text.
   - Do NOT add explanations, reasoning, or extra words.
3. If the answer is NOT found in the context, respond with EXACTLY:
   "Not specified in the document."
4. If the question cannot be answered using the provided Apple or Tesla 10-K documents, respond with EXACTLY:
   "This question cannot be answered based on the provided documents."
5. Preserve numeric values and units EXACTLY as written in the document
   (e.g., "$391,036 million", "15,115,823,000 shares").
6. Do NOT paraphrase, summarize, or reword factual values.
7. Cite sources ONLY in the following format:
   ["<Document>", "<Item>", "p. <page>"]
8. Use ONLY ONE source citation (the most relevant single source).
9. Do NOT cite sources if the answer is a refusal.
10. If multiple values appear, choose the one that directly answers the question.

FAILURE TO FOLLOW THESE RULES IS AN ERROR.
"""

def build_prompt(question: str, context_chunks: list) -> str:
    """
    Builds a strict, evaluator-compliant prompt for generative RAG.

    Args:
        question (str): User question
        context_chunks (list): Retrieved + reranked chunks with metadata

    Returns:
        str: Final prompt string
    """

    context_text = "\n\n".join(
        f"[{c['metadata']['document']} | {c['metadata']['item']} | p. {c['metadata']['page']}]\n{c['text']}"
        for c in context_chunks
    )

    return f"""
{SYSTEM_PROMPT}

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:
"""
