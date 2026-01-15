"""
prompt.py

STRICT extraction-only prompt for Apple & Tesla 10-K RAG system.

IMPORTANT:
- LLM does NOT decide refusal vs out-of-scope
- LLM does NOT emit sources
- LLM does NOT explain reasoning
- Python enforces all logic
"""

SYSTEM_PROMPT = """
You are a financial and legal document analysis assistant.

STRICT RULES (MUST FOLLOW EXACTLY):

1. Use ONLY the provided CONTEXT.
2. Output ONLY the exact answer text.
3. Do NOT add explanations, reasoning, context, or citations.
4. If the answer is NOT explicitly present in the CONTEXT, output EXACTLY:
   "Not specified in the document."
5. Do NOT guess, infer, summarize, or calculate.
6. Preserve numbers, currency, units, and wording EXACTLY as written.
"""

def build_prompt(question: str, context_chunks: list) -> str:
    """
    Builds a strict prompt for extractive document QA.
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
