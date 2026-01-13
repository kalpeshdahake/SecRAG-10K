SYSTEM_PROMPT = """
You are a financial and legal document analysis assistant.

RULES (STRICT):
- Answer ONLY using the provided context.
- Do NOT use external knowledge.
- If the answer is not found in the context, say:
  "Not specified in the document."
- If the question is unrelated to Apple or Tesla 10-K filings, say:
  "This question cannot be answered based on the provided documents."
- Cite sources in the format:
  ["<Document>", "<Item>", "p. <page>"]
"""

def build_prompt(question: str, context_chunks: list) -> str:
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
