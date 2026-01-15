from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from llm.generator import LLMGenerator
from llm.prompt import build_prompt


def clean_answer(text: str) -> str:
    """
    Cleans LLM output so ONLY the final answer text remains.
    Prevents CONTEXT / STRICT RULES / SOURCES leakage.
    """
    STOP_MARKERS = [
        "CONTEXT:",
        "STRICT RULES",
        "SOURCE:",
        "SOURCES:",
        "Document:",
        "[Apple",
        "[Tesla"
    ]

    for marker in STOP_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0]

    return text.strip()


def answer_question(query: str, collection) -> dict:
    q = query.lower()

    # HARD out-of-scope (DO NOT trust LLM for this)
    if not any(x in q for x in ["apple", "tesla"]):
        return {
            "answer": "This question cannot be answered based on the provided documents.",
            "sources": []
        }

    # Retrieve + rerank
    retriever = Retriever(collection, top_k=5)
    retrieved_chunks = retriever.retrieve(query)

    reranker = Reranker()
    top_chunks = reranker.rerank(query, retrieved_chunks, top_n=3)

    # No relevant context found → Not specified
    if not top_chunks:
        return {
            "answer": "Not specified in the document.",
            "sources": []
        }

    # Build STRICT prompt
    prompt = build_prompt(query, top_chunks)

    # Generate answer
    llm = LLMGenerator()
    output = llm.generate(prompt)

    # Post-process (STRICT)
    answer = output.split("ANSWER:")[-1].strip()
    answer = clean_answer(answer)

    # Refusal handling (FORCED)
    if "This question cannot be answered" in answer:
        return {
            "answer": "This question cannot be answered based on the provided documents.",
            "sources": []
        }

    if "Not specified in the document." in answer or len(answer) == 0:
        return {
            "answer": "Not specified in the document.",
            "sources": []
        }

    # Valid answer → add single source (top chunk)
    meta = top_chunks[0]["metadata"]
    source = f"{meta['document']}, {meta['item']}, p. {meta['page']}"

    return {
        "answer": answer,
        "sources": [source]
    }
