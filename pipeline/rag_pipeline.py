from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from llm.generator import LLMGenerator
from llm.prompt import build_prompt


# -------------------------------
# Helper: clean LLM output
# -------------------------------
def clean_answer(text: str) -> str:
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


# -------------------------------
# Helper: semantic out-of-scope
# -------------------------------
OUT_OF_SCOPE_PATTERNS = [
    "forecast",
    "prediction",
    "future",
    "as of 2025",
    "as of 2026",
    "color",
    "painted",
    "headquarters",
    "stock price"
]

def is_semantic_out_of_scope(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in OUT_OF_SCOPE_PATTERNS)


# -------------------------------
# Main RAG entry point
# -------------------------------
def answer_question(query: str, collection) -> dict:
    q = query.lower()

    # HARD lexical out-of-scope
    if not any(x in q for x in ["apple", "tesla"]):
        return {
            "answer": "This question cannot be answered based on the provided documents.",
            "sources": []
        }

    # HARD semantic out-of-scope (assignment-critical)
    if is_semantic_out_of_scope(query):
        return {
            "answer": "This question cannot be answered based on the provided documents.",
            "sources": []
        }

    # Retrieve + rerank
    retriever = Retriever(collection, top_k=5)
    retrieved_chunks = retriever.retrieve(query)

    reranker = Reranker()
    top_chunks = reranker.rerank(query, retrieved_chunks, top_n=3)

    # In-scope but not found
    if not top_chunks:
        return {
            "answer": "Not specified in the document.",
            "sources": []
        }

    # Build prompt
    prompt = build_prompt(query, top_chunks)

    # Generate
    llm = LLMGenerator()
    output = llm.generate(prompt)

    # Post-process
    answer = output.split("ANSWER:")[-1].strip()
    answer = clean_answer(answer)

    # Forced refusals
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

    # Valid answer → single source
    meta = top_chunks[0]["metadata"]
    source = f"{meta['document']}, {meta['item']}, p. {meta['page']}"

    return {
        "answer": answer,
        "sources": [source]
    }
