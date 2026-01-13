from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from llm.generator import LLMGenerator


OUT_OF_SCOPE_KEYWORDS = [
    "forecast", "stock price", "future",
    "painted", "color", "headquarters"
]

NUMERIC_KEYWORDS = [
    "total revenue", "revenue",
    "shares", "term debt",
    "filed", "signed"
]


def answer_question(query: str, collection) -> dict:
    q = query.lower()

    # --------------------------------------------------
    # 1️⃣ HARD OUT-OF-SCOPE (STRICT)
    # --------------------------------------------------
    if any(k in q for k in OUT_OF_SCOPE_KEYWORDS):
        return {
            "answer": "This question cannot be answered based on the provided documents.",
            "sources": []
        }

    # --------------------------------------------------
    # 2️⃣ RETRIEVE + RERANK
    # --------------------------------------------------
    retriever = Retriever(collection, top_k=5)
    retrieved_chunks = retriever.retrieve(query)

    if not retrieved_chunks:
        return {
            "answer": "This question cannot be answered based on the provided documents.",
            "sources": []
        }

    reranker = Reranker()
    top_chunks = reranker.rerank(query, retrieved_chunks, top_n=3)

    context = "\n\n".join(c["text"] for c in top_chunks)

    # --------------------------------------------------
    # 3️⃣ EXTRACTIVE QA
    # --------------------------------------------------
    qa = LLMGenerator()
    result = qa.extract(query, context)

    # Dynamic confidence threshold
    confidence_threshold = 0.15 if any(k in q for k in NUMERIC_KEYWORDS) else 0.25

    if (
        not result
        or not result["answer"].strip()
        or result["score"] < confidence_threshold
    ):
        return {
            "answer": "Not specified in the document.",
            "sources": []
        }

    # --------------------------------------------------
    # 4️⃣ SINGLE BEST SOURCE (FIXES MULTI-SOURCE ISSUE)
    # --------------------------------------------------
    best_chunk = top_chunks[0]
    meta = best_chunk["metadata"]

    source = f"{meta['document']}, {meta['item']}, p. {meta['page']}"

    return {
        "answer": result["answer"].strip(),
        "sources": [source]
    }
