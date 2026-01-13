from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from llm.generator import LLMGenerator

def answer_question(query: str, collection) -> dict:
    q = query.lower()

    # --------------------------------------------------
    # 1️⃣ HARD OUT-OF-SCOPE (MANDATORY)
    # --------------------------------------------------
    OUT_OF_SCOPE_KEYWORDS = [
        "stock price", "forecast", "prediction",
        "future", "color", "painted", "headquarters"
    ]

    if any(k in q for k in OUT_OF_SCOPE_KEYWORDS):
        return {
            "answer": "This question cannot be answered based on the provided documents.",
            "sources": []
        }

    # --------------------------------------------------
    # 2️⃣ RETRIEVE + RERANK
    # --------------------------------------------------
    retriever = Retriever(collection, top_k=5)
    chunks = retriever.retrieve(query)

    if not chunks:
        return {
            "answer": "This question cannot be answered based on the provided documents.",
            "sources": []
        }

    reranker = Reranker()
    top_chunks = reranker.rerank(query, chunks, top_n=3)

    context = "\n\n".join(c["text"] for c in top_chunks)

    # --------------------------------------------------
    # 3️⃣ EXTRACTIVE QA (NO GENERATION)
    # --------------------------------------------------
    qa = LLMGenerator()
    result = qa.extract(query, context)

    if not result or result["score"] < 0.25 or not result["answer"].strip():
        return {
            "answer": "Not specified in the document.",
            "sources": []
        }

    # --------------------------------------------------
    # 4️⃣ BUILD SOURCES
    # --------------------------------------------------
    sources = [
        f"{c['metadata']['document']}, {c['metadata']['item']}, p. {c['metadata']['page']}"
        for c in top_chunks
    ]

    return {
        "answer": result["answer"].strip(),
        "sources": sources
    }
