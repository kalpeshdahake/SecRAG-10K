from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from llm.prompt import build_prompt
from llm.generator import LLMGenerator

def answer_question(
    query: str,
    collection
) -> dict:
    """
    Answers a question using the RAG pipeline.
    """

    # Step 1: Retrieve
    retriever = Retriever(collection, top_k=5)
    retrieved_chunks = retriever.retrieve(query)

    if not retrieved_chunks:
        return {
            "answer": "This question cannot be answered based on the provided documents.",
            "sources": []
        }

    # Step 2: Re-rank
    reranker = Reranker()
    top_chunks = reranker.rerank(query, retrieved_chunks, top_n=3)

    # Step 3: Build prompt
    prompt = build_prompt(query, top_chunks)

    # Step 4: Generate answer
    llm = LLMGenerator()
    answer_text = llm.generate(prompt)

    # Step 5: Build citations
    sources = []
    for c in top_chunks:
        sources.append(
            f"{c['metadata']['document']}, {c['metadata']['item']}, p. {c['metadata']['page']}"
        )

    return {
        "answer": answer_text.strip(),
        "sources": sources
    }
