from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from llm.generator import LLMGenerator
from llm.prompt import build_prompt

def answer_question(query: str, collection) -> dict:
    q = query.lower()

    # 1️⃣ Hard out-of-scope
    if not any(x in q for x in ["apple", "tesla"]):
        return {
            "answer": "This question cannot be answered based on the provided documents.",
            "sources": []
        }

    # 2️⃣ Retrieve + rerank
    retriever = Retriever(collection, top_k=5)
    retrieved_chunks = retriever.retrieve(query)

    reranker = Reranker()
    top_chunks = reranker.rerank(query, retrieved_chunks, top_n=3)

    # 3️⃣ Build STRICT prompt
    prompt = build_prompt(query, top_chunks)

    # 4️⃣ Generate answer
    llm = LLMGenerator()
    output = llm.generate(prompt)

    # 5️⃣ Post-process
    answer = output.split("ANSWER:")[-1].strip()

    # Handle explicit refusals from model
    if answer.startswith("This question cannot be answered"):
        return {"answer": answer, "sources": []}

    if answer.startswith("Not specified"):
        return {"answer": answer, "sources": []}

    # 6️⃣ Source (top chunk – assignment allows this)
    meta = top_chunks[0]["metadata"]
    source = f"{meta['document']}, {meta['item']}, p. {meta['page']}"

    return {
        "answer": answer,
        "sources": [source]
    }

