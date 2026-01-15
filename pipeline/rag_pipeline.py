from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from llm.generator import LLMGenerator
from llm.prompt import build_prompt


# ------------------------------
# Scope Control
# ------------------------------

IN_SCOPE_TERMS = [
    "revenue",
    "term debt",
    "shares",
    "automotive",
    "vehicle",
    "lease",
    "staff comments",
    "form 10-k",
    "10-k"
]

def is_out_of_scope(query: str) -> bool:
    q = query.lower()

    # Must reference Apple or Tesla
    if not any(x in q for x in ["apple", "tesla"]):
        return True

    # Must relate to 10-K factual content
    if not any(k in q for k in IN_SCOPE_TERMS):
        return True

    return False


# ------------------------------
# LLM Output Cleaner
# ------------------------------

def clean_answer(text: str) -> str:
    """
    Remove leaked prompt/context text from LLM output.
    """
    stop_tokens = [
        "context:",
        "strict rules",
        "document:",
        "source",
        "explanation"
    ]

    lowered = text.lower()
    for token in stop_tokens:
        if token in lowered:
            text = text[:lowered.index(token)]

    return text.strip()


# ------------------------------
# Main RAG Entry Point
# ------------------------------

def answer_question(query: str, collection) -> dict:
    """
    Answers a question using the RAG pipeline.

    Returns ONLY:
    {
        "answer": "...",
        "sources": [...]
    }
    """

    # 1️⃣ HARD OUT-OF-SCOPE REFUSAL
    if is_out_of_scope(query):
        return {
            "answer": "This question cannot be answered based on the provided documents.",
            "sources": []
        }

    # 2️⃣ RETRIEVE
    retriever = Retriever(collection, top_k=5)
    retrieved_chunks = retriever.retrieve(query)

    # 3️⃣ RERANK
    reranker = Reranker()
    top_chunks = reranker.rerank(query, retrieved_chunks, top_n=3)

    if not top_chunks:
        return {
            "answer": "Not specified in the document.",
            "sources": []
        }

    # 4️⃣ BUILD PROMPT + GENERATE
    prompt = build_prompt(query, top_chunks)
    llm = LLMGenerator()

    raw_output = llm.generate(prompt)
    answer = clean_answer(raw_output)

    # 5️⃣ HANDLE EXPLICIT NULL ANSWERS
    if answer == "Not specified in the document.":
        return {
            "answer": answer,
            "sources": []
        }

    if answer.startswith("This question cannot be answered"):
        return {
            "answer": answer,
            "sources": []
        }

    # 6️⃣ CONTROLLED SOURCE (TOP CHUNK ONLY)
    meta = top_chunks[0]["metadata"]
    source = f"{meta['document']}, {meta['item']}, p. {meta['page']}"

    return {
        "answer": answer,
        "sources": [source]
    }
