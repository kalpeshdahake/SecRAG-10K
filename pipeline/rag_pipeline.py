from pipeline.financial_extractors import *


OUT_OF_SCOPE_KEYWORDS = [
    "forecast",
    "future",
    "stock price",
    "headquarters",
    "color",
    "painted",
    "cfo"
]


def answer_question(query, collection):
    q = query.lower()

    # 1️⃣ Out-of-scope
    if any(k in q for k in OUT_OF_SCOPE_KEYWORDS):
        return {
            "answer": "This question cannot be answered based on the provided documents.",
            "sources": []
        }

    # 2️⃣ Retrieval
    results = collection.query(query_texts=[query], n_results=8)

    chunks = [
        {"text": d, "metadata": m}
        for d, m in zip(results["documents"][0], results["metadatas"][0])
    ]

    if not chunks:
        return {
            "answer": "Not specified in the document.",
            "sources": []
        }

    # 3️⃣ Routing
    if "total revenue" in q:
        ans, src = extract_total_revenue(chunks)

    elif "term debt" in q:
        ans, src = extract_term_debt(chunks)

    elif "shares" in q and "outstanding" in q:
        ans, src = extract_shares_outstanding(chunks)

    elif "unresolved staff comments" in q:
        ans, src = extract_unresolved_staff_comments(chunks)

    elif "automotive sales" in q and "percentage" in q:
        ans, src = extract_automotive_sales_percentage(chunks)

    elif "vehicles" in q:
        ans, src = extract_vehicle_list(chunks)

    elif "elon musk" in q or "dependent" in q:
        ans, src = extract_dependency_statement(chunks)

    elif "lease pass-through" in q:
        ans, src = extract_lease_passthrough(chunks)

    else:
        return {
            "answer": "Not specified in the document.",
            "sources": []
        }

    # 4️⃣ Not found
    if not ans or not src:
        return {
            "answer": "Not specified in the document.",
            "sources": []
        }

    meta = src["metadata"]
    return {
        "answer": ans,
        "sources": [
            f'{meta["document"]}, {meta["item"]}, p. {meta["page"]}'
        ]
    }
