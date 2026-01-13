# RAG System for Financial & Legal Question Answering (Apple & Tesla 10-Ks)

## 📌 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers complex financial and legal questions using Apple Inc.'s FY2024 Form 10-K and Tesla, Inc.'s FY2023 Form 10-K filings.

The system retrieves relevant sections from the filings and generates accurate, well-sourced answers using open-source embedding models and LLMs, without relying on any closed or proprietary APIs.

## 📂 Input Documents

The following PDFs were provided as part of the assignment and used directly:

- **Apple Inc. Form 10-K** (Fiscal year ended September 28, 2024)
- **Tesla, Inc. Form 10-K** (Fiscal year ended December 31, 2023)

## 🧠 System Architecture

```
PDF Documents
    ↓
PDF Parsing (page-level)
    ↓
Section (Item) Detection
    ↓
Chunking with Overlap
    ↓
Embedding Generation (Open Source)
    ↓
Vector Database (ChromaDB)
    ↓
Similarity Search (Top-K)
    ↓
Cross-Encoder Re-Ranking
    ↓
LLM Answer Generation (Context-only)
    ↓
JSON Answer with Citations
```

## 🧩 Key Features

- Page-aware and section-aware citations
- Metadata-preserving chunking
- Open-source embeddings and LLMs only
- Strict hallucination prevention
- Clear refusal handling for out-of-scope queries
- Separate indexing for Apple and Tesla filings

## 🛠️ Technology Stack

| Component | Tool |
|-----------|------|
| PDF Parsing | pypdf |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Re-Ranking | Cross-Encoder (ms-marco-MiniLM) |
| LLM | Mistral / LLaMA 3 (open-access) |
| Runtime | Python 3.10+, Google Colab / Kaggle |

## 📁 Project Structure

```
rag-10k-system/
├── data/
│   ├── 10-Q4-2024-As-Filed.pdf
│   └── tsla-20231231-gen.pdf
│
├── ingestion/
│   ├── pdf_loader.py
│   ├── section_parser.py
│   └── chunker.py
│
├── embeddings/
│   ├── embedder.py
│   └── vector_store.py
│
├── retrieval/
│   ├── retriever.py
│   └── reranker.py
│
├── llm/
│   ├── prompt.py
│   └── generator.py
│
├── pipeline/
│   └── rag_pipeline.py
│
├── notebook/
│   └── run_rag.ipynb
│
├── requirements.txt
├── README.md
└── design_report.md
```

## 🔑 Required Interface

The system exposes a single callable function:

```python
answer_question(query: str) → dict
```

**Returns:**

- `answer`: the generated answer or a refusal message
- `sources`: list of citations in the format `["Apple 10-K", "Item 8", "p. 282"]`

## 🚀 How to Run (Colab / Kaggle)

1. Clone the GitHub repository
2. Install dependencies using requirements.txt
3. Run the notebook `notebook/run_rag.ipynb`
4. Use `answer_question(query)` to query the system

## ⚠️ Notes

- Closed APIs such as GPT-4 or Claude are not used.
- The LLM is strictly limited to retrieved context.
- Some early document pages do not belong to any Item section and are labeled as "Unknown".

## ✅ Compliance Summary

- ✔ Open-source models only
- ✔ Context-grounded answers
- ✔ Mandatory citations
- ✔ Explicit out-of-scope refusal
- ✔ Fully reproducible notebook
