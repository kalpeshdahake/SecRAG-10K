# Design Report: Retrieval-Augmented Generation System for SEC 10-K Filings

## 1. Objective

The objective of this project is to design and implement a Retrieval-Augmented Generation (RAG) system capable of answering complex financial and legal questions using Apple's FY2024 and Tesla's FY2023 Form 10-K filings.

The design emphasizes:

- **Accuracy and factual grounding**
- **Clear traceability via citations**
- **Prevention of hallucinations**
- **Use of open-source tools only**

## 2. Document Ingestion & Preprocessing

### PDF Parsing

The PDFs are parsed page by page to preserve page numbers required for citation. Each page is enriched with metadata including:

- Company name
- Document name
- Page number

### Section (Item) Detection

Section headers such as Item 1, Item 7, and Item 8 are detected using regex-based pattern matching. Once an Item is detected, it is carried forward to subsequent pages until a new Item header appears.

Pages prior to the first Item heading (e.g., cover and filing metadata pages) are labeled as "Unknown".

## 3. Chunking Strategy

### Approach

A sliding-window chunking strategy is used:

- **Chunk size:** ~900 characters
- **Overlap:** ~150 characters

### Justification

This approach:

- Preserves semantic continuity across long financial disclosures
- Prevents fragmentation of tables and notes
- Improves retrieval accuracy while maintaining citation fidelity

Each chunk retains the original page number and Item section metadata.

## 4. Embeddings & Vector Storage

### Embedding Model

The system uses the open-source embedding model:

```
sentence-transformers/all-MiniLM-L6-v2
```

**Rationale:**

- Lightweight and CPU-friendly
- Strong semantic retrieval performance
- Suitable for legal and financial text
- Fully open-source and assignment-compliant

### Vector Database

ChromaDB is used for vector storage due to:

- Native metadata filtering
- Persistent local storage
- Ease of debugging and inspection

Apple and Tesla filings are stored in separate collections to avoid cross-document contamination.

## 5. Retrieval & Re-Ranking

### Retrieval

A Top-K (k=5) similarity search is performed using vector embeddings to maximize recall.

### Re-Ranking

A cross-encoder re-ranker (ms-marco-MiniLM) is applied to improve precision by rescoring the retrieved chunks based on query relevance.

This two-stage approach balances recall and precision, which is critical for factual financial question answering.

## 6. LLM Integration & Guardrails

### Model Choice

An open-access instruction-tuned LLM (e.g., Mistral or LLaMA 3) is used for response generation.

### Prompt Design

The prompt enforces:

- Usage of retrieved context only
- Explicit source citations
- "Not specified in the document." responses when applicable
- Refusal of out-of-scope questions

## 7. Hallucination Prevention

Hallucinations are prevented through:

- **Context-only prompting**
- **Empty-context refusal logic**
- **Mandatory citation enforcement**
- **No external knowledge access**

## 8. Reproducibility

The system is designed to run end-to-end in a public Google Colab or Kaggle notebook, ensuring:

- Clean execution environment
- Reproducibility
- Transparent evaluation

## 9. Conclusion

This project demonstrates a production-ready RAG pipeline for regulatory document analysis, combining robust retrieval, precise citation handling, and strict answer validation using only open-source components.
