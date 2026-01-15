# Design Report: RAG System for SEC 10-K Financial Question Answering

## Executive Summary

This document outlines the architectural decisions for a Retrieval-Augmented Generation (RAG) system designed to answer complex financial and legal questions using Apple's FY2024 Form 10-K and Tesla's FY2023 Form 10-K filings. The system prioritizes **factual accuracy**, **transparent citations**, **hallucination prevention**, and **open-source compliance**.

## 1. Document Ingestion & Preprocessing

### PDF Parsing

The PDFs are parsed page by page to preserve page numbers required for citation. Each page is enriched with metadata including:

- Company name
- Document name
- Page number

### Section (Item) Detection

Section headers such as Item 1, Item 7, and Item 8 are detected using regex-based pattern matching. Once an Item is detected, it is carried forward to subsequent pages until a new Item header appears.

Pages prior to the first Item heading (e.g., cover and filing metadata pages) are labeled as "Unknown".

## 2. Chunking Strategy

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

## 3. Embeddings & Vector Storage

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

## 4. Retrieval & Re-Ranking

### Retrieval

A Top-K (k=5) similarity search is performed using vector embeddings to maximize recall.

### Re-Ranking

A cross-encoder re-ranker (ms-marco-MiniLM) is applied to improve precision by rescoring the retrieved chunks based on query relevance.

This two-stage approach balances recall and precision, which is critical for factual financial question answering.

## 5. LLM Integration & Guardrails

### Model Choice

An open-access instruction-tuned LLM (e.g., Mistral or LLaMA 3) is used for response generation.

### Prompt Design & Out-of-Scope Handling

The prompt enforces strict guardrails:

1. **Context-Only Responses**: The LLM is instructed to answer exclusively from retrieved context
2. **Explicit Citations**: All answers must cite sources in the format `["Apple 10-K", "Item X", "p. YYY"]`
3. **Out-of-Scope Refusal**: Responses outside the provided documents are refused with: *"This question cannot be answered based on the provided documents."*
4. **Unknown Information Handling**: When information is genuinely unavailable in the 10-K filings, the system responds: *"Not specified in the document."*

**Examples of Out-of-Scope Questions:**
- Stock price forecasts (forward-looking statements not in historical filings)
- Current 2025 executive roles (filings are historical)
- Subjective attributes (e.g., headquarters color)

The system validates that retrieved context is substantive (>100 characters) before generating answers. If no relevant context exists, an out-of-scope refusal is issued.

## 6. Hallucination Prevention

Hallucinations are prevented through:

- **Context-only prompting**
- **Empty-context refusal logic**
- **Mandatory citation enforcement**
- **No external knowledge access**

## 7. Reproducibility & Deployment

The system is designed for end-to-end reproducibility in cloud environments:

- **Platform**: Google Colab or Kaggle Notebooks (no local GPU required)
- **Execution**: All dependencies installed via `requirements.txt`
- **Data**: PDFs provided; auto-download from GitHub if needed
- **Indexing**: Full pipeline runs in single notebook cell
- **Inference**: Query via `answer_question(query)` function

The notebook includes evaluation cells for all 13 test questions with formatted JSON output.

## 8. Conclusion

This RAG system demonstrates a production-ready architecture for regulatory document analysis. By combining:
- Robust multi-stage retrieval (embedding search + re-ranking)
- Strict context-grounding with mandatory citations
- Explicit out-of-scope refusal logic
- Open-source compliance throughout

The system reliably answers factual financial questions while preventing hallucinations and maintaining transparency in answer generation. The design is fully reproducible and deployable in cloud notebooks without proprietary APIs.
