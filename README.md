# RAG System for SEC 10-K Financial Question Answering

## Overview

A **Retrieval-Augmented Generation (RAG) system** that answers financial and legal questions from Apple's FY2024 Form 10-K and Tesla's FY2023 Form 10-K. Uses only open-source models with semantic retrieval, cross-encoder re-ranking, and deterministic LLM generation.

**Key Features**: Transparent citations • Zero hallucinations • Out-of-scope detection • One-click Colab execution

## � Live Demo - Click to Run

### **[Open in Google Colab](https://colab.research.google.com/github/kalpeshdahake/SecRAG-10K/blob/main/run_rag.ipynb)**

1. Click the link above
2. **Change runtime to GPU**: Runtime → Change runtime type → T4 GPU
3. Click "Run all"
4. Wait ~10 minutes
5. See results with citations

No installation needed. Runs everything automatically.

---

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

- **Page-Aware Citations**: Every answer cites document, section (Item), and page number
- **Metadata Preservation**: Chunking maintains section (Item) and page metadata through pipeline
- **Open-Source Only**: All models are open-source; no proprietary APIs
- **Hallucination Prevention**: Context-only generation with empty-context refusal
- **Clear Refusal Logic**: 
  - Out-of-scope: *"This question cannot be answered based on the provided documents."*
  - Information missing: *"Not specified in the document."*
- **Separate Collections**: Apple and Tesla filings indexed independently to prevent cross-document contamination

## 🛠️ Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **PDF Parsing** | pypdf | Extracts text while preserving page structure |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Lightweight, open-source, strong semantic performance |
| **Vector DB** | ChromaDB | Persistent local storage, metadata filtering, debugging support |
| **Re-Ranking** | Cross-Encoder (ms-marco-MiniLM) | Balances recall (embedding search) with precision |
| **LLM** | Microsoft Phi-3-mini (4K-instruct) | Open-source, lightweight (3.8B), instruction-tuned, CPU inference |
| **Runtime** | Python 3.10+, PyTorch 2.0+, Google Colab | Cloud-native, fully reproducible, efficient CPU/GPU inference |

## 📦 Complete Technology & Library Stack

**All Open-Source Components Used:**

| Library | Version | Purpose |
|---------|---------|---------|
| pypdf | ≥4.0.0 | PDF text extraction |
| sentence-transformers | ≥2.2.0 | Semantic embeddings (all-MiniLM-L6-v2) |
| chromadb | ≥0.4.0 | Vector database for embeddings |
| torch | ≥2.0.0 | Deep learning framework (Phi-3 inference) |
| transformers | ≥4.30.0 | Hugging Face models (Phi-3, cross-encoder) |
| scikit-learn | ≥1.3.0 | Utilities and preprocessing |
| numpy | ≥1.24.0 | Numerical computing |
| pandas | ≥2.0.0 | Data manipulation |

**Pre-Trained Models (From Hugging Face Hub):**
- `microsoft/phi-3-mini-4k-instruct` – 3.8B instruction-tuned LLM
- `sentence-transformers/all-MiniLM-L6-v2` – 384-dim semantic embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` – Cross-encoder for re-ranking

**No Proprietary APIs or Closed Models Used** ✅

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

## 🔑 Core Interface

All queries are processed through a single function:

```python
def answer_question(query: str) -> dict:
    """
    Answers a question using the RAG pipeline.
    
    Args:
        query (str): The user question about Apple or Tesla 10-K filings.
    
    Returns:
        dict: {
            "answer": "Answer text or 'This question cannot be answered based on the provided documents.'",
            "sources": ["Apple 10-K", "Item 8", "p. 282"]  # Empty list if refused
        }
    """
```

**Example Usage:**
```python
result = answer_question("What was Apple's total revenue for FY2024?")
print(result)
# Output:
# {
#     "answer": "Apple's total revenue for the fiscal year ended September 28, 2024 was $391,036 million.",
#     "sources": ["Apple 10-K", "Item 8", "p. 282"]
# }
```

## 🚀 Quick Start

### Option 1: Run in Google Colab (Recommended - 1 Click!)

**[Click here to open the notebook in Google Colab →](https://colab.research.google.com/github/kalpeshdahake/SecRAG-10K/blob/main/run_rag.ipynb)**

The notebook will:
- Clone your repo automatically
- Install all dependencies
- Index the PDFs
- Run all 13 test questions
- Display interactive query mode

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-10k-system.git
cd rag-10k-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Option A: Run Jupyter notebook locally
jupyter notebook run_rag.ipynb

# Option B: Run pipeline script directly
python -m pipeline.rag_pipeline
```

### Option 3: Use the RAG Function Directly

```python
from pipeline.rag_pipeline import answer_question

# Answer a question
result = answer_question("What was Apple's revenue in 2024?", collection)
print(result)
# Output: {"answer": "...", "sources": [...]}
```

## 📊 Evaluation Results

The system is evaluated on 13 test questions:

| Q# | Category | Status | Details |
|---|----------|--------|---------|
| 1-5 | Apple Financial Data | ✅ Answerable | Revenue, shares, debt, filing date, SEC comments |
| 6-10 | Tesla Financial & Risk | ✅ Answerable | Revenue, breakdown, Elon dependency, vehicles, lease arrangements |
| 11-13 | Out-of-Scope | ✅ Correctly Refused | Stock forecast, 2025 CFO, HQ color (not in 10-K) |

See [evaluation results](run_rag.ipynb) in the notebook for detailed answers with sources.

## 📓 Notebook Structure (run_rag.ipynb)

**Section 1: Setup & Installation**
- Detect Colab vs. local environment
- Clone GitHub repo (Colab only)
- Install dependencies from requirements.txt
- Verify all imports

**Section 2: PDF Indexing Pipeline**
- Load Apple & Tesla PDFs
- Parse and section-detect
- Generate chunks with metadata
- Create embeddings
- Index in ChromaDB

**Section 3: Inference & Evaluation**
- Load 13 test questions
- Run RAG pipeline
- Output JSON results
- Display summary statistics

**Section 4: Interactive Mode**
- Custom query function
- Example queries
- Live testing interface

**Section 5: Download Results**
- Save results.json
- Download to local machine (Colab)

## 📋 Assignment Compliance Checklist

- [x] **Step 1: Document Ingestion & Indexing**
  - PDFs parsed with page-level and section-level metadata
  - Embeddings generated using `sentence-transformers/all-MiniLM-L6-v2`
  - Vector database: ChromaDB with separate collections per company
  
- [x] **Step 2: Retrieval Pipeline**
  - Top-5 similarity search for recall
  - Cross-encoder re-ranking (ms-marco-MiniLM) for precision
  - Design decisions documented in `design_report.md`
  
- [x] **Step 3: LLM Integration**
  - Open-source LLM: Microsoft Phi-3-mini-4k-instruct
  - No closed APIs used (transformers + torch from Hugging Face)
  - Temperature = 0.0 (deterministic, zero hallucination)
  - Dual out-of-scope filtering: lexical + semantic patterns
  - Mandatory single-source citations
  
- [x] **Step 4: Answer 13 Test Questions**
  - All 13 questions evaluated in `notebook/run_rag.ipynb`
  - JSON format output
  - Questions 1-10 answered; 11-13 correctly refused

- [x] **Cloud Deployment**
  - Runnable end-to-end in Google Colab
  - All dependencies in `requirements.txt`
  - Live notebook link in README
  
- [x] **Submission Package**
  - GitHub repository with all code
  - Design report: `design_report.md`
  - This README with live notebook link

## 🔧 Troubleshooting & FAQs

### Q: Why do some questions return "Not specified in the document"?
**A:** The 10-K filing may not contain that information. For example, forward-looking statements (stock price forecasts) and current-year roles are not in historical filings.

### Q: Why Phi-3-mini and not Mistral or LLaMA?
**A:** Phi-3-mini is lightweight (3.8B parameters), runs efficiently on CPU, and is specifically designed for factual, instruction-following tasks—critical for accurate financial Q&A.

### Q: Why temperature=0.0?
**A:** Deterministic generation (temperature=0.0, do_sample=False) prevents hallucinations and ensures consistent, reproducible answers for the same query.

### Q: Can I use GPT-4 or Claude instead?
**A:** No. The assignment explicitly requires open-source models only. Using closed APIs violates the assignment requirements.

### Q: How do I add more documents?
**A:** Follow the structure in `ingestion/pdf_loader.py`. Update `pdf_loader.py` to add new PDFs, then re-run the ingestion pipeline.

### Q: What if a question spans multiple pages or sections?
**A:** The re-ranker scores all retrieved chunks, and the LLM synthesizes answers from the top-ranked chunks while citing all relevant sources.

### Q: Why ChromaDB instead of FAISS?
**A:** ChromaDB offers native metadata filtering (to filter by company/section), simpler local persistence, and easier inspection for debugging.

## 📚 Design Philosophy

This RAG system follows **context-driven design**:

1. **Retrieve First**: Multi-stage retrieval (embedding + re-ranking) maximizes recall
2. **Cite Always**: Every answer requires source attribution
3. **Refuse Wisely**: Out-of-scope questions are detected and refused explicitly
4. **Ground Strictly**: The LLM has no access to external knowledge; answers are 100% derived from context

This prevents hallucinations while maintaining transparency and factual accuracy.

## 🚀 Live Demo & Run Now

**Click to Open in Google Colab (Easiest Way):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/rag-10k-system/blob/main/run_rag.ipynb)

**What the notebook does:**
1. ✅ Clones the GitHub repo automatically
2. ✅ Installs all dependencies
3. ✅ Indexes Apple & Tesla 10-K PDFs (10 minutes)
4. ✅ Runs inference on all 13 test questions
5. ✅ Displays results + allows custom queries
6. ✅ Downloads results as JSON

**No setup required** - just click the button above!

## 📖 Resources

- **[Notebook Code](run_rag.ipynb)** - End-to-end Colab notebook
- **[GitHub Repository](https://github.com/kalpeshdahake/SecRAG-10K)** - Full source code
- **[Design Report](design_report.md)** - Architecture decisions

For issues or questions:
1. Check the troubleshooting section above
2. Review the Colab notebook for inline documentation
3. See `design_report.md` for architectural details
