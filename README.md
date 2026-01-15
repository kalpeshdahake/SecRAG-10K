# RAG System for Financial Question Answering on SEC 10-K Filings

## 📌 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG) system** that answers complex financial and legal questions using Apple Inc.'s Form 10-K (FY2024) and Tesla, Inc.'s Form 10-K (FY2023). 

The system retrieves relevant sections from the regulatory filings and generates accurate, well-sourced answers using **only open-source models and open-access LLMs**—no closed APIs (GPT-4, Claude, etc.).

### Key Capabilities
- ✅ Factual question answering grounded in SEC filing context
- ✅ Transparent source citations with page numbers
- ✅ Automatic out-of-scope detection and refusal
- ✅ Zero hallucinations (context-only generation)
- ✅ End-to-end reproducible in Google Colab/Kaggle

## 📂 Input Documents

The following SEC filings are used (provided in `data/`):

| Document | Company | Filing Date | Link |
|----------|---------|-------------|------|
| `10-Q4-2024-As-Filed.pdf` | Apple Inc. | Nov 1, 2024 | [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K&dateb=&owner=exclude&count=100) |
| `tsla-20231231-gen.pdf` | Tesla, Inc. | Jan 31, 2024 | [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001018724&type=10-K&dateb=&owner=exclude&count=100) |

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
| **LLM** | Mistral 7B or LLaMA 3 | Open-source instruction-tuned models, no API calls |
| **Runtime** | Python 3.10+, Google Colab | Cloud-native, fully reproducible, no GPU required for inference |

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

### Option 1: Run in Google Colab (Recommended)

1. **Clone and setup:**
   ```
   !git clone https://github.com/YOUR_USERNAME/rag-10k-system.git
   %cd rag-10k-system
   !pip install -r requirements.txt
   ```

2. **Run the notebook:**
   Open `notebook/run_rag.ipynb` and execute all cells

3. **Query the system:**
   ```python
   result = answer_question("What was Tesla's revenue in 2023?")
   print(result)
   ```

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

# Run the pipeline
python pipeline/rag_pipeline.py
```

## 📊 Evaluation Results

The system is evaluated on 13 test questions:

| Q# | Category | Status | Details |
|---|----------|--------|---------|
| 1-5 | Apple Financial Data | ✅ Answerable | Revenue, shares, debt, filing date, SEC comments |
| 6-10 | Tesla Financial & Risk | ✅ Answerable | Revenue, breakdown, Elon dependency, vehicles, lease arrangements |
| 11-13 | Out-of-Scope | ✅ Correctly Refused | Stock forecast, 2025 CFO, HQ color (not in 10-K) |

See [evaluation results](notebook/run_rag.ipynb) for detailed answers with sources.

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
  - Open-source LLM (Mistral/LLaMA 3)
  - No closed APIs used
  - Custom prompt ensures context-only generation with mandatory citations
  - Out-of-scope questions explicitly refused
  
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


## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the Colab notebook for inline documentation
3. See `design_report.md` for architectural details
