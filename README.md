# RAG-QA - Knowledge Assistant

A production-ready Retrieval-Augmented Generation (RAG) system orchestrated with **LangChain**. It ingests PDF documents or benchmark datasets, creates semantic embeddings, and enables high-precision question-answering using the latest **Gemini 3** architecture with grounded source citations and automated performance observability.

## Overview

This system leverages a sophisticated **4-stage retrieval pipeline** to ensure extreme accuracy and cost-efficiency:

1. **Semantic Search** – Uses HuggingFace embeddings (`e5-large-v2`) with mandatory `query:` and `passage:` prefixing.
2. **Maximal Marginal Relevance (MMR)** – Diversifies results to ensure the LLM sees unique information rather than redundant chunks.
3. **Keyword Boosting** – Amplifies exact keyword matches and high-priority "box" content via a custom heuristic layer.
4. **Cross-Encoder Reranking** – Performs deep-semantic scoring of candidates to ensure "Source 1" is the most relevant.

The system is powered by **Google Gemini 3 Flash**, orchestrated via **LangChain**, and features a built-in **Performance Observability** suite for tracking latency, token usage, and costs.

## Project Structure

```text
src/
├── app/
│   └── ui.py                # Streamlit interface with chat history and performance metrics
├── embeddings/
│   └── vector_store.py      # ChromaDB management via LangChain wrappers
├── ingest/
│   ├── pipeline.py          # PDF processing orchestration and semantic chunking
│   └── pdf_parser.py        # Text, table, and box extraction via pdfplumber
├── qa/
│   └── chain.py             # Custom LangChain Retriever (MMR + Boosting + Rerank)
└── retrieval/
    ├── rerank.py            # Cross-encoder model logic (ms-marco-MiniLM)
    └── tracker.py           # Performance Observability (Latency, Tokens, Cost tracking)

data/
├── raw/                     # Input PDF files for policy Q&A
├── processed/               # Metadata tracking for processed documents
└── chroma_db/               # Local persistent vector store (Ignored by Git)

ingest_squad.py              # Dedicated CLI for SQuAD benchmark ingestion
run_pipeline.py              # CLI for batch PDF ingestion
requirements.txt             # Python dependencies
.env                         # Environment variables (Google API Key)

```

## Installation

```bash
pip install -r requirements.txt

```

## Quick Start

### ⚡ Step 1: Configuration

Ensure your `.env` file contains your Google API Key:

```text
GOOGLE_API_KEY=your_gemini_api_key_here

```

### ⚡ Step 2: Data Ingestion

To use the system for private PDF files (e.g., InCred internal policies):

1. Place PDF files in `data/raw/`.
2. Run the ingestion pipeline:

```bash
python run_pipeline.py

```

### ⚡ Step 3: Launch Assistant

```bash
# Set Python path to ensure 'src' is recognized
$env:PYTHONPATH = "."
streamlit run src/app/ui.py

```

---

## 🚀 Advanced Features

### Automated Performance Tracking

Every query is monitored in real-time. The sidebar displays:

* **Latency:** Exact time taken for the LLM to process context and generate an answer.
* **Token Usage:** Breakdown of input, output, and **Gemini 3 Thinking Tokens**.
* **Cost Efficiency:** Real-time USD cost calculation based on Feb 2026 pricing.
* **Persistence:** All metrics are automatically saved to `performance_log.csv` for audit and reporting.

### LangChain Orchestration

The system utilizes **LangChain Expression Language (LCEL)** for a modular, "pluggable" architecture. By subclassing `BaseRetriever`, the custom 4-stage logic remains compatible with any standard LangChain component, allowing for seamless model swapping (e.g., Gemini 3 Flash vs. Pro).

### Intelligent PDF Parsing

`pdf_parser.py` doesn't just extract text; it detects spatial "boxes" (often used for high-priority policy alerts) and converts tables to Markdown to preserve structural relationships for the LLM's reasoning engine.

---

## Configuration (`src/config.py`)

| Setting | Value | Purpose |
| --- | --- | --- |
| `TOP_K` | 3 | Final number of sources shown to the User |
| `RERANK_TOP_K` | 8 | Candidates sent to the Cross-Encoder for deep-scoring |
| `EMBED_MODEL` | e5-large-v2 | SOTA embedding model with prefix requirements |
| `LLM_MODEL` | gemini-3-flash | Next-gen model with 1M+ context window |

---

## Architecture Highlights

### Four-Stage Retrieval Flow

```text
      User Query
          ↓
[1] MMR Search (ChromaDB) → Diversity & Relevance
          ↓ (Top-8 candidates)
[2] Keyword & Priority Boosting → Heuristic Re-weighting
          ↓ (Prefix Stripping)
[3] Cross-Encoder Reranking → Deep Semantic Scoring
          ↓ (Top-3 sorted results)
[4] Gemini 3 Generation → Grounded Answer with Metadata Citations

```

## Technologies

| Component | Technology |
| --- | --- |
| **Orchestration** | **LangChain** |
| **LLM** | **Google Gemini 3 Flash / 3.1 Pro** |
| **Embeddings** | HuggingFace (`intfloat/e5-large-v2`) |
| **Vector DB** | ChromaDB (Local Persistence) |
| **Reranking** | Sentence-Transformers (`Cross-Encoder`) |
| **Parsing** | `pdfplumber` + `pandas` |
| **Monitoring** | Custom Performance Callbacks (CSV Logging) |

---