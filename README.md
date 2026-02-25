# RAG-QA - Knowledge Assistant

A production-ready Retrieval-Augmented Generation (RAG) system orchestrated with **LangChain**. It ingests PDF documents or benchmark datasets, creates semantic embeddings, and enables high-precision question-answering with grounded source citations and automated performance observability.

## Overview

This system combines four advanced retrieval techniques to answer questions with extreme accuracy:

1. **Semantic Search** – Uses HuggingFace embeddings (`e5-large-v2`) with mandatory `query:` and `passage:` prefixing.
2. **Maximal Marginal Relevance (MMR)** – Diversifies results to ensure the LLM sees unique information rather than redundant chunks.
3. **Keyword Boosting** – Amplifies exact keyword matches and high-priority "box" content for improved recall.
4. **Cross-Encoder Reranking** – Performs deep-semantic scoring of the top candidates to ensure "Source 1" is the most relevant.

The system uses **Google Gemini 3 Flash** to deliver ultra-fast, grounded responses with accurate citations and internal reasoning.

## Project Structure

```text
src/
├── app/
│   └── ui.py                # Streamlit interface with chat history and performance metrics
├── embeddings/
│   └── vector_store.py      # ChromaDB management and E5 prefixing logic
├── ingest/
│   ├── pipeline.py          # PDF processing orchestration and semantic chunking
│   └── pdf_parser.py        # Text, table, and box extraction via pdfplumber
├── qa/
│   └── chain.py             # Custom 4-stage retriever (MMR + Boosting + Rerank)
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

### ⚡ Option 1: Benchmarking with SQuAD (Recommended for Testing)

To test the system using the Stanford Question Answering Dataset (SQuAD) from Hugging Face:

```bash
# Step 1: Ingest SQuAD contexts into the vector store
python ingest_squad.py

# Step 2: Set Python path and Launch UI
$env:PYTHONPATH = "."
streamlit run src/app/ui.py

```

### 📁 Option 2: Using Your Own Policy Documents (PDF)

To use the system for private PDF files (e.g., Algebrik/InCred internal policies):

#### Step 1: Prepare Data

Place your PDF files into the `data/raw/` folder.

#### Step 2: Ingest Your Documents

```bash
python run_pipeline.py

```

#### Step 3: Launch the App

```bash
$env:PYTHONPATH = "."
streamlit run src/app/ui.py

```

---

## 📋 Workflow Summary

| Step | Command | Purpose |
| --- | --- | --- |
| 1 | `pip install -r requirements.txt` | Install system dependencies |
| 2 | `python ingest_squad.py` | Stream SQuAD dataset → Vectorize → Store |
| 3 | `python run_pipeline.py` | Parse PDFs → Extract Boxes/Tables → Store |
| 4 | `streamlit run src/app/ui.py` | Launch the Knowledge Assistant UI |

---

## Features

### Chat Interface & Observability

* **Performance Tracking**: Real-time monitoring of Latency, Token usage (including Thinking Tokens), and Cost.
* **CSV Logging**: Every query is logged to `performance_log.csv` for audit and reporting.
* **Interactive History**: Bubble-style chat history with lazy-loading optimization.

### Source Citations

* Expand **"📚 Source Citations"** to see:
* Exact **Page Numbers** or **SQuAD IDs**.
* **Modality Labels** (Text vs. Table vs. Box).
* Cleaned snippets (internal `passage:` prefixes are automatically stripped).

### Advanced Retrieval Pipeline

1. **Semantic Search**: Bi-Encoder finds top candidates via vector similarity.
2. **MMR (Diversity)**: Re-filters results to minimize redundancy.
3. **Keyword Boosting**: Multiplies scores for exact query matches and high-priority sections.
4. **Cross-Encoder Reranking**: Final precision pass using `ms-marco-MiniLM-L-6-v2`.

---

## Configuration

Edit `src/config.py` to customize behavior:

| Setting | Value | Purpose |
| --- | --- | --- |
| `TOP_K` | 3 | Final number of sources shown to the LLM/User |
| `RERANK_TOP_K` | 8 | Number of candidates sent to the Cross-Encoder |
| `EMBED_MODEL` | e5-large-v2 | SOTA embedding model with prefix requirements |
| `LLM_MODEL` | gemini-3-flash-preview | High-intelligence model with multimodal support |

---

## Architecture Highlights

### Four-Stage Retrieval Flow

```text
      User Query
          ↓
[1] MMR Search (ChromaDB) → Diversity & Relevance
          ↓ (Top-8 results)
[2] Keyword & Priority Boosting → Heuristic Check
          ↓ (Prefix Stripping)
[3] Cross-Encoder Reranking → Deep Semantic Score
          ↓ (Top-3 sorted results)
[4] Gemini 3 Generation → Grounded Answer with Citations

```

### Ingestion Logic

* **PDF Pipeline**: `pdf_parser.py` detects spatial "boxes" and converts tables to Markdown.
* **Orchestration**: Built using **LangChain LCEL** for modularity and easy component swapping.

## Performance

* **Ingestion**: ~4 seconds per PDF page.
* **Retrieval**: <1.5 seconds (Semantic + MMR + Rerank).
* **Latency tracking**: Built-in automated callbacks for monitoring production costs and speed.

## Technologies

| Component | Technology |
| --- | --- |
| **Orchestration** | **LangChain** |
| **LLM** | **Google Gemini 3 Flash** |
| **Embeddings** | HuggingFace (`intfloat/e5-large-v2`) |
| **Vector DB** | ChromaDB (Local Persistence) |
| **Reranking** | Sentence-Transformers (`Cross-Encoder`) |
| **Tracking** | Custom Callbacks with CSV Logging |

---