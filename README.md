# RAG-QA - Knowledge Assistant

A production-ready Retrieval-Augmented Generation (RAG) system orchestrated with **LangChain**. It ingests PDF documents or benchmark datasets, creates semantic embeddings, and enables high-precision question-answering with grounded source citations and automated performance observability.

The system now supports **dynamic, per-session RAG** directly from the Streamlit UI (each user uploads their own PDFs and gets an isolated vector database), while still allowing **offline batch ingestion** via CLI.

## Overview

This system combines four advanced retrieval techniques to answer questions with extreme accuracy:

1. **Semantic Search** – Uses HuggingFace embeddings (`intfloat/e5-large-v2`) with mandatory `query:` and `passage:` prefixing.
2. **Maximal Marginal Relevance (MMR)** – Diversifies results to ensure the LLM sees unique information rather than redundant chunks.
3. **Keyword Boosting** – Amplifies exact keyword matches and high-priority "box" content for improved recall.
4. **Cross-Encoder Reranking** – Performs deep-semantic scoring of the top candidates to ensure "Source 1" is the most relevant.

The LLM model is configurable in `src/config.py` (e.g. Gemini Flash or Llama 3.3 via Groq), and is used to deliver ultra-fast, grounded responses with accurate citations.

## Project Structure

```text
src/
├── app/
│   └── ui.py                # Streamlit interface with chat, uploads, SQL index views, and performance metrics
├── embeddings/
│   └── vector_store.py      # ChromaDB management, E5 prefixing logic, dynamic persist directories
├── ingest/
│   ├── pipeline.py          # PDF processing orchestration and semantic chunking
│   └── pdf_parser.py        # Text, table, and box extraction via pdfplumber
├── qa/
│   └── chain.py             # Custom 4-stage retriever (MMR + Boosting + Rerank)
├── retrieval/
│   ├── rerank.py            # Cross-encoder model logic (ms-marco-MiniLM)
│   └── tracker.py           # Performance observability (latency, tokens, cost tracking)
└── db/
    └── sql_index.py         # SQLite metadata index for SQL-style inspection

data/
├── raw/                     # Input PDF files for offline/CLI policy Q&A
├── processed/               # Metadata tracking for processed documents
├── chroma_db/
│   ├── sessions/            # Per-session ChromaDB vector stores (one directory per Streamlit session)
│   └── ...                  # (legacy/global) ChromaDB persistence
├── user_uploads/            # Per-session uploaded PDFs (one subfolder per Streamlit session)
└── index.sqlite             # SQLite metadata DB for SQL-style inspection (ignored by Git)

ingest_squad_dataset.py      # Dedicated CLI for SQuAD benchmark ingestion
run_pipeline.py              # CLI for batch PDF ingestion from data/raw
requirements.txt             # Python dependencies
.env                         # Environment variables (e.g. Google / Groq API Keys)

```

## Installation

```bash
pip install -r requirements.txt

```

## Quick Start

### ⚡ Option 1: Dynamic per-session PDF upload (recommended)

This mode lets each user upload one or more PDFs from the Streamlit UI. The app will build a **session-specific** vector database and restrict retrieval to those uploads only.

```bash
# From the project root
pip install -r requirements.txt

$env:PYTHONPATH = "."
streamlit run src/app/ui.py
```

Then in the Streamlit sidebar:

1. Upload one or more PDF files under **“Upload PDFs”**.
2. Click **“Process PDFs”**.
   - The app will parse pages/tables/boxes, chunk text, build a ChromaDB under `data/chroma_db/sessions/<session_id>/`, and index metadata into `data/index.sqlite`.
3. Once processing is complete, the chat input is enabled. Ask questions such as _“What are the eligibility conditions for X?”_ and the model will answer using only your uploaded documents.

You can click **“Reset Session (delete uploads + DB)”** in the sidebar to clear chat, delete your session’s uploads and vector DB, and start again with a fresh session.

### 📁 Option 2: Using your own policy documents via CLI (batch ingestion)

To use the system for private PDF files in **batch/CLI mode** (e.g., InCred internal policies):

#### Step 1: Prepare data

Place your PDF files into the `data/raw/` folder.

#### Step 2: Ingest your documents

```bash
python run_pipeline.py
```

This will parse all PDFs in `data/raw/` and add them to the **global** ChromaDB under `data/chroma_db/`.

#### Step 3: Launch the app

```bash
$env:PYTHONPATH = "."
streamlit run src/app/ui.py
```

In this mode, the app will query the global ChromaDB (or any pre-ingested data such as SQuAD; see below). You can still use the dynamic upload mode in the UI if you prefer session-specific isolation.

### 🧪 Option 3: Benchmarking with SQuAD

To test the system using the Stanford Question Answering Dataset (SQuAD) from Hugging Face:

```bash
# Step 1: Ingest SQuAD contexts into the vector store
python ingest_squad_dataset.py

# Step 2: Set Python path and launch UI
$env:PYTHONPATH = "."
streamlit run src/app/ui.py
```

---

## 📋 Workflow Summary

| Step | Command | Purpose |
| --- | --- | --- |
| 1 | `pip install -r requirements.txt` | Install system dependencies |
| 2 | `streamlit run src/app/ui.py` | Launch the Knowledge Assistant UI (upload PDFs in-app; per-session DB) |
| 3 | `python run_pipeline.py` | (Optional) Batch-ingest PDFs from `data/raw/` into the global vector DB |
| 4 | `python ingest_squad_dataset.py` | (Optional) Stream SQuAD dataset → Vectorize → Store in global DB |

---

## Features

### Chat Interface & Observability

* **Performance Tracking**: Real-time monitoring of latency, token usage (including thinking tokens), and cost via `PerformanceTracker`.
* **Interactive History**: Bubble-style chat history in Streamlit with expandable source citations.

### Source Citations

* Expand **"📚 Source Citations"** in each assistant reply to see:
* Exact **page numbers** or **SQuAD IDs**.
* **Modality labels** (Text vs. Table vs. Box).
* Cleaned snippets (internal `passage:` prefixes are automatically stripped).

### SQL-style Inspection (Manager-friendly)

* Every ingested chunk (via UI uploads or CLI pipeline) is also mirrored into a lightweight **SQLite index** at `data/index.sqlite`.
* The UI exposes safe, per-session views under **“🧾 SQL Index (this session)”**:
  * **List uploaded PDFs this session**
  * **Count chunks by modality**
  * **Top pages by chunk count**
* Advanced users can open `data/index.sqlite` directly in any SQLite client and run `SELECT` queries (scoping by `session_id`) to inspect what content was indexed.

### Advanced Retrieval Pipeline

1. **Semantic Search**: Bi-Encoder finds top candidates via vector similarity (E5 embeddings with `query:` / `passage:` prefixes).
2. **MMR (Diversity)**: Re-filters results to minimize redundancy.
3. **Keyword Boosting**: Multiplies scores for exact query matches and high-priority sections (especially box content).
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
          ↓ (Top-N results)
[2] Keyword & Priority Boosting → Heuristic Check (boxes & exact matches)
          ↓ (Prefix Stripping)
[3] Cross-Encoder Reranking → Deep Semantic Score
          ↓ (Top-K sorted results)
[4] LLM Generation → Grounded Answer with Citations

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
| **Orchestration** | **LangChain** (LCEL-style chains) |
| **LLM** | Configurable (e.g. Google Gemini Flash or Llama 3.3 via Groq) |
| **Embeddings** | HuggingFace (`intfloat/e5-large-v2`) |
| **Vector DB** | ChromaDB (local persistence; per-session + global) |
| **Reranking** | Sentence-Transformers (`Cross-Encoder`) |
| **SQL Index** | SQLite (`data/index.sqlite`) |
| **Tracking** | Custom callbacks (`PerformanceTracker`) |

---