from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_RAW_DIR
from src.embeddings.vector_store import add_documents
from src.ingest.pdf_parser import parse_pdf

logger = logging.getLogger(__name__)

def _chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text) # Just return the raw chunks

def process_pdf(pdf_path: Path) -> None:
    """Orchestrates PDF parsing and vector storage."""
    logger.info("Processing PDF: %s", pdf_path)
    try:
        elements = parse_pdf(pdf_path)
    except Exception as e:
        logger.error("Failed to parse %s: %s", pdf_path, e)
        return

    if not elements: return

    docs: List[str] = []
    metas: List[dict] = []

    for el in elements:
        content = el["content"]
        chunks = _chunk_text(content) if el["type"] == "text" else [content]
        
        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            metas.append({
                "source": pdf_path.name,
                "page": el["page_number"],
                "chunk": i,
                "modality": el["type"],
                "priority": "high" if el["type"] == "box" else "normal"
            })

    if docs:
        # Fixed: Removed the non-existent persist_immediately argument
        add_documents(docs, metas)
        logger.info(f"Added {len(docs)} chunks from {pdf_path.name}")

def run_pipeline() -> None:
    # This import is now safe because we added the function to vector_store.py
    from src.embeddings.vector_store import persist_vector_store
    
    pdf_files = sorted(Path(DATA_RAW_DIR).glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs found.")
        return

    for pdf in pdf_files:
        process_pdf(pdf)
    
    # Now this call won't fail
    persist_vector_store()
    logger.info("Pipeline complete!")