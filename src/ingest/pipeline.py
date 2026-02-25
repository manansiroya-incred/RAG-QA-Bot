from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Union
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

def process_pdf(
    pdf_path: Path,
    *,
    collection_name: str = "documents",
    persist_directory: Optional[Union[str, Path]] = None,
    session_id: str = "global",
) -> None:
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
        add_documents(docs, metas, collection_name=collection_name, persist_directory=persist_directory)
        logger.info(f"Added {len(docs)} chunks from {pdf_path.name}")

        try:
            from src.db.sql_index import insert_chunks, insert_document

            insert_document(
                session_id=session_id,
                source_file=pdf_path.name,
                uploaded_path=pdf_path,
            )
            chunk_rows = [
                (
                    int(m.get("page")) if m.get("page") is not None else None,
                    int(m.get("chunk")) if m.get("chunk") is not None else None,
                    str(m.get("modality")) if m.get("modality") is not None else None,
                    str(m.get("priority")) if m.get("priority") is not None else None,
                    str(d),
                    None,
                )
                for d, m in zip(docs, metas)
            ]
            insert_chunks(session_id=session_id, source_file=pdf_path.name, rows=chunk_rows)
        except Exception as exc:
            logger.warning("SQLite indexing skipped due to error: %s", exc)

def run_pipeline_for_files(
    pdf_files: Iterable[Path],
    *,
    collection_name: str = "documents",
    persist_directory: Optional[Union[str, Path]] = None,
    session_id: str = "global",
) -> None:
    from src.embeddings.vector_store import persist_vector_store

    pdf_list = [Path(p) for p in pdf_files]
    if not pdf_list:
        logger.warning("No PDFs provided.")
        return

    for pdf in pdf_list:
        process_pdf(
            pdf,
            collection_name=collection_name,
            persist_directory=persist_directory,
            session_id=session_id,
        )

    persist_vector_store()
    logger.info("Pipeline complete!")

def run_pipeline() -> None:
    pdf_files = sorted(Path(DATA_RAW_DIR).glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs found.")
        return

    run_pipeline_for_files(pdf_files)