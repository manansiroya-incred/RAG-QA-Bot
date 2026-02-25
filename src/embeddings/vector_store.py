from __future__ import annotations
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CHROMA_DB_DIR, EMBED_MODEL

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    logger.info("Loading embeddings model: %s", EMBED_MODEL)
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_vector_store(
    collection_name: str = "documents",
    persist_directory: Optional[Union[str, Path]] = None,
) -> Chroma:
    """Return a persistent Chroma vector store."""
    if persist_directory is None:
        persist_directory = CHROMA_DB_DIR
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=str(persist_path),
    )

def add_documents(
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    collection_name: str = "documents",
    persist_directory: Optional[Union[str, Path]] = None,
) -> None:
    # Add documents to the Chroma collection with mandatory E5 prefixes.
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas length must match texts length")

    # CRITICAL: E5 models require 'passage: ' prefix for document storage
    prefixed_texts = [f"passage: {t}" for t in texts]
    
    store = get_vector_store(collection_name=collection_name, persist_directory=persist_directory)
    store.add_texts(texts=prefixed_texts, metadatas=metadatas)
    
    store.persist()
    logger.info(f"Added {len(texts)} documents to {collection_name}")

def persist_vector_store() -> None:
    """
    In modern Chroma (0.4+), persistence is automatic.
    This function remains as a placeholder to prevent pipeline errors.
    """
    # If you specifically wanted to force a save in older versions, 
    # you would call store.persist(), but it's no longer required.
    logger.info("Vector store persistence handled automatically by Chroma.")