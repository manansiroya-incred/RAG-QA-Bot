from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional
import streamlit as st # Added for professional caching

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CHROMA_DB_DIR, EMBED_MODEL

logger = logging.getLogger(__name__)

@st.cache_resource
def _get_embeddings() -> HuggingFaceEmbeddings:
    logger.info("Loading embeddings model: %s", EMBED_MODEL)
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        # Some versions of langchain-huggingface prefer it here:
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_vector_store(collection_name: str = "documents") -> Chroma:
    """Return a persistent Chroma vector store."""
    Path(CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=str(CHROMA_DB_DIR),
    )

def add_documents(
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    collection_name: str = "documents",
) -> None:
    """
    Add documents to the Chroma collection with mandatory E5 prefixes.
    """
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas length must match texts length")

    # CRITICAL: E5 models require 'passage: ' prefix for document storage
    prefixed_texts = [f"passage: {t}" for t in texts]
    
    store = get_vector_store(collection_name)
    store.add_texts(texts=prefixed_texts, metadatas=metadatas)
    
    # In newer LangChain versions, persist() is often handled automatically
    # but we call it to be safe for your local setup
    store.persist()
    logger.info(f"Added {len(texts)} documents to {collection_name}")