from __future__ import annotations
import logging
from typing import List

import streamlit as st
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import CROSS_ENCODER_MODEL

logger = logging.getLogger(__name__)

@st.cache_resource # CRITICAL: Keeps the model in RAM across all app reruns
def _get_reranker_model() -> CrossEncoder:
    """Initialize and cache the Cross-Encoder model."""
    logger.info("Loading Cross-Encoder model: %s", CROSS_ENCODER_MODEL)
    # This model jointly encodes query and doc for high-precision scoring
    return CrossEncoder(CROSS_ENCODER_MODEL)

def rerank(query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
    """Rerank a list of Documents using a cached cross-encoder."""
    if not docs:
        return []

    # 1. Use the cached model instead of re-initializing
    model = _get_reranker_model()
    
    # 2. Prepare pairs for joint encoding
    # Cross-encoders are 'Bi-Directional', looking at query and doc tokens together
    pairs = [[query, doc.page_content.strip()] for doc in docs]
    
    # 3. Predict relevance scores
    scores = model.predict(pairs)
    
    # 4. Zip, sort by score (descending), and return top_k
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Reranked {len(docs)} documents. Top score: {scored[0][1]:.4f}")
    
    return [doc for doc, _ in scored[:top_k]]