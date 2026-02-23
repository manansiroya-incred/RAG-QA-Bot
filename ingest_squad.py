# ingest_squad.py
import logging
from datasets import load_dataset
from src.embeddings.vector_store import add_documents  # Removed deleted import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_squad_data(limit: int = 1000):
    """Downloads SQuAD and ingests it into ChromaDB."""
    logger.info(f"Loading SQuAD dataset (limit: {limit})...")
    dataset = load_dataset("rajpurkar/squad", split="train")
    
    docs = []
    metas = []
    
    # Contexts in SQuAD are often repeated; we grab a unique set
    unique_contexts = list(set(dataset["context"][:limit*2]))[:limit]

    for i, context in enumerate(unique_contexts):
        # We manually prefix here because we removed the query_instruction 
        # that was causing the ValidationError earlier
        docs.append(f"passage: {context}")
        metas.append({
            "source": "squad_dataset",
            "id": i,
            "modality": "text",
            "page": "Wikipedia" 
        })

    # This function now calls store.persist() internally as per our latest update
    add_documents(docs, metas)
    logger.info("SQuAD ingestion complete! The database has been persisted.")

if __name__ == "__main__":
    ingest_squad_data(100)