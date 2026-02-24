from __future__ import annotations

from typing import Any, List

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import os

from src.config import GOOGLE_API_KEY, LLM_MODEL, RERANK_TOP_K, TOP_K
from src.embeddings.vector_store import get_vector_store
from src.retrieval.rerank import rerank


def _boost_keyword_matches(query: str, docs: List[Document], boost_factor: float = 1.5) -> List[Document]:
    """
    Boost documents that contain exact query keywords.
    Also boost high-priority content (boxes).
    Returns reordered list with keyword matches prioritized.
    """
    query_lower = query.lower()
    # Extract key terms (remove common words)
    stop_words = {"what", "is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"}
    key_terms = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]
    
    scored = []
    for doc in docs:
        content_lower = doc.page_content.lower()
        score = 1.0
        
        # Boost high-priority content (boxes)
        priority = doc.metadata.get("priority", "normal")
        if priority == "high":
            score *= boost_factor * 1.5
        
        # Boost if exact phrase matches
        if query_lower in content_lower:
            score *= boost_factor * 2
        
        # Boost if key terms match
        term_matches = sum(1 for term in key_terms if term in content_lower)
        if term_matches > 0:
            score *= 1 + (term_matches / len(key_terms)) * (boost_factor - 1)
        
        # Extra boost for box modality
        if doc.metadata.get("modality") == "box":
            score *= boost_factor * 1.3
        
        scored.append((score, doc))
    
    # Sort by score (descending)
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored]


class CustomRetriever(BaseRetriever):
    """Custom retriever that combines vector search, boosting, and reranking."""
    
    vector_store: Any
    k: int = TOP_K
    
    def __init__(self, vector_store: Any, k: int = TOP_K):
        super().__init__(vector_store=vector_store, k=k)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve, clean, and let the Reranker decide the final Source 1."""
        e5_query = f"query: {query}"
        
        # 1. Use MMR to get a diverse pool (fetch 15, keep 8 for reranking)
        initial = self.vector_store.max_marginal_relevance_search(
            e5_query, 
            k=8, 
            fetch_k=15, 
            lambda_mult=0.6 
        )
        
        # 2. Cleanup: Strip prefix BEFORE anything else
        # This ensures both the Booster and Reranker see the raw text
        for doc in initial:
            if doc.page_content.startswith("passage: "):
                doc.page_content = doc.page_content.replace("passage: ", "", 1)

        # 3. Apply Keyword Boosting
        # We boost the 8 documents to help identify potential candidates
        boosted = _boost_keyword_matches(query, initial)
        
        # 4. FINAL RERANK: This is the source of truth.
        # It takes the boosted list and re-sorts it based on deep semantic relevance.
        # The result of this is what the UI will enumerate (1, 2, 3).
        final_results = rerank(query, boosted, top_k=self.k)
        
        return final_results


def get_qa_chain(collection_name: str = "documents"):
    """
    Create and return a QA chain with retrieval.
    
    Args:
        collection_name: Name of the Chroma collection to use
        
    Returns:
        A LangChain chain that can answer questions
    """
    vector_store = get_vector_store(collection_name)
    
    # Create custom retriever with boosting and reranking
    retriever = CustomRetriever(vector_store, k=TOP_K)
    
    # Initialize LLM (swapped to Groq Llama 3)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )
    
    # Define prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant that answers questions based on the provided context.

    Each context item includes metadata (page, source, type) at the top. Use this to cite correctly.

    Instructions:
    If the answer is present, provide a concise response and cite the source ID/Page.
    If the answer is absolutely NOT in the context, only then say you don't know.

    Context:
    {context}

    Question: {input}

    Answer:"""
    )
    
    # Custom function to format documents with explicit metadata
    def format_docs_with_metadata(docs: List[Document]) -> str:
        """Format documents showing page/source metadata explicitly."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            # Try multiple metadata key names for page number
            page = (
                doc.metadata.get("page") 
                or doc.metadata.get("page_number") 
                or doc.metadata.get("page_num")
                or "?"
            )
            source = doc.metadata.get("source", "Unknown")
            modality = doc.metadata.get("modality", "text")
            
            formatted.append(
                f"[Source {i}] {source} | Page {page} | Type: {modality}\n"
                f"{doc.page_content.replace('passage: ', '', 1) if doc.page_content.startswith('passage: ') else doc.page_content}"
            )
        return "\n---\n".join(formatted)
    
    # Create a custom chain that formats documents with metadata BEFORE LLM processing
    def chain_with_metadata_formatting(inputs: dict) -> dict:
        """Custom chain: retrieve docs, format with metadata, pass to LLM."""
        # Get context docs from retriever
        context_docs = retriever.invoke(inputs.get("input", ""))
        
        # Format docs with explicit metadata headers
        formatted_context = format_docs_with_metadata(context_docs)
        
        # Create the formatted prompt
        prompt_value = prompt.invoke({
            "context": formatted_context,
            "input": inputs["input"]
        })
        
        # Get response from LLM
        response = llm.invoke(prompt_value)
        
        return {
            "input": inputs["input"],
            "context": formatted_context,
            "output": response.content,
            "answer": response.content,
            "context_docs": context_docs  
        }
    
    # Return as a Runnable that mimics the retrieval chain interface
    from langchain_core.runnables import RunnableLambda
    qa_chain = RunnableLambda(chain_with_metadata_formatting)    
    return qa_chain