from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Setup project root
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.qa.chain import get_qa_chain

@st.cache_resource
def load_chain():
    """Load the QA chain once and keep it in memory."""
    return get_qa_chain()

def render_sources(context_docs: List[Any]):
    """Optimized rendering for source citations."""
    if not context_docs:
        st.info("No source documents available.")
        return

    for i, doc in enumerate(context_docs, start=1):
        meta = doc.metadata
        # Check for SQuAD ID or PDF Page
        source_id = meta.get("id") or meta.get("page") or meta.get("page_number") or "?"
        source_file = meta.get("source", "Policy Document")
        modality = meta.get("modality", "text")
        
        # Strip the 'passage: ' prefix for UI display
        clean_content = doc.page_content.replace("passage: ", "", 1)
        
        with st.expander(f"📚 Source {i}: {source_file} (ID/Page: {source_id})"):
            st.markdown(f"**Modality:** {modality.capitalize()}")
            st.markdown(f"**Snippet:**\n> {clean_content[:1000]}")
            if "image_path" in meta and os.path.exists(meta["image_path"]):
                st.image(meta["image_path"], caption="Reference Image", width=300)

def main():
    st.set_page_config(layout="wide", page_title="Algebrik AI Assistant")

    st.markdown("<h1 style='text-align: center;'>Knowledge Assistant</h1>", unsafe_allow_html=True)

    # Initialize session state for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar: Reset and Controls
    with st.sidebar:
        st.header("Controls")
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.info("Built for Algebrik & InCred Internal Policy Q&A")

    # Load Chain
    qa_chain = load_chain()

    # Display Chat History using modern bubble components
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "docs" in msg:
                render_sources(msg["docs"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about the policy..."):
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                try:
                    # Invoke the chain we optimized earlier
                    result = qa_chain.invoke({"input": prompt})
                    answer = result.get("answer", "I couldn't generate an answer.")
                    docs = result.get("context_docs", [])
                    
                    st.markdown(answer)
                    render_sources(docs)
                    
                    # Store in history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "docs": docs
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()