from __future__ import annotations
import os
import sys
import shutil
from uuid import uuid4
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

# Setup project root
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import LLM_MODEL, SESSION_CHROMA_ROOT, USER_UPLOADS_DIR, ensure_dirs
from src.retrieval.tracker import PerformanceTracker

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

    ensure_dirs()

    # Initialize session state for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())

    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready = False

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    session_id: str = st.session_state.session_id
    upload_dir = Path(USER_UPLOADS_DIR) / session_id
    session_db_dir = Path(SESSION_CHROMA_ROOT) / session_id
    
    # Sidebar: Reset and Controls
    with st.sidebar:
        st.header("Controls")
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

        if st.button("Reset Session (delete uploads + DB)"):
            try:
                st.session_state.messages = []
                st.session_state.rag_ready = False
                st.session_state.qa_chain = None

                if upload_dir.exists():
                    shutil.rmtree(upload_dir, ignore_errors=True)
                if session_db_dir.exists():
                    shutil.rmtree(session_db_dir, ignore_errors=True)

                st.session_state.session_id = str(uuid4())
            finally:
                st.rerun()
        st.divider()

        st.subheader("📄 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("Process PDFs", type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF.")
            else:
                try:
                    st.session_state.rag_ready = False
                    st.session_state.qa_chain = None

                    upload_dir.mkdir(parents=True, exist_ok=True)
                    if session_db_dir.exists():
                        shutil.rmtree(session_db_dir, ignore_errors=True)
                    session_db_dir.mkdir(parents=True, exist_ok=True)

                    saved_paths: List[Path] = []
                    for uf in uploaded_files:
                        dest = upload_dir / Path(uf.name).name
                        dest.write_bytes(uf.getvalue())
                        saved_paths.append(dest)

                    from src.ingest.pipeline import run_pipeline_for_files
                    from src.qa.chain import get_qa_chain

                    with st.status("Processing PDFs and building your vector database...", expanded=True) as status:
                        status.update(label="Ingesting PDFs...", state="running")
                        run_pipeline_for_files(
                            saved_paths,
                            persist_directory=session_db_dir,
                            session_id=session_id,
                        )
                        status.update(label="Initializing QA chain...", state="running")
                        st.session_state.qa_chain = get_qa_chain(persist_directory=session_db_dir)
                        st.session_state.rag_ready = True
                        status.update(label="Ready", state="complete")

                    st.success("Processing complete. You can now ask questions.")
                except Exception as e:
                    st.session_state.rag_ready = False
                    st.session_state.qa_chain = None
                    st.error(f"Processing failed: {str(e)}")

        with st.expander("🧾 SQL Index (this session)", expanded=False):
            try:
                from src.db.sql_index import (
                    count_chunks_by_modality,
                    get_documents_for_session,
                    top_pages_by_chunk_count,
                )

                if st.button("List uploaded PDFs"):
                    docs_rows = get_documents_for_session(session_id)
                    if docs_rows:
                        st.dataframe(docs_rows, use_container_width=True)
                    else:
                        st.info("No indexed PDFs yet for this session.")

                if st.button("Count chunks by modality"):
                    modality_rows = count_chunks_by_modality(session_id)
                    if modality_rows:
                        st.dataframe(modality_rows, use_container_width=True)
                    else:
                        st.info("No indexed chunks yet for this session.")

                if st.button("Top pages by chunk count"):
                    top_rows = top_pages_by_chunk_count(session_id, limit=10)
                    if top_rows:
                        st.dataframe(top_rows, use_container_width=True)
                    else:
                        st.info("No indexed chunks yet for this session.")
            except Exception as exc:
                st.caption(f"SQL index unavailable: {exc}")

        st.info("Built for InCred Internal Policy Q&A")
        st.subheader("📊 Performance Stats")
        metrics_container = st.container()
        st.caption(f"Session ID: `{session_id}`")

    # Display Chat History using bubble components
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "docs" in msg:
                render_sources(msg["docs"])

    if not st.session_state.rag_ready or st.session_state.qa_chain is None:
        st.info("Upload PDFs in the sidebar and click **Process PDFs** to start chatting.")
        return

    qa_chain = st.session_state.qa_chain

    # Chat Input (enabled only when ready)
    if prompt := st.chat_input("Ask a question about your uploaded PDFs..."):
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        

        # 2. Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                try:
                    tracker = PerformanceTracker(LLM_MODEL)
                    # Invoke the chain we optimized earlier
                    result = qa_chain.invoke({"input": prompt}, config={"callbacks": [tracker]})
                    answer = result.get("answer", "I couldn't generate an answer.")
                    docs = result.get("context_docs", [])
                    
                    st.markdown(answer)
                    render_sources(docs)
                    
                    if tracker.metrics and tracker.metrics.get("latency", 0) > 0:
                        with metrics_container:
                            st.write(f"**Model:** `{tracker.metrics['model']}`")
                            st.write(f"**Latency:** {tracker.metrics['latency']}s")
                            st.write(f"**Total Tokens:** {tracker.metrics['tokens']}")

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