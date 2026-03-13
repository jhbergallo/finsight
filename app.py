import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from src.ingestion import DocumentProcessor
from src.vector_store import VectorStore
from src.agent import FinancialAgent
from src.config import settings

load_dotenv()

st.set_page_config(
    page_title="FinSight — Financial Document Intelligence",
    page_icon="📊",
    layout="wide",
)

# --- Session State ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Your key is never stored — it lives only in this session.",
    )

    st.divider()
    st.subheader("📁 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload financial documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Supports earnings reports, 10-Ks, analyst notes, internal memos.",
    )

    chunk_size = st.slider("Chunk size (tokens)", 256, 1024, 512, step=64)
    chunk_overlap = st.slider("Chunk overlap", 0, 200, 50, step=10)

    if st.button("🔄 Ingest Documents", disabled=not uploaded_files or not api_key):
        with st.spinner("Processing documents..."):
            processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            vs = VectorStore(api_key=api_key)

            all_chunks = []
            doc_names = []

            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name

                chunks = processor.process(tmp_path, source_name=file.name)
                all_chunks.extend(chunks)
                doc_names.append(file.name)
                os.unlink(tmp_path)

            vs.add_documents(all_chunks)
            agent = FinancialAgent(vector_store=vs, api_key=api_key)

            st.session_state.vector_store = vs
            st.session_state.agent = agent
            st.session_state.ingested_docs = doc_names
            st.success(f"✅ Ingested {len(all_chunks)} chunks from {len(doc_names)} document(s).")

    if st.session_state.ingested_docs:
        st.divider()
        st.subheader("📄 Loaded Documents")
        for doc in st.session_state.ingested_docs:
            st.markdown(f"- `{doc}`")

    st.divider()
    if st.button("🗑️ Clear Session"):
        st.session_state.vector_store = None
        st.session_state.agent = None
        st.session_state.chat_history = []
        st.session_state.ingested_docs = []
        st.rerun()

# --- Main Interface ---
st.title("📊 FinSight")
st.caption("RAG-powered financial document intelligence with multi-step reasoning")

if not api_key:
    st.warning("Add your OpenAI API key in the sidebar to get started.")
    st.stop()

if not st.session_state.agent:
    st.info("Upload financial documents in the sidebar and click **Ingest Documents** to begin.")

    with st.expander("💡 Example questions you can ask"):
        st.markdown("""
        - *What was the revenue growth YoY mentioned in the earnings report?*
        - *Summarize the main risk factors from this 10-K.*
        - *What did management say about Q4 guidance?*
        - *Compare the gross margins across the documents I uploaded.*
        - *What are the key operational highlights from this period?*
        """)
    st.stop()

# --- Chat Interface ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['source']}** — chunk {src['chunk_id']}")
                    st.caption(src["content"][:300] + "...")

query = st.chat_input("Ask anything about your documents...")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.agent.run(
                query=query,
                chat_history=st.session_state.chat_history[:-1],
            )

        st.markdown(result["answer"])

        if result.get("sources"):
            with st.expander("📎 Sources"):
                for src in result["sources"]:
                    st.markdown(f"**{src['source']}** — chunk {src['chunk_id']}")
                    st.caption(src["content"][:300] + "...")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
    })
