"""
Streamlit application for the Document QA Chatbot.
Provides a premium chat-style interface where users can upload documents,
process them into a vector store, and ask questions answered via RAG.
"""

import streamlit as st
from document_processor import process_uploaded_files
from vector_store import create_vector_store, get_retriever
from rag_chain import create_rag_chain
import config

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📄 DocChat AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a premium glassmorphism look ──────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ───────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main header ─────────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #6C63FF 0%, #3F3D99 50%, #2D2B70 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.25);
    }
    .main-header h1 {
        color: #FFFFFF;
        margin: 0;
        font-weight: 700;
        font-size: 2rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 0.5rem 0 0 0;
        font-size: 1.05rem;
    }

    /* ── Chat bubbles ────────────────────────────────────────────────── */
    .chat-message {
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        line-height: 1.6;
        font-size: 0.95rem;
        animation: fadeIn 0.3s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background: linear-gradient(135deg, #6C63FF 0%, #5A52E0 100%);
        color: #FFFFFF;
        margin-left: 15%;
        border-bottom-right-radius: 4px;
    }
    .bot-message {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #E0E0E0;
        margin-right: 15%;
        border-bottom-left-radius: 4px;
        backdrop-filter: blur(10px);
    }

    /* ── Source chip ──────────────────────────────────────────────────── */
    .source-chip {
        display: inline-block;
        background: rgba(108, 99, 255, 0.15);
        color: #A8A3FF;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        border: 1px solid rgba(108, 99, 255, 0.25);
    }

    /* ── Sidebar styling ─────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #12141D;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ── Status badges ───────────────────────────────────────────────── */
    .status-ready {
        background: rgba(46, 213, 115, 0.12);
        color: #2ED573;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid rgba(46, 213, 115, 0.25);
    }
    .status-waiting {
        background: rgba(255, 165, 2, 0.12);
        color: #FFA502;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid rgba(255, 165, 2, 0.25);
    }

    /* ── Stat cards ──────────────────────────────────────────────────── */
    .stat-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
        margin-bottom: 0.75rem;
    }
    .stat-card .number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #6C63FF;
    }
    .stat-card .label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🗂️ Document Upload")
    st.markdown("---")

    # ── API key check ────────────────────────────────────────────────────
    if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key_here":
        st.error(
            "🔑 **Groq API key not configured!**\n\n"
            "1. Open the `.env` file in the project root\n"
            "2. Replace `your_groq_api_key_here` with your actual key\n"
            "3. Get a free key at [console.groq.com/keys](https://console.groq.com/keys)\n"
            "4. Restart the app"
        )

    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT",
    )

    process_btn = st.button("⚡ Process Documents", use_container_width=True)

    if process_btn and uploaded_files:
        try:
            with st.spinner("📖 Reading & chunking documents…"):
                chunks = process_uploaded_files(uploaded_files)

            if not chunks:
                st.warning(
                    "⚠️ No text could be extracted from the uploaded documents. "
                    "They may be empty or contain only images."
                )
            else:
                st.session_state.chunk_count = len(chunks)
                st.session_state.doc_count = len(uploaded_files)

                with st.spinner("🔢 Creating embeddings & vector store…"):
                    st.session_state.vector_store = create_vector_store(chunks)
                    retriever = get_retriever(st.session_state.vector_store)
                    st.session_state.rag_chain = create_rag_chain(retriever)

                st.success("✅ Documents processed successfully!")
        except Exception as e:
            st.error(f"❌ Error processing documents: {e}")

    elif process_btn and not uploaded_files:
        st.warning("Please upload at least one document first.")

    st.markdown("---")

    # ── Status & stats ───────────────────────────────────────────────────
    if st.session_state.vector_store is not None:
        st.markdown('<div class="status-ready">● Ready to Answer</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-waiting">○ Waiting for Documents</div>',
                    unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="number">{st.session_state.doc_count}</div>'
            f'<div class="label">Documents</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="number">{st.session_state.chunk_count}</div>'
            f'<div class="label">Chunks</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        f"**Model:** `{config.LLM_MODEL_NAME}`\n\n"
        f"**Embeddings:** `{config.EMBEDDING_MODEL_NAME}`\n\n"
        f"**Chunk size:** {config.CHUNK_SIZE} · overlap {config.CHUNK_OVERLAP}"
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="main-header">'
    "<h1>📄 DocChat AI</h1>"
    "<p>Upload documents and ask questions — answers are grounded in your files.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Chat history ─────────────────────────────────────────────────────────────
for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        st.markdown(
            f'<div class="chat-message user-message">🧑 {entry["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-message bot-message">🤖 {entry["content"]}</div>',
            unsafe_allow_html=True,
        )
        # Show source chips if available
        if entry.get("sources"):
            sources_html = "".join(
                f'<span class="source-chip">📎 {s}</span>' for s in entry["sources"]
            )
            st.markdown(sources_html, unsafe_allow_html=True)

# ── Question input ───────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your documents…")

if question:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": question})

    if st.session_state.rag_chain is None:
        answer = "⚠️ Please upload and process documents first using the sidebar."
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": []}
        )
    else:
        with st.spinner("🔍 Searching documents & generating answer…"):
            result = st.session_state.rag_chain.invoke({"query": question})
            answer = result["result"]

            # Collect unique source file names
            sources = list(
                {doc.metadata.get("source", "unknown")
                 for doc in result.get("source_documents", [])}
            )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

    st.rerun()
