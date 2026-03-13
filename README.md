# 📄 DocChat AI — Document Question-Answering Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload documents and ask questions answered **only** from the document content. Built with **LangChain**, **Groq** (Llama 3), **FAISS**, and **Streamlit**.

---

## System Architecture

```
┌─────────────────── Streamlit UI ───────────────────┐
│  Upload Documents  │  Ask Questions  │  View Answers │
└────────┬───────────┴───────┬────────┴───────────────┘
         │                   │
    ┌────▼────┐         ┌────▼────┐
    │ Document │         │Retriever│
    │ Loader   │         │  (FAISS)│
    └────┬────┘         └────┬────┘
         │                   │
    ┌────▼────┐         ┌────▼─────────┐
    │  Text   │         │ Groq LLM     │
    │Splitter │         │ (Llama 3 70B)│
    └────┬────┘         └──────────────┘
         │
    ┌────▼──────────────┐
    │ HuggingFace       │
    │ Embeddings        │
    │ (all-MiniLM-L6-v2)│
    └────┬──────────────┘
         │
    ┌────▼────┐
    │  FAISS  │
    │  Index  │
    └─────────┘
```

### How the RAG Pipeline Works

1. **Upload** — User uploads PDF, DOCX, or TXT files via the sidebar.
2. **Load** — LangChain document loaders extract raw text from each file.
3. **Chunk** — `RecursiveCharacterTextSplitter` breaks the text into ~1 000-char overlapping chunks.
4. **Embed** — Each chunk is embedded locally with HuggingFace `all-MiniLM-L6-v2`.
5. **Index** — Embeddings are stored in an in-memory FAISS vector database.
6. **Retrieve** — When the user asks a question, the top-k most similar chunks are retrieved.
7. **Generate** — The chunks + question are sent to Groq's Llama 3 70B, which produces a grounded answer.

---

## Folder Structure

```
chat_bot/
├── .streamlit/
│   └── config.toml          # Dark theme
├── .env.example              # Env-var template
├── .env                      # Your actual keys (gitignored)
├── requirements.txt
├── README.md
├── config.py                 # Settings & constants
├── document_processor.py     # Load & chunk documents
├── vector_store.py           # FAISS embeddings & retrieval
├── rag_chain.py              # LangChain + Groq RAG chain
└── app.py                    # Streamlit UI
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- A free **Groq API key** → [console.groq.com](https://console.groq.com)

### 2. Install Dependencies

```bash
cd chat_bot
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
copy .env.example .env
```

Open `.env` and replace `your_groq_api_key_here` with your actual key:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**.

### 5. Use It

1. **Upload documents** via the sidebar (PDF, DOCX, or TXT).
2. Click **⚡ Process Documents**.
3. Type a question in the chat input.
4. Read the AI-generated answer grounded in your documents.

---

## Configuration Options

All settings can be overridden via environment variables in `.env`:

| Variable              | Default            | Description                        |
|-----------------------|--------------------|------------------------------------|
| `GROQ_API_KEY`        | —                  | Your Groq API key (required)       |
| `LLM_MODEL_NAME`      | `llama3-70b-8192`  | Groq model identifier              |
| `LLM_TEMPERATURE`     | `0.2`              | LLM temperature                    |
| `LLM_MAX_TOKENS`      | `1024`             | Max response tokens                |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | HuggingFace embedding model       |
| `CHUNK_SIZE`           | `1000`             | Chunk size in characters           |
| `CHUNK_OVERLAP`        | `200`              | Overlap between chunks             |
| `RETRIEVER_K`          | `4`                | Number of chunks to retrieve       |

---

## Tech Stack

| Component       | Technology                                  |
|-----------------|---------------------------------------------|
| Language        | Python 3.10+                                |
| LLM             | Groq API → Llama 3 70B                      |
| Framework       | LangChain                                   |
| Embeddings      | HuggingFace `all-MiniLM-L6-v2` (local)      |
| Vector DB       | FAISS (in-memory)                            |
| UI              | Streamlit                                   |
| Doc Loaders     | PyPDF, docx2txt, TextLoader                 |
