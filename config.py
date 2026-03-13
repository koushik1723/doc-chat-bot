"""
Configuration module for the Document QA Chatbot.
Loads environment variables and exposes all settings as constants.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ── Groq LLM ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")

# Validate API key early
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    import warnings
    warnings.warn(
        "⚠️  GROQ_API_KEY is not set! "
        "Copy .env.example to .env and add your Groq API key. "
        "Get one free at https://console.groq.com/keys"
    )
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# ── Text Splitting ───────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Retriever ────────────────────────────────────────────────────────────────
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))
