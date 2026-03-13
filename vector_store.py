"""
Vector store module.
Creates HuggingFace embeddings, builds a FAISS index from document chunks,
and exposes a retriever for similarity search.
"""

import os

# ── Prevent Keras 3 / TensorFlow crash ───────────────────────────────────────
# sentence-transformers imports transformers, which tries to import TF utils.
# If Keras 3 is installed (comes with TensorFlow ≥ 2.16), this causes:
#   RuntimeError: "Your currently installed version of Keras is Keras 3,
#   but this is not yet supported in Transformers."
# Setting this env var tells the transformers library to skip TF entirely,
# since we only use the PyTorch backend via sentence-transformers.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # suppress TF warnings

from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

import config


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a HuggingFace embedding model instance.
    Uses `all-MiniLM-L6-v2` by default — a lightweight, fast model that
    runs entirely on CPU with no API key required.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(chunks: List[Document]) -> FAISS:
    """
    Build a FAISS vector store from a list of document chunks.
    Raises ValueError if the chunk list is empty.
    """
    if not chunks:
        raise ValueError(
            "No text chunks to embed. The uploaded documents may be empty "
            "or contain only images/non-text content."
        )

    # Filter out any empty or whitespace-only chunks before embedding
    valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    if not valid_chunks:
        raise ValueError(
            "All chunks were empty after filtering whitespace. "
            "The uploaded documents may not contain extractable text."
        )

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(valid_chunks, embeddings)
    return vector_store


def get_retriever(vector_store: FAISS, k: int | None = None):
    """
    Return a similarity-search retriever from the FAISS vector store.
    """
    k = k or config.RETRIEVER_K
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
