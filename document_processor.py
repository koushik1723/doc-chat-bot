"""
Document processing module.
Handles loading PDF, DOCX, and TXT files and splitting them into chunks
for embedding.
"""

import os
import tempfile
import logging
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.schema import Document

import config

logger = logging.getLogger(__name__)


def load_document(uploaded_file) -> List[Document]:
    """
    Load a single uploaded file and return a list of LangChain Documents.

    Supported formats: .pdf, .docx, .txt
    The file is written to a temp directory so that LangChain loaders
    (which expect filesystem paths) can read it.
    """
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    # Write the uploaded bytes to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif file_extension == ".txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        documents = loader.load()

        # Tag every document with the original filename for source tracking
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name

        return documents

    finally:
        # Clean up temp file (ignore errors on Windows where file may be locked)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of Documents into smaller, overlapping chunks suitable
    for embedding. Strips whitespace from document text first and filters
    out any empty chunks.
    """
    # Strip leading/trailing whitespace from each document's text
    for doc in documents:
        doc.page_content = doc.page_content.strip()

    # Remove docs that are completely empty after stripping
    documents = [doc for doc in documents if doc.page_content]

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Filter out any empty/whitespace-only chunks produced by splitting
    chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

    return chunks


def process_uploaded_files(uploaded_files) -> List[Document]:
    """
    Convenience wrapper: loads *and* chunks a list of uploaded files.
    Returns all chunks ready for embedding.
    Continues processing remaining files if one file fails to load.
    """
    all_chunks: List[Document] = []
    errors: List[str] = []

    for uploaded_file in uploaded_files:
        try:
            docs = load_document(uploaded_file)
            chunks = chunk_documents(docs)
            all_chunks.extend(chunks)
        except Exception as e:
            error_msg = f"Failed to process '{uploaded_file.name}': {e}"
            logger.warning(error_msg)
            errors.append(error_msg)

    if errors and not all_chunks:
        raise RuntimeError(
            "All documents failed to process:\n" + "\n".join(errors)
        )

    return all_chunks
