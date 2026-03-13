"""
RAG chain module.
Builds a LangChain Retrieval-Augmented Generation chain that uses
Groq-hosted Llama 3 (or any other Groq-supported model) to answer
questions grounded in document context.
"""

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import config


# ── Prompt Template ──────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions
based ONLY on the provided context. If the context does not contain enough
information to answer the question, say "I don't have enough information in
the uploaded documents to answer this question."

Do NOT make up information. Always base your answer strictly on the context.

Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def get_llm() -> ChatGroq:
    """
    Instantiate the Groq-hosted LLM.
    """
    return ChatGroq(
        groq_api_key=config.GROQ_API_KEY,
        model_name=config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )


def create_rag_chain(retriever) -> RetrievalQA:
    """
    Build a RetrievalQA chain that:
      1. Retrieves relevant chunks from the vector store.
      2. Passes them as context to the Groq LLM.
      3. Generates a grounded answer.
    """
    llm = get_llm()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # concatenate all retrieved docs into one prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    return chain
