"""Retrieval-Augmented Generation (RAG) helpers for AI-WebScout.

Uses a pretrained Transformer sentence-embedding model through LangChain,
then retrieves the most relevant chunks from crawled pages. Gemini remains
responsible for generating the final answer in extractor.py.
"""
from __future__ import annotations

from typing import Any

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def retrieve_relevant_context(
    pages: list[dict[str, Any]],
    question: str,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    top_k: int = 5,
) -> str:
    """Chunk crawled pages and return source-labelled passages relevant to question.

    LangChain's text splitter creates chunks, Hugging Face sentence-transformers
    creates Transformer embeddings, and FAISS performs vector similarity search.
    Imports are lazy so the rest of the crawler can still run if optional RAG
    dependencies have not been installed.
    """
    if not pages:
        return ""
    if not question or not question.strip():
        return ""
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise RuntimeError(
            "RAG dependencies are missing. Install requirements.txt, including "
            "langchain-community, langchain-text-splitters, sentence-transformers, and faiss-cpu."
        ) from exc

    documents = []
    for page in pages:
        text = (page.get("text") or "").strip()
        if not text:
            continue
        metadata = {
            "title": str(page.get("title") or "Untitled page"),
            "source": str(page.get("url") or ""),
        }
        documents.append(Document(page_content=text, metadata=metadata))

    if not documents:
        return ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        return ""

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": min(max(int(top_k), 1), len(chunks))},
    )
    try:
        relevant_docs = retriever.invoke(question.strip())
    except AttributeError:  # Compatibility with older LangChain releases
        relevant_docs = retriever.get_relevant_documents(question.strip())

    formatted = []
    for index, doc in enumerate(relevant_docs, start=1):
        title = doc.metadata.get("title", "Untitled page")
        source = doc.metadata.get("source", "")
        formatted.append(
            f"[Retrieved passage {index}]\nTitle: {title}\nSource URL: {source}\n"
            f"Passage:\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(formatted)
