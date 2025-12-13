"""
RAG (Retrieval-Augmented Generation) Module

This module provides:
- Document loading (PDF, text)
- Text chunking
- Embeddings using sentence-transformers
- FAISS vector database
- Document retrieval
- RAG agent/tool for LangGraph
"""

from .document_loader import DocumentLoader
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .retriever import RAGRetriever
from .rag_agent import RAGAgent

__all__ = [
    "DocumentLoader",
    "EmbeddingModel",
    "VectorStore",
    "RAGRetriever",
    "RAGAgent",
]

