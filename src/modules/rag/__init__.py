"""
RAG (Retrieval-Augmented Generation) Module for Henry Bot M2.

Provides document retrieval, embeddings, and vector storage capabilities.
"""

from src.modules.rag.retriever import RAGRetriever

__all__ = [
    "RAGRetriever",
]