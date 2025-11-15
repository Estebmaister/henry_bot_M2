"""
RAG Retriever for Henry Bot M2.

Provides semantic document retrieval using FAISS vector database
and sentence transformer embeddings for context-aware responses.
"""

import json
import os
import faiss
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import logging

from src.core.exceptions import RAGError
from src.core.config import settings
from src.modules.rag.chunking import DocumentChunk
from src.modules.logging.logger import log_error

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    RAG retriever that provides semantic document search capabilities.

    Uses FAISS for vector similarity search and sentence transformers
    for embedding generation to find relevant document context.
    """

    def __init__(self):
        """Initialize RAG retriever with vector database and embedding service."""
        self.vector_store_path = settings.vector_store_path
        self.chunk_store_path = settings.chunk_store_path
        self.is_available = False

        # Initialize these lazily to avoid initialization issues
        self._embedding_service = None
        self._vector_store = None
        self._chunk_store = {}
        self._index_to_id: Dict[int, str] = {}

        try:
            # Try to initialize the RAG system
            self._initialize_rag_system()
        except Exception as e:
            logger.warning(f"RAG system initialization failed: {e}")
            self.is_available = False

    def _initialize_rag_system(self):
        """Initialize the RAG system components."""
        try:
            # Initialize embedding service
            from src.modules.rag.processor import create_embedding_service
            self._embedding_service = create_embedding_service()

            # Load vector store if available
            self._load_vector_store()

            # Load chunk store if available
            self._load_chunk_store()

            # Check if system is ready
            self.is_ready()

        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            self.is_available = False

    def _load_vector_store(self):
        """Load FAISS vector store from disk."""
        if not os.path.exists(self.vector_store_path):
            logger.info(f"Vector store not found at {self.vector_store_path}")
            return
        try:
            self._vector_store = faiss.read_index(self.vector_store_path)
            # Cargar el mapeo índice→ID si existe
            mapping_path = self.vector_store_path.replace(
                '.faiss', '_index_mapping.json')
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                self._index_to_id = {
                    int(k): v for k, v in mapping_data.items()}
            else:
                self._index_to_id = {}
            logger.info(
                f"Loaded vector store with {self._vector_store.ntotal} vectors")

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            self._vector_store = None
            self._index_to_id = {}

    def _load_chunk_store(self):
        """Load chunk metadata from JSON store."""
        if not os.path.exists(self.chunk_store_path):
            logger.info(f"Chunk store not found at {self.chunk_store_path}")
            return

        try:
            with open(self.chunk_store_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)

            # Handle both dictionary and list formats
            self._chunk_store = {}

            if isinstance(chunk_data, dict):
                # If it's already a dictionary with chunk_ids as keys, use it directly
                self._chunk_store = chunk_data
                logger.info(
                    f"Loaded {len(self._chunk_store)} chunks from store (dict format)")
            elif isinstance(chunk_data, list):
                # If it's a list, rebuild with chunk_id as key
                for chunk_info in chunk_data:
                    if isinstance(chunk_info, dict):
                        chunk_id = chunk_info.get('chunk_id')
                        if chunk_id:
                            self._chunk_store[chunk_id] = chunk_info
                logger.info(
                    f"Loaded {len(self._chunk_store)} chunks from store (list format)")
            else:
                logger.error(
                    f"Unexpected chunk store format: {type(chunk_data)}")
                self._chunk_store = {}

        except Exception as e:
            logger.error(f"Failed to load chunk store: {e}", exc_info=True)
            self._chunk_store = {}

    async def retrieve_context(self, query: str, top_k: int = 3) -> Tuple[Optional[str], Optional[float]]:
        """
        Retrieve relevant document context using semantic search.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            Tuple of (context_text, similarity_score)
        """
        if not self.is_ready():
            logger.info("RAG system not ready, returning no context")
            return None, None

        try:
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(query)
            if query_embedding is None:
                return None, None

            # Search for similar chunks
            similarities, indices = await self._search_similar_chunks(query_embedding, top_k)
            if similarities is None or indices is None:
                return None, None

            # Retrieve and format context
            context_text, avg_score = await self._format_retrieved_context(similarities, indices)

            return context_text, avg_score

        except Exception as e:
            log_error(
                error_type="RAGRetrievalError",
                error_message=str(e),
                user_question=query
            )
            return None, None

    async def _generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for the query."""
        try:
            if not self._embedding_service:
                logger.warning("Embedding service not available")
                return None

            # Use the embedding service's async embed_text method
            result = await self._embedding_service.embed_text(query)

            if result and result.embeddings is not None and len(result.embeddings) > 0:
                # result.embeddings is already a numpy array with shape (1, embedding_dim)
                # Ensure it's float32 for FAISS
                embedding = result.embeddings.astype('float32')

                # Reshape to (1, dim) if needed
                if len(embedding.shape) == 1:
                    embedding = embedding.reshape(1, -1)

                logger.debug(
                    f"Generated query embedding with shape: {embedding.shape}")
                return embedding

            logger.warning("Embedding generation returned no results")
            return None

        except Exception as e:
            logger.error(
                f"Failed to generate query embedding: {e}", exc_info=True)
            return None

    async def _search_similar_chunks(self, query_embedding: np.ndarray, top_k: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Search for similar chunks in the vector store."""
        try:
            if self._vector_store is None:
                return None, None

            # Ensure we don't request more results than available
            available_vectors = self._vector_store.ntotal
            actual_k = min(top_k, available_vectors)

            if actual_k == 0:
                return None, None

            # Search vector store
            similarities, indices = self._vector_store.search(
                query_embedding, actual_k)

            return similarities, indices

        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            return None, None

    async def _format_retrieved_context(self, similarities: np.ndarray, indices: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """Format retrieved chunks into context text."""
        try:
            context_parts = []
            total_score = 0.0

            for similarity, idx in zip(similarities[0], indices[0]):
                chunk_id = self._index_to_id.get(int(idx), str(int(idx)))

                # Get chunk info from store
                chunk_info = self._chunk_store.get(chunk_id)
                if chunk_info:
                    chunk_content = chunk_info.get('content', '')
                    metadata = chunk_info.get('metadata', {})
                    document_source = metadata.get(
                        'document_source', f'document_{chunk_id}')

                    # Format chunk with source information
                    formatted_chunk = f"[Source: {document_source}]\n{chunk_content}"
                    context_parts.append(formatted_chunk)
                    total_score += float(similarity)
                else:
                    logger.warning(
                        f"Chunk {chunk_id} not found in chunk store")

            if not context_parts:
                return None, None

            # Join chunks with separators
            context_text = "\n\n---\n\n".join(context_parts)
            avg_score = total_score / len(context_parts)

            return context_text, avg_score

        except Exception as e:
            logger.error(f"Failed to format retrieved context: {e}")
            return None, None

    def is_ready(self) -> bool:
        """
        Check if RAG system is ready for retrieval.

        Returns:
            True if vector store and chunk store are loaded
        """
        was_available = self.is_available

        # Check if all components are available
        embedding_ready = self._embedding_service is not None
        vector_ready = self._vector_store is not None and (
            self._vector_store.ntotal > 0)
        chunk_ready = len(self._chunk_store) > 0

        self.is_available = embedding_ready and vector_ready and chunk_ready

        # Log availability changes
        if was_available != self.is_available:
            if self.is_available:
                logger.info(
                    f"RAG system is ready with {self._vector_store.ntotal if self._vector_store else 0} vectors and {len(self._chunk_store)} chunks")
            else:
                logger.info("RAG system is not ready")

        return self.is_available

    async def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return {
            "is_available": self.is_ready(),
            "vector_count": self._vector_store.ntotal if self._vector_store else 0,
            "chunk_count": len(self._chunk_store),
            "vector_store_path": self.vector_store_path,
            "chunk_store_path": self.chunk_store_path
        }
