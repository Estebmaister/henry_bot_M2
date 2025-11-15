"""
Storage interfaces and implementations for RAG system.

Provides vector storage for embeddings and document metadata storage
with support for different storage backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import faiss


@dataclass
class SearchResult:
    """Result of vector similarity search."""
    chunks: List[Dict[str, Any]]
    scores: List[float]
    metadata: List[Dict[str, Any]]
    query_embedding: Optional[np.ndarray] = None
    search_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunks": self.chunks,
            "scores": self.scores,
            "metadata": self.metadata,
            "search_time_ms": self.search_time_ms,
            "result_count": len(self.chunks),
        }


class VectorStore(ABC):
    """Abstract base class for vector storage implementations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""
        pass

    @abstractmethod
    async def add_texts(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> List[str]:
        """Add texts with embeddings to the store."""
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResult:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        pass


class DocumentStore(ABC):
    """Abstract base class for document metadata storage."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the document store."""
        pass

    @abstractmethod
    async def store_document(
        self,
        document_id: str,
        source: str,
        document_type: str,
        chunk_count: int,
        metadata: Dict[str, Any]
    ) -> None:
        """Store document metadata."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get document store statistics."""
        pass

    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        pass

    @abstractmethod
    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List documents with pagination and filtering."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete document by ID."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the document store is healthy."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation."""

    def __init__(self, dimension: int, index_path: Optional[str] = None):
        """Initialize FAISS vector store."""
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.texts = []
        self.metadatas = []
        self.ids = []
        self.index_to_id = {}  # Maps FAISS index (int) to chunk_id (str)

    async def initialize(self) -> None:
        """Initialize FAISS index."""
        # Create index
        # Inner product for cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)

        # Load existing index if path provided
        if self.index_path and Path(self.index_path).exists():
            await self._load_index()

    async def add_texts(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> List[str]:
        """Add texts with embeddings to FAISS index."""
        if self.index is None:
            await self.initialize()

        # Normalize embeddings for cosine similarity
        embeddings = embeddings / \
            np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Get starting index before adding
        start_idx = self.index.ntotal

        # Add to index
        self.index.add(embeddings.astype(np.float32))

        # Store metadata
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

        for i, chunk_id in enumerate(ids):
            faiss_idx = start_idx + i
            self.index_to_id[faiss_idx] = chunk_id

        # Save index if path provided
        if self.index_path:
            await self._save_index()

        return ids

    async def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResult:
        """Search FAISS index for similar vectors."""
        if self.index is None or self.index.ntotal == 0:
            return SearchResult(chunks=[], scores=[], metadata=[])

        import time
        start_time = time.time()

        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(
            query_embedding, min(k, self.index.ntotal))

        # Process results
        chunks = []
        result_scores = []
        result_metadata = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            if score_threshold is not None and score < score_threshold:
                continue

            # Apply filters if provided
            if filter_dict and not self._matches_filter(self.metadatas[idx], filter_dict):
                continue

            chunks.append(self.texts[idx])
            result_scores.append(float(score))
            result_metadata.append(self.metadatas[idx])

        search_time = (time.time() - start_time) * 1000

        return SearchResult(
            chunks=chunks,
            scores=result_scores,
            metadata=result_metadata,
            query_embedding=query_embedding.flatten(),
            search_time_ms=search_time
        )

    async def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs (not implemented for basic FAISS)."""
        # Basic FAISS doesn't support deletion efficiently
        # This would require rebuilding the index
        return False

    async def health_check(self) -> bool:
        """Check if FAISS index is accessible."""
        try:
            return self.index is not None
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics."""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_embeddings": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": "IndexFlatIP",
            "stored_texts": len(self.texts),
            "stored_metadatas": len(self.metadatas),
            "index_mappings": len(self.index_to_id),
        }

    async def _save_index(self) -> None:
        """Save FAISS index to disk."""
        if not self.index_path:
            return

        import faiss
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

        # Save index
        faiss.write_index(self.index, self.index_path)

        # Save metadata
        metadata_path = self.index_path.replace('.faiss', '_metadata.json')
        metadata = {
            "texts": self.texts,
            "metadatas": [m.to_dict() if hasattr(m, 'to_dict') else m for m in self.metadatas],
            "ids": self.ids,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        index_mapping_path = self.index_path.replace(
            '.faiss', '_index_mapping.json')
        # Convert int keys to strings for JSON serialization
        mapping_for_json = {str(k): v for k, v in self.index_to_id.items()}
        with open(index_mapping_path, 'w') as f:
            json.dump(mapping_for_json, f, indent=2)

    async def _load_index(self) -> None:
        """Load FAISS index from disk."""
        if not self.index_path:
            return

        import faiss

        # Load index
        self.index = faiss.read_index(self.index_path)

        # Load metadata
        metadata_path = self.index_path.replace('.faiss', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.texts = metadata.get("texts", [])
            self.metadatas = metadata.get("metadatas", [])
            self.ids = metadata.get("ids", [])

        index_mapping_path = self.index_path.replace(
            '.faiss', '_index_mapping.json')
        if Path(index_mapping_path).exists():
            with open(index_mapping_path, 'r') as f:
                mapping_data = json.load(f)
            # Convert string keys back to integers
            self.index_to_id = {int(k): v for k, v in mapping_data.items()}
        else:
            # Fallback: create mapping from current order
            self.index_to_id = {i: chunk_id for i,
                                chunk_id in enumerate(self.ids)}

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True


class JSONDocumentStore(DocumentStore):
    """JSON file-based document store implementation."""

    def __init__(self, storage_path: str = "./data/documents.json"):
        """Initialize JSON document store."""
        self.storage_path = Path(storage_path)
        self.documents = {}
        self._loaded = False

    async def initialize(self) -> None:
        """Load documents from JSON file."""
        await self._load_documents()

    async def store_document(
        self,
        document_id: str,
        source: str,
        document_type: str,
        chunk_count: int,
        metadata: Dict[str, Any]
    ) -> None:
        """Store document metadata in JSON."""
        await self._load_documents()

        self.documents[document_id] = {
            "document_id": document_id,
            "source": source,
            "document_type": document_type,
            "chunk_count": chunk_count,
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        await self._save_documents()

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        await self._load_documents()
        return self.documents.get(document_id)

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List documents with pagination and filtering."""
        await self._load_documents()

        documents = list(self.documents.values())

        # Apply filters
        if filters:
            documents = [
                doc for doc in documents
                if self._matches_document_filter(doc, filters)
            ]

        # Sort by created_at descending
        documents.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # Apply pagination
        return documents[offset:offset + limit]

    async def delete_document(self, document_id: str) -> bool:
        """Delete document by ID."""
        await self._load_documents()

        if document_id in self.documents:
            del self.documents[document_id]
            await self._save_documents()
            return True

        return False

    async def health_check(self) -> bool:
        """Check if document store is accessible."""
        try:
            await self._load_documents()
            return True
        except Exception:
            return False

    async def _load_documents(self) -> None:
        """Load documents from JSON file."""
        if self._loaded:
            return

        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    self.documents = json.load(f)
            else:
                self.documents = {}
        except Exception:
            self.documents = {}

        self._loaded = True

    async def _save_documents(self) -> None:
        """Save documents to JSON file."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self.documents, f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Get document store statistics."""
        total_documents = len(self.documents)
        processed_documents = sum(
            1 for doc in self.documents.values() if doc.get("chunk_count", 0) > 0)
        total_chunks = sum(doc.get("chunk_count", 0)
                           for doc in self.documents.values())

        # Document type distribution
        type_counts = {}
        for doc in self.documents.values():
            doc_type = doc.get("document_type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        return {
            "total_documents": total_documents,
            "processed_documents": processed_documents,
            "total_chunks": total_chunks,
            "document_types": type_counts,
            "storage_path": str(self.storage_path)
        }

    def _matches_document_filter(self, document: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document matches filter criteria."""
        for key, value in filters.items():
            if key in document and document[key] != value:
                return False
            if key in document.get("metadata", {}) and document["metadata"][key] != value:
                return False
        return True


# Factory functions
def create_vector_store(store_type: str = "faiss", **kwargs) -> VectorStore:
    """
    Factory function to create vector stores.

    Args:
        store_type: Type of vector store to create
        **kwargs: Store-specific configuration

    Returns:
        Configured VectorStore instance
    """
    if store_type == "faiss":
        dimension = kwargs.get("dimension", 384)
        index_path = kwargs.get("index_path", "./data/vector_store.faiss")
        return FAISSVectorStore(dimension=dimension, index_path=index_path)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")


def create_document_store(store_type: str = "json", **kwargs) -> DocumentStore:
    """
    Factory function to create document store.

    Args:
        store_type: Type of document store to create
        **kwargs: Store-specific configuration

    Returns:
        Configured DocumentStore instance
    """
    if store_type == "json":
        storage_path = kwargs.get("storage_path", "./data/documents.json")
        return JSONDocumentStore(storage_path=storage_path)
    else:
        raise ValueError(f"Unknown document store type: {store_type}")
