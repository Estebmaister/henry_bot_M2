"""
JSON-based chunk storage for visibility and debugging.

Stores document chunks in JSON format alongside the vector store
for easy inspection and debugging of chunking results.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .chunking import DocumentChunk


@dataclass
class ChunkStorageEntry:
    """Storage entry for a document chunk."""
    chunk_id: str
    document_id: str
    source: str
    document_type: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "document_type": self.document_type.value if hasattr(self.document_type, 'value') else self.document_type,
            "content": self.content,
            "metadata": self.metadata.to_dict() if hasattr(self.metadata, 'to_dict') else self.metadata,
            "embedding": self.embedding,
            "created_at": self.created_at
        }


class JSONChunkStore:
    """
    JSON-based storage for document chunks.

    Provides visibility into chunking results and allows for easy
    inspection and debugging of the RAG system.
    """

    def __init__(self, storage_path: str = "./data/chunks.json"):
        """Initialize JSON chunk store."""
        self.storage_path = Path(storage_path)
        self.chunks: Dict[str, ChunkStorageEntry] = {}
        self._lock = asyncio.Lock()
        self._loaded = False

    async def initialize(self) -> None:
        """Load chunks from JSON file."""
        async with self._lock:
            await self._load_chunks()

    async def store_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: Optional[List[List[float]]] = None
    ) -> None:
        """
        Store chunks in JSON format.

        Args:
            chunks: List of document chunks to store
            embeddings: Optional list of embeddings for each chunk
        """
        async with self._lock:
            await self._load_chunks()

            for i, chunk in enumerate(chunks):
                embedding = embeddings[i] if embeddings and i < len(embeddings) else None

                entry = ChunkStorageEntry(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.metadata.document_id,
                    source=chunk.metadata.document_source,
                    document_type=chunk.metadata.document_type.value,
                    content=chunk.content,
                    metadata=chunk.metadata.to_dict(),
                    embedding=embedding
                )

                self.chunks[chunk.chunk_id] = entry

            await self._save_chunks()

    async def get_chunk(self, chunk_id: str) -> Optional[ChunkStorageEntry]:
        """Get a specific chunk by ID."""
        async with self._lock:
            await self._load_chunks()
            return self.chunks.get(chunk_id)

    async def get_document_chunks(self, document_id: str) -> List[ChunkStorageEntry]:
        """Get all chunks for a specific document."""
        async with self._lock:
            await self._load_chunks()
            return [
                chunk for chunk in self.chunks.values()
                if chunk.document_id == document_id
            ]

    async def list_chunks(
        self,
        limit: int = 100,
        offset: int = 0,
        document_id: Optional[str] = None
    ) -> List[ChunkStorageEntry]:
        """List chunks with pagination."""
        async with self._lock:
            await self._load_chunks()

            chunks = list(self.chunks.values())

            # Filter by document ID if provided
            if document_id:
                chunks = [c for c in chunks if c.document_id == document_id]

            # Sort by created_at descending
            chunks.sort(key=lambda x: x.created_at, reverse=True)

            # Apply pagination
            return chunks[offset:offset + limit]

    async def search_chunks(
        self,
        query: str,
        limit: int = 10
    ) -> List[ChunkStorageEntry]:
        """
        Simple text search in chunk contents.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of chunks matching the query
        """
        async with self._lock:
            await self._load_chunks()

            query_lower = query.lower()
            matching_chunks = []

            for chunk in self.chunks.values():
                # Simple content matching
                if query_lower in chunk.content.lower():
                    matching_chunks.append(chunk)
                    continue

                # Search in metadata keywords
                keywords = chunk.metadata.get("keywords", [])
                if any(query_lower in keyword.lower() for keyword in keywords):
                    matching_chunks.append(chunk)
                    continue

                # Search in named entities
                entities = chunk.metadata.get("named_entities", [])
                if any(query_lower in entity.lower() for entity in entities):
                    matching_chunks.append(chunk)

            # Sort by relevance (simple keyword count)
            matching_chunks.sort(
                key=lambda x: x.content.lower().count(query_lower),
                reverse=True
            )

            return matching_chunks[:limit]

    async def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        async with self._lock:
            await self._load_chunks()

            chunk_ids_to_delete = [
                chunk_id for chunk_id, chunk in self.chunks.items()
                if chunk.document_id == document_id
            ]

            for chunk_id in chunk_ids_to_delete:
                del self.chunks[chunk_id]

            if chunk_ids_to_delete:
                await self._save_chunks()

            return len(chunk_ids_to_delete)

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        # Since this is now a synchronous method but needs async components,
        # we'll use asyncio.run to handle the async operations
        import asyncio

        async def _async_get_stats():
            async with self._lock:
                await self._load_chunks()

                total_chunks = len(self.chunks)
                documents = set(chunk.document_id for chunk in self.chunks.values())

                # Document type distribution
                type_counts = {}
                for chunk in self.chunks.values():
                    doc_type = chunk.document_type
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

                # Chunks with embeddings
                chunks_with_embeddings = sum(
                    1 for chunk in self.chunks.values()
                    if chunk.embedding is not None
                )

                return {
                    "total_chunks": total_chunks,
                    "total_documents": len(documents),
                    "document_types": type_counts,
                    "chunks_with_embeddings": chunks_with_embeddings,
                    "storage_path": str(self.storage_path),
                    "last_updated": max(
                        (chunk.created_at for chunk in self.chunks.values()),
                        default="Never"
                    )
                }

        # Handle both running and non-running event loop scenarios
        try:
            loop = asyncio.get_running_loop()
            # Create a future and run it in the existing loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _async_get_stats())
                return future.result()
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(_async_get_stats())

    async def export_chunks(
        self,
        output_path: str,
        document_id: Optional[str] = None
    ) -> None:
        """Export chunks to a JSON file."""
        async with self._lock:
            await self._load_chunks()

            if document_id:
                chunks_to_export = [
                    chunk for chunk in self.chunks.values()
                    if chunk.document_id == document_id
                ]
            else:
                chunks_to_export = list(self.chunks.values())

            export_data = {
                "exported_at": datetime.now().isoformat(),
                "total_chunks": len(chunks_to_export),
                "chunks": [chunk.to_dict() for chunk in chunks_to_export]
            }

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)

    async def _load_chunks(self) -> None:
        """Load chunks from JSON file."""
        if self._loaded:
            return

        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)

                # Handle both old format (list) and new format (dict by chunk_id)
                if isinstance(data, list):
                    # Legacy format - convert to dict
                    for item in data:
                        entry = ChunkStorageEntry(**item)
                        self.chunks[entry.chunk_id] = entry
                elif isinstance(data, dict):
                    # New format
                    if "chunks" in data:
                        # Export format
                        for item in data["chunks"]:
                            entry = ChunkStorageEntry(**item)
                            self.chunks[entry.chunk_id] = entry
                    else:
                        # Direct chunk storage format
                        for chunk_id, item in data.items():
                            entry = ChunkStorageEntry(**item)
                            self.chunks[chunk_id] = entry
            else:
                self.chunks = {}
        except Exception as e:
            print(f"Error loading chunks from {self.storage_path}: {e}")
            self.chunks = {}

        self._loaded = True

    async def _save_chunks(self) -> None:
        """Save chunks to JSON file."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as dict by chunk_id for efficient lookups
        data = {
            chunk_id: entry.to_dict()
            for chunk_id, entry in self.chunks.items()
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)


# Factory function
def create_chunk_store(storage_path: str = "./data/chunks.json") -> JSONChunkStore:
    """
    Factory function to create JSON chunk store.

    Args:
        storage_path: Path to JSON storage file

    Returns:
        Configured JSONChunkStore instance
    """
    return JSONChunkStore(storage_path=storage_path)