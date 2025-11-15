"""
Unit tests for JSON chunk storage.

Tests the JSONChunkStore functionality including storing,
retrieving, searching, and managing document chunks.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from modules.rag.chunk_store import JSONChunkStore, create_chunk_store
from modules.rag.chunking import DocumentChunk, ChunkMetadata, DocumentType


@pytest.fixture
async def temp_chunk_store():
    """Create a temporary JSON chunk store for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir) / "test_chunks.json"
        store = create_chunk_store(str(storage_path))
        await store.initialize()
        yield store


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    metadata1 = ChunkMetadata(
        chunk_id="chunk_1",
        document_id="doc_1",
        document_type=DocumentType.TEXT,
        keywords=["machine", "learning"],
        chunk_index=0,
        token_count=50,
        char_count=200
    )

    metadata2 = ChunkMetadata(
        chunk_id="chunk_2",
        document_id="doc_1",
        document_type=DocumentType.TEXT,
        keywords=["neural", "networks"],
        chunk_index=1,
        token_count=45,
        char_count=180
    )

    chunk1 = DocumentChunk(
        chunk_id="chunk_1",
        content="Machine learning is a subset of artificial intelligence.",
        metadata=metadata1
    )

    chunk2 = DocumentChunk(
        chunk_id="chunk_2",
        content="Neural networks are inspired by biological brain structures.",
        metadata=metadata2
    )

    return [chunk1, chunk2]


class TestJSONChunkStore:
    """Test cases for JSONChunkStore functionality."""

    @pytest.mark.asyncio
    async def test_store_chunks(self, temp_chunk_store, sample_chunks):
        """Test storing chunks in JSON format."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        await temp_chunk_store.store_chunks(sample_chunks, embeddings)

        # Verify chunks were stored
        stored_chunk = await temp_chunk_store.get_chunk("chunk_1")
        assert stored_chunk is not None
        assert stored_chunk.content == "Machine learning is a subset of artificial intelligence."
        assert stored_chunk.embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_store_chunks_without_embeddings(self, temp_chunk_store, sample_chunks):
        """Test storing chunks without embeddings."""
        await temp_chunk_store.store_chunks(sample_chunks)

        stored_chunk = await temp_chunk_store.get_chunk("chunk_1")
        assert stored_chunk is not None
        assert stored_chunk.embedding is None

    @pytest.mark.asyncio
    async def test_get_document_chunks(self, temp_chunk_store, sample_chunks):
        """Test retrieving all chunks for a document."""
        await temp_chunk_store.store_chunks(sample_chunks)

        doc_chunks = await temp_chunk_store.get_document_chunks("doc_1")
        assert len(doc_chunks) == 2
        chunk_ids = [chunk.chunk_id for chunk in doc_chunks]
        assert "chunk_1" in chunk_ids
        assert "chunk_2" in chunk_ids

    @pytest.mark.asyncio
    async def test_list_chunks_with_pagination(self, temp_chunk_store, sample_chunks):
        """Test listing chunks with pagination."""
        await temp_chunk_store.store_chunks(sample_chunks)

        # Test first page
        first_page = await temp_chunk_store.list_chunks(limit=1, offset=0)
        assert len(first_page) == 1

        # Test second page
        second_page = await temp_chunk_store.list_chunks(limit=1, offset=1)
        assert len(second_page) == 1

        # Test pagination by document ID
        doc_chunks = await temp_chunk_store.list_chunks(document_id="doc_1")
        assert len(doc_chunks) == 2

    @pytest.mark.asyncio
    async def test_search_chunks(self, temp_chunk_store, sample_chunks):
        """Test searching chunks by content."""
        await temp_chunk_store.store_chunks(sample_chunks)

        # Search by content
        results = await temp_chunk_store.search_chunks("machine learning")
        assert len(results) >= 1
        assert "machine learning" in results[0].content.lower()

        # Search by keywords
        results = await temp_chunk_store.search_chunks("neural")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_delete_document_chunks(self, temp_chunk_store, sample_chunks):
        """Test deleting all chunks for a document."""
        await temp_chunk_store.store_chunks(sample_chunks)

        # Verify chunks exist
        doc_chunks = await temp_chunk_store.get_document_chunks("doc_1")
        assert len(doc_chunks) == 2

        # Delete chunks
        deleted_count = await temp_chunk_store.delete_document_chunks("doc_1")
        assert deleted_count == 2

        # Verify chunks are deleted
        doc_chunks = await temp_chunk_store.get_document_chunks("doc_1")
        assert len(doc_chunks) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, temp_chunk_store, sample_chunks):
        """Test getting storage statistics."""
        await temp_chunk_store.store_chunks(sample_chunks)

        stats = await temp_chunk_store.get_stats()
        assert stats["total_chunks"] == 2
        assert stats["total_documents"] == 1
        assert "text" in stats["document_types"]
        assert stats["document_types"]["text"] == 2

    @pytest.mark.asyncio
    async def test_export_chunks(self, temp_chunk_store, sample_chunks):
        """Test exporting chunks to a JSON file."""
        await temp_chunk_store.store_chunks(sample_chunks)

        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "exported_chunks.json"

            await temp_chunk_store.export_chunks(str(export_path))

            # Verify export file exists and contains expected data
            assert export_path.exists()

            with open(export_path, 'r') as f:
                export_data = json.load(f)

            assert "exported_at" in export_data
            assert export_data["total_chunks"] == 2
            assert len(export_data["chunks"]) == 2

    @pytest.mark.asyncio
    async def test_persistence(self, sample_chunks):
        """Test that chunks persist across store instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "persistent_chunks.json"

            # Store chunks in first instance
            store1 = create_chunk_store(str(storage_path))
            await store1.initialize()
            await store1.store_chunks(sample_chunks)

            # Create new instance and verify persistence
            store2 = create_chunk_store(str(storage_path))
            await store2.initialize()

            stored_chunk = await store2.get_chunk("chunk_1")
            assert stored_chunk is not None
            assert stored_chunk.content == "Machine learning is a subset of artificial intelligence."

    @pytest.mark.asyncio
    async def test_empty_storage(self, temp_chunk_store):
        """Test operations on empty storage."""
        # Test getting non-existent chunk
        chunk = await temp_chunk_store.get_chunk("non_existent")
        assert chunk is None

        # Test getting chunks for non-existent document
        chunks = await temp_chunk_store.get_document_chunks("non_existent_doc")
        assert len(chunks) == 0

        # Test deleting from empty storage
        deleted_count = await temp_chunk_store.delete_document_chunks("non_existent_doc")
        assert deleted_count == 0

        # Test searching empty storage
        results = await temp_chunk_store.search_chunks("query")
        assert len(results) == 0


class TestChunkStoreFactory:
    """Test cases for chunk store factory function."""

    def test_create_chunk_store_default_path(self):
        """Test creating chunk store with default path."""
        store = create_chunk_store()
        assert store.storage_path.name == "chunks.json"

    def test_create_chunk_store_custom_path(self):
        """Test creating chunk store with custom path."""
        custom_path = "./test_custom_chunks.json"
        store = create_chunk_store(custom_path)
        assert str(store.storage_path) == custom_path


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])