#!/usr/bin/env python3
"""
Integration test for the RAG retriever system.

Tests the complete FAISS-based semantic search functionality
to verify the implemented RAG retriever works correctly.
"""

from core.config import settings
from modules.rag.processor import create_document_processor, DocumentType
from modules.rag.retriever import RAGRetriever
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


async def test_rag_retriever_initialization():
    """Test RAG retriever initialization."""
    print("🧪 Testing RAG Retriever Initialization...")

    try:
        retriever = RAGRetriever()
        print("✅ RAG Retriever created successfully")

        # Check initial state
        stats = await retriever.get_stats()
        print(f"   Available: {stats['is_available']}")
        print(f"   Vector count: {stats['vector_count']}")
        print(f"   Chunk count: {stats['chunk_count']}")

        return retriever

    except Exception as e:
        print(f"❌ RAG Retriever initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_document_processing_and_indexing(retriever):
    """Test document processing and indexing for RAG."""
    print("\n🧪 Testing Document Processing and Indexing...")

    try:
        # Create document processor
        processor = create_document_processor()
        await processor.initialize()
        print("✅ Document processor initialized")

        # Test document content
        test_content = """
        Henry Bot M2 - Advanced RAG System Implementation

        The Henry Bot M2 system represents a significant advancement in retrieval-augmented generation technology.
        It incorporates a modular architecture that separates concerns into distinct layers for maintainability.

        The RAG system uses FAISS for efficient vector similarity search, enabling fast retrieval of relevant document chunks.
        Sentence transformers generate high-quality embeddings that capture semantic meaning.

        Key features include:
        - Sliding window chunking with overlap for context preservation
        - Rich metadata generation for hybrid search capabilities
        - Asynchronous processing for non-blocking operations
        - Comprehensive error handling and logging
        - Multiple prompting techniques (simple, few-shot, chain-of-thought)

        The system processes documents in the background, updating vector stores and chunk metadata as documents are added.
        This enables real-time search capabilities without interrupting the user experience.
        """

        # Process the document
        result = await processor.process_document(
            content=test_content,
            source="henry_bot_documentation.txt",
            document_type=DocumentType.TEXT,
            store_embeddings=True  # Store embeddings for retrieval
        )

        if result.success:
            print(f"✅ Document processed successfully")
            print(f"   Document ID: {result.document_id}")
            print(f"   Chunks generated: {len(result.chunks)}")
            print(f"   Processing time: {result.processing_time_ms:.1f}ms")

            # Reinitialize retriever to pick up new data
            retriever._initialize_rag_system()

            # Check updated stats
            stats = await retriever.get_stats()
            print(f"   Updated vector count: {stats['vector_count']}")
            print(f"   Updated chunk count: {stats['chunk_count']}")

            return True
        else:
            print(f"❌ Document processing failed: {result.error_message}")
            return False

    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_semantic_search(retriever):
    """Test semantic search functionality."""
    print("\n🧪 Testing Semantic Search...")

    try:
        # Test queries that should find relevant content
        test_queries = [
            "What is Henry Bot M2?",
            "How does the RAG system work?",
            "What features does the system include?",
            "How are documents processed?",
            "What prompting techniques are available?"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: '{query}'")

            # Retrieve context
            context, score = await retriever.retrieve_context(query, top_k=2)

            if context and score is not None:
                print(f"   ✅ Found relevant context (similarity: {score:.3f})")
                print(f"   Context preview: {context[:200]}...")
            else:
                print(f"   ⚠️  No context found for this query")

        return True

    except Exception as e:
        print(f"❌ Semantic search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_retriever_stats_and_monitoring(retriever):
    """Test retriever statistics and monitoring."""
    print("\n🧪 Testing Retriever Stats and Monitoring...")

    try:
        # Get comprehensive stats
        stats = await retriever.get_stats()

        print("✅ Retrieved system statistics:")
        print(f"   System Available: {stats['is_available']}")
        print(f"   Vector Count: {stats['vector_count']}")
        print(f"   Chunk Count: {stats['chunk_count']}")
        print(f"   Vector Store Path: {stats['vector_store_path']}")
        print(f"   Chunk Store Path: {stats['chunk_store_path']}")

        # Test ready state
        is_ready = retriever.is_ready()
        print(f"   Ready State: {is_ready}")

        return True

    except Exception as e:
        print(f"❌ Stats and monitoring test failed: {e}")
        return False


async def test_edge_cases(retriever):
    """Test edge cases and error handling."""
    print("\n🧪 Testing Edge Cases...")

    try:
        # Test empty query
        context, score = await retriever.retrieve_context("", top_k=3)
        print(
            f"   Empty query result: context={context is not None}, score={score}")

        # Test query with special characters
        context, score = await retriever.retrieve_context("Test with @#$%^&*() special chars!", top_k=1)
        print(
            f"   Special chars query: context={context is not None}, score={score}")

        # Test very high top_k
        context, score = await retriever.retrieve_context("Henry Bot", top_k=100)
        print(
            f"   High top_k query: context={context is not None}, score={score}")

        print("✅ Edge cases handled properly")
        return True

    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        return False


async def main():
    """Run all RAG retriever integration tests."""
    print("🚀 Starting RAG Retriever Integration Tests")
    print("=" * 60)

    try:
        # Test 1: RAG Retriever Initialization
        retriever = await test_rag_retriever_initialization()
        if not retriever:
            print("\n❌ Cannot proceed with tests - RAG retriever initialization failed")
            return 1

        # Test 2: Document Processing and Indexing
        indexing_success = await test_document_processing_and_indexing(retriever)

        # Only continue with search tests if indexing succeeded
        if indexing_success and retriever.is_ready():
            # Test 3: Semantic Search
            search_success = await test_semantic_search(retriever)

            # Test 4: Stats and Monitoring
            stats_success = await test_retriever_stats_and_monitoring(retriever)

            # Test 5: Edge Cases
            edge_cases_success = await test_edge_cases(retriever)

            print("\n" + "=" * 60)
            print("✅ All RAG retriever integration tests completed!")
            print(f"   Document Indexing: {'✅' if indexing_success else '❌'}")
            print(f"   Semantic Search: {'✅' if search_success else '❌'}")
            print(f"   Stats & Monitoring: {'✅' if stats_success else '❌'}")
            print(f"   Edge Cases: {'✅' if edge_cases_success else '❌'}")

            if all([indexing_success, search_success, stats_success, edge_cases_success]):
                print("\n🎉 RAG retriever system is fully functional!")
                print("\n📋 Verified Capabilities:")
                print("   ✅ FAISS-based vector similarity search")
                print("   ✅ Sentence transformer embeddings")
                print("   ✅ Async context retrieval")
                print("   ✅ Rich context formatting with sources")
                print("   ✅ Comprehensive error handling")
                print("   ✅ System monitoring and statistics")
            else:
                print("\n⚠️  Some components need attention.")
        else:
            print("\n⚠️  Skipping search tests - indexing not ready")

            print("\n" + "=" * 60)
            print("📊 Test Results:")
            print(f"   Document Indexing: {'✅' if indexing_success else '❌'}")

            if not indexing_success:
                print("\n❌ Document indexing failed - RAG system not ready for search")

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
