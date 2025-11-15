#!/usr/bin/env python3
"""
Test script for RAG chunking system.

Tests the sliding window chunking strategy and metadata generation
to verify the implementation works correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from modules.rag.chunking import SlidingWindowChunker, DocumentType, create_chunker
from modules.rag.embeddings import create_embedding_service, EmbeddingModelType
from modules.rag.processor import DocumentProcessor, create_document_processor
from modules.rag.storage import create_vector_store, create_document_store


async def test_sliding_window_chunking():
    """Test the sliding window chunking strategy."""
    print("🧪 Testing Sliding Window Chunking...")

    # Test document
    test_content = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.

    Machine learning is a method of teaching computers to learn from data without being explicitly programmed. It is a subset of artificial intelligence that focuses on the development of algorithms that can learn and make predictions or decisions based on data.

    Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. These neural networks are inspired by the structure and function of the human brain.

    Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. It bridges the gap between human communication and computer understanding.

    Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.
    """

    # Create chunker with small chunk size for testing
    chunker = SlidingWindowChunker(
        chunk_size=300,
        overlap_size=50,
        respect_sentence_boundaries=True,
        include_metadata_keywords=True
    )

    # Chunk the document
    chunks = chunker.chunk_document(
        document_content=test_content,
        document_id="test_doc_001",
        document_source="test_content.txt",
        document_type=DocumentType.TEXT
    )

    print(f"✅ Generated {len(chunks)} chunks")
    print(f"   Chunk size: 300 chars, Overlap: 50 chars")
    print(f"   Respect sentence boundaries: True")

    # Print chunk details
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n   Chunk {i + 1}:")
        print(f"   ID: {chunk.chunk_id}")
        print(f"   Length: {chunk.metadata.char_count} chars")
        print(f"   Words: {chunk.metadata.word_count}")
        print(f"   Density Score: {chunk.metadata.density_score:.3f}")
        print(f"   Keywords: {chunk.metadata.keywords[:3]}")  # Show first 3 keywords
        print(f"   Preview: {chunk.content[:100]}...")

    # Test with different parameters
    print(f"\n🔧 Testing with different parameters...")
    chunker2 = create_chunker("sliding_window", chunk_size=200, overlap_size=30)
    chunks2 = chunker2.chunk_document(
        document_content=test_content,
        document_id="test_doc_002",
        document_source="test_content.txt"
    )
    print(f"   Generated {len(chunks2)} chunks with smaller size")

    return chunks


async def test_embedding_service():
    """Test the embedding service."""
    print("\n🧪 Testing Embedding Service...")

    try:
        # Create embedding service
        embedding_service = create_embedding_service()
        await embedding_service.initialize()

        print("✅ Embedding service initialized successfully")

        # Test embedding generation
        test_texts = [
            "Artificial intelligence is a branch of computer science.",
            "Machine learning algorithms learn from data.",
            "Deep learning uses neural networks with multiple layers."
        ]

        result = await embedding_service.embed_text(test_texts)

        print(f"✅ Generated embeddings for {len(test_texts)} texts")
        print(f"   Model: {result.model_name}")
        print(f"   Dimension: {result.embedding_dimension}")
        print(f"   Processing time: {result.processing_time_ms:.1f}ms")
        print(f"   Total tokens: {result.total_tokens}")

        # Show first few embedding values
        print(f"   Sample embedding values: {result.embeddings[0][:5]}")

        # Get service stats
        stats = embedding_service.get_stats()
        print(f"   Service stats: {stats['total_texts']} texts processed")

        return True

    except Exception as e:
        print(f"❌ Embedding service test failed: {e}")
        return False


async def test_document_processor():
    """Test the complete document processing pipeline."""
    print("\n🧪 Testing Document Processor...")

    try:
        # Create document processor
        processor = create_document_processor()
        await processor.initialize()

        print("✅ Document processor initialized")

        # Test document processing
        test_content = """
        Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.

        Python's design philosophy emphasizes code readability with its notable use of significant indentation. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.

        The Python standard library is extensive and contains a wide range of built-in modules that provide access to various functionalities. These include modules for string manipulation, file I/O, networking, and more.
        """

        result = await processor.process_document(
            content=test_content,
            source="python_overview.txt",
            document_type=DocumentType.TEXT,
            store_embeddings=False  # Don't store for this test
        )

        if result.success:
            print(f"✅ Document processed successfully")
            print(f"   Document ID: {result.document_id}")
            print(f"   Chunks generated: {len(result.chunks)}")
            print(f"   Processing time: {result.processing_time_ms:.1f}ms")
            print(f"   Embeddings generated: {result.embeddings is not None}")

            if result.embeddings:
                print(f"   Embedding shape: {result.embeddings.embeddings.shape}")

            # Show chunk metadata
            if result.chunks:
                chunk = result.chunks[0]
                print(f"   First chunk metadata:")
                print(f"     Keywords: {chunk.metadata.keywords}")
                print(f"     Technical terms: {chunk.metadata.technical_terms}")
                print(f"     Coherence score: {chunk.metadata.coherence_score:.3f}")

        else:
            print(f"❌ Document processing failed: {result.error_message}")
            return False

        # Get processing stats
        stats = processor.get_stats()
        print(f"\n📊 Processing Statistics:")
        print(f"   Documents processed: {stats['processing_stats']['total_documents']}")
        print(f"   Chunks created: {stats['processing_stats']['total_chunks']}")
        print(f"   Success rate: {stats['processing_stats']['success_rate']:.1%}")

        return True

    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_hybrid_search_metadata():
    """Test metadata generation for hybrid search."""
    print("\n🧪 Testing Hybrid Search Metadata...")

    # Create technical content with various elements
    tech_content = """
    # Machine Learning Algorithms

    ## Supervised Learning

    Supervised learning algorithms learn from labeled training data. Popular algorithms include:

    - **Linear Regression**: Predicts continuous values using linear relationships
    - **Random Forest**: Ensemble method using multiple decision trees
    - **Neural Networks**: Inspired by biological neural networks

    ```python
    def train_model(X, y):
        model = RandomForestClassifier()
        model.fit(X, y)
        return model
    ```

    ## Unsupervised Learning

    Companies like Google, OpenAI, and Microsoft use unsupervised learning for pattern recognition.

    The function `kmeans_clustering()` is commonly used for customer segmentation.
    """

    chunker = SlidingWindowChunker(
        chunk_size=400,
        overlap_size=100,
        include_metadata_keywords=True
    )

    chunks = chunker.chunk_document(
        document_content=tech_content,
        document_id="tech_doc_001",
        document_source="ml_guide.md",
        document_type=DocumentType.MARKDOWN
    )

    print(f"✅ Processed technical markdown content")
    print(f"   Generated {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        print(f"\n   Chunk {i + 1} Metadata:")
        print(f"     Document Type: {chunk.metadata.document_type.value}")
        print(f"     Is Code Block: {chunk.metadata.is_code_block}")
        print(f"     Is Table Content: {chunk.metadata.is_table_content}")
        print(f"     Keywords: {chunk.metadata.keywords}")
        print(f"     Named Entities: {chunk.metadata.named_entities}")
        print(f"     Technical Terms: {chunk.metadata.technical_terms}")
        print(f"     Density Score: {chunk.metadata.density_score:.3f}")
        print(f"     Coherence Score: {chunk.metadata.coherence_score:.3f}")

    return True


async def main():
    """Run all RAG chunking tests."""
    print("🚀 Starting RAG Chunking System Tests")
    print("=" * 50)

    try:
        # Test 1: Sliding Window Chunking
        await test_sliding_window_chunking()

        # Test 2: Embedding Service
        embedding_success = await test_embedding_service()

        # Test 3: Document Processor
        processor_success = await test_document_processor()

        # Test 4: Hybrid Search Metadata
        await test_hybrid_search_metadata()

        print("\n" + "=" * 50)
        print("✅ All RAG chunking tests completed!")
        print(f"   Embedding Service: {'✅' if embedding_success else '❌'}")
        print(f"   Document Processor: {'✅' if processor_success else '❌'}")

        if embedding_success and processor_success:
            print("\n🎉 RAG chunking system is ready for production!")
        else:
            print("\n⚠️  Some components need attention.")

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)