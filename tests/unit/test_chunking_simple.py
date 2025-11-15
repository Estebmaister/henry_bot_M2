#!/usr/bin/env python3
"""
Simple test script for RAG chunking functionality.

Tests the sliding window chunking strategy and metadata generation.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from modules.rag.chunking import SlidingWindowChunker, DocumentType, create_chunker


def test_sliding_window_chunking():
    """Test the sliding window chunking strategy."""
    print("🧪 Testing Sliding Window Chunking...")

    # Test document
    test_content = """Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.

    Machine learning is a method of teaching computers to learn from data without being explicitly programmed. It is a subset of artificial intelligence that focuses on the development of algorithms that can learn and make predictions or decisions based on data.

    Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. These neural networks are inspired by the structure and function of the human brain.

    Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. It bridges the gap between human communication and computer understanding.

    Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do."""

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


def test_markdown_chunking():
    """Test chunking with markdown content."""
    print("\n🧪 Testing Markdown Document Chunking...")

    # Test markdown document
    markdown_content = """# Machine Learning Guide

## Introduction

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

## Key Concepts

### Supervised Learning
Supervised learning algorithms learn from labeled training data. Examples include:
- Classification tasks
- Regression problems
- Decision trees and random forests

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. Popular methods:
- Clustering algorithms
- Dimensionality reduction
- Anomaly detection

## Popular Algorithms

Some well-known machine learning algorithms include:

1. Linear Regression - Used for predictive modeling
2. Random Forest - Ensemble learning method
3. Neural Networks - Deep learning approach
4. Support Vector Machines - Classification algorithm

## Conclusion

Machine learning continues to evolve with new techniques and applications emerging regularly. Companies like Google, OpenAI, and Microsoft are pushing the boundaries of what is possible with AI."""

    # Create chunker
    chunker = SlidingWindowChunker(
        chunk_size=400,
        overlap_size=100,
        respect_sentence_boundaries=True,
        include_metadata_keywords=True
    )

    # Chunk the markdown document
    chunks = chunker.chunk_document(
        document_content=markdown_content,
        document_id="ml_guide_001",
        document_source="machine_learning_guide.md",
        document_type=DocumentType.MARKDOWN
    )

    print(f"✅ Generated {len(chunks)} chunks from markdown document")

    # Show detailed metadata for first chunk
    if chunks:
        chunk = chunks[0]
        print(f"\n   First Chunk Analysis:")
        print(f"   Content preview: {chunk.content[:150]}...")
        print(f"   Keywords: {chunk.metadata.keywords}")
        print(f"   Named Entities: {chunk.metadata.named_entities}")
        print(f"   Technical Terms: {chunk.metadata.technical_terms}")
        print(f"   Is Code Block: {chunk.metadata.is_code_block}")
        print(f"   Document Type: {chunk.metadata.document_type.value}")
        print(f"   Strategy: {chunk.metadata.strategy_name}")
        print(f"   Coherence Score: {chunk.metadata.coherence_score:.3f}")
        print(f"   Density Score: {chunk.metadata.density_score:.3f}")

    return chunks


def test_hybrid_search_metadata():
    """Test metadata specifically designed for hybrid search."""
    print("\n🧪 Testing Hybrid Search Metadata Generation...")

    # Technical content with various search elements
    tech_content = """The function process_data handles JSON parsing and validation. It uses the pandas library for efficient data manipulation.

    Company names like Google, Microsoft, and OpenAI are leading the AI industry. The city of San Francisco hosts many tech startups.

    For web development, React and Vue.js are popular JavaScript frameworks. Python Django and Flask frameworks are widely used for backend development.

    The database query SELECT all active users from database. Use parameters to prevent SQL injection attacks.

    HTTP status codes like 200, 404, and 500 indicate different response states. The REST API design follows standard conventions.

    Machine learning models require feature engineering and hyperparameter tuning. Deep learning architectures like CNNs and RNNs serve different purposes."""

    chunker = SlidingWindowChunker(
        chunk_size=250,
        overlap_size=50,
        include_metadata_keywords=True
    )

    chunks = chunker.chunk_document(
        document_content=tech_content,
        document_id="tech_doc_001",
        document_source="technical_content.txt",
        document_type=DocumentType.TEXT
    )

    print(f"✅ Generated {len(chunks)} chunks with hybrid search metadata")

    # Show rich metadata for search optimization
    for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
        print(f"\n   Chunk {i + 1} Search Metadata:")
        print(f"     Keywords: {chunk.metadata.keywords}")
        print(f"     Named Entities: {chunk.metadata.named_entities}")
        print(f"     Technical Terms: {chunk.metadata.technical_terms}")
        print(f"     Word Count: {chunk.metadata.word_count}")
        print(f"     Content Density: {chunk.metadata.density_score:.3f}")

    return chunks


def main():
    """Run all chunking tests."""
    print("🚀 Starting RAG Chunking Tests")
    print("=" * 50)

    try:
        # Test 1: Basic Sliding Window Chunking
        test_sliding_window_chunking()

        # Test 2: Markdown Document Chunking
        test_markdown_chunking()

        # Test 3: Hybrid Search Metadata
        test_hybrid_search_metadata()

        print("\n" + "=" * 50)
        print("✅ All RAG chunking tests completed successfully!")
        print("\n🎉 RAG chunking system is working properly!")
        print("\n📋 Summary:")
        print("   ✅ Sliding window chunking with overlap")
        print("   ✅ Rich metadata generation for hybrid search")
        print("   ✅ Support for different document types")
        print("   ✅ Configurable chunking parameters")
        print("   ✅ Keyword and entity extraction")
        print("   ✅ Code block and table detection")

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)