"""
Document chunking strategies for RAG system.

Provides modular chunking approaches with support for sliding windows
and extensible design for future semantic chunking strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime
from enum import Enum

from src.core.config import settings


class DocumentType(Enum):
    """Supported document types for chunking."""
    TEXT = "text"
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"


@dataclass
class ChunkMetadata:
    """
    Rich metadata for document chunks supporting hybrid search.

    Designed to support both keyword and semantic search strategies
    with extensibility for future search enhancements.
    """
    # Core identification
    chunk_id: str
    document_id: str
    document_type: DocumentType
    document_source: str  # filename, URL, etc.

    # Position information
    chunk_index: int
    start_char: int
    end_char: int
    page_number: Optional[int] = None  # For PDF documents

    # Content statistics
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0

    # Structure information (for semantic chunking future)
    section_title: Optional[str] = None
    hierarchy_level: int = 0  # H1, H2, H3, etc.
    is_code_block: bool = False
    is_table_content: bool = False

    # Keyword search metadata
    keywords: List[str] = field(default_factory=list)
    # People, places, organizations
    named_entities: List[str] = field(default_factory=list)
    technical_terms: List[str] = field(default_factory=list)

    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)

    # Chunking strategy metadata
    strategy_name: str = ""
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    # Quality indicators
    density_score: float = 0.0  # Information density (0-1)
    coherence_score: float = 0.0  # Local coherence (0-1)

    # Context information for overlapping windows
    overlap_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_type": self.document_type.value,
            "document_source": self.document_source,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "page_number": self.page_number,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "section_title": self.section_title,
            "hierarchy_level": self.hierarchy_level,
            "is_code_block": self.is_code_block,
            "is_table_content": self.is_table_content,
            "keywords": self.keywords,
            "named_entities": self.named_entities,
            "technical_terms": self.technical_terms,
            "created_at": self.created_at.isoformat(),
            "strategy_name": self.strategy_name,
            "strategy_params": self.strategy_params,
            "density_score": self.density_score,
            "coherence_score": self.coherence_score,
            "overlap_context": self.overlap_context,
        }


@dataclass
class DocumentChunk:
    """
    A chunk of document text with rich metadata.

    This is the fundamental unit that will be embedded and stored
    in the vector database for retrieval.
    """
    content: str
    metadata: ChunkMetadata

    @property
    def chunk_id(self) -> str:
        """Get chunk ID for convenience."""
        return self.metadata.chunk_id

    @property
    def text_for_embedding(self) -> str:
        """
        Get text optimized for embedding generation.

        May include context or preprocessing based on strategy.
        """
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict(),
        }


class ChunkingStrategy(ABC):
    """
    Abstract base class for document chunking strategies.

    Provides a common interface for different chunking approaches
    while allowing strategy-specific parameters and behaviors.
    """

    @abstractmethod
    def chunk_document(
        self,
        document_content: str,
        document_id: str,
        document_source: str,
        document_type: DocumentType = DocumentType.TEXT,
        **kwargs
    ) -> List[DocumentChunk]:
        """
        Split document into chunks using the specific strategy.

        Args:
            document_content: Full text content of the document
            document_id: Unique identifier for the document
            document_source: Source filename or URL
            document_type: Type of document for specialized handling
            **kwargs: Strategy-specific parameters

        Returns:
            List of DocumentChunk objects with rich metadata
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this chunking strategy."""
        pass

    @abstractmethod
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get the parameters used by this strategy."""
        pass


class SlidingWindowChunker(ChunkingStrategy):
    """
    Sliding window chunking strategy with configurable overlap.

    Simple but effective strategy that creates overlapping chunks
    to ensure context continuity across chunk boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap_size: int = 200,
        min_chunk_size: int = 100,
        respect_sentence_boundaries: bool = True,
        include_metadata_keywords: bool = True,
    ):
        """
        Initialize sliding window chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            overlap_size: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a chunk to be considered valid
            respect_sentence_boundaries: Try to break at sentence boundaries
            include_metadata_keywords: Extract keywords for hybrid search
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.respect_sentence_boundaries = respect_sentence_boundaries
        self.include_metadata_keywords = include_metadata_keywords

        # Validate parameters
        if overlap_size >= chunk_size:
            raise ValueError("overlap_size must be less than chunk_size")
        if min_chunk_size >= chunk_size:
            raise ValueError("min_chunk_size must be less than chunk_size")

    def get_strategy_name(self) -> str:
        return "sliding_window"

    def get_strategy_params(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "overlap_size": self.overlap_size,
            "min_chunk_size": self.min_chunk_size,
            "respect_sentence_boundaries": self.respect_sentence_boundaries,
            "include_metadata_keywords": self.include_metadata_keywords,
        }

    def chunk_document(
        self,
        document_content: str,
        document_id: str,
        document_source: str,
        document_type: DocumentType = DocumentType.TEXT,
        **kwargs
    ) -> List[DocumentChunk]:
        """Split document using sliding window strategy."""
        if not document_content or not document_content.strip():
            return []

        processed_content = self._preprocess_content(
            document_content, document_type)
        chunks = []
        start_pos = 0
        chunk_index = 0

        while start_pos < len(processed_content):
            # Determine end position for this chunk
            end_pos = min(start_pos + self.chunk_size, len(processed_content))

            # If we're not at the end, try to find a good break point
            if end_pos < len(processed_content):
                end_pos = self._find_break_point(
                    processed_content, start_pos, end_pos)

            # Extract chunk content
            chunk_content = processed_content[start_pos:end_pos].strip()

            # Calculate next start position BEFORE any continue statements
            next_start_pos = end_pos - self.overlap_size

            # Ensure we're making progress (prevent infinite loop)
            if next_start_pos <= start_pos:
                next_start_pos = start_pos + \
                    max(1, self.chunk_size - self.overlap_size)

            # Skip if too small (unless it's the last chunk)
            if len(chunk_content) < self.min_chunk_size and end_pos < len(processed_content):
                start_pos = next_start_pos
                continue

            # Create metadata
            metadata = self._create_metadata(
                chunk_content=chunk_content,
                document_id=document_id,
                document_source=document_source,
                document_type=document_type,
                chunk_index=chunk_index,
                start_char=start_pos,
                end_char=end_pos,
                original_content=document_content,
            )

            # Create chunk
            chunk = DocumentChunk(content=chunk_content, metadata=metadata)
            chunks.append(chunk)
            chunk_index += 1

            # Move to next position
            start_pos = next_start_pos

        return chunks

    def _preprocess_content(self, content: str, document_type: DocumentType) -> str:
        """Preprocess content based on document type."""
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)

        if document_type == DocumentType.MARKDOWN:
            # Preserve paragraph structure but normalize spacing
            content = re.sub(r'\n\s*\n', '\n\n', content)
            # Remove excessive heading markers but keep structure info
            content = re.sub(r'^(#{1,6})\s+', '', content, flags=re.MULTILINE)

        elif document_type == DocumentType.TEXT:
            # Simple text normalization
            content = content.strip()

        return content

    def _find_break_point(self, content: str, start_pos: int, target_end: int) -> int:
        """Find optimal break point for chunk boundary."""
        if not self.respect_sentence_boundaries:
            return target_end

        # Search backwards for sentence boundary
        search_text = content[start_pos:target_end]

        # Look for sentence endings in reverse order
        sentence_patterns = [
            r'[.!?]\s+',  # Sentence ending with space
            r'[.!?]\s*"',  # Sentence ending with quote
            r'[:;]\s+',  # Clause endings
            r'\n\s*\n',  # Paragraph breaks
        ]

        for pattern in sentence_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # Use the last match that gives us a reasonable chunk size
                for match in reversed(matches):
                    break_pos = start_pos + match.end()
                    if break_pos - start_pos >= self.min_chunk_size:
                        return break_pos

        # Fallback: look for word boundaries
        if target_end < len(content):
            # Find next space after target position
            next_space = content.find(' ', target_end)
            if next_space != -1 and next_space - start_pos < self.chunk_size * 1.2:
                return next_space

        return target_end

    def _create_metadata(
        self,
        chunk_content: str,
        document_id: str,
        document_source: str,
        document_type: DocumentType,
        chunk_index: int,
        start_char: int,
        end_char: int,
        original_content: str,
    ) -> ChunkMetadata:
        """Create rich metadata for the chunk."""
        chunk_id = f"{document_id}_chunk_{chunk_index:04d}"

        # Basic statistics
        char_count = len(chunk_content)
        word_count = len(re.findall(r'\b\w+\b', chunk_content))
        sentence_count = len(re.findall(r'[.!?]+', chunk_content))

        # Extract keywords and entities
        keywords = []
        named_entities = []
        technical_terms = []

        if self.include_metadata_keywords:
            keywords, named_entities, technical_terms = self._extract_keywords_and_entities(
                chunk_content, document_type
            )

        # Detect structure (for future semantic chunking)
        is_code_block = self._is_code_block(chunk_content, document_type)
        is_table_content = self._is_table_content(chunk_content, document_type)

        # Calculate quality scores
        density_score = self._calculate_density_score(chunk_content)
        coherence_score = self._calculate_coherence_score(chunk_content)

        # Overlap context
        overlap_context = {
            "has_overlap": chunk_index > 0,
            "overlap_size": self.overlap_size if chunk_index > 0 else 0,
        }

        return ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_id,
            document_type=document_type,
            document_source=document_source,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
            is_code_block=is_code_block,
            is_table_content=is_table_content,
            keywords=keywords,
            named_entities=named_entities,
            technical_terms=technical_terms,
            strategy_name=self.get_strategy_name(),
            strategy_params=self.get_strategy_params(),
            density_score=density_score,
            coherence_score=coherence_score,
            overlap_context=overlap_context,
        )

    def _extract_keywords_and_entities(
        self, content: str, document_type: DocumentType
    ) -> Tuple[List[str], List[str], List[str]]:
        """Extract keywords, named entities, and technical terms (optimized version)."""
        # For performance, limit content analysis for large chunks
        max_content_length = 2000  # Limit analysis to first 2000 chars
        analysis_content = content[:max_content_length] if len(
            content) > max_content_length else content

        # Simple keyword extraction with performance optimization
        words = re.findall(r'\b\w+\b', analysis_content.lower())

        # Filter common stop words (limit to first 100 words for performance)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        # Get word frequencies (limit processing for performance)
        word_freq = {}
        processed_words = 0
        for word in words:
            if processed_words >= 200:  # Limit to first 200 words
                break
            if len(word) > 2 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            processed_words += 1

        # Get top keywords (limit to 3 for performance)
        keywords = [word for word, freq in sorted(
            word_freq.items(), key=lambda x: x[1], reverse=True)[:3]]

        # Simple patterns for named entities (limit matches for performance)
        named_entities_matches = re.findall(
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', analysis_content)
        named_entities = list(set(named_entities_matches[:5]))

        # Technical terms (limit patterns and matches for performance)
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms only
        ]

        technical_terms = []
        for pattern in technical_patterns:
            matches = re.findall(pattern, analysis_content)
            technical_terms.extend(matches)

        technical_terms = list(set(technical_terms[:3]))

        return keywords, named_entities, technical_terms

    def _is_code_block(self, content: str, document_type: DocumentType) -> bool:
        """Detect if content contains code blocks."""
        if document_type == DocumentType.MARKDOWN:
            return bool(re.search(r'```[\s\S]*?```|`[^`]+`', content))

        # General code indicators
        code_indicators = [
            r'\bdef\s+\w+\s*\(',  # Python functions
            r'\bfunction\s+\w+\s*\(',  # JavaScript functions
            r'\bclass\s+\w+',  # Class definitions
            r'\bimport\s+\w+',  # Import statements
            r'<[^>]+>',  # HTML/XML tags
        ]

        return any(re.search(pattern, content) for pattern in code_indicators)

    def _is_table_content(self, content: str, document_type: DocumentType) -> bool:
        """Detect if content contains table data."""
        if document_type == DocumentType.MARKDOWN:
            return bool(re.search(r'\|.*\|', content))

        # General table indicators
        return bool(re.search(r'\t{2,}| {4,}|\s+\|\s+|\s{2,}\w+\s{2,}', content))

    def _calculate_density_score(self, content: str) -> float:
        """Calculate information density score (0-1)."""
        if not content:
            return 0.0

        # Density based on ratio of content words to total words
        words = re.findall(r'\b\w+\b', content)
        if not words:
            return 0.0

        # Content words vs function words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been'
        }

        content_words = [w for w in words if w.lower() not in stop_words]
        return len(content_words) / len(words)

    def _calculate_coherence_score(self, content: str) -> float:
        """Calculate local coherence score (0-1)."""
        if not content:
            return 0.0

        # Simple coherence based on sentence structure
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return 1.0

        # Check for pronouns and connectors that indicate coherence
        coherence_indicators = [
            'it', 'this', 'that', 'these', 'those', 'he', 'she', 'they',
            'however', 'therefore', 'moreover', 'furthermore', 'consequently',
            'also', 'additionally', 'finally', 'first', 'second', 'third'
        ]

        indicator_count = sum(
            sentence.count(indicator)
            for sentence in sentences
            for indicator in coherence_indicators
        )

        return min(1.0, indicator_count / len(sentences))


def create_chunker(strategy: str = "sliding_window", **kwargs) -> ChunkingStrategy:
    """
    Factory function to create chunking strategies.

    Args:
        strategy: Name of the chunking strategy
        **kwargs: Strategy-specific parameters

    Returns:
        Configured chunking strategy instance

    Raises:
        ValueError: If strategy is not supported
    """
    strategies = {
        "sliding_window": SlidingWindowChunker,
    }

    if strategy not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(
            f"Unknown strategy '{strategy}'. Available: {available}")

    chunker_class = strategies[strategy]

    # Use default parameters from settings if not provided
    if strategy == "sliding_window":
        default_params = {
            "chunk_size": getattr(settings, 'chunk_size', 1000),
            "overlap_size": getattr(settings, 'chunk_overlap', 200),
            "min_chunk_size": getattr(settings, 'min_chunk_size', 100),
            "respect_sentence_boundaries": getattr(settings, 'respect_sentence_boundaries', True),
            "include_metadata_keywords": getattr(settings, 'include_metadata_keywords', True),
        }
        default_params.update(kwargs)
        return chunker_class(**default_params)

    return chunker_class(**kwargs)
