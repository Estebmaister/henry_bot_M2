"""
Embedding generation and management for RAG system.

Provides model-agnostic embedding interface with support for
different embedding models and batch processing capabilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import numpy as np
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import torch

from src.core.config import settings


class EmbeddingModelType(Enum):
    """Supported embedding model types."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    LOCAL = "local"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str
    model_type: EmbeddingModelType
    embedding_dimension: int
    max_sequence_length: int = 512
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: str = "cpu"  # cpu, cuda, mps

    # Model-specific settings
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}


@dataclass
class EmbeddingResult:
    """Result of embedding generation with metadata."""
    embeddings: np.ndarray
    model_name: str
    embedding_dimension: int
    processing_time_ms: float
    total_tokens: int
    batch_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "embeddings": self.embeddings.tolist(),
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "processing_time_ms": self.processing_time_ms,
            "total_tokens": self.total_tokens,
            "batch_size": self.batch_size,
        }


class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.

    Provides a consistent interface for different embedding providers
    while allowing provider-specific optimizations and features.
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize embedding model with configuration."""
        self.config = config
        self.model_name = config.model_name
        self.embedding_dimension = config.embedding_dimension
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model and any required resources."""
        pass

    @abstractmethod
    async def embed_text(self, texts: Union[str, List[str]]) -> EmbeddingResult:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts to embed

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        pass

    @abstractmethod
    async def embed_chunks(self, chunks: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for document chunks.

        Optimized for processing multiple chunks with batch processing.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        pass

    async def health_check(self) -> bool:
        """Check if the embedding model is available and working."""
        try:
            if not self.is_initialized:
                await self.initialize()
            # Test with simple text
            result = await self.embed_text("test")
            return result.embeddings.shape[1] == self.embedding_dimension
        except Exception:
            return False


class SentenceTransformerModel(EmbeddingModel):
    """
    Sentence Transformers embedding model.

    Uses the sentence-transformers library for high-quality embeddings
    with support for various pre-trained models.
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def initialize(self) -> None:
        """Initialize the sentence transformer model."""
        if self.is_initialized:
            return

        try:
            # Import here to avoid dependency issues if not needed
            from sentence_transformers import SentenceTransformer

            # Load model synchronously - it's a one-time operation
            # Running in the main thread avoids segfault issues
            self.model = SentenceTransformer(
                self.model_name,
                device=self.config.device,
                **self.config.model_kwargs
            )

            # Set number of threads to avoid conflicts
            torch.set_num_threads(1)

            self.is_initialized = True

        except ImportError:
            raise ImportError(
                "sentence-transformers package is required. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize sentence transformer model: {e}")

    async def embed_text(self, texts: Union[str, List[str]]) -> EmbeddingResult:
        """Generate embeddings using sentence transformers."""
        if not self.is_initialized:
            await self.initialize()

        # Normalize input
        if isinstance(texts, str):
            texts = [texts]

        start_time = time.time()

        # Use asyncio.to_thread() instead of ThreadPoolExecutor
        # This is safer and avoids segfaults
        embeddings = await asyncio.to_thread(self._encode_batch, texts)

        processing_time = (time.time() - start_time) * 1000

        # Calculate tokens (rough estimate)
        total_tokens = sum(len(text.split()) for text in texts)

        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.model_name,
            embedding_dimension=self.embedding_dimension,
            processing_time_ms=processing_time,
            total_tokens=total_tokens,
            batch_size=len(texts)
        )

    async def embed_chunks(self, chunks: List[str]) -> EmbeddingResult:
        """Generate embeddings for document chunks with batch processing."""
        return await self.embed_text(chunks)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the sentence transformer model."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True  # Ensure numpy output
        )
        return np.array(embeddings)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the sentence transformer model."""
        return {
            "model_type": EmbeddingModelType.SENTENCE_TRANSFORMERS.value,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": self.config.max_sequence_length,
            "normalize_embeddings": self.config.normalize_embeddings,
            "device": self.config.device,
            "is_initialized": self.is_initialized,
        }


class OpenAIEmbeddingModel(EmbeddingModel):
    """
    OpenAI embedding model using their API.

    Provides access to OpenAI's embedding models with proper error handling
    and rate limiting support.
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.client = None

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if self.is_initialized:
            return

        try:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base
            )
            self.is_initialized = True

        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    async def embed_text(self, texts: Union[str, List[str]]) -> EmbeddingResult:
        """Generate embeddings using OpenAI's API."""
        if not self.is_initialized:
            await self.initialize()

        # Normalize input
        if isinstance(texts, str):
            texts = [texts]

        start_time = time.time()

        # Process in batches to respect API limits
        all_embeddings = []
        total_tokens = 0

        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]

            try:
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts,
                    **self.config.model_kwargs
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                total_tokens += response.usage.total_tokens

            except Exception as e:
                raise RuntimeError(f"OpenAI API error: {e}")

        processing_time = (time.time() - start_time) * 1000

        return EmbeddingResult(
            embeddings=np.array(all_embeddings),
            model_name=self.model_name,
            embedding_dimension=self.embedding_dimension,
            processing_time_ms=processing_time,
            total_tokens=total_tokens,
            batch_size=len(texts)
        )

    async def embed_chunks(self, chunks: List[str]) -> EmbeddingResult:
        """Generate embeddings for document chunks."""
        return await self.embed_text(chunks)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OpenAI model."""
        return {
            "model_type": EmbeddingModelType.OPENAI.value,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "api_base": self.config.api_base,
            "batch_size": self.config.batch_size,
            "is_initialized": self.is_initialized,
        }


class EmbeddingService:
    """
    High-level service for embedding generation and management.

    Provides a unified interface for embedding operations with
    model management, caching, and performance monitoring.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize embedding service with configuration."""
        self.config = config or self._create_default_config()
        self.model: Optional[EmbeddingModel] = None
        self._stats = {
            "total_embeddings": 0,
            "total_processing_time_ms": 0,
            "total_texts": 0,
            "total_tokens": 0,
        }

    @staticmethod
    def _create_default_config() -> EmbeddingConfig:
        """Create default embedding configuration from settings."""
        model_name = getattr(settings, 'embedding_model', 'all-MiniLM-L6-v2')

        # Determine model type from name
        if model_name.startswith('text-embedding'):
            model_type = EmbeddingModelType.OPENAI
            dimension = 1536  # Default for OpenAI embeddings
        elif '/' in model_name or model_name in ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']:
            model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
            # Default dimensions for common models
            dimension_map = {
                'all-MiniLM-L6-v2': 384,
                'all-mpnet-base-v2': 768,
            }
            dimension = dimension_map.get(model_name, 384)
        else:
            model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
            dimension = 384

        return EmbeddingConfig(
            model_name=model_name,
            model_type=model_type,
            embedding_dimension=dimension,
            max_sequence_length=getattr(settings, 'max_sequence_length', 512),
            batch_size=getattr(settings, 'embedding_batch_size', 32),
            normalize_embeddings=getattr(
                settings, 'normalize_embeddings', True),
            device=getattr(settings, 'embedding_device', 'cpu'),
            api_key=getattr(settings, 'embeddings_api_key', None),
            api_base=getattr(settings, 'embeddings_base_url', None),
        )

    async def initialize(self) -> None:
        """Initialize the embedding service and model."""
        if self.model is not None:
            return

        # Create model based on type
        if self.config.model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS:
            self.model = SentenceTransformerModel(self.config)
        elif self.config.model_type == EmbeddingModelType.OPENAI:
            self.model = OpenAIEmbeddingModel(self.config)
        else:
            raise ValueError(
                f"Unsupported model type: {self.config.model_type}")

        # Initialize the model
        await self.model.initialize()

    async def embed_text(self, texts: Union[str, List[str]]) -> EmbeddingResult:
        """Generate embeddings for text(s)."""
        if self.model is None:
            await self.initialize()

        result = await self.model.embed_text(texts)

        # Update statistics
        self._update_stats(result)

        return result

    async def embed_chunks(self, chunks: List[str]) -> EmbeddingResult:
        """Generate embeddings for document chunks."""
        if self.model is None:
            await self.initialize()

        result = await self.model.embed_chunks(chunks)

        # Update statistics
        self._update_stats(result)

        return result

    def _update_stats(self, result: EmbeddingResult) -> None:
        """Update service statistics."""
        self._stats["total_embeddings"] += result.embeddings.shape[0]
        self._stats["total_processing_time_ms"] += result.processing_time_ms
        self._stats["total_texts"] += result.batch_size
        self._stats["total_tokens"] += result.total_tokens

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        avg_time = (
            self._stats["total_processing_time_ms"] /
            self._stats["total_texts"]
            if self._stats["total_texts"] > 0 else 0
        )

        return {
            **self._stats,
            "average_processing_time_ms": avg_time,
            "model_info": self.model.get_model_info() if self.model else None,
        }

    async def health_check(self) -> bool:
        """Check if the embedding service is healthy."""
        if self.model is None:
            try:
                await self.initialize()
            except Exception:
                return False

        return await self.model.health_check()

    async def switch_model(self, new_config: EmbeddingConfig) -> None:
        """Switch to a different embedding model."""
        self.config = new_config
        self.model = None
        await self.initialize()

    def get_supported_models(self) -> List[Dict[str, Any]]:
        """Get list of supported embedding models."""
        return [
            {
                "name": "all-MiniLM-L6-v2",
                "type": EmbeddingModelType.SENTENCE_TRANSFORMERS.value,
                "dimension": 384,
                "description": "Fast and efficient multilingual model",
            },
            {
                "name": "all-mpnet-base-v2",
                "type": EmbeddingModelType.SENTENCE_TRANSFORMERS.value,
                "dimension": 768,
                "description": "High-quality English model",
            },
            {
                "name": "text-embedding-3-small",
                "type": EmbeddingModelType.OPENAI.value,
                "dimension": 1536,
                "description": "OpenAI's small embedding model",
            },
            {
                "name": "text-embedding-3-large",
                "type": EmbeddingModelType.OPENAI.value,
                "dimension": 3072,
                "description": "OpenAI's large embedding model",
            },
        ]


def create_embedding_service(**kwargs) -> EmbeddingService:
    """
    Factory function to create embedding service.

    Args:
        **kwargs: Configuration overrides

    Returns:
        Configured embedding service
    """
    # Create default config using settings
    config = EmbeddingService._create_default_config()

    # Override with provided kwargs
    if "model_name" in kwargs:
        config.model_name = kwargs["model_name"]
    if "model_type" in kwargs:
        config.model_type = EmbeddingModelType(kwargs["model_type"])
    if "embedding_dimension" in kwargs:
        config.embedding_dimension = kwargs["embedding_dimension"]
    if "batch_size" in kwargs:
        config.batch_size = kwargs["batch_size"]
    if "device" in kwargs:
        config.device = kwargs["device"]
    if "api_key" in kwargs:
        config.api_key = kwargs["api_key"]

    return EmbeddingService(config)
