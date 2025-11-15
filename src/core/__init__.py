"""
Core module for Henry Bot M2.

Contains fundamental components including configuration,
exceptions, and core agent logic.
"""

from src.core.config import settings, Settings
from src.core.exceptions import (
    HenryBotError,
    ConfigurationError,
    AuthenticationError,
    RAGError,
    DocumentProcessingError,
    EmbeddingError,
    RetrievalError,
    LLMError,
    PromptingError,
    SafetyError,
    ValidationError,
    MetricsError,
)

__all__ = [
    "settings",
    "Settings",
    "HenryBotError",
    "ConfigurationError",
    "AuthenticationError",
    "RAGError",
    "DocumentProcessingError",
    "EmbeddingError",
    "RetrievalError",
    "LLMError",
    "PromptingError",
    "SafetyError",
    "ValidationError",
    "MetricsError",
]
