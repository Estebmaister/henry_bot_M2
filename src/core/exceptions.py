"""
Custom exceptions for Henry Bot M2.

Defines specific exception types for better error handling
and debugging throughout the application.
"""


class HenryBotError(Exception):
    """Base exception for all Henry Bot M2 errors."""
    pass


class ConfigurationError(HenryBotError):
    """Raised when there's a configuration issue."""
    pass


class AuthenticationError(HenryBotError):
    """Raised when API authentication fails."""
    pass


class RAGError(HenryBotError):
    """Raised when RAG system encounters an issue."""
    pass


class DocumentProcessingError(RAGError):
    """Raised when document processing fails."""
    pass


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""
    pass


class RetrievalError(RAGError):
    """Raised when document retrieval fails."""
    pass


class LLMError(HenryBotError):
    """Raised when LLM API call fails."""
    pass


class PromptingError(HenryBotError):
    """Raised when prompt engineering fails."""
    pass


class SafetyError(HenryBotError):
    """Raised when safety checks detect issues."""
    pass


class ValidationError(HenryBotError):
    """Raised when input validation fails."""
    pass


class MetricsError(HenryBotError):
    """Raised when metrics tracking fails."""
    pass
