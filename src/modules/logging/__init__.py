"""
Enhanced Logging Module for Henry Bot M2.

Based on M1 logging with RAG performance tracking and structured logging.
Provides comprehensive CSV-based logging for the application.
"""

from src.modules.logging.logger import (
    setup_logging,
    log_metrics_from_tracker,
    log_error,
    log_rag_performance,
    log_api_request
)

__all__ = [
    "setup_logging",
    "log_metrics_from_tracker",
    "log_error",
    "log_rag_performance",
    "log_api_request",
]