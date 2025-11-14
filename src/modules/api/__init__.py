"""
API Module for Henry Bot M2.

Provides RESTful API endpoints with Swagger documentation,
authentication, and RAG-augmented chat functionality.
"""

from src.modules.api.server import create_app
from src.modules.api.schemas import ChatRequest, ChatResponse, DocumentUploadResponse, HealthResponse

__all__ = [
    "create_app",
    "ChatRequest",
    "ChatResponse",
    "DocumentUploadResponse",
    "HealthResponse",
]