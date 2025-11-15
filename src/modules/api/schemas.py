"""
Pydantic schemas for Henry Bot M2 API.

Defines request/response models with validation and Swagger documentation.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from datetime import datetime


class MetricsSchema(BaseModel):
    """Schema for performance metrics."""
    latency_ms: int = Field(..., description="Response time in milliseconds")
    tokens_total: int = Field(..., description="Total tokens used")
    cost_usd: float = Field(..., description="Estimated API cost in USD")


class RAGSchema(BaseModel):
    """Schema for RAG system information."""
    context_used: bool = Field(..., description="Whether RAG context was used")
    retrieval_score: Optional[float] = Field(
        None, description="RAG retrieval similarity score")
    context_length: int = Field(..., description="Length of retrieved context")
    context_preview: str = Field(...,
                                 description="Preview of retrieved context")


class ChatRequest(BaseModel):
    """Schema for chat requests."""
    question: str = Field(..., min_length=1, max_length=2000,
                          description="User's question")
    prompt_technique: Optional[str] = Field(
        "few_shot",
        description="Prompting technique to use",
        pattern="^(few_shot|simple|chain_of_thought)$"
    )
    use_rag: Optional[bool] = Field(
        True, description="Whether to use RAG system")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Tell me about the Pandora Experience program.",
                "prompt_technique": "few_shot",
                "use_rag": True
            }
        }
    }


class ChatResponse(BaseModel):
    """Schema for chat responses."""
    answer: str = Field(..., description="AI-generated answer")
    reasoning: Optional[str] = Field(
        None, description="Reasoning for complex answers")
    metrics: MetricsSchema = Field(..., description="Performance metrics")
    rag: Optional[RAGSchema] = Field(
        None, description="RAG system information")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "Pandora Experience is a personal transformation experience that integrates conscious breathing, guided meditation, and ice immersion therapy. It is designed for individuals undergoing change, feeling emotionally stuck, or seeking to enhance their resilience and mental focus. It helps to release fear, unlock repressed emotions, and reconnect with inner strength. The experience focuses on transforming pain into clarity, strength, and peace by embracing discomfort and discovering strength in vulnerability.",
                "reasoning": "null",
                "metrics": {
                    "latency_ms": 1695,
                    "tokens_total": 710,
                    "cost_usd": 0
                },
                "rag": {
                    "context_used": "true",
                    "retrieval_score": 0.34603750705718994,
                    "context_length": 1814,
                    "context_preview": "[Source: ¿QUÉ ES PANDORA EXPERIENCE_.docx]\n¿QUÉ ES PANDORA EXPERIENCE? 🌬️ UNA EXPERIENCIA QUE TRANSFORMA EL FRÍO EN CLARIDAD Pandora Experience es una jornada vivencial de transformación personal que..."
                },
                "timestamp": "2025-11-15T13:35:33.863763"
            }
        }
    }


class DocumentUploadResponse(BaseModel):
    """Schema for document upload responses."""
    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Status message")
    documents_processed: int = Field(...,
                                     description="Number of documents processed")
    total_chunks: int = Field(...,
                              description="Total number of text chunks created")
    processing_time_ms: float = Field(...,
                                      description="Processing time in milliseconds")
    errors: List[str] = Field(default_factory=list,
                              description="Any processing errors")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "message": "Documents processed successfully",
                "documents_processed": 3,
                "total_chunks": 15,
                "processing_time_ms": 1250,
                "errors": []
            }
        }
    }


class HealthResponse(BaseModel):
    """Schema for health check responses."""
    status: str = Field(..., description="System status")
    model: str = Field(..., description="LLM model being used")
    rag_available: bool = Field(...,
                                description="Whether RAG system is available")
    prompting_technique: str = Field(...,
                                     description="Default prompting technique")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="API version")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model": "google/gemini-2.0-flash-exp:free",
                "rag_available": True,
                "prompting_technique": "few_shot",
                "uptime_seconds": 3600.5,
                "version": "2.0.0"
            }
        }
    }


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Error timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "Invalid API key provided",
                "details": {"code": "AUTH_ERROR"},
                "timestamp": "2025-11-10T15:30:00"
            }
        }
    }


class AdversarialResponse(BaseModel):
    """Schema for adversarial prompt detection responses."""
    error: str = Field(...,
                       description="Error message for adversarial prompts")
    adversarial_info: Optional[Dict[str, Any]] = Field(
        None, description="Adversarial detection info")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "I cannot process this request. Please rephrase your question in a more appropriate way.",
                "adversarial_info": {
                    "patterns_detected": 2,
                    "severity_score": 0.8
                },
                "timestamp": "2025-11-10T15:30:00"
            }
        }
    }


class DocumentProcessingStatus(BaseModel):
    """Schema for individual document processing status."""
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    status: str = Field(...,
                        description="Processing status: pending, processing, completed, failed")
    document_type: str = Field(..., description="Document type")
    chunk_count: int = Field(
        default=0, description="Number of chunks generated")
    embedding_status: str = Field(
        default="pending", description="Embedding generation status")
    created_at: datetime = Field(...,
                                 description="Document creation timestamp")
    processing_started_at: Optional[datetime] = Field(
        None, description="Processing start time")
    completed_at: Optional[datetime] = Field(
        None, description="Completion timestamp")
    error_message: Optional[str] = Field(
        None, description="Error message if failed")


class PipelineStatusResponse(BaseModel):
    """Schema for RAG pipeline status responses."""
    pipeline_status: str = Field(
        ..., description="Overall pipeline status: idle, processing, completed")
    total_documents: int = Field(...,
                                 description="Total documents in the system")
    processing_documents: int = Field(
        ..., description="Number of documents currently being processed")
    completed_documents: int = Field(
        ..., description="Number of successfully processed documents")
    failed_documents: int = Field(...,
                                  description="Number of failed document processing attempts")
    total_chunks: int = Field(...,
                              description="Total chunks generated across all documents")
    total_embeddings: int = Field(...,
                                  description="Total embeddings generated")
    storage_status: Dict[str,
                         bool] = Field(..., description="Status of storage components")
    recent_documents: List[DocumentProcessingStatus] = Field(
        ..., description="Recently processed documents")
    uptime_seconds: Optional[float] = Field(
        None, description="Server uptime in seconds")
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Status last updated timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pipeline_status": "processing",
                "total_documents": 15,
                "processing_documents": 3,
                "completed_documents": 12,
                "failed_documents": 0,
                "total_chunks": 234,
                "total_embeddings": 234,
                "storage_status": {
                    "vector_store": True,
                    "chunk_store": True,
                    "document_store": True
                },
                "recent_documents": [
                    {
                        "document_id": "doc_123",
                        "filename": "example.docx",
                        "status": "completed",
                        "document_type": "DOCX",
                        "chunk_count": 18,
                        "embedding_status": "completed",
                        "created_at": "2025-11-14T18:30:00",
                        "processing_started_at": "2025-11-14T18:30:01",
                        "completed_at": "2025-11-14T18:32:15",
                        "error_message": None
                    }
                ],
                "uptime_seconds": 3600.5,
                "last_updated": "2025-11-14T18:35:00"
            }
        }
    }


class APIKeyResponse(BaseModel):
    """Schema for API key generation responses."""
    api_key: str = Field(..., description="Generated API key")
    status: str = Field(..., description="Status of the API key generation")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(
        default_factory=datetime.now, description="API key creation timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "api_key": "henry_bot_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
                "status": "generated",
                "message": "API key generated successfully",
                "created_at": "2025-01-14T15:30:00"
            }
        }
    }
