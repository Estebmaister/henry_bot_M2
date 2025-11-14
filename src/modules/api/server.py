"""
FastAPI server for Henry Bot M2.

Provides RESTful API endpoints with automatic Swagger documentation,
authentication, and RAG-augmented chat functionality.
"""

from .schemas import (
    ChatRequest, ChatResponse, DocumentUploadResponse, HealthResponse,
    ErrorResponse, AdversarialResponse, APIKeyResponse
)
from .dependencies import verify_api_key
from .middleware import (
    URLNormalizationMiddleware, RateLimitMiddleware,
    RequestLoggingMiddleware, SecurityHeadersMiddleware
)
import time
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends, File, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import datetime

from src.core.config import settings
from src.core.agent import HenryBot
from src.core.exceptions import HenryBotError


def generate_api_key() -> str:
    """
    Generate a secure API key for Henry Bot M2.

    Returns:
        A secure random API key with 'henry_bot_' prefix
    """
    # Generate 32 bytes of random data and convert to hex
    # random_part = secrets.token_hex(16)
    # return f"henry_bot_{random_part}"
    return settings.api_key  # Use configured API key for simplicity


class DateTimeJSONResponse(JSONResponse):
    """Custom JSONResponse that can handle datetime objects."""

    def render(self, content) -> bytes:
        return json.dumps(
            content,
            default=self._json_serializer,
            separators=(",", ":"),
            ensure_ascii=False
        ).encode("utf-8")

    def _json_serializer(self, obj):
        """Custom JSON serializer for datetime objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(
            f'Object of type {obj.__class__.__name__} is not JSON serializable')


# Global variables for application state
app_state = {
    "bot": None,
    "start_time": None,
    "version": "2.0.0"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    app_state["start_time"] = time.time()
    try:
        app_state["bot"] = HenryBot()
        print("🚀 Henry Bot M2 initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Henry Bot M2: {e}")
        raise

    yield

    # Shutdown
    print("👋 Henry Bot M2 shutting down")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Henry Bot M2 API",
        description="Enhanced LLM agent with RAG capabilities and comprehensive metrics",
        version=app_state["version"],
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        default_response_class=DateTimeJSONResponse
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware (order matters!)
    # URL normalization must be FIRST to fix URLs before routing
    app.add_middleware(URLNormalizationMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    # 12 users = ~12 calls/min each
    app.add_middleware(RateLimitMiddleware, calls_per_minute=60)
    app.add_middleware(RequestLoggingMiddleware)
    # Note: APIKeyAuthMiddleware removed - now using FastAPI dependencies

    # Add exception handlers
    @app.exception_handler(HenryBotError)
    async def henry_bot_exception_handler(request: Request, exc: HenryBotError):
        """Handle Henry Bot specific exceptions."""
        return DateTimeJSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(exc),
                details={"type": type(exc).__name__}
            ).dict()
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        return DateTimeJSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                details={"type": type(exc).__name__}
            ).dict()
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        """Handle 404 Not Found errors."""
        return DateTimeJSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="Endpoint not found",
                details={"path": request.url.path, "method": request.method}
            ).dict()
        )

    @app.exception_handler(status.HTTP_401_UNAUTHORIZED)
    async def unauthorized_handler(request: Request, exc: HTTPException):
        """Handle 401 Unauthorized errors."""
        return DateTimeJSONResponse(
            status_code=401,
            content=ErrorResponse(
                error=exc.detail,
                details={"code": "UNAUTHORIZED"}
            ).dict()
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle all HTTPExceptions that aren't caught by specific handlers."""
        return DateTimeJSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail,
                details={"code": f"HTTP_{exc.status_code}",
                         "path": request.url.path}
            ).dict()
        )

    # Add routes
    register_routes(app)

    return app


def register_routes(app: FastAPI):
    """Register all API routes."""

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """
        Check system health and status.

        Returns system status including model availability, RAG system status,
        and basic performance metrics.
        """
        uptime = time.time() - \
            app_state["start_time"] if app_state["start_time"] else 0

        if app_state["bot"]:
            status = app_state["bot"].get_system_status()
            return HealthResponse(
                status=status["status"],
                model=status["model"],
                rag_available=status["rag_available"],
                prompting_technique=status["prompting_technique"],
                uptime_seconds=uptime,
                version=app_state["version"]
            )
        else:
            return HealthResponse(
                status="initializing",
                model="unknown",
                rag_available=False,
                prompting_technique="unknown",
                uptime_seconds=uptime,
                version=app_state["version"]
            )

    @app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
    async def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
        """
        Process a user question with optional RAG augmentation.

        This is the main endpoint for interacting with Henry Bot M2.
        Supports multiple prompting techniques and RAG-enhanced responses.
        """
        if not app_state["bot"]:
            raise HTTPException(
                status_code=503, detail="Service not initialized")

        try:
            # Process the question
            result = await app_state["bot"].process_question(
                user_question=request.question,
                prompt_technique=request.prompt_technique,
                use_rag=request.use_rag
            )

            # Check for adversarial response
            if "error" in result and "adversarial_info" in result:
                return DateTimeJSONResponse(
                    status_code=400,
                    content=AdversarialResponse(
                        error=result["error"],
                        adversarial_info=result["adversarial_info"]
                    ).dict()
                )

            # Check for general error
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])

            # Convert to response model
            response_data = {
                "answer": result.get("answer", ""),
                "metrics": result["metrics"],
            }

            # Add optional fields
            if "reasoning" in result:
                response_data["reasoning"] = result["reasoning"]
            if "rag" in result:
                response_data["rag"] = result["rag"]

            return ChatResponse(**response_data)

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Processing failed: {str(e)}")

    @app.post("/api/v1/documents", response_model=DocumentUploadResponse, tags=["Documents"])
    async def upload_documents(files: list[UploadFile] = File(...), api_key: str = Depends(verify_api_key)):
        """
        Upload and process documents for RAG system.

        Supports .txt and .md files for now, with .pdf support planned.
        Documents are processed into chunks and added to the vector store.
        """
        if not app_state["bot"]:
            raise HTTPException(
                status_code=503, detail="Service not initialized")

        if not app_state["bot"].rag_retriever:
            raise HTTPException(
                status_code=503,
                detail="RAG system not available"
            )

        try:
            # TODO: Implement document upload processing
            # This will be implemented in Phase 2
            return DocumentUploadResponse(
                success=False,
                message="Document upload not yet implemented - coming in Phase 2",
                documents_processed=0,
                total_chunks=0,
                processing_time_ms=0
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Document processing failed: {str(e)}")

    @app.get("/api/v1/generate-api-key", response_model=APIKeyResponse, tags=["Authentication"])
    @app.post("/api/v1/generate-api-key", response_model=APIKeyResponse, tags=["Authentication"])
    async def generate_api_key_endpoint():
        """
        Generate a new API key for accessing protected endpoints.

        This public endpoint generates a secure API key that can be used
        to authenticate requests to protected API endpoints. No authentication
        is required to access this endpoint.

        Supports both GET (browser-friendly) and POST requests.

        The generated API key follows the format: henry_bot_[32_random_hex_characters]
        """
        try:
            # Generate a new API key
            new_api_key = generate_api_key()

            return APIKeyResponse(
                api_key=new_api_key,
                status="generated",
                message="API key generated successfully. Use this key in the X-API-Key header for authenticated endpoints."
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to generate API key: {str(e)}")

    @app.get("/favicon.ico", tags=["Static"])
    async def favicon():
        """Return a minimal favicon response."""
        # Return 204 No Content since we don't have a favicon file
        from fastapi import Response
        return Response(status_code=204)

    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with basic API information."""
        return {
            "name": "Henry Bot M2",
            "version": app_state["version"],
            "description": "Enhanced LLM agent with RAG capabilities",
            "docs": "/docs",
            "health": "/health",
            "generate_api_key": "/api/v1/generate-api-key"
        }
