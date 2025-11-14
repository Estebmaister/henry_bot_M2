"""
API middleware for Henry Bot M2.

Provides authentication, rate limiting, and request logging middleware.
"""

import time
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint

from src.core.config import settings
from src.modules.logging.logger import log_api_request


class URLNormalizationMiddleware(BaseHTTPMiddleware):
    """Middleware to normalize URLs before they reach FastAPI routing."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Normalize URLs by removing double slashes and fixing other common URL issues.

        This middleware runs BEFORE all other middleware to prevent malformed URLs
        from causing routing conflicts in FastAPI.

        Args:
            request: Incoming request
            call_next: Next middleware in chain

        Returns:
            Response from next middleware
        """
        # Get the original path
        original_path = request.url.path

        # Normalize the path:
        # 1. Remove double slashes anywhere in the path
        # 2. Remove trailing slash (except for root path)
        # 3. Ensure path starts with exactly one slash
        normalized_path = original_path

        # Replace multiple consecutive slashes with single slashes
        while "//" in normalized_path:
            normalized_path = normalized_path.replace("//", "/")

        # Remove trailing slash (but keep single slash for root)
        if normalized_path != "/" and normalized_path.endswith("/"):
            normalized_path = normalized_path.rstrip("/")

        # If the path was normalized, create a new request object
        if original_path != normalized_path:
            # Clone the scope and modify the path-related fields
            new_scope = dict(request.scope)
            new_scope["path"] = normalized_path
            new_scope["raw_path"] = normalized_path.encode()

            # Update the query string if needed
            if "query_string" in new_scope:
                # The query string stays the same, only the path changes
                pass

            # Create a new request with the normalized scope
            # Note: We don't pass send/receive as they're not needed for URL normalization
            normalized_request = Request(new_scope)

            # Call next middleware with the normalized request
            return await call_next(normalized_request)

        # If no normalization was needed, use the original request
        return await call_next(request)




class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""

    def __init__(self, app, calls_per_minute: int = 60):
        """
        Initialize rate limiting middleware.

        Args:
            app: FastAPI application
            calls_per_minute: Maximum calls per minute per client
        """
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.clients = {}  # Simple in-memory store

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Apply rate limiting based on client IP.

        Args:
            request: Incoming request
            call_next: Next middleware in chain

        Returns:
            Response from next middleware or HTTP 429 if rate limited
        """
        # Get client IP
        client_ip = request.client.host
        current_time = time.time()

        # Clean old entries (older than 1 minute)
        cutoff_time = current_time - 60
        if client_ip in self.clients:
            self.clients[client_ip] = [
                call_time for call_time in self.clients[client_ip]
                if call_time > cutoff_time
            ]

        # Check rate limit
        if client_ip not in self.clients:
            self.clients[client_ip] = []

        self.clients[client_ip].append(current_time)

        if len(self.clients[client_ip]) > self.calls_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {self.calls_per_minute} calls per minute.",
            )

        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging API requests."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Log request details and timing.

        Args:
            request: Incoming request
            call_next: Next middleware in chain

        Returns:
            Response from next middleware with logging side effects
        """
        start_time = time.time()

        # Get request details
        method = request.method
        endpoint = str(request.url.path)
        user_agent = request.headers.get("User-Agent", "")
        ip_address = request.client.host

        # Process request
        response = await call_next(request)

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Log the request (async but we don't await to avoid blocking)
        log_api_request(
            endpoint=endpoint,
            method=method,
            user_agent=user_agent,
            ip_address=ip_address,
            response_status=response.status_code,
            response_time_ms=response_time_ms
        )

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Add security headers to responses.

        Args:
            request: Incoming request
            call_next: Next middleware in chain

        Returns:
            Response with added security headers
        """
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response