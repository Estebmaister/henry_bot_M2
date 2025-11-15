"""
FastAPI dependencies for Henry Bot M2 authentication.

This file contains FastAPI dependencies that replace the middleware-based
authentication system with FastAPI's native dependency injection approach.
"""

from typing import Optional
from fastapi import HTTPException, Header, status, Depends
from src.core.config import settings


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    FastAPI dependency for API key authentication.

    This replaces the APIKeyAuthMiddleware and provides clean dependency injection
    for authentication without the ExceptionGroup issues we encountered with middleware.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        Valid API key string

    Raises:
        HTTPException: 401 if API key is missing or invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Use X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not settings.validate_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return x_api_key


def get_current_api_key(api_key: str = Depends(verify_api_key)) -> str:
    """
    Dependency to get the current validated API key.

    This can be used in endpoints that need access to the actual API key value
    for logging or user identification purposes.

    Args:
        api_key: Previously validated API key from verify_api_key

    Returns:
        The validated API key string
    """
    return api_key


# Public endpoints dependency that doesn't require authentication
async def no_auth() -> None:
    """
    Dependency for public endpoints that don't require authentication.

    This can be used to explicitly mark endpoints as public.
    """
    pass
