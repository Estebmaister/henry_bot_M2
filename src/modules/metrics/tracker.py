"""
Enhanced Metrics Tracker for Henry Bot M2.

Based on M1 metrics with RAG performance tracking capabilities.
Tracks LLM API calls and RAG system performance metrics.
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing information for LLM models."""
    prompt_price_per_1k: float  # Cost per 1K prompt tokens
    completion_price_per_1k: float  # Cost per 1K completion tokens


# Enhanced model pricing (in USD per 1K tokens)
# Updated with current OpenRouter rates
MODEL_PRICING = {
    "openai/gpt-3.5-turbo": ModelPricing(0.0005, 0.0015),
    "openai/gpt-4": ModelPricing(0.03, 0.06),
    "openai/gpt-4-turbo": ModelPricing(0.01, 0.03),
    "openai/gpt-4o": ModelPricing(0.005, 0.015),
    "openai/gpt-4o-mini": ModelPricing(0.00015, 0.0006),
    "anthropic/claude-3-haiku": ModelPricing(0.00025, 0.00125),
    "anthropic/claude-3-sonnet": ModelPricing(0.003, 0.015),
    "anthropic/claude-3-opus": ModelPricing(0.015, 0.075),
    "anthropic/claude-3-5-sonnet": ModelPricing(0.003, 0.015),
    "meta-llama/llama-3-8b": ModelPricing(0.0001, 0.0001),
    "meta-llama/llama-3-70b": ModelPricing(0.0007, 0.0007),
    "google/gemini-2.0-flash-exp:free": ModelPricing(0.0, 0.0),  # Free model
    "google/gemini-pro": ModelPricing(0.0005, 0.0015),
    # Default pricing for unknown models
    "default": ModelPricing(0.001, 0.002)
}


class MetricsTracker:
    """Enhanced metrics tracker for LLM API calls with RAG performance support."""

    def __init__(self, model: str = "openai/gpt-3.5-turbo"):
        """
        Initialize the enhanced metrics tracker.

        Args:
            model: The model identifier being used
        """
        self.model = model
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0

        # RAG-specific metrics (M2 enhancement)
        self.rag_used: bool = False
        self.rag_retrieval_time_ms: int = 0
        self.rag_similarity_score: Optional[float] = None
        self.rag_context_length: int = 0
        self.rag_retrieval_count: int = 0

        # Additional metadata (M2 enhancement)
        self.prompt_technique: Optional[str] = None
        self.success: bool = True
        self.error_type: Optional[str] = None

    def start(self) -> None:
        """Start tracking latency."""
        self.start_time = time.time()

    def stop(self) -> None:
        """Stop tracking latency."""
        self.end_time = time.time()

    def start_rag_timing(self) -> None:
        """Start RAG retrieval timing."""
        self.rag_start_time = time.time()

    def stop_rag_timing(self) -> None:
        """Stop RAG retrieval timing and calculate duration."""
        if hasattr(self, 'rag_start_time'):
            rag_latency_seconds = time.time() - self.rag_start_time
            self.rag_retrieval_time_ms = round(rag_latency_seconds * 1000)

    def set_token_usage(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        """
        Set token usage from API response.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            total_tokens: Total tokens used
        """
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

    def set_rag_metrics(
        self,
        rag_used: bool,
        similarity_score: Optional[float] = None,
        context_length: int = 0,
        retrieval_count: int = 0
    ) -> None:
        """
        Set RAG performance metrics.

        Args:
            rag_used: Whether RAG was used for this request
            similarity_score: The highest similarity score from retrieval
            context_length: Length of retrieved context in characters
            retrieval_count: Number of documents retrieved
        """
        self.rag_used = rag_used
        self.rag_similarity_score = similarity_score
        self.rag_context_length = context_length
        self.rag_retrieval_count = retrieval_count

    def set_metadata(self, prompt_technique: str = None, success: bool = True, error_type: str = None) -> None:
        """
        Set additional metadata for the request.

        Args:
            prompt_technique: The prompting technique used
            success: Whether the request was successful
            error_type: Type of error if unsuccessful
        """
        self.prompt_technique = prompt_technique
        self.success = success
        self.error_type = error_type

    def get_latency_ms(self) -> int:
        """
        Calculate latency in milliseconds.

        Returns:
            Latency in milliseconds (rounded to nearest integer)
        """
        if self.start_time is None or self.end_time is None:
            return 0

        latency_seconds = self.end_time - self.start_time
        return round(latency_seconds * 1000)

    def calculate_cost(self) -> float:
        """
        Calculate estimated API cost in USD.

        Returns:
            Estimated cost in USD (rounded to 5 decimal places)
        """
        # Get pricing for the model or use default
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING["default"])

        # Calculate costs
        prompt_cost = (self.prompt_tokens / 1000) * pricing.prompt_price_per_1k
        completion_cost = (self.completion_tokens / 1000) * pricing.completion_price_per_1k
        total_cost = prompt_cost + completion_cost

        return round(total_cost, 5)

    def get_metrics(self) -> Dict[str, any]:
        """
        Get all tracked metrics including RAG performance.

        Returns:
            Dictionary containing latency, token usage, cost, and RAG metrics
        """
        base_metrics = {
            "latency_ms": self.get_latency_ms(),
            "tokens_total": self.total_tokens,
            "tokens_prompt": self.prompt_tokens,
            "tokens_completion": self.completion_tokens,
            "cost_usd": self.calculate_cost(),
            "model": self.model,
        }

        # Add RAG metrics if available (M2 enhancement)
        rag_metrics = {}
        if self.rag_used:
            rag_metrics = {
                "rag_used": True,
                "rag_retrieval_time_ms": self.rag_retrieval_time_ms,
                "rag_similarity_score": self.rag_similarity_score,
                "rag_context_length": self.rag_context_length,
                "rag_retrieval_count": self.rag_retrieval_count,
            }

        # Add metadata
        metadata = {
            "prompt_technique": self.prompt_technique,
            "success": self.success,
            "error_type": self.error_type,
        }

        return {
            **base_metrics,
            **rag_metrics,
            "metadata": metadata
        }

    def get_summary_metrics(self) -> Dict[str, any]:
        """
        Get summary metrics compatible with M1 format.

        Returns:
            Dictionary with key metrics only
        """
        return {
            "latency_ms": self.get_latency_ms(),
            "tokens_total": self.total_tokens,
            "cost_usd": self.calculate_cost()
        }

    def get_rag_performance_score(self) -> Optional[float]:
        """
        Calculate RAG performance score (0-1 scale).

        Returns:
            RAG performance score or None if RAG not used
        """
        if not self.rag_used:
            return None

        # Simple scoring based on similarity score and retrieval efficiency
        if self.rag_similarity_score is not None:
            # Weight similarity more heavily
            similarity_weight = 0.7
            efficiency_weight = 0.3

            # Efficiency score based on retrieval time (lower is better)
            # Assuming 500ms as a baseline for "good" retrieval
            efficiency_score = max(0, 1 - (self.rag_retrieval_time_ms / 1000))

            return (
                similarity_weight * self.rag_similarity_score +
                efficiency_weight * efficiency_score
            )

        return None


def track_api_call(model: str = "openai/gpt-3.5-turbo") -> MetricsTracker:
    """
    Create a metrics tracker for an API call.

    Args:
        model: The model identifier being used

    Returns:
        Initialized MetricsTracker instance
    """
    tracker = MetricsTracker(model)
    tracker.start()
    return tracker