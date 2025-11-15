"""
Enhanced Metrics Tracking Module for Henry Bot M2.

Based on M1 metrics with RAG performance tracking and enhanced analytics.
Tracks latency, token usage, costs, and RAG system performance.
"""

from src.modules.metrics.tracker import (
    MetricsTracker,
    track_api_call,
    ModelPricing,
    MODEL_PRICING
)

__all__ = [
    "MetricsTracker",
    "track_api_call",
    "ModelPricing",
    "MODEL_PRICING",
]
