"""
Enhanced Logging System for Henry Bot M2.

Based on M1 logging with RAG performance tracking and improved structure.
Uses CSV-based logging for metrics and structured logging for events.
"""

import csv
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.core.config import settings
from src.modules.metrics.tracker import MetricsTracker


def setup_logging():
    """Setup structured logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Henry Bot M2 logging system initialized")


def _ensure_csv_headers(csv_file: str, headers: List[str]):
    """Ensure CSV file has proper headers."""
    file_exists = os.path.exists(csv_file)

    if not file_exists:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)


def sanitize_field(field):
    """Sanitize field for single-line CSV output, hiding internal system info."""
    field_str = str(field) if field is not None else ""

    # Remove newlines and replace with spaces
    field_str = field_str.replace('\n', ' ').replace('\r', ' ')

    # Sanitize file paths to hide internal system structure
    # Replace absolute paths with generic placeholders
    field_str = re.sub(r'/[^/\s]+/venv/', '/[project]/venv/', field_str)
    field_str = re.sub(r'/Users/[^/\s]+/', '/[project]/', field_str)
    field_str = re.sub(r'"[^"]*venv[^"]*"', '"[project]/venv/"', field_str)

    # Remove any remaining newlines and multiple spaces
    field_str = re.sub(r'\s+', ' ', field_str).strip()

    # Limit length to prevent overly long entries and truncate cleanly
    max_length = 200  # Even shorter limit for cleaner logs
    if len(field_str) > max_length:
        field_str = field_str[:max_length-3] + "..."

    return field_str


def sanitize_error_message(error_msg):
    """Sanitize error messages to remove verbose internal details."""
    if not error_msg:
        return ""

    # Convert to string and extract key information
    error_str = str(error_msg)

    # Remove verbose OpenRouter/metadata that exposes internal details
    error_str = re.sub(r'Error code: \d+ - \{.*?\}', 'API error', error_str)
    error_str = re.sub(r'\{[^}]*\'metadata\'\: \{[^}]*\'raw\'\: [^}]*\}', '', error_str)
    error_str = re.sub(r'\'user_id\'\: [^}]*\}', '', error_str)
    error_str = re.sub(r'\'provider_name\'\: [^}]*\}', '', error_str)
    error_str = re.sub(r'\{\}[^}]*$', '', error_str)  # Clean up remaining JSON fragments

    return sanitize_field(error_msg)[:150]  # Even further limit for error messages


def log_metrics_from_tracker(
    tracker: MetricsTracker,
    prompt_technique: str = "unknown",
    success: bool = True,
    rag_used: bool = False,
    rag_score: Optional[float] = None
):
    """
    Log metrics using the enhanced tracker with RAG support.

    Args:
        tracker: The MetricsTracker instance with data
        prompt_technique: The prompting technique used
        success: Whether the request was successful
        rag_used: Whether RAG was used
        rag_score: RAG similarity score if applicable
    """
    metrics = tracker.get_metrics()

    # Enhanced headers for M2 with RAG support
    headers = [
        'timestamp', 'model', 'latency_ms', 'tokens_total', 'tokens_prompt',
        'tokens_completion', 'cost_usd', 'prompt_technique', 'success',
        'rag_used', 'rag_similarity_score', 'rag_retrieval_time_ms',
        'rag_context_length', 'rag_retrieval_count', 'rag_performance_score'
    ]

    _ensure_csv_headers(settings.metrics_csv, headers)

    row = [
        datetime.now().isoformat(),
        metrics['model'],
        metrics['latency_ms'],
        metrics['tokens_total'],
        metrics['tokens_prompt'],
        metrics['tokens_completion'],
        metrics['cost_usd'],
        prompt_technique,
        success,
        rag_used,
        metrics.get('rag_similarity_score', ''),
        metrics.get('rag_retrieval_time_ms', ''),
        metrics.get('rag_context_length', ''),
        metrics.get('rag_retrieval_count', ''),
        tracker.get_rag_performance_score() or ''
    ]

    try:
        with open(settings.metrics_csv, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    except Exception as e:
        logging.error(f"Failed to write metrics to CSV: {e}")


def log_error(
    error_type: str,
    error_message: str,
    model: str = "unknown",
    user_question: str = "",
    stack_trace: str = "",
    rag_used: bool = False
):
    """
    Log error details to CSV with sanitized single-line format.

    Args:
        error_type: Type of error that occurred
        error_message: Error message details
        model: Model being used when error occurred
        user_question: User question that triggered the error
        stack_trace: Full stack trace if available
        rag_used: Whether RAG was being used
    """
    error_headers = [
        'timestamp', 'error_type', 'error_message', 'model',
        'user_question', 'stack_trace', 'rag_used'
    ]

    error_csv = settings.metrics_csv.replace('.csv', '_errors.csv')
    _ensure_csv_headers(error_csv, error_headers)

    # Sanitize and limit all fields
    sanitized_error_message = sanitize_error_message(error_message)
    sanitized_user_question = sanitize_field(user_question)

    # Special handling for stack traces - extract only key info
    if stack_trace:
        # Extract just the first meaningful line of the stack trace
        lines = stack_trace.strip().split('\n')
        if lines:
            # Take the first line and limit to key info
            first_line = lines[0]
            if len(first_line) > 100:
                # Extract just the file name and function name
                import re
                match = re.search(r'File "[^"]*"[^,]*", line \d+, in (\w+)', first_line)
                if match:
                    first_line = f"{match.group(1)}: Error in {match.group(2)}"
                else:
                    first_line = first_line[:100]
            sanitized_stack_trace = sanitize_field(first_line)[:150]
        else:
            sanitized_stack_trace = "Stack trace unavailable"
    else:
        sanitized_stack_trace = ""

    row = [
        datetime.now().isoformat(),
        error_type,
        sanitized_error_message,
        model,
        sanitized_user_question,
        sanitized_stack_trace,
        rag_used
    ]

    try:
        with open(error_csv, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    except Exception as e:
        logging.error(f"Failed to write error to CSV: {e}")


def log_rag_performance(
    query: str,
    retrieved_docs: int,
    similarity_scores: List[float],
    retrieval_time_ms: int,
    context_used: bool,
    scoring_threshold: float
):
    """
    Log detailed RAG performance metrics (M2 enhancement).

    Args:
        query: The search query
        retrieved_docs: Number of documents retrieved
        similarity_scores: List of similarity scores
        retrieval_time_ms: Time taken for retrieval
        context_used: Whether context was actually used in response
        scoring_threshold: Threshold used for filtering
    """
    rag_headers = [
        'timestamp', 'query', 'retrieved_docs', 'avg_similarity',
        'max_similarity', 'retrieval_time_ms', 'context_used',
        'scoring_threshold', 'above_threshold_count'
    ]

    rag_csv = settings.metrics_csv.replace('.csv', '_rag.csv')
    _ensure_csv_headers(rag_csv, rag_headers)

    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    max_similarity = max(similarity_scores) if similarity_scores else 0
    above_threshold = sum(1 for score in similarity_scores if score >= scoring_threshold)

    row = [
        datetime.now().isoformat(),
        query[:200],  # Limit query length
        retrieved_docs,
        round(avg_similarity, 4),
        round(max_similarity, 4),
        retrieval_time_ms,
        context_used,
        scoring_threshold,
        above_threshold
    ]

    try:
        with open(rag_csv, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    except Exception as e:
        logging.error(f"Failed to write RAG metrics to CSV: {e}")


def log_api_request(
    endpoint: str,
    method: str,
    user_agent: str = "",
    ip_address: str = "",
    response_status: int = 200,
    response_time_ms: int = 0
):
    """
    Log API request metrics (M2 enhancement).

    Args:
        endpoint: API endpoint called
        method: HTTP method used
        user_agent: User agent string
        ip_address: Client IP address
        response_status: HTTP response status code
        response_time_ms: Response time in milliseconds
    """
    api_headers = [
        'timestamp', 'endpoint', 'method', 'user_agent',
        'ip_address', 'response_status', 'response_time_ms'
    ]

    api_csv = settings.metrics_csv.replace('.csv', '_api.csv')
    _ensure_csv_headers(api_csv, api_headers)

    row = [
        datetime.now().isoformat(),
        endpoint,
        method,
        user_agent[:200],  # Limit length
        ip_address,
        response_status,
        response_time_ms
    ]

    try:
        with open(api_csv, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    except Exception as e:
        logging.error(f"Failed to write API metrics to CSV: {e}")


class MetricsAnalytics:
    """Utility class for analyzing logged metrics (M2 enhancement)."""

    @staticmethod
    def get_performance_summary(csv_file: str = None) -> Dict[str, Any]:
        """
        Get performance summary from metrics CSV.

        Args:
            csv_file: Path to metrics CSV file

        Returns:
            Dictionary with performance summary
        """
        if csv_file is None:
            csv_file = settings.metrics_csv

        if not os.path.exists(csv_file):
            return {"error": "No metrics data found"}

        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                rows = list(reader)

            if not rows:
                return {"error": "No metrics data found"}

            # Calculate summary statistics
            total_requests = len(rows)
            successful_requests = sum(1 for row in rows if row.get('success', '').lower() == 'true')
            rag_usage = sum(1 for row in rows if row.get('rag_used', '').lower() == 'true')

            avg_latency = sum(float(row.get('latency_ms', 0)) for row in rows) / total_requests
            total_cost = sum(float(row.get('cost_usd', 0)) for row in rows)
            total_tokens = sum(int(row.get('tokens_total', 0)) for row in rows)

            return {
                "total_requests": total_requests,
                "success_rate": successful_requests / total_requests * 100,
                "rag_usage_rate": rag_usage / total_requests * 100,
                "avg_latency_ms": round(avg_latency, 2),
                "total_cost_usd": round(total_cost, 4),
                "total_tokens": total_tokens,
                "avg_tokens_per_request": round(total_tokens / total_requests, 1)
            }

        except Exception as e:
            return {"error": f"Failed to analyze metrics: {str(e)}"}