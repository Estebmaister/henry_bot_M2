"""
Prompting module for Henry Bot M2.

Enhanced from M1 with RAG-aware prompt engineering capabilities.
"""

from src.modules.prompting.engine import create_prompt
from src.modules.prompting.safety import check_adversarial_prompt

__all__ = [
    "create_prompt",
    "check_adversarial_prompt",
]