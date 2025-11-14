"""
Enhanced Adversarial Prompt Detection Module for Henry Bot M2.

Based on M1 safety module with improved pattern detection and logging.
Detects and flags potentially malicious or manipulative prompts.
"""

import re
from typing import Dict, List, Tuple, Optional
from src.core.exceptions import SafetyError


class AdversarialPromptDetector:
    """Detects adversarial patterns in user prompts with enhanced accuracy."""

    def __init__(self):
        """Initialize the detector with adversarial patterns."""
        # Enhanced patterns for prompt injection attempts
        self.injection_patterns = [
            r'ignore\s+(all|the|your|previous|above)?\s*instructions?',
            r'forget\s+(all|the|your|previous|above)?\s*instructions?',
            r'disregard\s+(all|the|your|previous|above)?\s*instructions?',
            r'override\s+(all|the|your|previous|above)?\s*instructions?',
            r'new\s+instructions?:',
            r'system\s+prompt',
            r'reveal\s+(your\s+)?(system|instructions?|prompt)',
            r'show\s+(me\s+)?(your\s+)?(system|instructions?|prompt)',
            r'what\s+(are|is)\s+(your\s+)?(system|instructions?|prompt)',
            r'print\s+(your\s+)?(system|instructions?|prompt)',
            r'display\s+(your\s+)?(system|instructions?|prompt)',
            r'dump\s+(your\s+)?(system|instructions?|prompt)',
            r'previous\s+(messages?|conversation)',
            r'conversation\s+history',
            r'tell\s+me\s+what\s+you\s+were\s+told',
        ]

        # Enhanced patterns for sensitive information requests
        self.sensitive_patterns = [
            r'(api|secret|private|access)\s+key',
            r'password',
            r'credentials?',
            r'access\s+token',
            r'authentication\s+token',
            r'database\s+connection',
            r'env(ironment)?\s+variable',
            r'config(uration)?\s+file',
            r'secret\s+(file|key)',
            r'internal\s+(documentation|process)',
            r'proprietary\s+information',
        ]

        # Enhanced patterns for role manipulation
        self.role_manipulation_patterns = [
            r'you\s+are\s+now',
            r'act\s+as\s+(a\s+)?',
            r'pretend\s+(to\s+be|you\s+are)',
            r'simulate\s+(a\s+)?',
            r'roleplay\s+as',
            r'from\s+now\s+on\s+you\s+are',
            r'you\s+will\s+act\s+as',
            r'you\s+must\s+respond\s+as',
            r'you\s+should\s+behave\s+like',
        ]

        # Patterns for jailbreak attempts (new in M2)
        self.jailbreak_patterns = [
            r'develop\s+a\s+character',
            r'create\s+a\s+persona',
            r'imagine\s+you\s+are',
            r'write\s+a\s+story\s+about',
            r'hypothetically\s+(speaking|talking)',
            r'let\'s\s+pretend',
            r'as\s+an\s+experiment',
            r'this\s+is\s+just\s+a\s+fictional\s+scenario',
            r'for\s+educational\s+purposes',
        ]

        # Patterns for encoding tricks (new in M2)
        self.encoding_patterns = [
            r'base64',
            r'rot13',
            r'hexadecimal',
            r'unicode',
            r'encode',
            r'decode',
            r'cipher',
            r'encryption',
        ]

    def detect(self, user_input: str) -> Tuple[bool, List[str]]:
        """
        Detect adversarial patterns in user input.

        Args:
            user_input: The user's input text to analyze

        Returns:
            Tuple of (is_adversarial, list_of_detected_patterns)
        """
        detected_patterns = []
        user_input_lower = user_input.lower()

        # Check for prompt injection
        for pattern in self.injection_patterns:
            if re.search(pattern, user_input_lower, re.IGNORECASE):
                detected_patterns.append(f"Prompt injection: {pattern}")

        # Check for sensitive information requests
        for pattern in self.sensitive_patterns:
            if re.search(pattern, user_input_lower, re.IGNORECASE):
                detected_patterns.append(f"Sensitive info request: {pattern}")

        # Check for role manipulation
        for pattern in self.role_manipulation_patterns:
            if re.search(pattern, user_input_lower, re.IGNORECASE):
                detected_patterns.append(f"Role manipulation: {pattern}")

        # Check for jailbreak attempts (M2 enhancement)
        for pattern in self.jailbreak_patterns:
            if re.search(pattern, user_input_lower, re.IGNORECASE):
                detected_patterns.append(f"Jailbreak attempt: {pattern}")

        # Check for encoding tricks (M2 enhancement)
        for pattern in self.encoding_patterns:
            if re.search(pattern, user_input_lower, re.IGNORECASE):
                detected_patterns.append(f"Encoding trick: {pattern}")

        # Additional heuristic checks (M2 enhancement)
        self._check_heuristics(user_input, detected_patterns)

        is_adversarial = len(detected_patterns) > 0
        return is_adversarial, detected_patterns

    def _check_heuristics(self, user_input: str, detected_patterns: List[str]):
        """
        Apply heuristic-based detection for subtle adversarial prompts.

        Args:
            user_input: The user's input text
            detected_patterns: List to append detected patterns to
        """
        # Check for excessive repetition (common in jailbreak attempts)
        words = user_input.lower().split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # If any word appears more than 30% of the time, it's suspicious
            for word, count in word_counts.items():
                if count / len(words) > 0.3 and len(word) > 3:
                    detected_patterns.append(f"Heuristic: Excessive repetition of '{word}'")
                    break

        # Check for very long prompts (potential context flooding)
        if len(user_input) > 2000:
            detected_patterns.append("Heuristic: Unusually long prompt")

        # Check for special character patterns
        if re.search(r'[^\w\s\.,!?;:\'"()\-]{10,}', user_input):
            detected_patterns.append("Heuristic: Excessive special characters")

    def get_safe_response(self, patterns: List[str] = None) -> Dict[str, str]:
        """
        Return a safe error response for adversarial prompts.

        Args:
            patterns: List of detected patterns (for logging purposes)

        Returns:
            Dictionary with error message
        """
        # Don't reveal the specific patterns detected to the user
        return {
            "error": "I cannot process this request. Please rephrase your question in a more appropriate way."
        }

    def get_severity_score(self, patterns: List[str]) -> float:
        """
        Calculate a severity score for the detected patterns.

        Args:
            patterns: List of detected pattern descriptions

        Returns:
            Severity score between 0.0 (low) and 1.0 (high)
        """
        if not patterns:
            return 0.0

        severity_weights = {
            "Prompt injection": 0.8,
            "Jailbreak attempt": 0.9,
            "Sensitive info request": 0.7,
            "Role manipulation": 0.6,
            "Encoding trick": 0.5,
            "Heuristic": 0.4,
        }

        max_severity = 0.0
        for pattern in patterns:
            for keyword, weight in severity_weights.items():
                if keyword.lower() in pattern.lower():
                    max_severity = max(max_severity, weight)
                    break

        return min(max_severity, 1.0)


def check_adversarial_prompt(user_input: str) -> Tuple[bool, Dict]:
    """
    Convenience function to check if a prompt is adversarial.

    Args:
        user_input: The user's input text

    Returns:
        Tuple of (is_adversarial, response_dict)
    """
    try:
        detector = AdversarialPromptDetector()
        is_adversarial, patterns = detector.detect(user_input)

        if is_adversarial:
            # Calculate severity for potential logging
            severity = detector.get_severity_score(patterns)

            # Note: In M2, logging will be handled by the main agent
            # to maintain consistency with the new architecture
            return True, {
                **detector.get_safe_response(patterns),
                "adversarial_info": {
                    "patterns_detected": len(patterns),
                    "severity_score": severity
                }
            }

        return False, {}

    except Exception as e:
        # If safety check fails, log it but don't block the request
        # This ensures the system remains functional even if safety checks fail
        print(f"Warning: Safety check failed: {e}")
        return False, {}