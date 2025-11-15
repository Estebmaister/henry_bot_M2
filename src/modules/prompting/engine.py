"""
Enhanced Prompt Engineering Module for Henry Bot M2.

Based on M1 prompting with RAG-aware enhancements and better modularity.
Supports multiple techniques: few-shot, simple, and chain-of-thought.
"""

from pathlib import Path
from typing import List, Dict, Optional


class PromptBuilder:
    """Builds engineered prompts using simple, few-shot and chain-of-thought techniques with RAG awareness."""

    def __init__(self):
        """Initialize the prompt builder with system instructions."""
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Get the system prompt from file or use default."""
        try:
            prompt_path = Path(__file__).parent / \
                "templates" / "system_prompt.txt"
            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

        # Fallback system prompt
        return """You are a helpful AI assistant that provides accurate, structured responses.
Always respond in valid JSON format. Be concise but thorough."""

    def build_few_shot_prompt(
        self,
        user_question: str,
        rag_context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build a few-shot prompt with examples to guide the model.

        Enhanced for M2 to include RAG context when available.

        Args:
            user_question: The user's question to answer
            rag_context: Optional retrieved context from RAG system

        Returns:
            List of message dictionaries formatted for the OpenRouter API
        """
        # Enhance system prompt based on whether RAG context is available
        system_prompt = self.system_prompt
        if rag_context:
            system_prompt += "\n\nUse the provided context to answer the user's question accurately. If the context doesn't contain the answer, say so clearly."

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        # Add context if available (RAG enhancement)
        if rag_context:
            messages.append({
                "role": "system",
                "content": f"Context: {rag_context}"
            })

        # Add few-shot examples
        examples = [
            {
                "user": "What is the capital of France?",
                "assistant": '{"answer": "Paris"}'
            },
            {
                "user": "What is 2 + 2?",
                "assistant": '{"answer": "4"}'
            },
            {
                "user": "Who wrote Romeo and Juliet?",
                "assistant": '{"answer": "William Shakespeare"}'
            }
        ]

        for example in examples:
            messages.extend([
                {"role": "user", "content": example["user"]},
                {"role": "assistant", "content": example["assistant"]}
            ])

        # Add the actual user question
        messages.append({
            "role": "user",
            "content": user_question
        })

        return messages

    def build_simple_prompt(
        self,
        user_question: str,
        rag_context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build a simple prompt without examples.

        Args:
            user_question: The user's question to answer
            rag_context: Optional retrieved context from RAG system

        Returns:
            List of message dictionaries formatted for the OpenRouter API
        """
        system_prompt = self.system_prompt + \
            "\nRespond with JSON in this format: {\"answer\": \"your answer here\"}"

        if rag_context:
            system_prompt += "\n\nUse the provided context to answer the user's question accurately. If the context doesn't contain the answer, say so clearly."

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        # Add context if available
        if rag_context:
            messages.append({
                "role": "system",
                "content": f"Context: {rag_context}"
            })

        messages.append({
            "role": "user",
            "content": user_question
        })

        return messages

    def build_chain_of_thought_prompt(
        self,
        user_question: str,
        rag_context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build a chain-of-thought prompt for complex reasoning.

        Enhanced for M2 to include RAG context when available.

        Args:
            user_question: The user's question to answer
            rag_context: Optional retrieved context from RAG system

        Returns:
            List of message dictionaries formatted for the OpenRouter API
        """
        enhanced_system_prompt = self.system_prompt + """

When answering complex questions, break down your reasoning step by step.
Always provide your final answer in JSON format: {"answer": "your answer", "reasoning": "brief explanation"}"""

        if rag_context:
            enhanced_system_prompt += "\n\nUse the provided context to answer the user's question accurately. Reference the context in your reasoning. If the context doesn't contain the answer, say so clearly."

        messages = [
            {
                "role": "system",
                "content": enhanced_system_prompt
            }
        ]

        # Add context if available
        if rag_context:
            messages.append({
                "role": "system",
                "content": f"Context: {rag_context}"
            })

        messages.append({
            "role": "user",
            "content": user_question
        })

        return messages


def create_prompt(
    user_question: str,
    technique: str = "few_shot",
    rag_context: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Convenience function to create a prompt using specified technique.

    Enhanced for M2 to support RAG context integration.

    Args:
        user_question: The user's question
        technique: Prompting technique to use ('few_shot', 'simple', or 'chain_of_thought')
        rag_context: Optional retrieved context from RAG system

    Returns:
        List of formatted messages for the API
    """
    builder = PromptBuilder()

    if technique == "few_shot":
        return builder.build_few_shot_prompt(user_question, rag_context)
    elif technique == "simple":
        return builder.build_simple_prompt(user_question, rag_context)
    elif technique == "chain_of_thought":
        return builder.build_chain_of_thought_prompt(user_question, rag_context)
    else:
        # Default to few_shot
        return builder.build_few_shot_prompt(user_question, rag_context)
