"""
Core Henry Bot M2 agent.

Main orchestrator that connects all system modules.
This replaces the monolithic approach from M1 with a modular design.
"""

import json
import traceback
from typing import Dict, Optional, List
from openai import OpenAI

from src.core.config import settings
from src.core.exceptions import LLMError, HenryBotError, ConfigurationError
from src.modules.prompting.engine import create_prompt
from src.modules.prompting.safety import check_adversarial_prompt
from src.modules.metrics.tracker import track_api_call
from src.modules.rag.retriever import RAGRetriever
from src.modules.rag.processor import create_document_processor
from src.modules.logging.logger import log_metrics_from_tracker, log_error


class HenryBot:
    """
    Main Henry Bot agent that orchestrates all system modules.

    This class serves as the central coordinator, connecting:
    - LLM API interactions (inherited from M1)
    - RAG system (new in M2)
    - Prompt engineering (enhanced from M1)
    - Safety checks (inherited from M1)
    - Metrics tracking (enhanced from M1)
    """

    def __init__(self):
        """Initialize the Henry Bot agent with all modules."""
        self.model = settings.model_name
        self.api_key = settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self.prompting_technique = settings.prompting_technique

        # Validate API key
        if not self.api_key:
            raise ConfigurationError(
                "OPENROUTER_API_KEY not found. Please set it in your environment or .env file.\n"
                "Get your API key from: https://openrouter.ai/settings/keys"
            )

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # Initialize RAG retriever
        try:
            self.rag_retriever = RAGRetriever()
        except Exception as e:
            # RAG system can be optional for basic functionality
            self.rag_retriever = None
            print(f"Warning: RAG system not initialized: {e}")

        # Initialize document processor
        try:
            self.document_processor = create_document_processor()
        except Exception as e:
            # Document processor can be optional for basic functionality
            self.document_processor = None
            print(f"Warning: Document processor not initialized: {e}")

    async def process_question(
        self,
        user_question: str,
        prompt_technique: Optional[str] = None,
        use_rag: bool = True
    ) -> Dict:
        """
        Process a user question with optional RAG augmentation.

        Args:
            user_question: The user's question
            prompt_technique: Prompting technique to use
            use_rag: Whether to use RAG system for context

        Returns:
            Dictionary containing the answer, metrics, and RAG scores
        """
        # Use default prompting technique if not specified
        if prompt_technique is None:
            prompt_technique = self.prompting_technique

        # Step 1: Check for adversarial prompts (inherited from M1)
        is_adversarial, adversarial_response = check_adversarial_prompt(
            user_question)
        if is_adversarial:
            return adversarial_response

        # Step 2: Retrieve relevant context using RAG (if available and enabled)
        rag_context = None
        rag_score = None
        if use_rag and self.rag_retriever:
            try:
                rag_context, rag_score = await self.rag_retriever.retrieve_context(
                    user_question,
                    top_k=settings.similarity_top_k
                )
            except Exception as e:
                # Log RAG error but continue without context
                log_error(
                    error_type="RAGRetrievalError",
                    error_message=str(e),
                    model=self.model,
                    user_question=user_question
                )
                rag_context = None
                rag_score = None

        # Step 3: Build the prompt using prompt engineering (enhanced from M1)
        messages = create_prompt(
            user_question,
            technique=prompt_technique,
            rag_context=rag_context
        )

        # Step 4: Start metrics tracking (enhanced from M1)
        tracker = track_api_call(model=self.model)

        try:
            # Step 5: Call the LLM API (inherited from M1)
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/estebmaister/henry_bot_M2",
                    "X-Title": "henry_bot_M2"
                },
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Step 6: Stop metrics tracking
            tracker.stop()

            # Step 7: Extract token usage
            usage = response.usage
            tracker.set_token_usage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens
            )

            # Step 8: Parse the response
            answer_text = response.choices[0].message.content

            # Try to parse as JSON
            try:
                answer_json = json.loads(answer_text)
            except json.JSONDecodeError:
                # If not valid JSON, wrap it
                answer_json = {"answer": answer_text}

            # Step 9: Add metrics and RAG scores to response
            result = {
                **answer_json,
                "metrics": tracker.get_summary_metrics()
            }

            # Add RAG information if available
            if rag_context is not None:
                result["rag"] = {
                    "context_used": True,
                    "retrieval_score": rag_score,
                    "context_length": len(rag_context),
                    "context_preview": rag_context[:200] + "..." if len(rag_context) > 200 else rag_context
                }

            # Step 10: Log successful metrics to CSV (enhanced from M1)
            log_metrics_from_tracker(
                tracker,
                prompt_technique=prompt_technique,
                success=True,
                rag_used=rag_context is not None,
                rag_score=rag_score
            )

            return result

        except Exception as e:
            # Handle any errors during LLM processing
            tracker.stop()
            log_error(
                error_type="LLMProcessingError",
                error_message=str(e),
                model=self.model,
                user_question=user_question
            )

            log_metrics_from_tracker(
                tracker,
                prompt_technique=prompt_technique,
                success=False,
                rag_used=rag_context is not None,
                rag_score=rag_score
            )

            return {
                "error": f"Processing failed: {str(e)}",
                "metrics": tracker.get_summary_metrics(),
                "rag": {
                    "context_used": rag_context is not None,
                    "retrieval_score": rag_score,
                    "context_length": len(rag_context) if rag_context else 0,
                    "context_preview": rag_context[:200] + "..." if rag_context and len(rag_context) > 200 else rag_context
                } if rag_context is not None else None
            }

    async def process_uploaded_document(
        self,
        file_content: str,
        filename: str,
        store_embeddings: bool = True
    ) -> Dict:
        """
        Process an uploaded document through the RAG pipeline.

        Args:
            file_content: Content of the uploaded file
            filename: Name of the uploaded file
            store_embeddings: Whether to store embeddings in vector store

        Returns:
            Dictionary with processing results and statistics
        """
        if not self.document_processor:
            raise HenryBotError("Document processor not available")

        try:
            # Process the document using the new RAG processor
            result = await self.document_processor.process_document(
                content=file_content,
                source=filename,
                store_embeddings=store_embeddings
            )

            # Convert to API response format
            return {
                "success": result.success,
                "document_id": result.document_id,
                "filename": result.source,
                "document_type": result.document_type.value,
                "chunks_generated": len(result.chunks),
                "embeddings_generated": result.embeddings is not None,
                "processing_time_ms": result.processing_time_ms,
                "error_message": result.error_message,
                "chunking_strategy": result.chunks[0].metadata.strategy_name if result.chunks else None,
                "total_characters": sum(chunk.metadata.char_count for chunk in result.chunks),
                "total_words": sum(chunk.metadata.word_count for chunk in result.chunks),
            }

        except Exception as e:
            raise HenryBotError(f"Document processing failed: {str(e)}")

    def get_document_processor_stats(self) -> Dict:
        """
        Get statistics from the document processor.

        Returns:
            Dictionary with processing statistics
        """
        if not self.document_processor:
            return {"available": False, "message": "Document processor not initialized"}

        try:
            return {
                "available": True,
                **self.document_processor.get_stats()
            }
        except Exception as e:
            return {
                "available": False,
                "message": f"Failed to get stats: {str(e)}"
            }

    def get_system_status(self) -> Dict:
        """
        Get the current system status including module availability.

        Returns:
            Dictionary with system status information
        """
        return {
            "status": "healthy",
            "model": self.model,
            "rag_available": self.rag_retriever is not None,
            "document_processor_available": self.document_processor is not None,
            "prompting_technique": self.prompting_technique,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
