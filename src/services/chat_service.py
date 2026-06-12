import sys
import os
from typing import Dict, Any, List
from langchain_community.vectorstores import FAISS

# Ensure root directory is on the path to resolve imports cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.llm_service import GeminiClient
from src.prompts.templates import CHAT_RAG_PROMPT
from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

class ChatServiceError(Exception):
    """Custom exception raised for errors during ChatService execution."""
    pass

class ChatService:
    """Service to orchestrate RAG chat query pipelines over uploaded PDF vectors."""

    def __init__(self, api_key: str = None) -> None:
        """
        Initialize ChatService.
        
        Args:
            api_key: Optional API key override. Defaults to value in settings.
        """
        self.api_key = api_key or settings.gemini_api_key
        if not self.api_key:
            logger.warning("ChatService initialized without a configured GEMINI_API_KEY.")

    def query(self, query: str, vector_store: FAISS, k: int = 4) -> str:
        """
        Retrieves context chunks from FAISS and runs Gemini with the strict RAG prompt.
        
        Args:
            query: The user's chat question.
            vector_store: Instantiated FAISS vector store database.
            k: Number of similarity documents to retrieve.
            
        Returns:
            The parsed answer from the LLM or "Information not found in uploaded document."
            
        Raises:
            ChatServiceError: If input parameters or operations fail.
        """
        if not query or not query.strip():
            raise ChatServiceError("User query cannot be empty.")
            
        if not vector_store:
            raise ChatServiceError("Vector store must be provided for RAG retrieval.")
            
        if not self.api_key:
            raise ChatServiceError("GEMINI_API_KEY must be configured to run RAG query.")

        logger.info(f"Initiating RAG chat query: '{query}' (k={k})")
        try:
            # Step 1: Perform similarity search on FAISS index
            retrieved_docs = vector_store.similarity_search(query, k=k)
            
            if not retrieved_docs:
                logger.info("No relevant context found in FAISS. Returning default message.")
                return "Information not found in uploaded document."
                
            # Step 2: Build context block
            context_blocks = []
            for doc in retrieved_docs:
                page_num = doc.metadata.get("page", "N/A")
                content = doc.page_content.strip()
                if content:
                    context_blocks.append(f"[Page {page_num}]: {content}")
                    
            if not context_blocks:
                logger.info("Retrieved documents contain empty content. Returning default message.")
                return "Information not found in uploaded document."
                
            full_context = "\n\n".join(context_blocks)
            
            # Step 3: Initialize reusable Gemini client
            client = GeminiClient(
                api_key=self.api_key,
                model_name=settings.gemini_model,
                temperature=0.0  # Keep answering extremely factual
            )
            
            # Step 4: Execute prompt template
            logger.info("Executing chat RAG prompt template via Gemini.")
            response = client.execute_prompt(
                prompt_template=CHAT_RAG_PROMPT,
                input_variables={
                    "context": full_context,
                    "question": query
                }
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"RAG chat query execution failed: {str(e)}", exc_info=True)
            raise ChatServiceError(f"Error executing RAG chat query: {str(e)}") from e
