import os
import logging
from typing import Any, Dict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class LLMServiceError(Exception):
    """Custom exception raised for errors during LLM service operations."""
    pass

class GeminiClient:
    """Reusable wrapper for interacting with the Google Gemini API using LangChain."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        **kwargs: Any
    ) -> None:
        """
        Initializes the Gemini Client.
        
        Args:
            api_key: Google Gemini API key. Defaults to GEMINI_API_KEY environment variable.
            model_name: Name of the Gemini model (default: 'gemini-1.5-flash').
            temperature: Sampling temperature.
            **kwargs: Additional arguments passed to ChatGoogleGenerativeAI.
            
        Raises:
            LLMServiceError: If the API key is missing.
        """
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMServiceError(
                "GEMINI_API_KEY must be provided or configured as an environment variable."
            )
            
        self.model_name = model_name
        self.temperature = temperature
        
        try:
            logger.info(f"Initializing ChatGoogleGenerativeAI model: {model_name}")
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=key,
                temperature=temperature,
                **kwargs
            )
            logger.info("GeminiClient initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to create ChatGoogleGenerativeAI instance: {str(e)}", exc_info=True)
            raise LLMServiceError(f"LLM Client initialization error: {str(e)}") from e

    def execute_prompt(
        self,
        prompt_template: PromptTemplate,
        input_variables: Dict[str, Any]
    ) -> str:
        """
        Formats and runs a LangChain prompt template through Gemini and returns the string response.
        
        Args:
            prompt_template: A LangChain PromptTemplate instance.
            input_variables: Dictionary mapping placeholders to values.
            
        Returns:
            The generated response string.
            
        Raises:
            LLMServiceError: If the execution fails.
        """
        if not isinstance(prompt_template, PromptTemplate):
            raise LLMServiceError(
                f"prompt_template must be a LangChain 'PromptTemplate', "
                f"received: {type(prompt_template).__name__}"
            )
            
        try:
            logger.info(f"Executing prompt template on model '{self.model_name}'")
            chain = prompt_template | self.llm | StrOutputParser()
            response = chain.invoke(input_variables)
            return response
        except Exception as e:
            logger.error(f"Failed to execute prompt: {str(e)}", exc_info=True)
            raise LLMServiceError(f"Error during prompt execution: {str(e)}") from e

    def execute_messages(self, messages: List[Any]) -> str:
        """
        Executes a raw list of chat messages.
        
        Args:
            messages: List of BaseMessage objects.
            
        Returns:
            Generated response content.
            
        Raises:
            LLMServiceError: If execution fails.
        """
        try:
            logger.info("Invoking Gemini with messages list.")
            response = self.llm.invoke(messages)
            return str(response.content)
        except Exception as e:
            logger.error(f"Failed to execute messages: {str(e)}", exc_info=True)
            raise LLMServiceError(f"Error during chat messages invocation: {str(e)}") from e
