import sys
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# Ensure root directory is on the path to resolve imports cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.vector_store import get_embeddings_model
from modules.llm_service import GeminiClient
from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

class LLMService:
    """Service to wrapper Google Gemini API interactions using LangChain components."""

    def __init__(self) -> None:
        """Initialize LLMService by checking configurations."""
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY is not configured in the environment.")

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Retrieves the Hugging Face Embeddings generator model instance.
        
        Returns:
            HuggingFaceEmbeddings instance.
        """
        logger.info("Initializing HuggingFaceEmbeddings model via LLMService.")
        return get_embeddings_model(
            api_key=settings.gemini_api_key,
            model_name=settings.embedding_model
        )

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """
        Retrieves the Google Gemini Chat LLM model instance.
        
        Returns:
            ChatGoogleGenerativeAI instance.
        """
        logger.info("Initializing ChatGoogleGenerativeAI model.")
        client = GeminiClient(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model
        )
        return client.llm

    def execute_qa_chain(
        self, 
        vector_store: FAISS, 
        query: str, 
        prompt_template: PromptTemplate
    ) -> str:
        """
        Executes a RetrievalQA chain using the Gemini LLM and FAISS retriever.
        
        Args:
            vector_store: Instantiated FAISS vector database.
            query: The user question.
            prompt_template: The LangChain PromptTemplate to apply.
            
        Returns:
            Response text from the LLM.
        """
        logger.info(f"Executing RetrievalQA chain for query: {query}")
        try:
            llm = self.get_llm()
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt_template}
            )
            response = qa_chain.invoke({"query": query})
            return response.get("result", "")
        except Exception as e:
            logger.error(f"Error executing QA retrieval chain: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to execute QA chain: {str(e)}") from e

    def execute_summarize(self, text: str, prompt_template: PromptTemplate) -> str:
        """
        Executes an LLM summary call using the custom modules/llm_service wrapper.
        
        Args:
            text: Scientific text chunk to summarize.
            prompt_template: Summarization instruction template.
            
        Returns:
            Summarized text from the LLM.
        """
        logger.info("Running summarization template via GeminiClient.")
        try:
            client = GeminiClient(
                api_key=settings.gemini_api_key,
                model_name=settings.gemini_model
            )
            return client.execute_prompt(prompt_template, {"text": text})
        except Exception as e:
            logger.error(f"Error during academic summarization: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to summarize text: {str(e)}") from e
