import sys
import os
from typing import List, Optional

# Ensure root directory is on the path to resolve imports cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from modules.vector_store import create_vector_store, save_vector_store, load_vector_store
from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

class VectorService:
    """Service to manage the lifecycle and operations of the FAISS vector database."""

    def __init__(self, embeddings: Embeddings):
        """
        Initialize the VectorService.
        
        Args:
            embeddings: An instance of LangChain Embeddings.
        """
        self.embeddings = embeddings

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Creates a new FAISS vector database from a list of chunked documents.
        
        Args:
            documents: List of LangChain Document objects.
            
        Returns:
            A FAISS vector store instance.
        """
        logger.info("Creating FAISS vector database via VectorService.")
        return create_vector_store(documents, self.embeddings)

    def save_vector_store(self, vector_store: FAISS, index_name: str = "faiss_index") -> None:
        """
        Persists the FAISS index files locally to data/vector_store.
        
        Args:
            vector_store: The FAISS instance to save.
            index_name: Name of the folder/index file to save.
        """
        logger.info("Saving FAISS vector database via VectorService.")
        save_vector_store(vector_store, settings.vector_store_dir, index_name)

    def load_vector_store(self, index_name: str = "faiss_index") -> Optional[FAISS]:
        """
        Loads a persisted FAISS vector index from local disk.
        
        Args:
            index_name: Name of the folder/index file to load.
            
        Returns:
            A FAISS vector store instance, or None if it does not exist.
        """
        logger.info("Loading FAISS vector database via VectorService.")
        return load_vector_store(settings.vector_store_dir, self.embeddings, index_name)

    def merge_vector_stores(self, db1: FAISS, db2: FAISS) -> FAISS:
        """
        Merges the second FAISS database into the first FAISS database.
        
        Args:
            db1: The target FAISS vector store to merge into.
            db2: The source FAISS vector store to extract vectors from.
            
        Returns:
            The merged FAISS vector store.
        """
        logger.info("Merging FAISS databases via VectorService.")
        db1.merge_from(db2)
        return db1
