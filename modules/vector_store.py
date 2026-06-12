import os
import logging
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    """Custom exception raised for errors during vector store operations."""
    pass

def get_embeddings_model(
    api_key: Optional[str] = None, 
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> HuggingFaceEmbeddings:
    """
    Creates and returns a HuggingFaceEmbeddings model.
    
    Args:
        api_key: Unused parameter kept for backward compatibility.
        model_name: Name of the Hugging Face embedding model.
        
    Returns:
        HuggingFaceEmbeddings instance.
        
    Raises:
        VectorStoreError: If the model name is missing or loading fails.
    """
    if not model_name:
        raise VectorStoreError("Embedding model name must be configured.")
    try:
        logger.info(f"Initializing HuggingFaceEmbeddings model: {model_name}")
        return HuggingFaceEmbeddings(
            model_name=model_name
        )
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFaceEmbeddings: {str(e)}", exc_info=True)
        raise VectorStoreError(f"Embedding initialization error: {str(e)}") from e

def create_vector_store(
    documents: List[Document], 
    embeddings: Embeddings
) -> FAISS:
    """
    Creates a FAISS vector database from a list of documents.
    
    Args:
        documents: List of LangChain Document objects.
        embeddings: Embeddings model instance.
        
    Returns:
        FAISS vector store instance.
        
    Raises:
        VectorStoreError: If validation fails or FAISS initialization fails.
    """
    if not isinstance(documents, list):
        raise VectorStoreError(f"Input 'documents' must be a list, received: {type(documents).__name__}")
    if not documents:
        raise VectorStoreError("Cannot create a vector store from an empty list of documents.")
        
    try:
        logger.info(f"Creating FAISS vector database with {len(documents)} document chunks.")
        db = FAISS.from_documents(documents, embeddings)
        logger.info("Successfully created FAISS vector store.")
        return db
    except Exception as e:
        logger.error(f"Failed to create FAISS vector store: {str(e)}", exc_info=True)
        raise VectorStoreError(f"Vector store creation error: {str(e)}") from e

def save_vector_store(
    db: FAISS, 
    folder_path: str, 
    index_name: str = "index"
) -> None:
    """
    Saves a FAISS vector database locally to disk.
    
    Args:
        db: FAISS vector database instance.
        folder_path: Folder path where files will be stored.
        index_name: Subfolder or index file prefix name.
        
    Raises:
        VectorStoreError: If directory creation or saving fails.
    """
    if not isinstance(db, FAISS):
        raise VectorStoreError(f"Input 'db' must be a FAISS instance, received: {type(db).__name__}")
        
    try:
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Saving FAISS index locally to folder '{folder_path}' with name '{index_name}'.")
        db.save_local(folder_path, index_name)
        logger.info("Successfully saved FAISS index locally.")
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {str(e)}", exc_info=True)
        raise VectorStoreError(f"Vector store save error: {str(e)}") from e

def load_vector_store(
    folder_path: str, 
    embeddings: Embeddings, 
    index_name: str = "index"
) -> Optional[FAISS]:
    """
    Loads a local FAISS index from disk.
    
    Args:
        folder_path: Folder path where files are stored.
        embeddings: Embeddings model instance.
        index_name: Subfolder or index file prefix name.
        
    Returns:
        Loaded FAISS vector store, or None if files do not exist.
        
    Raises:
        VectorStoreError: If index deserialization or loading fails.
    """
    faiss_file = os.path.join(folder_path, f"{index_name}.faiss")
    if not os.path.exists(faiss_file):
        logger.warning(f"No FAISS index files found in '{folder_path}' with name '{index_name}'.")
        return None
        
    try:
        logger.info(f"Loading FAISS index from folder '{folder_path}' with name '{index_name}'.")
        db = FAISS.load_local(
            folder_path, 
            embeddings, 
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
        logger.info("Successfully loaded FAISS index.")
        return db
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {str(e)}", exc_info=True)
        raise VectorStoreError(f"Vector store load error: {str(e)}") from e
