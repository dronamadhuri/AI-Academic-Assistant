import logging
from typing import List, Union, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class TextSplitterError(Exception):
    """Custom exception raised for errors during text splitting."""
    pass

def get_text_splitter(
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> RecursiveCharacterTextSplitter:
    """
    Creates and returns a configured RecursiveCharacterTextSplitter instance.
    
    Args:
        chunk_size: Size of characters in each chunk.
        chunk_overlap: Overlap of characters between adjacent chunks.
        
    Returns:
        Configured RecursiveCharacterTextSplitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

def split_text(
    text: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[str]:
    """
    Splits a single string of text into smaller text chunks.
    
    Args:
        text: Raw text string to split.
        chunk_size: Character length limit per chunk.
        chunk_overlap: Shared boundary overlap length.
        
    Returns:
        List of raw string chunks.
        
    Raises:
        TextSplitterError: If input validation fails or splitter engine fails.
    """
    if not isinstance(text, str):
        raise TextSplitterError(
            f"Input 'text' must be of type 'str', received: '{type(text).__name__}'"
        )
        
    if chunk_size <= 0:
        raise TextSplitterError(f"chunk_size must be a positive integer, received: {chunk_size}")
        
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise TextSplitterError(
            f"chunk_overlap must be non-negative and strictly less than chunk_size, "
            f"received: {chunk_overlap} (chunk_size: {chunk_size})"
        )

    try:
        logger.info(f"Splitting raw text (length={len(text)}) with chunk_size={chunk_size}, overlap={chunk_overlap}")
        splitter = get_text_splitter(chunk_size, chunk_overlap)
        chunks = splitter.split_text(text)
        logger.info(f"Generated {len(chunks)} text chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Failed to split text: {str(e)}", exc_info=True)
        raise TextSplitterError(f"Error during text splitting operation: {str(e)}") from e

def split_documents(
    documents: List[Document], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits a list of LangChain Document objects into smaller Document chunks.
    
    Args:
        documents: List of LangChain Document objects.
        chunk_size: Character length limit per chunk.
        chunk_overlap: Shared boundary overlap length.
        
    Returns:
        List of split LangChain Document objects.
        
    Raises:
        TextSplitterError: If document collection validation fails or splitter fails.
    """
    if not isinstance(documents, list):
        raise TextSplitterError(
            f"Input 'documents' must be a list, received: '{type(documents).__name__}'"
        )
        
    for idx, doc in enumerate(documents):
        if not isinstance(doc, Document):
            raise TextSplitterError(
                f"Element at index {idx} in 'documents' list is not a LangChain 'Document' "
                f"object (type: '{type(doc).__name__}')."
            )

    try:
        logger.info(f"Splitting {len(documents)} documents with chunk_size={chunk_size}, overlap={chunk_overlap}")
        splitter = get_text_splitter(chunk_size, chunk_overlap)
        split_docs = splitter.split_documents(documents)
        logger.info(f"Generated {len(split_docs)} document chunks.")
        return split_docs
    except Exception as e:
        logger.error(f"Failed to split documents: {str(e)}", exc_info=True)
        raise TextSplitterError(f"Error during document splitting operation: {str(e)}") from e
