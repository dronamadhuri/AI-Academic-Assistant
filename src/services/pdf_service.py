import sys
import os
from typing import List, Union
import io

# Ensure root directory is on the path to resolve imports cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_core.documents import Document
from modules.pdf_loader import load_pdf
from modules.text_splitter import split_documents
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PDFService:
    """Service to handle PDF text extraction, document chunking, and metadata parsing."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDFService.
        
        Args:
            chunk_size: Size of characters in each chunk.
            chunk_overlap: Overlap of characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_input: Union[str, bytes, io.BytesIO]) -> str:
        """
        Extracts raw text from a PDF file or stream.
        
        Args:
            pdf_input: Can be a file path (str), raw bytes, or a file-like stream (BytesIO).
            
        Returns:
            Extracted text as a single string.
        """
        logger.info("Extracting text from PDF input via PDFService.")
        pdf_doc = load_pdf(pdf_input)
        return pdf_doc.text

    def load_and_split_pdf(
        self, 
        pdf_input: Union[str, bytes, io.BytesIO], 
        source_name: str = "pdf_document"
    ) -> List[Document]:
        """
        Loads a PDF, builds Document objects per page with metadata, and splits them.
        
        Args:
            pdf_input: Can be a file path (str), raw bytes, or a file-like stream (BytesIO).
            source_name: Reference name of the source (e.g. filename) for metadata tracking.
            
        Returns:
            A list of chunked LangChain Document objects with serializable metadata.
        """
        logger.info(f"Loading and chunking PDF from source: {source_name}")
        pdf_doc = load_pdf(pdf_input)
        
        # Build individual LangChain Documents for each page
        page_documents = []
        for idx, page_text in enumerate(pdf_doc.pages):
            if not page_text.strip():
                continue
                
            # Base metadata to trace search citations
            doc_metadata = {
                "source": source_name,
                "page": idx + 1,
                "total_pages": pdf_doc.page_count,
                "is_scanned": pdf_doc.is_scanned,
            }
            
            # Incorporate PDF-level metadata attributes from PyMuPDF
            for k, v in pdf_doc.metadata.items():
                if isinstance(v, (str, int, float, bool)) and v:
                    doc_metadata[f"pdf_{k}"] = v
            
            page_documents.append(
                Document(page_content=page_text, metadata=doc_metadata)
            )
            
        # Split documents using modules/text_splitter.py
        split_docs = split_documents(
            page_documents, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        logger.info(f"Successfully split PDF into {len(split_docs)} chunks using modules/text_splitter.")
        return split_docs
