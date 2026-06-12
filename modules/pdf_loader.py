import io
import logging
import re
from typing import Union, Dict, Any, List
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

class PDFLoaderError(Exception):
    """Custom exception raised for errors during PDF loading and parsing."""
    pass

class PDFDocument:
    """Represents a successfully parsed PDF document and its extracted content."""
    
    def __init__(
        self,
        text: str,
        pages: List[str],
        metadata: Dict[str, Any],
        is_scanned: bool,
        is_empty: bool
    ):
        self.text = text
        self.pages = pages
        self.metadata = metadata
        self.is_scanned = is_scanned
        self.is_empty = is_empty
        self.page_count = len(pages)

    def __repr__(self) -> str:
        return (
            f"<PDFDocument pages={self.page_count} "
            f"is_scanned={self.is_scanned} is_empty={self.is_empty}>"
        )


def clean_text(text: str) -> str:
    """
    Cleans extracted text by normalizing whitespace and removing control characters.
    
    Args:
        text: Raw extracted text.
        
    Returns:
        Cleaned and normalized text.
    """
    if not text:
        return ""
    
    # Replace null bytes and other common non-printable control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]", "", text)
    
    # Normalize different newline styles to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Replace multiple consecutive spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)
    
    # Remove spacing around line breaks, but keep line breaks
    text = re.sub(r" * \n *", "\n", text)
    
    # Replace more than two consecutive newlines with exactly two newlines (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def load_pdf(
    pdf_input: Union[str, bytes, io.BytesIO],
    min_char_threshold_per_page: int = 50
) -> PDFDocument:
    """
    Loads a PDF document and extracts its text contents using PyMuPDF.
    
    Args:
        pdf_input: Can be a file path (str), raw bytes, or a file-like stream (BytesIO).
        min_char_threshold_per_page: Threshold characters per page to detect scanned/empty documents.
        
    Returns:
        A PDFDocument object containing cleaned text, page breakdowns, and status.
        
    Raises:
        PDFLoaderError: If PyMuPDF fails to open or parse the document.
    """
    doc = None
    try:
        if isinstance(pdf_input, str):
            logger.info(f"Opening PDF file from path: {pdf_input}")
            doc = fitz.open(pdf_input)
        elif isinstance(pdf_input, bytes):
            logger.info("Opening PDF file from bytes stream.")
            doc = fitz.open(stream=pdf_input, filetype="pdf")
        elif hasattr(pdf_input, "read"):
            logger.info("Opening PDF file from file-like stream.")
            # Read bytes from file-like object (e.g. BytesIO)
            content = pdf_input.read()
            # If the stream pointer was already at the end, make sure to handle it
            if not content and hasattr(pdf_input, "seek"):
                pdf_input.seek(0)
                content = pdf_input.read()
            doc = fitz.open(stream=content, filetype="pdf")
        else:
            raise PDFLoaderError(
                f"Unsupported PDF input type: {type(pdf_input)}. "
                f"Must be file path (str), bytes, or file-like object."
            )

        page_count = len(doc)
        logger.info(f"Successfully opened PDF with {page_count} pages.")
        
        # Standard document metadata from PyMuPDF
        meta = doc.metadata or {}
        
        pages_text: List[str] = []
        total_chars = 0
        pages_with_significant_text = 0
        
        for page_idx, page in enumerate(doc):
            raw_page_text = page.get_text("text")
            cleaned_page_text = clean_text(raw_page_text)
            pages_text.append(cleaned_page_text)
            
            char_count = len(cleaned_page_text)
            total_chars += char_count
            
            if char_count >= min_char_threshold_per_page:
                pages_with_significant_text += 1
                
            logger.debug(f"Page {page_idx + 1}/{page_count}: Extracted {char_count} characters.")

        # Determine if document is empty or scanned
        is_empty = total_chars == 0
        
        # A PDF is considered scanned if it has pages but very few contain searchable text
        is_scanned = False
        if page_count > 0 and not is_empty:
            # If less than 20% of pages contain significant text, flag as scanned
            ratio = pages_with_significant_text / page_count
            if ratio < 0.2:
                is_scanned = True
                logger.warning(
                    f"PDF flagged as potentially scanned/image-only. "
                    f"Only {pages_with_significant_text}/{page_count} pages contain "
                    f"at least {min_char_threshold_per_page} characters of text."
                )
        elif is_empty:
            is_scanned = True  # Empty PDF has no text; behaves like a scanned document without OCR
            logger.warning("PDF contains no extractable text. It may be empty or scanned.")

        full_cleaned_text = "\n\n".join(pages_text)
        
        return PDFDocument(
            text=full_cleaned_text,
            pages=pages_text,
            metadata=meta,
            is_scanned=is_scanned,
            is_empty=is_empty
        )

    except Exception as e:
        logger.error(f"Error occurred while processing PDF: {str(e)}", exc_info=True)
        raise PDFLoaderError(f"Failed to load and parse PDF: {str(e)}") from e
        
    finally:
        if doc is not None:
            doc.close()
            logger.info("PDF document handle closed.")
