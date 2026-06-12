import re
import logging
from typing import Dict, Optional
from modules.llm_service import GeminiClient
from src.config import get_settings
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)
settings = get_settings()

CITATION_EXTRACT_TEMPLATE = """You are an expert academic metadata extraction assistant.
Analyze the following text snippet from the beginning of a scientific publication and extract the citation details.
If any value is missing or cannot be inferred, write "Unknown".

Snippet:
{snippet}

Respond EXACTLY in the following format:
TITLE: [Inferred Title]
AUTHORS: [Inferred Authors, separated by commas]
YEAR: [Inferred Publication Year, e.g. 2026]
PUBLISHER: [Inferred Journal, Conference, or Publisher name]
"""

CITATION_EXTRACT_PROMPT = PromptTemplate(
    template=CITATION_EXTRACT_TEMPLATE,
    input_variables=["snippet"]
)

class CitationService:
    """Service to extract publication metadata and compile standard APA/IEEE academic citations."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize CitationService.
        
        Args:
            api_key: Optional Google Gemini API key override.
        """
        self.api_key = api_key or settings.gemini_api_key

    def extract_metadata(self, pdf_text: str, filename: str, fallback_metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Extracts document details (title, authors, year, publisher) using Gemini analysis or fallbacks.
        
        Args:
            pdf_text: Raw or clean document text.
            filename: Name of the uploaded PDF file.
            fallback_metadata: Dictionary of PDF headers from PyMuPDF.
            
        Returns:
            Dict containing keys: title, authors, year, publisher.
        """
        meta = {
            "title": "Unknown Title",
            "authors": "Unknown Authors",
            "year": "2026",
            "publisher": "Academic publication"
        }
        
        # Determine fallback values from PDF metadata
        if fallback_metadata.get("title") and len(fallback_metadata.get("title").strip()) > 3:
            meta["title"] = fallback_metadata["title"].strip()
        else:
            # Clean filename to act as title fallback
            cleaned_filename = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
            meta["title"] = cleaned_filename.replace("_", " ").replace("-", " ").title()
            
        if fallback_metadata.get("author"):
            meta["authors"] = fallback_metadata["author"].strip()
            
        # Try to infer year from date metadata or filename
        year_match = re.search(r"\b(19|20)\d{2}\b", filename)
        if year_match:
            meta["year"] = year_match.group(0)
        elif fallback_metadata.get("creationDate"):
            # Format: D:YYYYMMDD...
            date_str = fallback_metadata["creationDate"]
            if date_str.startswith("D:") and len(date_str) >= 6:
                meta["year"] = date_str[2:6]

        # Use Gemini for smart extraction from first page (snippet)
        if self.api_key and pdf_text.strip():
            try:
                # Take first 1500 characters containing title page text
                snippet = pdf_text[:1500].strip()
                logger.info("Extracting citation metadata using Gemini.")
                client = GeminiClient(
                    api_key=self.api_key,
                    model_name=settings.gemini_model,
                    temperature=0.0
                )
                
                response = client.execute_prompt(
                    prompt_template=CITATION_EXTRACT_PROMPT,
                    input_variables={"snippet": snippet}
                )
                
                # Parse key-values from response
                lines = response.split("\n")
                for line in lines:
                    if line.startswith("TITLE:"):
                        val = line.replace("TITLE:", "").strip()
                        if val.lower() != "unknown" and len(val) > 4:
                            meta["title"] = val
                    elif line.startswith("AUTHORS:"):
                        val = line.replace("AUTHORS:", "").strip()
                        if val.lower() != "unknown" and len(val) > 2:
                            meta["authors"] = val
                    elif line.startswith("YEAR:"):
                        val = line.replace("YEAR:", "").strip()
                        if val.lower() != "unknown" and val.isdigit():
                            meta["year"] = val
                    elif line.startswith("PUBLISHER:"):
                        val = line.replace("PUBLISHER:", "").strip()
                        if val.lower() != "unknown" and len(val) > 3:
                            meta["publisher"] = val
                            
            except Exception as e:
                logger.warning(f"Gemini citation metadata extraction failed: {str(e)}. Falling back to file metadata.")
                
        return meta

    def generate_citations(self, metadata: Dict[str, str]) -> Dict[str, str]:
        """
        Compiles APA and IEEE formatted citation reference strings.
        
        Args:
            metadata: Extracted dictionary containing title, authors, year, publisher.
            
        Returns:
            Dict containing keys: 'apa', 'ieee'.
        """
        title = metadata.get("title", "Unknown Title").strip()
        authors = metadata.get("authors", "Unknown Authors").strip()
        year = metadata.get("year", "2026").strip()
        publisher = metadata.get("publisher", "Academic Publication").strip()
        
        # Split into individual author names
        if "," in authors:
            parts = [p.strip() for p in authors.split(",")]
        else:
            parts = [authors.strip()]
            
        # Formatting APA:
        cleaned_parts_apa = []
        for part in parts:
            name_words = part.split()
            if len(name_words) > 1:
                last = name_words[-1]
                first_initials = "".join([f" {w[0]}." for w in name_words[:-1]])
                cleaned_parts_apa.append(f"{last},{first_initials}")
            else:
                cleaned_parts_apa.append(part)
        
        if len(cleaned_parts_apa) > 1:
            apa_authors = ", & ".join(cleaned_parts_apa)
        else:
            apa_authors = cleaned_parts_apa[0]
            
        apa_citation = f"{apa_authors} ({year}). {title}. {publisher}."
        
        # Formatting IEEE:
        cleaned_parts_ieee = []
        for part in parts:
            name_words = part.split()
            if len(name_words) > 1:
                last = name_words[-1]
                first_initials = "".join([f"{w[0]}." for w in name_words[:-1]])
                cleaned_parts_ieee.append(f"{first_initials} {last}")
            else:
                cleaned_parts_ieee.append(part)
                
        if len(cleaned_parts_ieee) > 1:
            ieee_authors = ", ".join(cleaned_parts_ieee[:-1]) + " and " + cleaned_parts_ieee[-1]
        else:
            ieee_authors = cleaned_parts_ieee[0]
            
        ieee_citation = f'{ieee_authors}, "{title}," {publisher}, {year}.'
        
        return {
            "apa": apa_citation,
            "ieee": ieee_citation
        }
