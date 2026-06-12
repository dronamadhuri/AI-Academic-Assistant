import sys
import os
from typing import List, Dict, Any

# Ensure root directory is on the path to resolve imports cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.llm_service import GeminiClient
from src.prompts.templates import PPT_GENERATION_PROMPT
from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

class PPTServiceError(Exception):
    """Custom exception raised for errors during PPT service operations."""
    pass

class PPTService:
    """Service to handle the creation and structured parsing of presentation slides using Gemini."""

    def __init__(self, api_key: str = None) -> None:
        """
        Initialize PPTService.
        
        Args:
            api_key: Optional API key override. Defaults to value in settings.
        """
        self.api_key = api_key or settings.gemini_api_key
        if not self.api_key:
            logger.warning("PPTService initialized without a configured GEMINI_API_KEY.")

    def parse_ppt_response(self, text: str) -> List[Dict[str, Any]]:
        """
        Parses structured slide response text using a state-machine parser.
        Handles titles, list bullets, and multi-line speaker notes.
        
        Args:
            text: Raw output string from LLM.
            
        Returns:
            List of dictionaries containing parsed slide attributes (title, bullets, speaker_notes).
        """
        text = text.replace("\r\n", "\n")
        lines = text.split("\n")
        
        slides: List[Dict[str, Any]] = []
        current_slide: Dict[str, Any] = {}
        current_state = None
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            # Check for header tag identifiers
            if stripped.upper().startswith("SLIDE_TITLE:"):
                if current_slide and "title" in current_slide:
                    slides.append(current_slide)
                current_slide = {
                    "title": stripped[len("SLIDE_TITLE:"):].strip(),
                    "bullets": [],
                    "speaker_notes": ""
                }
                current_state = "title"
            elif not current_slide:
                continue
            elif stripped.upper().startswith("BULLET_POINTS:"):
                current_state = "bullet_points"
            elif stripped.upper().startswith("SPEAKER_NOTES:"):
                current_slide["speaker_notes"] = stripped[len("SPEAKER_NOTES:"):].strip()
                current_state = "speaker_notes"
            else:
                # Append to current tag section if multi-line content
                if current_state == "title":
                    current_slide["title"] += " " + stripped
                elif current_state == "bullet_points":
                    if stripped.startswith("-") or stripped.startswith("*"):
                        bullet_content = stripped[1:].strip()
                        if bullet_content:
                            current_slide["bullets"].append(bullet_content)
                    else:
                        if current_slide["bullets"]:
                            current_slide["bullets"][-1] += " " + stripped
                        else:
                            current_slide["bullets"].append(stripped)
                elif current_state == "speaker_notes":
                    current_slide["speaker_notes"] += "\n" + stripped
                    
        # Append final slide block
        if current_slide and "title" in current_slide:
            slides.append(current_slide)
            
        return slides

    def generate_slides(self, pdf_text: str) -> List[Dict[str, Any]]:
        """
        Generates exactly 10 presentation slides from PDF text using Gemini.
        
        Args:
            pdf_text: Cleaned text extracted from the PDF paper.
            
        Returns:
            List of parsed slide dictionary elements.
            
        Raises:
            PPTServiceError: If input text is empty or API execution fails.
        """
        if not pdf_text or not pdf_text.strip():
            raise PPTServiceError("Cannot generate presentation slides from empty text content.")
            
        if not self.api_key:
            raise PPTServiceError("GEMINI_API_KEY must be configured to use PPTService.")

        logger.info(f"Generating presentation slides for text content (length={len(pdf_text)}).")
        try:
            # Reusable Gemini client
            client = GeminiClient(
                api_key=self.api_key,
                model_name=settings.gemini_model,
                temperature=0.4  # Slightly creative for design layout
            )
            
            # Execute prompt template
            raw_response = client.execute_prompt(
                prompt_template=PPT_GENERATION_PROMPT,
                input_variables={"text": pdf_text}
            )
            
            logger.info("PPT slide prompt execution completed. Parsing output blocks.")
            parsed_slides = self.parse_ppt_response(raw_response)
            logger.info(f"Successfully parsed {len(parsed_slides)} presentation slides.")
            return parsed_slides
            
        except Exception as e:
            logger.error(f"PPT generation service failed: {str(e)}", exc_info=True)
            raise PPTServiceError(f"Error executing PPT generator service: {str(e)}") from e
