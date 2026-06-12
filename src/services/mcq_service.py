import sys
import os
import re
from typing import List, Dict, Any, Union

# Ensure root directory is on the path to resolve imports cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.llm_service import GeminiClient
from src.prompts.templates import MCQ_QUESTIONS_PROMPT
from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

class MCQServiceError(Exception):
    """Custom exception raised for errors during MCQ service operations."""
    pass

class MCQService:
    """Service to handle the creation and structured parsing of multiple choice questions (MCQs) using Gemini."""

    def __init__(self, api_key: str = None) -> None:
        """
        Initialize MCQService.
        
        Args:
            api_key: Optional API key override. Defaults to value in settings.
        """
        self.api_key = api_key or settings.gemini_api_key
        if not self.api_key:
            logger.warning("MCQService initialized without a configured GEMINI_API_KEY.")

    def parse_mcq_response(self, text: str) -> List[Dict[str, Any]]:
        """
        Parses structured MCQ response text using a state-machine parser.
        Handles multi-line text, tags, options, correct answers, and explanations.
        
        Args:
            text: Raw output string from LLM.
            
        Returns:
            List of dictionaries containing parsed MCQ attributes.
        """
        text = text.replace("\r\n", "\n")
        lines = text.split("\n")
        
        questions: List[Dict[str, Any]] = []
        current_q: Dict[str, Any] = {}
        current_state = None
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            # Check for header tag identifiers
            if stripped.upper().startswith("QUESTION:"):
                if current_q and "question" in current_q:
                    questions.append(current_q)
                current_q = {
                    "question": stripped[len("QUESTION:"):].strip(),
                    "options": {},
                    "answer": "",
                    "explanation": ""
                }
                current_state = "question"
            elif not current_q:
                continue
            elif stripped.upper().startswith("A)"):
                current_q["options"]["A"] = stripped[2:].strip()
                current_state = "option_a"
            elif stripped.upper().startswith("B)"):
                current_q["options"]["B"] = stripped[2:].strip()
                current_state = "option_b"
            elif stripped.upper().startswith("C)"):
                current_q["options"]["C"] = stripped[2:].strip()
                current_state = "option_c"
            elif stripped.upper().startswith("D)"):
                current_q["options"]["D"] = stripped[2:].strip()
                current_state = "option_d"
            elif stripped.upper().startswith("ANSWER:"):
                current_q["answer"] = stripped[len("ANSWER:"):].strip().upper()
                current_state = "answer"
            elif stripped.upper().startswith("EXPLANATION:"):
                current_q["explanation"] = stripped[len("EXPLANATION:"):].strip()
                current_state = "explanation"
            else:
                # Append to current tag section if multi-line content
                if current_state == "question":
                    current_q["question"] += " " + stripped
                elif current_state == "option_a":
                    current_q["options"]["A"] += " " + stripped
                elif current_state == "option_b":
                    current_q["options"]["B"] += " " + stripped
                elif current_state == "option_c":
                    current_q["options"]["C"] += " " + stripped
                elif current_state == "option_d":
                    current_q["options"]["D"] += " " + stripped
                elif current_state == "explanation":
                    current_q["explanation"] += "\n" + stripped
                    
        # Append final question block
        if current_q and "question" in current_q:
            questions.append(current_q)
            
        return questions

    def generate_mcqs(self, pdf_text: str) -> List[Dict[str, Any]]:
        """
        Generates exactly 25 multiple choice questions from PDF text using Gemini.
        
        Args:
            pdf_text: Cleaned text extracted from the PDF paper.
            
        Returns:
            List of parsed MCQ dictionary elements.
            
        Raises:
            MCQServiceError: If input text is empty or API execution fails.
        """
        if not pdf_text or not pdf_text.strip():
            raise MCQServiceError("Cannot generate MCQs from empty text content.")
            
        if not self.api_key:
            raise MCQServiceError("GEMINI_API_KEY must be configured to use MCQService.")

        logger.info(f"Generating academic Multiple Choice Questions for text content (length={len(pdf_text)}).")
        try:
            # Reusable Gemini client
            client = GeminiClient(
                api_key=self.api_key,
                model_name=settings.gemini_model,
                temperature=0.3  # Structured and factual
            )
            
            # Execute prompt template
            raw_response = client.execute_prompt(
                prompt_template=MCQ_QUESTIONS_PROMPT,
                input_variables={"text": pdf_text}
            )
            
            logger.info("MCQ prompt execution completed. Parsing output blocks.")
            parsed_questions = self.parse_mcq_response(raw_response)
            logger.info(f"Successfully parsed {len(parsed_questions)} Multiple Choice Questions.")
            return parsed_questions
            
        except Exception as e:
            logger.error(f"MCQ generation service failed: {str(e)}", exc_info=True)
            raise MCQServiceError(f"Error executing MCQ generator service: {str(e)}") from e
