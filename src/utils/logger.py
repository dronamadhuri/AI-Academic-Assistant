import logging
import sys
from src.config import get_settings

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a structured logger instance with custom levels.
    
    Args:
        name: Name of the logger module instance.
        
    Returns:
        Configured logging.Logger object.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger is imported in multiple modules
    if not logger.handlers:
        # Resolve log level dynamically from settings
        try:
            settings = get_settings()
            log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
        except Exception:
            log_level = logging.INFO
            
        logger.setLevel(log_level)
        
        # Configure standard system stdout logging format
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
