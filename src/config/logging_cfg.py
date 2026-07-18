import logging
import sys

def setup_logging():
    """Configure structured logging for the application."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Simple formatter for development and production logs
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers = []
        
    logger.addHandler(handler)
