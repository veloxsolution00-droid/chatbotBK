"""Shared durable storage initialization and persistent logging."""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from app.core.config import settings

_initialized = False

def initialize_storage():
    global _initialized
    paths = [settings.uploads_path, settings.extracted_path, settings.chroma_path,
             settings.metadata_path, settings.logs_path, Path(settings.sqlite_path).parent]
    for value in paths:
        Path(value).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    if not _initialized:
        logger = logging.getLogger("document_chatbot")
        logger.setLevel(logging.INFO)
        if not any(getattr(h, "_document_chatbot_handler", False) for h in logger.handlers):
            handler = RotatingFileHandler(Path(settings.logs_path) / "application.log", maxBytes=5_000_000,
                                          backupCount=5, encoding="utf-8")
            handler._document_chatbot_handler = True
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        _initialized = True
    return logging.getLogger("document_chatbot")

def logger():
    initialize_storage()
    return logging.getLogger("document_chatbot")
