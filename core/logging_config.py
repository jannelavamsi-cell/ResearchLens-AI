"""
Logging configuration for ResearchLens AI.
Provides clean console output and persistent file logging while preventing
accidental leaking of full document text into logs at INFO level.
"""

import logging
import sys
from pathlib import Path

DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_filename: str = "researchlens.log",
) -> logging.Logger:
    """Configures structured root logging with file and stream handlers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file_path = Path(log_dir) / log_filename

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger("researchlens")
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Persistent File Handler
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Suppress overly chatty external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Gets a named child logger under the researchlens namespace."""
    return logging.getLogger(f"researchlens.{name}")
