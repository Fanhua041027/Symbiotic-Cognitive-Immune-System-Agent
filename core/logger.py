"""Logging configuration for the immune agent system."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def _resolve_log_level() -> str:
    """Read LOG_LEVEL at call time so config changes take effect without restart."""
    return os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logger(name: str) -> logging.Logger:
    """Set up a logger with consistent formatting."""
    os.makedirs(LOG_DIR, exist_ok=True)

    level_name = _resolve_log_level()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "[%(name)s] %(levelname)s %(message)s",
    ))
    logger.addHandler(console)

    # File handler
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "immune_agent.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(file_handler)

    return logger
