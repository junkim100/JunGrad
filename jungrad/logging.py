"""Logging utilities for jungrad."""

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a configured logger for the given name.

    Args:
        name: Logger name. If None, returns root logger.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name or "jungrad")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
    return logger
