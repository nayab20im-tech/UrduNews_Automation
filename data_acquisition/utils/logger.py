"""
utils/logger.py
================
One shared, consistently-formatted logger for the whole Data Acquisition
layer. Every module calls `get_logger(__name__)` instead of configuring
logging itself, so log output (console + rotating file) stays uniform and
duplicate handlers never get attached.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .. import config

_CONFIGURED_LOGGERS: set[str] = set()

# Handlers registered via add_global_handler(); attached to every logger
# configured so far and to every logger get_logger() creates from now on.
# Used by Section 6's Logs DB (storage/logs_db.py) to mirror system logs
# into SQLite without touching each call site.
_GLOBAL_HANDLERS: list[logging.Handler] = []

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for `name` (typically __name__)."""
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(config.LOG_LEVEL)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        config.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    for handler in _GLOBAL_HANDLERS:
        logger.addHandler(handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger


def add_global_handler(handler: logging.Handler) -> None:
    """Attach `handler` to every configured logger and to all loggers
    created by `get_logger()` from now on."""
    _GLOBAL_HANDLERS.append(handler)
    for name in _CONFIGURED_LOGGERS:
        logging.getLogger(name).addHandler(handler)
