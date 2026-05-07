"""Logging setup for file and terminal diagnostics."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from agent.config.settings import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structured-enough local logging with rotation."""

    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.logs_dir / "agent.log"

    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
