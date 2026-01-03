from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(*, state_dir: Path) -> None:
    """Configure daemon/control-plane logging.

    Uvicorn config is environment-dependent; we add a stable Redesmyn logger that:
    - always logs to stderr, and
    - writes a rotating file under the repo state dir for post-mortem debugging.
    """

    log_dir = state_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "redesmyn.log"

    logger = logging.getLogger("redesmyn")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        stream = logging.StreamHandler()
        stream.setLevel(logging.INFO)
        stream.setFormatter(formatter)
        logger.addHandler(stream)

    if not any(
        isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == log_path
        for h in logger.handlers
    ):
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Avoid double logging when uvicorn config attaches handlers upstream.
    logger.propagate = False
