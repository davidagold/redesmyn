from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog
from structlog.contextvars import merge_contextvars
from structlog.processors import JSONRenderer, TimeStamper, format_exc_info
from structlog.stdlib import ProcessorFormatter, add_logger_name


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

    pre_chain = [
        merge_contextvars,
        structlog.stdlib.add_log_level,
        add_logger_name,
        TimeStamper(fmt="iso", utc=True),
    ]

    # Configure structlog once (idempotent-ish).
    structlog.configure(
        processors=[
            *pre_chain,
            structlog.processors.StackInfoRenderer(),
            format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    console_formatter = ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
        foreign_pre_chain=pre_chain,
    )

    file_formatter = ProcessorFormatter(
        processor=JSONRenderer(sort_keys=True),
        foreign_pre_chain=pre_chain,
    )

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        stream = logging.StreamHandler()
        stream.setLevel(logging.INFO)
        stream.setFormatter(console_formatter)
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
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Avoid double logging when uvicorn config attaches handlers upstream.
    logger.propagate = False
