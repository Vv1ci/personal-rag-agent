from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from rag.config import settings


LOGGER_ROOT_NAME = "personal_rag"


def configure_logging() -> logging.Logger:
    """
    Configure one application logger with console and
    rotating-file output.
    """

    application_logger = logging.getLogger(
        LOGGER_ROOT_NAME
    )

    if getattr(
        application_logger,
        "_personal_rag_configured",
        False,
    ):
        return application_logger

    settings.log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    application_logger.setLevel(
        getattr(
            logging,
            settings.log_level,
        )
    )

    application_logger.propagate = False

    formatter = logging.Formatter(
        (
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        )
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        settings.log_dir / "app.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    application_logger.addHandler(
        console_handler
    )

    application_logger.addHandler(
        file_handler
    )

    setattr(
        application_logger,
        "_personal_rag_configured",
        True,
    )

    return application_logger


def get_logger(
    module_name: str,
) -> logging.Logger:
    """
    Return a child logger for one project module.
    """

    configure_logging()

    return logging.getLogger(
        f"{LOGGER_ROOT_NAME}.{module_name}"
    )