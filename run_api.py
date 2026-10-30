from __future__ import annotations

import uvicorn

from rag.config import settings
from rag.logging_config import get_logger


logger = get_logger(__name__)


def main() -> None:
    reload_enabled = (
        settings.app_env == "development"
    )

    logger.info(
        (
            "Starting API | "
            "host=%s | "
            "port=%s | "
            "environment=%s"
        ),
        settings.api_host,
        settings.api_port,
        settings.app_env,
    )

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()