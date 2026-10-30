from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load variables from project-root/.env.
load_dotenv(PROJECT_ROOT / ".env")


def _read_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """
    Read and validate an integer environment variable.
    """

    raw_value = os.getenv(name)

    if raw_value is None:
        value = default

    else:
        try:
            value = int(raw_value)

        except ValueError as exc:
            raise ValueError(
                f"{name} must be an integer, "
                f"received: {raw_value!r}"
            ) from exc

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"{name} must be at most {maximum}."
        )

    return value


def _read_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """
    Read and validate a floating-point environment variable.
    """

    raw_value = os.getenv(name)

    if raw_value is None:
        value = default

    else:
        try:
            value = float(raw_value)

        except ValueError as exc:
            raise ValueError(
                f"{name} must be a number, "
                f"received: {raw_value!r}"
            ) from exc

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"{name} must be at most {maximum}."
        )

    return value


def _resolve_project_path(
    environment_name: str,
    default_relative_path: str,
) -> Path:
    """
    Convert a relative environment path into an absolute
    path below the project root.

    Absolute paths are preserved.
    """

    raw_value = os.getenv(
        environment_name,
        default_relative_path,
    )

    path = Path(raw_value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def _read_extensions() -> tuple[str, ...]:
    """
    Read comma-separated upload extensions.

    Example:
        .txt,.md,.pdf
    """

    raw_value = os.getenv(
        "ALLOWED_UPLOAD_EXTENSIONS",
        ".txt,.md,.pdf",
    )

    extensions: list[str] = []

    for item in raw_value.split(","):
        extension = item.strip().lower()

        if not extension:
            continue

        if not extension.startswith("."):
            extension = "." + extension

        extensions.append(extension)

    if not extensions:
        raise ValueError(
            "ALLOWED_UPLOAD_EXTENSIONS cannot be empty."
        )

    return tuple(dict.fromkeys(extensions))


def _read_log_level() -> str:
    level = os.getenv(
        "LOG_LEVEL",
        "INFO",
    ).strip().upper()

    allowed_levels = {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    if level not in allowed_levels:
        raise ValueError(
            "LOG_LEVEL must be one of: "
            + ", ".join(sorted(allowed_levels))
        )

    return level


@dataclass(
    frozen=True,
    slots=True,
)
class Settings:
    """
    Central application settings.
    """

    app_name: str
    app_env: str

    api_host: str
    api_port: int
    api_base_url: str

    ollama_base_url: str
    ollama_model: str
    embedding_model: str

    default_top_k: int
    min_relevance_score: float
    memory_max_turns: int

    project_root: Path
    data_raw_dir: Path
    data_processed_dir: Path
    chunks_path: Path
    chroma_db_dir: Path
    log_dir: Path

    allowed_upload_extensions: tuple[str, ...]
    log_level: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Build the settings object once and reuse it.
    """

    api_host = os.getenv(
        "API_HOST",
        "127.0.0.1",
    ).strip()

    api_port = _read_int(
        "API_PORT",
        8000,
        minimum=1,
        maximum=65535,
    )

    default_api_base_url = (
        f"http://{api_host}:{api_port}"
    )

    api_base_url = os.getenv(
        "API_BASE_URL",
        default_api_base_url,
    ).strip().rstrip("/")

    data_processed_dir = _resolve_project_path(
        "DATA_PROCESSED_DIR",
        "data/processed",
    )

    return Settings(
        app_name=os.getenv(
            "APP_NAME",
            "Personal RAG Agent",
        ).strip(),
        app_env=os.getenv(
            "APP_ENV",
            "development",
        ).strip().lower(),
        api_host=api_host,
        api_port=api_port,
        api_base_url=api_base_url,
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434",
        ).strip().rstrip("/"),
        ollama_model=os.getenv(
            "OLLAMA_MODEL",
            "qwen2.5:1.5b",
        ).strip(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            (
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
        ).strip(),
        default_top_k=_read_int(
            "DEFAULT_TOP_K",
            3,
            minimum=1,
            maximum=20,
        ),
        min_relevance_score=_read_float(
            "MIN_RELEVANCE_SCORE",
            0.42,
            minimum=0.0,
            maximum=1.0,
        ),
        memory_max_turns=_read_int(
            "MEMORY_MAX_TURNS",
            4,
            minimum=1,
            maximum=100,
        ),
        project_root=PROJECT_ROOT,
        data_raw_dir=_resolve_project_path(
            "DATA_RAW_DIR",
            "data/raw",
        ),
        data_processed_dir=data_processed_dir,
        chunks_path=(
            data_processed_dir
            / "all_chunks.json"
        ),
        chroma_db_dir=_resolve_project_path(
            "CHROMA_DB_DIR",
            "chroma_db",
        ),
        log_dir=_resolve_project_path(
            "LOG_DIR",
            "logs",
        ),
        allowed_upload_extensions=(
            _read_extensions()
        ),
        log_level=_read_log_level(),
    )


settings = get_settings()