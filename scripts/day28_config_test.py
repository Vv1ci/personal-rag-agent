from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag.config import settings
from rag.logging_config import get_logger


TestFunction = Callable[[], None]


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def test_project_paths() -> None:
    require(
        settings.project_root.is_absolute(),
        "Project root must be an absolute path.",
    )

    require(
        settings.data_raw_dir.is_absolute(),
        "Data raw path must be absolute.",
    )

    require(
        settings.chunks_path.name
        == "all_chunks.json",
        "The chunks filename is incorrect.",
    )


def test_numeric_settings() -> None:
    require(
        1 <= settings.default_top_k <= 20,
        "DEFAULT_TOP_K is outside its range.",
    )

    require(
        0.0
        <= settings.min_relevance_score
        <= 1.0,
        (
            "MIN_RELEVANCE_SCORE is "
            "outside its range."
        ),
    )

    require(
        settings.memory_max_turns >= 1,
        "MEMORY_MAX_TURNS must be positive.",
    )


def test_env_example() -> None:
    env_example_path = (
        settings.project_root
        / ".env.example"
    )

    require(
        env_example_path.exists(),
        ".env.example does not exist.",
    )

    content = env_example_path.read_text(
        encoding="utf-8-sig"
    )

    required_names = [
        "API_BASE_URL",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "DEFAULT_TOP_K",
        "MIN_RELEVANCE_SCORE",
        "MEMORY_MAX_TURNS",
        "LOG_LEVEL",
    ]

    for name in required_names:
        require(
            f"{name}=" in content,
            f"{name} is missing from .env.example.",
        )


def test_env_is_ignored() -> None:
    gitignore_path = (
        settings.project_root
        / ".gitignore"
    )

    require(
        gitignore_path.exists(),
        ".gitignore does not exist.",
    )

    content = gitignore_path.read_text(
        encoding="utf-8-sig"
    )

    lines = {
        line.strip()
        for line in content.splitlines()
    }

    require(
        ".env" in lines,
        ".env is not protected by .gitignore.",
    )

    require(
        "!.env.example" in lines,
        (
            ".env.example is not explicitly "
            "allowed in Git."
        ),
    )


def test_logging() -> None:
    marker = (
        "day28-test-"
        + str(uuid4())
    )

    logger = get_logger(
        "scripts.day28_config_test"
    )

    logger.warning(marker)

    application_logger = logging.getLogger(
        "personal_rag"
    )

    for handler in application_logger.handlers:
        handler.flush()

    log_path = settings.log_dir / "app.log"

    require(
        log_path.exists(),
        "The application log file was not created.",
    )

    content = log_path.read_text(
        encoding="utf-8"
    )

    require(
        marker in content,
        "The test log message was not written.",
    )


def run_test(
    name: str,
    test_function: TestFunction,
) -> bool:
    try:
        test_function()
        print(f"[PASS] {name}")
        return True

    except Exception as exc:
        print(f"[FAIL] {name}")
        print(
            f"       {type(exc).__name__}: {exc}"
        )
        return False


def main() -> None:
    tests: list[
        tuple[str, TestFunction]
    ] = [
        (
            "Project paths",
            test_project_paths,
        ),
        (
            "Numeric settings",
            test_numeric_settings,
        ),
        (
            ".env.example",
            test_env_example,
        ),
        (
            ".env Git protection",
            test_env_is_ignored,
        ),
        (
            "Logging output",
            test_logging,
        ),
    ]

    passed = 0

    for name, test_function in tests:
        if run_test(name, test_function):
            passed += 1

    total = len(tests)
    failed = total - passed

    print()
    print("=" * 50)
    print("Day 28 Configuration Test Summary")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {failed}/{total}")

    if failed:
        raise SystemExit(1)

    print(
        "All Day 28 configuration tests passed."
    )


if __name__ == "__main__":
    main()