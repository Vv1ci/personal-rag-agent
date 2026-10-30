from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag.agent.agent_controller import run_agent
from rag.agent.tools.document_stats_tool import (
    build_document_statistics,
    run_document_stats_tool,
)


TestFunction = Callable[[], None]


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def create_test_knowledge_base(
    root: Path,
) -> tuple[Path, Path]:
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"

    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    (raw_dir / "alpha.txt").write_text(
        "Alpha raw document.",
        encoding="utf-8",
    )

    (raw_dir / "beta.md").write_text(
        "# Beta",
        encoding="utf-8",
    )

    # This document exists in data/raw but has no chunks.
    (raw_dir / "unindexed.pdf").write_bytes(b"test")

    chunks = [
        {
            "text": "alpha one",
            "metadata": {
                "source": "alpha.txt",
            },
        },
        {
            "text": "alpha two",
            "metadata": {
                "source": "alpha.txt",
            },
        },
        {
            "content": "beta",
            "source": "beta.md",
        },
    ]

    chunks_path = processed_dir / "all_chunks.json"

    chunks_path.write_text(
        json.dumps(
            chunks,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return raw_dir, chunks_path


def test_whole_knowledge_base_statistics() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)

        raw_dir, chunks_path = (
            create_test_knowledge_base(root)
        )

        statistics = build_document_statistics(
            raw_dir=raw_dir,
            chunks_path=chunks_path,
        )

        require(
            statistics["raw_document_count"] == 3,
            "Expected three raw documents.",
        )

        require(
            statistics["indexed_document_count"] == 2,
            "Expected two indexed documents.",
        )

        require(
            statistics["total_chunk_count"] == 3,
            "Expected three chunks.",
        )

        require(
            statistics["unindexed_documents"]
            == ["unindexed.pdf"],
            "The unindexed PDF was not detected.",
        )


def test_specific_document_statistics() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)

        raw_dir, chunks_path = (
            create_test_knowledge_base(root)
        )

        result = run_document_stats_tool(
            "alpha.txt",
            raw_dir=raw_dir,
            chunks_path=chunks_path,
        )

        require(
            result["success"] is True,
            f"Tool failed: {result['error']}",
        )

        require(
            result["data"]["chunk_count"] == 2,
            "alpha.txt should contain two chunks.",
        )

        require(
            result["data"]["character_count"] == 18,
            "Expected 18 total chunk characters.",
        )


def test_missing_document() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)

        raw_dir, chunks_path = (
            create_test_knowledge_base(root)
        )

        result = run_document_stats_tool(
            "missing.txt",
            raw_dir=raw_dir,
            chunks_path=chunks_path,
        )

        require(
            result["success"] is False,
            "A missing document should return an error.",
        )


def fake_document_stats_handler(
    target: str,
    *,
    language: str = "en",
) -> dict[str, Any]:
    return {
        "success": True,
        "answer": (
            f"Fake statistics for: "
            f"{target or 'all documents'}"
        ),
        "data": {
            "target": target,
            "language": language,
        },
        "error": None,
    }


def test_agent_routes_to_document_stats() -> None:
    result = run_agent(
        "/stats alpha.txt",
        document_stats_handler=(
            fake_document_stats_handler
        ),
    )

    require(
        result["metadata"]["agent_route"] == "tool",
        "The agent did not choose the tool route.",
    )

    require(
        result["metadata"]["tool_used"]
        == "document_stats",
        "The wrong tool was selected.",
    )

    require(
        result["sources"] == [],
        "Document statistics should not return RAG sources.",
    )


def fake_rag_handler(
    question: str,
    **_: Any,
) -> dict[str, Any]:
    return {
        "answer": f"RAG answer: {question}",
        "sources": [
            {
                "source": "test.txt",
            }
        ],
        "metadata": {},
    }


def test_normal_question_still_uses_rag() -> None:
    result = run_agent(
        "What is ChromaDB?",
        rag_handler=fake_rag_handler,
    )

    require(
        result["metadata"]["agent_route"] == "rag",
        "Normal questions should continue to use RAG.",
    )

    require(
        len(result["sources"]) == 1,
        "The fake RAG source should be preserved.",
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
    tests: list[tuple[str, TestFunction]] = [
        (
            "Whole knowledge base statistics",
            test_whole_knowledge_base_statistics,
        ),
        (
            "Specific document statistics",
            test_specific_document_statistics,
        ),
        (
            "Missing document handling",
            test_missing_document,
        ),
        (
            "Agent document stats routing",
            test_agent_routes_to_document_stats,
        ),
        (
            "RAG fallback routing",
            test_normal_question_still_uses_rag,
        ),
    ]

    passed = 0

    for name, test_function in tests:
        if run_test(name, test_function):
            passed += 1

    total = len(tests)
    failed = total - passed

    print()
    print("=" * 55)
    print("Day 26 Document Stats Test Summary")
    print("=" * 55)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {failed}/{total}")

    if failed:
        raise SystemExit(1)

    print("All Day 26 document stats tests passed.")


if __name__ == "__main__":
    main()