from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag.agent.agent_controller import run_agent
from rag.agent.tools.calculator_tool import calculate_expression


TestFunction = Callable[[], None]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_operator_precedence() -> None:
    result = calculate_expression("2 + 3 * 4")

    require(
        result == 14,
        f"Expected 14, received {result}.",
    )


def test_parentheses() -> None:
    result = calculate_expression("(10 - 2) / 4")

    require(
        result == 2,
        f"Expected 2, received {result}.",
    )


def test_unsafe_expression_is_rejected() -> None:
    try:
        calculate_expression("__import__('os').system('echo unsafe')")
    except ValueError:
        return

    raise AssertionError(
        "Unsafe function calls should have been rejected."
    )


def test_agent_routes_to_calculator() -> None:
    result = run_agent("/calc (12 + 8) * 3")

    require(
        result["metadata"]["tool_used"] == "calculator",
        "The agent did not select the calculator tool.",
    )

    require(
        result["metadata"]["result"] == 60,
        f"Expected 60, received {result['metadata']['result']}.",
    )

    require(
        result["sources"] == [],
        "Calculator answers should not contain document sources.",
    )


def fake_rag_handler(
    question: str,
    **_: Any,
) -> dict[str, Any]:
    return {
        "answer": f"Fake RAG answer for: {question}",
        "sources": [
            {
                "source": "fake_document.txt",
                "relevance_score": 0.99,
            }
        ],
        "metadata": {
            "api_mode": "fake_test",
        },
    }


def test_agent_routes_to_rag() -> None:
    result = run_agent(
        "What is ChromaDB?",
        rag_handler=fake_rag_handler,
    )

    require(
        result["metadata"]["agent_route"] == "rag",
        "The agent did not select the RAG route.",
    )

    require(
        result["metadata"]["tool_used"] is None,
        "A normal knowledge question should not use a tool.",
    )

    require(
        len(result["sources"]) == 1,
        "The fake RAG source was not preserved.",
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
        print(f"       {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    tests: list[tuple[str, TestFunction]] = [
        ("Operator precedence", test_operator_precedence),
        ("Parentheses", test_parentheses),
        ("Unsafe expression rejection", test_unsafe_expression_is_rejected),
        ("Calculator routing", test_agent_routes_to_calculator),
        ("RAG fallback routing", test_agent_routes_to_rag),
    ]

    passed = 0

    for name, test_function in tests:
        if run_test(name, test_function):
            passed += 1

    total = len(tests)
    failed = total - passed

    print()
    print("=" * 50)
    print("Day 25 Agent Test Summary")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {failed}/{total}")

    if failed:
        raise SystemExit(1)

    print("All Day 25 agent tests passed.")


if __name__ == "__main__":
    main()