from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag.agent.agent_controller import run_agent
from rag.agent.memory import ConversationMemory


TestFunction = Callable[[], None]


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def test_memory_window_limit() -> None:
    memory = ConversationMemory(max_turns=3)

    for number in range(5):
        memory.add_turn(
            "session-a",
            user=f"Question {number}",
            assistant=f"Answer {number}",
            route="rag",
        )

    turns = memory.get_turns("session-a")

    require(
        len(turns) == 3,
        "The memory window should contain three turns.",
    )

    require(
        turns[0].user == "Question 2",
        "The oldest turns were not removed correctly.",
    )


def test_follow_up_question_uses_memory() -> None:
    memory = ConversationMemory(max_turns=4)
    captured_questions: list[str] = []

    def fake_rag_handler(
        question: str,
        **_: Any,
    ) -> dict[str, Any]:
        captured_questions.append(question)

        return {
            "answer": "Fake RAG answer.",
            "sources": [
                {
                    "source": "fake.txt",
                }
            ],
            "metadata": {},
        }

    run_agent(
        "What is ChromaDB?",
        session_id="session-a",
        memory_store=memory,
        rag_handler=fake_rag_handler,
    )

    second_result = run_agent(
        "What is it used for?",
        session_id="session-a",
        memory_store=memory,
        rag_handler=fake_rag_handler,
    )

    contextualised_question = (
        captured_questions[-1]
    )

    require(
        "What is ChromaDB?"
        in contextualised_question,
        "The previous question was not added.",
    )

    require(
        "What is it used for?"
        in contextualised_question,
        "The current question was not preserved.",
    )

    require(
        second_result["metadata"]["memory_used"]
        is True,
        "The second request should use memory.",
    )


def test_sessions_are_isolated() -> None:
    memory = ConversationMemory(max_turns=4)

    memory.add_turn(
        "session-a",
        user="What is ChromaDB?",
        assistant="A vector database.",
        route="rag",
    )

    session_b_context = (
        memory.build_contextualised_question(
            "session-b",
            "What is it used for?",
        )
    )

    require(
        "ChromaDB" not in session_b_context,
        "Conversation data leaked between sessions.",
    )


def test_clear_memory_command() -> None:
    memory = ConversationMemory(max_turns=4)

    memory.add_turn(
        "session-a",
        user="Question",
        assistant="Answer",
        route="rag",
    )

    result = run_agent(
        "/clear-memory",
        session_id="session-a",
        memory_store=memory,
    )

    require(
        memory.count("session-a") == 0,
        "The session memory was not cleared.",
    )

    require(
        result["metadata"]["removed_turns"] == 1,
        "The removed turn count is incorrect.",
    )


def test_memory_can_be_disabled() -> None:
    memory = ConversationMemory(max_turns=4)
    captured_questions: list[str] = []

    memory.add_turn(
        "session-a",
        user="Previous private question",
        assistant="Previous answer",
        route="rag",
    )

    def fake_rag_handler(
        question: str,
        **_: Any,
    ) -> dict[str, Any]:
        captured_questions.append(question)

        return {
            "answer": "Fake answer.",
            "sources": [],
            "metadata": {},
        }

    result = run_agent(
        "Current independent question",
        session_id="session-a",
        use_memory=False,
        memory_store=memory,
        rag_handler=fake_rag_handler,
    )

    require(
        captured_questions[-1]
        == "Current independent question",
        "History should not be included when disabled.",
    )

    require(
        result["metadata"]["memory_used"] is False,
        "memory_used should be false.",
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
            "Memory window limit",
            test_memory_window_limit,
        ),
        (
            "Follow-up question uses memory",
            test_follow_up_question_uses_memory,
        ),
        (
            "Session isolation",
            test_sessions_are_isolated,
        ),
        (
            "Clear memory command",
            test_clear_memory_command,
        ),
        (
            "Disable memory",
            test_memory_can_be_disabled,
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
    print("Day 27 Memory Test Summary")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {failed}/{total}")

    if failed:
        raise SystemExit(1)

    print("All Day 27 memory tests passed.")


if __name__ == "__main__":
    main()