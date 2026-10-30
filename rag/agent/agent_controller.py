from __future__ import annotations

import inspect
from typing import Any, Callable

from rag.agent.tools.calculator_tool import run_calculator_tool

from rag.agent.tools.document_stats_tool import (
    run_document_stats_tool,
)

from rag.agent.memory import (
    GLOBAL_MEMORY,
    ConversationMemory,
)

RagHandler = Callable[..., Any]
DocumentStatsHandler = Callable[..., dict[str, Any]]

_CALCULATOR_PREFIXES = (
    "/calc",
    "calculate:",
    "calculate ",
    "计算：",
    "计算:",
    "计算 ",
)
_DOCUMENT_STATS_PREFIXES = (
    "/stats",
    "/docstats",
    "stats:",
    "统计：",
    "统计:",
)

def extract_calculator_expression(question: str) -> str | None:
    """
    Return the arithmetic expression if the user explicitly requests
    the calculator tool.

    Return None when the question should not use the calculator.
    """

    cleaned_question = question.strip()
    lowered_question = cleaned_question.lower()

    for prefix in _CALCULATOR_PREFIXES:
        if lowered_question.startswith(prefix.lower()):
            return cleaned_question[len(prefix):].strip()

    return None

def extract_document_stats_target(
    question: str,
) -> str | None:
    """
    Return:

    ""              for /stats
    "example.txt"   for /stats example.txt
    None            when this is not a stats request
    """

    cleaned_question = question.strip()
    lowered_question = cleaned_question.lower()

    for prefix in _DOCUMENT_STATS_PREFIXES:
        lowered_prefix = prefix.lower()

        if lowered_question == lowered_prefix:
            return ""

        if prefix.endswith((':', '：')):
            if lowered_question.startswith(lowered_prefix):
                return cleaned_question[len(prefix):].strip()

        elif lowered_question.startswith(
            lowered_prefix + " "
        ):
            return cleaned_question[len(prefix):].strip()

    return None

def _get_default_rag_handler() -> RagHandler:
    """
    Import lazily so calculator requests do not immediately load
    the embedding model or RAG pipeline.
    """

    from rag.chains.rag_chain import answer_with_rag

    return answer_with_rag


def _call_rag_handler(
    handler: RagHandler,
    *,
    question: str,
    top_k: int,
    language: str,
    min_relevance_score: float,
    debug: bool,
) -> Any:
    """
    Call different versions of answer_with_rag by inspecting
    which parameters the function currently accepts.
    """

    signature = inspect.signature(handler)
    parameter_names = set(signature.parameters)

    candidate_values = {
        "question": question,
        "query": question,
        "top_k": top_k,
        "language": language,
        "min_relevance_score": min_relevance_score,
        "debug": debug,
    }

    accepted_arguments = {
        name: value
        for name, value in candidate_values.items()
        if name in parameter_names
    }

    if "question" in parameter_names or "query" in parameter_names:
        return handler(**accepted_arguments)

    # Fallback for a function whose first parameter has a different name.
    remaining_arguments = {
        name: value
        for name, value in accepted_arguments.items()
        if name not in {"question", "query"}
    }

    return handler(
        question,
        **remaining_arguments,
    )


def _normalize_rag_result(raw_result: Any) -> dict[str, Any]:
    """
    Convert different possible RAG return formats into one dictionary.
    """

    if isinstance(raw_result, dict):
        result = dict(raw_result)

    elif isinstance(raw_result, str):
        result = {
            "answer": raw_result,
            "sources": [],
            "metadata": {},
        }

    else:
        result = {
            "answer": str(raw_result),
            "sources": [],
            "metadata": {},
        }

    result.setdefault("answer", "")
    result.setdefault("sources", [])

    metadata = result.get("metadata")

    if not isinstance(metadata, dict):
        metadata = {}

    metadata["agent_route"] = "rag"
    metadata["tool_used"] = None

    result["metadata"] = metadata

    return result

def _finalise_agent_result(
    result: dict[str, Any],
    *,
    question: str,
    session_id: str,
    use_memory: bool,
    memory_store: ConversationMemory,
    memory_used: bool,
) -> dict[str, Any]:
    """
    Add memory metadata and store the completed turn.
    """

    metadata = result.get("metadata")

    if not isinstance(metadata, dict):
        metadata = {}

    metadata["session_id"] = session_id
    metadata["memory_used"] = memory_used
    metadata["memory_turns_before_response"] = (
        memory_store.count(session_id)
    )

    result["metadata"] = metadata

    if use_memory:
        memory_store.add_turn(
            session_id,
            user=question,
            assistant=str(
                result.get("answer", "")
            ),
            route=str(
                metadata.get(
                    "agent_route",
                    "unknown",
                )
            ),
            tool_used=metadata.get("tool_used"),
        )

    metadata["memory_turns_after_response"] = (
        memory_store.count(session_id)
    )

    return result

def run_agent(
    question: str,
    *,
    top_k: int = 3,
    language: str = "en",
    min_relevance_score: float = 0.42,
    debug: bool = False,
    rag_handler: RagHandler | None = None,
    document_stats_handler: (
        DocumentStatsHandler | None
    ) = None,
    session_id: str = "default",
    use_memory: bool = True,
    memory_store: ConversationMemory = GLOBAL_MEMORY,
) -> dict[str, Any]:
    """
    Route the request to a command, tool, or RAG chain,
    while maintaining short-term session memory.
    """

    cleaned_question = question.strip()
    lowered_question = cleaned_question.lower()

    # -------------------------------------------------
    # Memory command: display the current session
    # -------------------------------------------------

    if lowered_question == "/memory":
        return {
            "answer": memory_store.format_memory(
                session_id,
                language=language,
            ),
            "sources": [],
            "metadata": {
                "agent_route": "tool",
                "tool_used": "memory",
                "session_id": session_id,
                "memory_used": False,
                "memory_turns": memory_store.count(
                    session_id
                ),
            },
        }

    # -------------------------------------------------
    # Memory command: clear the current session
    # -------------------------------------------------

    if lowered_question in {
        "/clear-memory",
        "/new-session",
    }:
        removed_turns = memory_store.clear(
            session_id
        )

        answer = (
            (
                "Conversation memory cleared. "
                f"Removed {removed_turns} turns."
            )
            if language == "en"
            else (
                "当前会话记忆已清除，"
                f"共删除 {removed_turns} 轮对话。"
            )
        )

        return {
            "answer": answer,
            "sources": [],
            "metadata": {
                "agent_route": "tool",
                "tool_used": "memory",
                "session_id": session_id,
                "memory_used": False,
                "removed_turns": removed_turns,
                "memory_turns": 0,
            },
        }

    # -------------------------------------------------
    # Route 1: Calculator Tool
    # -------------------------------------------------

    calculator_expression = (
        extract_calculator_expression(
            cleaned_question
        )
    )

    if calculator_expression is not None:
        tool_result = run_calculator_tool(
            calculator_expression
        )

        if tool_result["success"]:
            result = {
                "answer": tool_result["answer"],
                "sources": [],
                "metadata": {
                    "agent_route": "tool",
                    "tool_used": "calculator",
                    "expression": (
                        tool_result["expression"]
                    ),
                    "result": tool_result["result"],
                },
            }

        else:
            result = {
                "answer": (
                    (
                        "Calculator error: "
                        f"{tool_result['error']}"
                    )
                    if language == "en"
                    else (
                        "计算器错误："
                        f"{tool_result['error']}"
                    )
                ),
                "sources": [],
                "metadata": {
                    "agent_route": "tool",
                    "tool_used": "calculator",
                    "expression": (
                        tool_result["expression"]
                    ),
                    "error": tool_result["error"],
                },
            }

        return _finalise_agent_result(
            result,
            question=cleaned_question,
            session_id=session_id,
            use_memory=use_memory,
            memory_store=memory_store,
            memory_used=False,
        )

    # -------------------------------------------------
    # Route 2: Document Statistics Tool
    # -------------------------------------------------

    stats_target = extract_document_stats_target(
        cleaned_question
    )

    if stats_target is not None:
        selected_stats_handler = (
            document_stats_handler
            or run_document_stats_tool
        )

        tool_result = selected_stats_handler(
            stats_target,
            language=language,
        )

        if tool_result["success"]:
            result = {
                "answer": tool_result["answer"],
                "sources": [],
                "metadata": {
                    "agent_route": "tool",
                    "tool_used": "document_stats",
                    "statistics": tool_result["data"],
                },
            }

        else:
            result = {
                "answer": (
                    (
                        "Document statistics error: "
                        f"{tool_result['error']}"
                    )
                    if language == "en"
                    else (
                        "文档统计错误："
                        f"{tool_result['error']}"
                    )
                ),
                "sources": [],
                "metadata": {
                    "agent_route": "tool",
                    "tool_used": "document_stats",
                    "error": tool_result["error"],
                },
            }

        return _finalise_agent_result(
            result,
            question=cleaned_question,
            session_id=session_id,
            use_memory=use_memory,
            memory_store=memory_store,
            memory_used=False,
        )

    # -------------------------------------------------
    # Route 3: Memory-aware RAG
    # -------------------------------------------------

    previous_turn_count = memory_store.count(
        session_id
    )

    memory_should_be_used = (
        use_memory
        and previous_turn_count > 0
    )

    if memory_should_be_used:
        rag_question = (
            memory_store
            .build_contextualised_question(
                session_id,
                cleaned_question,
            )
        )
    else:
        rag_question = cleaned_question

    selected_rag_handler = (
        rag_handler
        or _get_default_rag_handler()
    )

    raw_rag_result = _call_rag_handler(
        selected_rag_handler,
        question=rag_question,
        top_k=top_k,
        language=language,
        min_relevance_score=min_relevance_score,
        debug=debug,
    )

    result = _normalize_rag_result(
        raw_rag_result
    )

    result["metadata"]["original_question"] = (
        cleaned_question
    )

    return _finalise_agent_result(
        result,
        question=cleaned_question,
        session_id=session_id,
        use_memory=use_memory,
        memory_store=memory_store,
        memory_used=memory_should_be_used,
    )