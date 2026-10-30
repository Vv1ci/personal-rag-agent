from __future__ import annotations
from rag.config import settings
from collections import deque
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any


@dataclass(slots=True)
class ChatTurn:
    """
    One completed user-assistant interaction.
    """

    user: str
    assistant: str
    route: str
    tool_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConversationMemory:
    """
    Store a limited number of recent conversation turns
    for each session.
    """

    def __init__(self, max_turns: int = 4) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")

        self.max_turns = max_turns

        self._sessions: dict[
            str,
            deque[ChatTurn],
        ] = {}

        self._lock = RLock()

    @staticmethod
    def _normalise_session_id(
        session_id: str,
    ) -> str:
        cleaned_session_id = session_id.strip()

        if not cleaned_session_id:
            return "default"

        return cleaned_session_id

    def _get_or_create_session(
        self,
        session_id: str,
    ) -> deque[ChatTurn]:
        normalised_id = self._normalise_session_id(
            session_id
        )

        if normalised_id not in self._sessions:
            self._sessions[normalised_id] = deque(
                maxlen=self.max_turns
            )

        return self._sessions[normalised_id]

    def add_turn(
        self,
        session_id: str,
        *,
        user: str,
        assistant: str,
        route: str,
        tool_used: str | None = None,
    ) -> None:
        """
        Add one completed interaction to a session.
        """

        if not user.strip():
            return

        turn = ChatTurn(
            user=user.strip(),
            assistant=assistant.strip(),
            route=route,
            tool_used=tool_used,
        )

        with self._lock:
            session = self._get_or_create_session(
                session_id
            )
            session.append(turn)

    def get_turns(
        self,
        session_id: str,
    ) -> list[ChatTurn]:
        """
        Return a copy of the session's recent turns.
        """

        normalised_id = self._normalise_session_id(
            session_id
        )

        with self._lock:
            session = self._sessions.get(
                normalised_id
            )

            if session is None:
                return []

            return list(session)

    def count(
        self,
        session_id: str,
    ) -> int:
        return len(self.get_turns(session_id))

    def clear(
        self,
        session_id: str,
    ) -> int:
        """
        Clear one session and return the number
        of removed turns.
        """

        normalised_id = self._normalise_session_id(
            session_id
        )

        with self._lock:
            session = self._sessions.pop(
                normalised_id,
                None,
            )

        return len(session) if session else 0

    @staticmethod
    def _truncate(
        text: str,
        limit: int = 500,
    ) -> str:
        cleaned_text = text.strip()

        if len(cleaned_text) <= limit:
            return cleaned_text

        return cleaned_text[:limit] + "..."

    def build_contextualised_question(
        self,
        session_id: str,
        current_question: str,
        *,
        history_turns: int = 2,
    ) -> str:
        """
        Add recent conversation context to the current
        question so references such as 'it' can be resolved.
        """

        turns = self.get_turns(session_id)

        if not turns:
            return current_question

        recent_turns = turns[-history_turns:]

        lines = [
            (
                "Use the recent conversation only to "
                "resolve references in the current question."
            ),
            "",
            "Recent conversation:",
        ]

        for turn in recent_turns:
            lines.append(
                "User: "
                + self._truncate(turn.user)
            )
            lines.append(
                "Assistant: "
                + self._truncate(turn.assistant)
            )

        lines.extend(
            [
                "",
                (
                    "Current user question: "
                    + current_question.strip()
                ),
            ]
        )

        return "\n".join(lines)

    def format_memory(
        self,
        session_id: str,
        *,
        language: str = "en",
    ) -> str:
        turns = self.get_turns(session_id)

        if not turns:
            return (
                "当前会话没有记忆。"
                if language == "zh"
                else "The current session has no memory."
            )

        heading = (
            "当前会话记忆："
            if language == "zh"
            else "Current Session Memory:"
        )

        lines = [heading]

        for index, turn in enumerate(
            turns,
            start=1,
        ):
            lines.extend(
                [
                    "",
                    f"Turn {index}",
                    f"User: {turn.user}",
                    f"Assistant: {turn.assistant}",
                    f"Route: {turn.route}",
                    (
                        "Tool: "
                        f"{turn.tool_used or 'None'}"
                    ),
                ]
            )

        return "\n".join(lines)


GLOBAL_MEMORY = ConversationMemory(
    max_turns=settings.memory_max_turns
)