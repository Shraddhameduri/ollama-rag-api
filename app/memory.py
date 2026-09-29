"""In-memory per-session conversation history with a fixed window."""

from __future__ import annotations

import threading


class ConversationMemory:
    """Thread-safe store of recent turns keyed by ``session_id``."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._lock = threading.Lock()

    def add(self, session_id: str, role: str, content: str) -> None:
        """Append one turn to a session's history."""
        with self._lock:
            self._sessions.setdefault(session_id, []).append(
                {"role": role, "content": content}
            )

    def get(self, session_id: str, window: int) -> list[dict[str, str]]:
        """Return up to the last ``window`` turns for a session."""
        with self._lock:
            history = self._sessions.get(session_id, [])
            return list(history[-window:]) if window > 0 else []

    def clear(self, session_id: str) -> None:
        """Drop all history for a session."""
        with self._lock:
            self._sessions.pop(session_id, None)
