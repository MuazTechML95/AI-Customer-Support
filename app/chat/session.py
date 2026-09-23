"""
session.py
----------
WHAT THIS FILE DOES:
Manages conversation history PER SESSION, so that two different users (or
two different browser tabs) never see each other's conversation - this is
the "session isolation" requirement.

HOW IT WORKS:
- Each session is identified by a `session_id` string (in the Streamlit UI,
  this is a random UUID generated once per browser session and stored in
  st.session_state - see ui/streamlit_app.py).
- `SessionStore` keeps an in-memory dictionary: {session_id: [messages...]}.
  Because Python dictionaries key strictly by the session_id string, there
  is NO way for session A's code path to accidentally read session B's data -
  they are simply different dictionary entries.

LIMITATION (documented honestly, per project rules):
This store is IN-MEMORY ONLY. If the server process restarts, all session
history is lost. This is an intentional, documented trade-off for this
version of the project (simplicity, zero extra infrastructure). See the
README "Known Limitations" section.

HOW TO UPGRADE THIS LATER:
Replace the in-memory dict with a Redis or SQLite-backed store. Because all
access goes through the SessionStore class methods (get_history,
append_message, clear_session), you would only need to change THIS file -
no other file directly touches the underlying storage.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Message:
    role: str        # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sources: list[str] = field(default_factory=list)  # only populated for assistant messages


class SessionStore:
    """
    In-memory store of conversation history, keyed by session_id.
    A single instance of this class is created once and shared across the
    app (see services/support_service.py) so history persists across
    multiple user messages within the same running process.
    """

    def __init__(self):
        self._sessions: dict[str, list[Message]] = {}

    def get_history(self, session_id: str) -> list[Message]:
        """Returns the full message list for a session (empty list if new)."""
        return self._sessions.setdefault(session_id, [])

    def append_message(self, session_id: str, role: str, content: str, sources: list[str] | None = None) -> None:
        """Adds one message (user or assistant) to a session's history."""
        history = self._sessions.setdefault(session_id, [])
        history.append(Message(role=role, content=content, sources=sources or []))

    def get_recent_history_text(self, session_id: str, max_turns: int) -> str:
        """
        Formats the last `max_turns` message pairs as plain text, suitable for
        inserting into the LLM prompt (see chat/prompts.py build_user_prompt).
        A "turn" here roughly means one message (not strictly user+assistant
        pairs) - max_turns *2 messages are kept at most for simplicity.
        """
        history = self._sessions.get(session_id, [])
        recent = history[-(max_turns * 2):] if history else []
        lines = []
        for msg in recent:
            speaker = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{speaker}: {msg.content}")
        return "\n".join(lines)

    def clear_session(self, session_id: str) -> None:
        """Wipes a session's history (used by the 'Clear Conversation' button)."""
        self._sessions[session_id] = []

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions
