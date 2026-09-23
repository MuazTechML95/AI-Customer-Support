"""
test_session_isolation.py
----------------------------
WHAT THIS TESTS:
That two different session_ids never share conversation history - Session A's
messages must not be visible when reading Session B's history.

HOW TO RUN:
    pytest tests/test_session_isolation.py -v

NOTE: This test works directly against the SessionStore (app/chat/session.py)
and does not require the vector index or an LLM API key, since it's testing
the storage/isolation mechanism itself, not the full RAG pipeline.
"""

import uuid

from app.chat.session import SessionStore


def test_sessions_do_not_share_history():
    store = SessionStore()
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    store.append_message(session_a, "user", "What is your return policy?")
    store.append_message(session_a, "assistant", "Returns are accepted within 30 days.")

    # Session B has never sent a message - its history must be empty.
    history_b = store.get_history(session_b)
    assert history_b == [], "Session B should not see Session A's messages"

    # Session A's history should be intact and unaffected by session B being created.
    history_a = store.get_history(session_a)
    assert len(history_a) == 2
    assert history_a[0].content == "What is your return policy?"


def test_clearing_one_session_does_not_affect_another():
    store = SessionStore()
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    store.append_message(session_a, "user", "Hello from A")
    store.append_message(session_b, "user", "Hello from B")

    store.clear_session(session_a)

    assert store.get_history(session_a) == []
    assert len(store.get_history(session_b)) == 1
    assert store.get_history(session_b)[0].content == "Hello from B"
