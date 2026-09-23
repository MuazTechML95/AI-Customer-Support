"""
test_unsupported_query.py
--------------------------
WHAT THIS TESTS:
That a question with NO relation to the knowledge base (e.g. asking about
Bitcoin price) is safely handled with the canned "not available" message,
instead of the LLM hallucinating an answer.

HOW TO RUN:
    pytest tests/test_unsupported_query.py -v

PREREQUISITE: vector index must be built first (see test_retrieval.py notes).
"""

import uuid

from app.services import support_service
from app.chat.prompts import UNSUPPORTED_QUESTION_MESSAGE


def test_unsupported_question_returns_safe_fallback():
    session_id = str(uuid.uuid4())
    result = support_service.ask(session_id, "Can you tell me today's Bitcoin price?")

    assert result["grounded"] is False, "An off-topic question should never be marked as grounded"
    assert result["sources"] == [], "No sources should be cited for an unsupported question"
    assert result["answer"] == UNSUPPORTED_QUESTION_MESSAGE
