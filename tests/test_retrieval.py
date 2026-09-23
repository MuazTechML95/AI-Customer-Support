"""
test_retrieval.py
------------------
WHAT THIS TESTS:
That a question CLEARLY covered by the knowledge base (e.g. about the return
policy) produces a grounded answer with at least one valid source name.

HOW TO RUN:
    pytest tests/test_retrieval.py -v

PREREQUISITE:
The vector index must already be built before running tests:
    python -m app.rag.vector_store
This is documented in the README "Testing" section.

NOTE ON LLM CALLS: this test calls the real configured LLM (needs LLM_API_KEY
set in .env). If you want to test WITHOUT spending API calls, see
test_prompt_injection.py and test_session_isolation.py, which don't require
an LLM call to verify their behavior.
"""

import uuid
import pytest

from app.services import support_service


def test_supported_question_returns_grounded_answer_with_sources():
    session_id = str(uuid.uuid4())
    result = support_service.ask(session_id, "What is your return policy?")

    assert result["grounded"] is True, "Expected a grounded (KB-backed) answer"
    assert len(result["sources"]) > 0, "Expected at least one source to be cited"
    assert "Return" in " ".join(result["sources"]) or "Refund" in " ".join(result["sources"]), (
        "Expected the Return & Refund Policy document to be cited as a source"
    )
    assert len(result["answer"]) > 0
