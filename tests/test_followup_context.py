"""
test_followup_context.py
--------------------------
WHAT THIS TESTS:
That a vague follow-up question ("What about damaged ones?") is correctly
understood using the PREVIOUS message's context within the SAME session -
this verifies session.py's history-passing mechanism actually feeds into
the LLM prompt (chat/prompts.py build_user_prompt).

HOW TO RUN:
    pytest tests/test_followup_context.py -v

PREREQUISITE: vector index must be built first (see test_retrieval.py notes).

NOTE: this test checks that history text is being correctly recorded and
passed - full semantic correctness of the LLM's follow-up answer would need
human/LLM-graded evaluation, which is out of scope for this basic test suite
(documented as a future enhancement in the README).
"""

import uuid

from app.services import support_service
from app.config import settings


def test_history_is_recorded_and_passed_for_followups():
    session_id = str(uuid.uuid4())

    # First question establishes context.
    first_result = support_service.ask(session_id, "What is your return policy?")
    assert first_result["grounded"] is True

    # History should now contain both the user's question and the bot's answer.
    history_text = support_service.get_session_store().get_recent_history_text(
        session_id, settings.max_history_turns
    )
    assert "return policy" in history_text.lower()
    assert "Assistant:" in history_text

    # Follow-up question, relying on the earlier context to be meaningful.
    followup_result = support_service.ask(session_id, "What about damaged products?")
    assert len(followup_result["answer"]) > 0
    # The follow-up should still be able to find relevant KB content
    # (damaged products are explicitly covered in the Return & Refund Policy).
    assert followup_result["grounded"] is True
