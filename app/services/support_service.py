"""
support_service.py
-------------------
WHAT THIS FILE DOES:
This is the GLUE layer that ties everything together into one simple
function the UI can call: `ask(session_id, question)`.

It orchestrates, in order:
  1. Guardrails check (safety/guardrails.py) - block prompt injection BEFORE
     touching the LLM or vector store.
  2. Session history lookup (chat/session.py) - for follow-up context.
  3. Retrieval (rag/retriever.py) - find relevant knowledge base chunks.
  4. Generation (rag/generator.py) - produce a grounded answer (or safe
     fallback if nothing relevant was found).
  5. Output safety scan (safety/guardrails.py) - catch any accidental
     secret leakage in the LLM's response.
  6. Save both the user's question and the bot's answer to session history.

WHY THIS LAYER EXISTS SEPARATELY FROM THE UI:
Per project requirement #13 (clean architecture) - the Streamlit UI file
(ui/streamlit_app.py) should contain ONLY display/interaction code, not
business logic. This makes the core chatbot logic testable without needing
to run Streamlit at all (see tests/ folder, which imports directly from
this module).

MODEL CACHING:
The embedding model and session store are expensive/stateful, so they are
created ONCE at module import time and reused across every request, rather
than being re-created per call.
"""

import logging

from app.config import settings
from app.chat.session import SessionStore
from app.chat import prompts
from app.rag.embeddings import EmbeddingModel
from app.rag.retriever import retrieve
from app.rag.generator import generate_answer, LLMGenerationError
from app.safety.guardrails import is_prompt_injection, scan_output_for_leakage

logger = logging.getLogger("supportai.support_service")

# Created once and reused (loading the embedding model repeatedly would be slow).
_embedding_model: EmbeddingModel | None = None
_session_store = SessionStore()


def _get_embedding_model() -> EmbeddingModel:
    """Lazily creates the embedding model singleton on first use."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel(settings.embedding_model)
    return _embedding_model


def get_session_store() -> SessionStore:
    """Exposes the shared session store (used by the UI for 'Clear Conversation')."""
    return _session_store


def ask(session_id: str, question: str) -> dict:
    """
    Main function called by the UI (and by tests) for every user message.

    Returns a dict:
        {
            "answer": str,
            "sources": list[str],
            "grounded": bool,
            "blocked_reason": str | None   # "injection" | "error" | None
        }
    """
    question = (question or "").strip()
    if not question:
        return {
            "answer": "Please type a question so I can help you.",
            "sources": [],
            "grounded": False,
            "blocked_reason": None,
        }

    # ------------------------------------------------------------------
    # STEP 1: Guardrail check (pre-LLM). No LLM call is made if this trips.
    # ------------------------------------------------------------------
    if is_prompt_injection(question):
        logger.warning("Blocked a prompt-injection attempt in session %s", session_id)
        _session_store.append_message(session_id, "user", question)
        _session_store.append_message(session_id, "assistant", prompts.INJECTION_REFUSAL_MESSAGE)
        return {
            "answer": prompts.INJECTION_REFUSAL_MESSAGE,
            "sources": [],
            "grounded": False,
            "blocked_reason": "injection",
        }

    # ------------------------------------------------------------------
    # STEP 2: Gather recent conversation history for follow-up context.
    # ------------------------------------------------------------------
    history_text = _session_store.get_recent_history_text(session_id, settings.max_history_turns)

    # ------------------------------------------------------------------
    # STEP 3 + 4: Retrieve relevant chunks, then generate a grounded answer.
    # Wrapped in try/except so vector-store or LLM failures degrade gracefully
    # instead of crashing the whole app (project requirement #17).
    # ------------------------------------------------------------------
    try:
        embedding_model = _get_embedding_model()
        retrieval_result = retrieve(question, embedding_model)
        result = generate_answer(question, retrieval_result, history_text)
    except RuntimeError as exc:
        # Raised by vector_store.get_collection() if the index hasn't been built.
        logger.error("Retrieval failed: %s", exc)
        answer = (
            "The knowledge base index isn't ready yet. "
            "An administrator needs to build the index before I can answer questions."
        )
        _session_store.append_message(session_id, "user", question)
        _session_store.append_message(session_id, "assistant", answer)
        return {"answer": answer, "sources": [], "grounded": False, "blocked_reason": "error"}
    except LLMGenerationError as exc:
        logger.error("Generation failed: %s", exc)
        _session_store.append_message(session_id, "user", question)
        _session_store.append_message(session_id, "assistant", str(exc))
        return {"answer": str(exc), "sources": [], "grounded": False, "blocked_reason": "error"}

    # ------------------------------------------------------------------
    # STEP 5: Output safety scan (catches accidental secret leakage).
    # ------------------------------------------------------------------
    is_safe, safe_answer = scan_output_for_leakage(result["answer"])
    if not is_safe:
        logger.warning("Blocked an LLM response that matched a secret-leak pattern in session %s", session_id)
        result["answer"] = safe_answer
        result["sources"] = []
        result["grounded"] = False

    # ------------------------------------------------------------------
    # STEP 6: Persist to session history (isolated per session_id).
    # ------------------------------------------------------------------
    _session_store.append_message(session_id, "user", question)
    _session_store.append_message(session_id, "assistant", result["answer"], sources=result["sources"])

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "grounded": result["grounded"],
        "blocked_reason": None,
    }
