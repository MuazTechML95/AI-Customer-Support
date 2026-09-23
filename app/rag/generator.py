"""
generator.py
------------
WHAT THIS FILE DOES:
Takes a user's question + retrieved knowledge base chunks + conversation
history, builds the final prompt (using chat/prompts.py templates), and
calls the configured LLM provider to get a grounded answer.

THIS IS ALSO WHERE THE "NO CONTEXT => NO HALLUCINATION" RULE IS ENFORCED:
If retriever.py determined that no chunk was similar enough to the question
(has_sufficient_context=False), this file does NOT call the LLM at all - it
returns the canned UNSUPPORTED_QUESTION_MESSAGE directly.

HOW TO SWITCH LLM PROVIDERS:
Set LLM_PROVIDER=openai, LLM_PROVIDER=groq, or LLM_PROVIDER=anthropic in .env
(see app/config.py). This file branches on that value in `_call_llm()`.
Groq uses the OpenAI-compatible SDK with a different base_url, since Groq
hosts open models (like Llama) behind an OpenAI-style API for free.

ERROR HANDLING:
Any failure calling the LLM API (timeout, invalid key, rate limit, network
error) is logged internally and raised so the actual error can be diagnosed
during development.
"""

import logging

from app.config import settings
from app.chat import prompts
from app.rag.retriever import RetrievalResult

logger = logging.getLogger("supportai.generator")


class LLMGenerationError(Exception):
    """Raised when the LLM call fails for any reason (network, auth, timeout)."""
    pass


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Makes the actual API call to whichever provider is configured.
    Isolated into its own function so swapping providers or adding retry
    logic doesn't touch the rest of generate_answer().
    """
    try:
        if settings.llm_provider in ("openai", "groq"):
            from openai import OpenAI

            if settings.llm_provider == "groq":
                # Groq hosts open models (Llama, etc.) behind an
                # OpenAI-compatible API — same SDK, different base_url.
                client = OpenAI(
                    api_key=settings.llm_api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
            else:
                client = OpenAI(api_key=settings.llm_api_key)

            response = client.chat.completions.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            return response.choices[0].message.content.strip()

        elif settings.llm_provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=settings.llm_api_key)

            response = client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )

            return response.content[0].text.strip()

        else:
            raise LLMGenerationError(
                f"Unsupported LLM_PROVIDER: {settings.llm_provider}"
            )

    except LLMGenerationError:
        raise

    except Exception as exc:
        logger.error(
            "LLM call failed: %s",
            exc,
            exc_info=True
        )
        raise


def generate_answer(
    question: str,
    retrieval_result: RetrievalResult,
    history_text: str
) -> dict:
    """
    Main entry point used by services/support_service.py.

    Returns a dict:
        {
            "answer": str,
            "sources": list[str],   # empty if no KB context was used
            "grounded": bool,       # False when we returned the safe fallback
        }
    """

    if (
        not retrieval_result.has_sufficient_context
        or not retrieval_result.chunks
    ):
        return {
            "answer": prompts.UNSUPPORTED_QUESTION_MESSAGE,
            "sources": [],
            "grounded": False,
        }

    context_chunks = [
        c.text for c in retrieval_result.chunks
    ]

    user_prompt = prompts.build_user_prompt(
        question,
        context_chunks,
        history_text
    )

    answer_text = _call_llm(
        prompts.SYSTEM_PROMPT,
        user_prompt
    )

    # The LLM is instructed (SYSTEM_PROMPT rule #2) to say it doesn't have
    # approved information when the retrieved context doesn't actually
    # answer the question. Even when retrieval found chunks above the
    # similarity threshold, they may not be relevant enough for the LLM to
    # answer from - so we trust the LLM's own refusal wording here, not
    # just the fact that some chunks were retrieved.
    refusal_markers = (
        "don't have approved information",
        "do not have approved information",
        "i don't have information",
        "i do not have information",
    )
    if any(marker in answer_text.lower() for marker in refusal_markers):
        return {
            "answer": answer_text,
            "sources": [],
            "grounded": False,
        }

    seen = set()
    sources = []

    for chunk in retrieval_result.chunks:
        if chunk.document_name not in seen:
            sources.append(chunk.document_name)
            seen.add(chunk.document_name)

    return {
        "answer": answer_text,
        "sources": sources,
        "grounded": True
    }