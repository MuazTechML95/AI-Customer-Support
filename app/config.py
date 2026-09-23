"""
config.py
---------
WHAT THIS FILE DOES:
This is the SINGLE SOURCE OF TRUTH for all configuration in the project.
Every other file that needs a setting (API key, model name, folder path, etc.)
imports the `settings` object from this file instead of reading os.environ
directly. This keeps configuration centralized and easy to change.

WHERE VALUES COME FROM:
Values are loaded from a `.env` file (see .env.example for the template) using
python-dotenv. This means secrets (like API keys) never get hard-coded into
the source code and never get committed to git (.env is in .gitignore).

HOW TO CHANGE SETTINGS:
1. Copy `.env.example` to `.env`
2. Fill in your own values (API key, model name, etc.)
3. Restart the app - config.py reads .env once at startup.

WHAT TO EDIT IF YOU WANT TO...
- Switch LLM provider (OpenAI <-> Groq <-> Anthropic): change LLM_PROVIDER and LLM_MODEL in .env
- Switch embedding model: change EMBEDDING_MODEL in .env
- Change how many chunks are retrieved: change RETRIEVAL_TOP_K in .env
- Change the "confidence" cutoff for unsupported questions: change SIMILARITY_THRESHOLD in .env
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load variables from a .env file in the project root into the process environment.
# If .env does not exist, this simply does nothing (no crash) - useful for Docker
# environments where env vars are injected directly instead of via a file.
load_dotenv()

# Providers supported by app/rag/generator.py's _call_llm(). Kept as a single
# constant here so config validation and generator branching never drift apart.
SUPPORTED_LLM_PROVIDERS = ("openai", "groq", "anthropic")


def _get_bool(env_name: str, default: bool) -> bool:
    """Small helper: reads an env var and converts 'true'/'false' strings to bool."""
    val = os.getenv(env_name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # ------------------------------------------------------------------
    # LLM SETTINGS
    # ------------------------------------------------------------------
    # Which LLM provider to call. Supported: "openai", "groq", or "anthropic".
    # CHANGE THIS if you want to switch providers without touching code -
    # generator.py reads this value and branches accordingly.
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))

    # The actual model name sent to the provider's API.
    # Examples: "gpt-4o-mini" (OpenAI), "llama-3.3-70b-versatile" (Groq),
    # "claude-sonnet-4-5" (Anthropic)
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))

    # API key for whichever provider you selected above.
    # NEVER hard-code this. NEVER print/log this value anywhere.
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))

    # Max tokens the LLM is allowed to generate per answer (keeps cost/latency bounded).
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "500")))

    # Temperature controls randomness. Kept LOW (0-0.3) on purpose because this is a
    # factual support bot, not a creative writer - we want consistent, grounded answers.
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2")))

    # ------------------------------------------------------------------
    # EMBEDDING SETTINGS
    # ------------------------------------------------------------------
    # Local, free, offline embedding model (via sentence-transformers).
    # CHANGE THIS if you want a different embedding model. If you switch to an
    # API-based embedding model (e.g. OpenAI's text-embedding-3-small), you would
    # also need to update embeddings.py to call that API instead of a local model.
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # ------------------------------------------------------------------
    # VECTOR STORE SETTINGS
    # ------------------------------------------------------------------
    # Folder where the Chroma persistent database lives on disk.
    vector_store_path: str = field(default_factory=lambda: os.getenv("VECTOR_STORE_PATH", "vectorstore"))

    # Name of the Chroma "collection" (like a table name inside the vector DB).
    collection_name: str = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "novacart_kb"))

    # ------------------------------------------------------------------
    # RAG / RETRIEVAL SETTINGS
    # ------------------------------------------------------------------
    # Folder containing the raw knowledge base documents (.md files).
    knowledge_base_path: str = field(default_factory=lambda: os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base"))

    # Chunk size and overlap (measured in characters here, for simplicity/predictability).
    # CHANGE THESE if your documents are much longer/shorter than typical FAQ/policy text.
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))

    # How many chunks to retrieve per query. Higher = more context but more tokens/cost.
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "4")))

    # Similarity threshold (0-1, higher = stricter). Chroma returns a "distance" - we
    # convert it to a similarity score in retriever.py. If the BEST chunk's similarity
    # is below this threshold, we treat the query as "not covered by the knowledge base"
    # and skip calling the LLM for a KB-grounded answer (this is what prevents
    # hallucination on off-topic questions).
    # CHANGE THIS if the bot is either too strict (rejecting valid questions) or too
    # lenient (answering things it shouldn't) - raise to be stricter, lower to be looser.
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.35")))

    # ------------------------------------------------------------------
    # SESSION SETTINGS
    # ------------------------------------------------------------------
    # How many previous turns (user+bot pairs) to include as conversation context
    # when answering a follow-up question. CHANGE THIS to give the bot a longer or
    # shorter "memory" window.
    max_history_turns: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY_TURNS", "5")))

    # ------------------------------------------------------------------
    # APP SETTINGS
    # ------------------------------------------------------------------
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "SupportAI"))
    company_name: str = field(default_factory=lambda: os.getenv("COMPANY_NAME", "NovaCart"))
    debug: bool = field(default_factory=lambda: _get_bool("DEBUG", False))

    def validate(self) -> list[str]:
        """
        Checks that required settings are present and sane.
        Returns a list of human-readable error strings (empty list = all good).
        CALLED FROM: app/main.py at startup, and shown in the Streamlit UI if it fails,
        so the user gets a friendly message instead of a random crash/stack trace.
        """
        errors = []
        if not self.llm_api_key:
            errors.append(
                "LLM_API_KEY is not set. Copy .env.example to .env and add your API key."
            )
        if self.llm_provider not in SUPPORTED_LLM_PROVIDERS:
            errors.append(
                f"LLM_PROVIDER must be one of {SUPPORTED_LLM_PROVIDERS}, got '{self.llm_provider}'."
            )
        if self.chunk_overlap >= self.chunk_size:
            errors.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            errors.append("SIMILARITY_THRESHOLD must be between 0 and 1.")
        return errors


# A single shared instance imported everywhere else in the app, e.g.:
#   from app.config import settings
#   print(settings.llm_model)
settings = Settings()