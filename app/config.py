""""
config.py
---------
Central configuration for the SupportAI application.

Configuration priority:
1. Environment variables / Streamlit Cloud Secrets
2. Local .env file
3. Safe default values

For production, API keys must NEVER be hard-coded.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------

# Loads values from local .env when running locally.
# In Streamlit Cloud, values can be provided through Secrets/environment vars.
load_dotenv()


# ---------------------------------------------------------------------------
# SUPPORTED LLM PROVIDERS
# ---------------------------------------------------------------------------

SUPPORTED_LLM_PROVIDERS = (
    "openai",
    "groq",
    "anthropic",
)


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def _get_bool(env_name: str, default: bool) -> bool:
    """Read an environment variable and convert it to a boolean."""

    value = os.getenv(env_name)

    if value is None:
        return default

    return value.strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

@dataclass
class Settings:

    # ------------------------------------------------------------------
    # LLM SETTINGS
    # ------------------------------------------------------------------

    # Groq is the default provider for this project.
    llm_provider: str = field(
        default_factory=lambda: os.getenv(
            "LLM_PROVIDER",
            "groq",
        )
    )

    # Groq model used by the application.
    #
    # This can be overridden through:
    # LLM_MODEL
    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL",
            "openai/gpt-oss-120b",
        )
    )

    # API key.
    #
    # NEVER hard-code the real API key here.
    llm_api_key: str = field(
        default_factory=lambda: os.getenv(
            "LLM_API_KEY",
            "",
        )
    )

    # Maximum number of generated tokens.
    llm_max_tokens: int = field(
        default_factory=lambda: int(
            os.getenv(
                "LLM_MAX_TOKENS",
                "500",
            )
        )
    )

    # Lower temperature keeps customer-support answers
    # more consistent and factual.
    llm_temperature: float = field(
        default_factory=lambda: float(
            os.getenv(
                "LLM_TEMPERATURE",
                "0.2",
            )
        )
    )

    # ------------------------------------------------------------------
    # EMBEDDING SETTINGS
    # ------------------------------------------------------------------

    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )

    # ------------------------------------------------------------------
    # VECTOR STORE SETTINGS
    # ------------------------------------------------------------------

    vector_store_path: str = field(
        default_factory=lambda: os.getenv(
            "VECTOR_STORE_PATH",
            "vectorstore",
        )
    )

    collection_name: str = field(
        default_factory=lambda: os.getenv(
            "COLLECTION_NAME",
            "novacart_kb",
        )
    )

    # ------------------------------------------------------------------
    # RAG / RETRIEVAL SETTINGS
    # ------------------------------------------------------------------

    knowledge_base_path: str = field(
        default_factory=lambda: os.getenv(
            "KNOWLEDGE_BASE_PATH",
            "data/knowledge_base",
        )
    )

    chunk_size: int = field(
        default_factory=lambda: int(
            os.getenv(
                "CHUNK_SIZE",
                "800",
            )
        )
    )

    chunk_overlap: int = field(
        default_factory=lambda: int(
            os.getenv(
                "CHUNK_OVERLAP",
                "100",
            )
        )
    )

    retrieval_top_k: int = field(
        default_factory=lambda: int(
            os.getenv(
                "RETRIEVAL_TOP_K",
                "4",
            )
        )
    )

    similarity_threshold: float = field(
        default_factory=lambda: float(
            os.getenv(
                "SIMILARITY_THRESHOLD",
                "0.35",
            )
        )
    )

    # ------------------------------------------------------------------
    # SESSION SETTINGS
    # ------------------------------------------------------------------

    max_history_turns: int = field(
        default_factory=lambda: int(
            os.getenv(
                "MAX_HISTORY_TURNS",
                "5",
            )
        )
    )

    # ------------------------------------------------------------------
    # APP SETTINGS
    # ------------------------------------------------------------------

    app_name: str = field(
        default_factory=lambda: os.getenv(
            "APP_NAME",
            "SupportAI",
        )
    )

    company_name: str = field(
        default_factory=lambda: os.getenv(
            "COMPANY_NAME",
            "NovaCart",
        )
    )

    debug: bool = field(
        default_factory=lambda: _get_bool(
            "DEBUG",
            False,
        )
    )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Validate application configuration.

        Returns:
            list[str]: Empty list when configuration is valid.
        """

        errors = []

        # API key must exist.
        if not self.llm_api_key:
            errors.append(
                "LLM_API_KEY is not set. "
                "Add your API key through .env locally "
                "or Streamlit Secrets in production."
            )

        # Provider must be supported.
        if self.llm_provider not in SUPPORTED_LLM_PROVIDERS:
            errors.append(
                f"LLM_PROVIDER must be one of "
                f"{SUPPORTED_LLM_PROVIDERS}, "
                f"got '{self.llm_provider}'."
            )

        # Chunk configuration.
        if self.chunk_overlap >= self.chunk_size:
            errors.append(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
            )

        # Similarity threshold.
        if not 0.0 <= self.similarity_threshold <= 1.0:
            errors.append(
                "SIMILARITY_THRESHOLD must be between 0 and 1."
            )

        # Token configuration.
        if self.llm_max_tokens <= 0:
            errors.append(
                "LLM_MAX_TOKENS must be greater than 0."
            )

        # Temperature configuration.
        if not 0.0 <= self.llm_temperature <= 2.0:
            errors.append(
                "LLM_TEMPERATURE must be between 0 and 2."
            )

        return errors


# ---------------------------------------------------------------------------
# SHARED SETTINGS INSTANCE
# ---------------------------------------------------------------------------

settings = Settings()

