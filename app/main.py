"""
main.py
-------
WHAT THIS FILE DOES:
Provides `run_startup_checks()`, called once when the Streamlit app boots
(see ui/streamlit_app.py). It validates configuration and checks whether the
vector index exists, returning a clear list of problems (if any) so the UI
can show a friendly setup message instead of crashing halfway through a
chat interaction.

This file intentionally does NOT contain business logic (that lives in
services/support_service.py) or UI code (that lives in ui/streamlit_app.py) -
it is purely a startup/bootstrapping helper, per the clean-architecture
requirement.
"""

import logging

from app.config import settings
from app.rag.vector_store import get_index_status

# Basic structured logging setup. Only goes to console/stdout by default -
# in Docker this is captured by `docker logs`. Never logs secrets (see
# app/config.py - llm_api_key is never passed to logger calls anywhere).
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def run_startup_checks() -> dict:
    """
    Returns:
        {
            "config_errors": list[str],   # from settings.validate()
            "index_status": dict,          # from vector_store.get_index_status()
            "ready": bool                    # True only if no errors AND index exists
        }
    """
    config_errors = settings.validate()
    index_status = get_index_status()

    ready = (len(config_errors) == 0) and index_status["exists"]

    return {
        "config_errors": config_errors,
        "index_status": index_status,
        "ready": ready,
    }
