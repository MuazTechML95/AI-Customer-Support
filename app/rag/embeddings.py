"""
embeddings.py
-------------
WHAT THIS FILE DOES:
Wraps whatever embedding model we're using behind one simple class,
`EmbeddingModel`, so the rest of the codebase (vector_store.py, retriever.py)
never has to know or care WHICH embedding model is being used underneath.

CURRENT MODEL:
sentence-transformers/all-MiniLM-L6-v2
  - Runs 100% locally (no API key, no internet needed after first download).
  - Free, fast, good enough quality for FAQ/policy-style short text.
  - Produces 384-dimensional vectors.

HOW TO SWITCH EMBEDDING MODELS:
Option A (another local sentence-transformers model):
  - Change EMBEDDING_MODEL in .env to any model name from
    https://www.sbert.net/docs/pretrained_models.html
  - No code changes needed.

Option B (switch to an API-based embedding model, e.g. OpenAI):
  - You WOULD need to edit the `embed_texts()` and `embed_query()` methods
    below to call that API instead of self._model.encode(...).
  - Everything else in the pipeline (vector_store, retriever) stays the same
    because they only depend on this class's public methods.

WHY EMBEDDING LOGIC IS SEPARATED FROM APPLICATION LOGIC:
Per project requirement #3 - this file has ONE job (turn text into vectors).
It is not mixed into ingestion, retrieval, or UI code, so it can be tested,
swapped, or upgraded independently.
"""

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Thin wrapper around a sentence-transformers model.

    USAGE:
        model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
        vectors = model.embed_texts(["hello world", "return policy"])
        query_vector = model.embed_query("what is your return policy?")
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        # Loading the model is somewhat slow (a few seconds) - this class is
        # meant to be instantiated ONCE and reused, not re-created per request.
        # See services/support_service.py for how it's cached.
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a BATCH of texts at once (used during index building - much
        faster than embedding one chunk at a time).
        Returns a list of vectors (list of floats), one per input text.
        """
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embeds a SINGLE user query at request time.
        Kept as a separate method (even though it just calls embed_texts under
        the hood) because some embedding models use a different mode/prefix
        for queries vs. documents - this is the single place you'd add that.
        """
        return self.embed_texts([query])[0]
