"""
retriever.py
------------
WHAT THIS FILE DOES:
Given a user's question, this is the piece that:
  1. Embeds the question.
  2. Queries Chroma for the top-k most similar chunks.
  3. Converts Chroma's raw "distance" score into an intuitive 0-1 similarity
     score.
  4. Decides whether the retrieved chunks are actually relevant enough to
     answer with (using SIMILARITY_THRESHOLD from config) - this is the
     KEY anti-hallucination safeguard: if nothing relevant was found, we
     tell the caller "no_context", and generator.py will NOT ask the LLM
     to answer from thin air.

HOW TO TUNE RETRIEVAL QUALITY:
- RETRIEVAL_TOP_K (in .env): how many chunks to fetch. More = more context,
  more tokens, slightly higher chance of noise.
- SIMILARITY_THRESHOLD (in .env): raise it to make the bot MORE cautious
  (more "I don't have that information" answers), lower it to make the bot
  MORE willing to attempt an answer from loosely-related chunks.
- Deduplication: if multiple top chunks come from the exact same document
  and say near-identical things, we keep them (Chroma already ranks by
  relevance) but you could add stricter dedup logic here if needed.
"""

from dataclasses import dataclass

from app.config import settings
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import get_collection


@dataclass
class RetrievedChunk:
    text: str
    document_name: str
    category: str
    source: str
    chunk_id: str
    similarity: float  # 0.0 (unrelated) to 1.0 (identical meaning)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    has_sufficient_context: bool  # False => trigger the "unsupported question" flow


def _distance_to_similarity(distance: float) -> float:
    """
    Chroma's default distance metric is squared L2 (or cosine distance
    depending on config) - lower distance = more similar. We convert it to a
    friendlier "similarity" score in roughly the 0-1 range so the rest of the
    app (and SIMILARITY_THRESHOLD in .env) can be reasoned about intuitively
    as "0 = unrelated, 1 = perfect match".

    NOTE: this is an approximation, not a calibrated probability - it's a
    practical, documented heuristic, not a fabricated "confidence score".
    """
    return max(0.0, 1.0 - distance)


def retrieve(query: str, embedding_model: EmbeddingModel, top_k: int | None = None) -> RetrievalResult:
    """
    Main retrieval function. Called by generator.py for every user message.

    Returns a RetrievalResult with:
      - chunks: the retrieved chunks (may be empty)
      - has_sufficient_context: whether the BEST chunk clears the similarity
        threshold. If False, the caller should NOT attempt a KB-grounded
        answer (see generator.py's handling of this flag).
    """
    top_k = top_k or settings.retrieval_top_k
    collection = get_collection()  # raises a friendly error if index doesn't exist

    query_vector = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )

    # Chroma returns parallel lists (documents, metadatas, distances), each
    # wrapped one level deeper because .query() supports batched queries -
    # we only ever send one query at a time, hence the [0] indexing.
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks: list[RetrievedChunk] = []
    for doc_text, meta, distance in zip(documents, metadatas, distances):
        chunks.append(
            RetrievedChunk(
                text=doc_text,
                document_name=meta.get("document_name", "Unknown"),
                category=meta.get("category", "general"),
                source=meta.get("source", ""),
                chunk_id=meta.get("chunk_id", ""),
                similarity=_distance_to_similarity(distance),
            )
        )

    best_similarity = chunks[0].similarity if chunks else 0.0
    has_sufficient_context = best_similarity >= settings.similarity_threshold

    return RetrievalResult(chunks=chunks, has_sufficient_context=has_sufficient_context)
