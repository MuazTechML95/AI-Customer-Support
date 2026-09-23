"""
vector_store.py
----------------
WHAT THIS FILE DOES:
Manages the persistent Chroma vector database: building it from scratch,
loading it for querying, rebuilding it when documents change, and reporting
its status (used by the Admin page).

WHY CHROMA:
- Simple, embedded (no separate server process needed), stores data on local
  disk (VECTOR_STORE_PATH in .env), good metadata filtering support.
- Trade-off: not built for massive scale (millions of vectors) - if this
  product ever needs that, swap Chroma for Pinecone/Weaviate/Qdrant. Because
  this file is the ONLY place that talks to Chroma directly, that swap would
  only require changes here, not in retriever.py or generator.py.

DISTANCE METRIC:
The collection is created with cosine distance ("hnsw:space": "cosine"), so
Chroma returns distance in the 0-2 range and (1 - distance) is a meaningful
similarity score in retriever.py. Chroma's default is squared L2, which does
NOT work with 1 - distance. If you change this setting, you MUST rebuild
the index (an existing collection keeps the metric it was created with).

HOW TO REBUILD THE INDEX (e.g. after editing knowledge base files):
- From the Admin page in the UI: click "Rebuild Index" button.
- From the command line: `python -m app.rag.vector_store`
Both paths call `rebuild_index()` below.

METADATA STORED PER CHUNK:
  document_name, category, source, chunk_id, doc_id
This is what lets the UI show "Sources: Return & Refund Policy" instead of
a raw chunk of text.
"""

import shutil
import os
import chromadb

from app.config import settings
from app.rag.ingestion import load_knowledge_base
from app.rag.chunking import chunk_documents
from app.rag.embeddings import EmbeddingModel


def _get_client() -> chromadb.PersistentClient:
    """Creates (or reconnects to) the on-disk Chroma database."""
    os.makedirs(settings.vector_store_path, exist_ok=True)
    return chromadb.PersistentClient(path=settings.vector_store_path)


def build_index(embedding_model: EmbeddingModel) -> dict:
    """
    Builds the vector index FROM SCRATCH:
      knowledge base files -> clean -> chunk -> embed -> store in Chroma.

    This is safe to call even if a collection already exists - it deletes
    the old collection first (see rebuild_index()) to avoid duplicate/stale
    entries (project requirement: "no unnecessary duplicate indexes").

    Returns a small summary dict (used for the Admin "Index Status" display).
    """
    client = _get_client()

    # Drop existing collection if present, so rebuilding never creates duplicates.
    existing = [c.name for c in client.list_collections()]
    if settings.collection_name in existing:
        client.delete_collection(settings.collection_name)

    collection = client.create_collection(
        name=settings.collection_name,
        metadata={
            "description": f"{settings.company_name} knowledge base",
            # ensures distance is cosine distance (0-2 range), so
            # 1 - distance is a meaningful similarity score
            "hnsw:space": "cosine",
        },
    )

    # Step 1-2: load + clean/chunk documents.
    documents = load_knowledge_base(settings.knowledge_base_path)
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)

    if not chunks:
        raise ValueError("No chunks were produced from the knowledge base - check your documents.")

    # Step 3: embed all chunks in one batch call (efficient).
    texts = [c.text for c in chunks]
    vectors = embedding_model.embed_texts(texts)

    # Step 4: store in Chroma with full metadata for later citation.
    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[
            {
                "document_name": c.document_name,
                "category": c.category,
                "source": c.source,
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
            }
            for c in chunks
        ],
    )

    return {
        "documents_indexed": len(documents),
        "chunks_indexed": len(chunks),
        "collection_name": settings.collection_name,
    }


def rebuild_index(embedding_model: EmbeddingModel) -> dict:
    """
    Public entry point for "Rebuild Index" - just calls build_index(), which
    already handles wiping the old collection. Kept as a separate named
    function so the intent is clear at call sites (Admin UI, CLI, tests).
    """
    return build_index(embedding_model)


def get_collection(client=None):
    """
    Returns the existing Chroma collection for querying.
    Raises a clear error if the index hasn't been built yet, instead of a
    confusing low-level Chroma exception - the UI catches this and shows a
    friendly "please build the index first" message.
    """
    client = client or _get_client()
    existing = [c.name for c in client.list_collections()]
    if settings.collection_name not in existing:
        raise RuntimeError(
            f"Vector index '{settings.collection_name}' does not exist yet. "
            "Go to the Admin page and click 'Rebuild Index' first."
        )
    return client.get_collection(settings.collection_name)


def get_index_status() -> dict:
    """
    Returns basic stats about the current index (used by Admin > Index Status).
    Does not raise if the index is missing - returns a status flag instead,
    so the Admin page can render an informative empty state.
    """
    try:
        client = _get_client()
        collection = get_collection(client)
        count = collection.count()
        return {"exists": True, "chunk_count": count, "collection_name": settings.collection_name}
    except RuntimeError:
        return {"exists": False, "chunk_count": 0, "collection_name": settings.collection_name}


def wipe_index() -> None:
    """
    Deletes the ENTIRE vectorstore folder from disk. Used for a full clean
    reset (e.g. in tests, or if the index becomes corrupted). Not exposed in
    the normal Admin UI flow (rebuild_index() is the safer everyday action).
    """
    if os.path.isdir(settings.vector_store_path):
        shutil.rmtree(settings.vector_store_path)


if __name__ == "__main__":
    # Allows rebuilding the index directly from the command line:
    #   python -m app.rag.vector_store
    print(f"Rebuilding index using embedding model: {settings.embedding_model}")
    model = EmbeddingModel(settings.embedding_model)
    result = rebuild_index(model)
    print(f"Done. Indexed {result['documents_indexed']} documents "
          f"into {result['chunks_indexed']} chunks.")