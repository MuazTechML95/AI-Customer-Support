"""
chunking.py
-----------
WHAT THIS FILE DOES:
Takes each KBDocument (a whole markdown file's content) and:
  1. Cleans the raw text (removes extra whitespace/markdown noise).
  2. Splits it into smaller overlapping "chunks" that are small enough to embed
     and retrieve accurately, but large enough to keep sentences/clauses intact.

WHY CHUNKING MATTERS:
Embedding an entire document as one vector loses precision - a query about
"return shipping cost" would get diluted by an entire policy document's worth
of unrelated content. Chunking lets us retrieve just the most relevant part.

HOW TO CHANGE CHUNKING BEHAVIOR:
- Chunk size / overlap are controlled by CHUNK_SIZE / CHUNK_OVERLAP in .env
  (see app/config.py) - NOT hard-coded here. Change those values, then
  rebuild the index, to make chunks bigger/smaller.
- If you want smarter splitting (e.g. by markdown headers instead of raw
  character count), modify `split_text()` below - the rest of the pipeline
  (embeddings, vector_store) does not need to change.
"""

import re
from dataclasses import dataclass

from app.rag.ingestion import KBDocument


@dataclass
class Chunk:
    """A single chunk of text, ready to be embedded and stored."""
    chunk_id: str      # unique id, e.g. "return_refund_policy_0"
    text: str            # the cleaned chunk text
    document_name: str  # human-readable source document title (for citations)
    category: str        # inherited from parent document
    source: str           # relative file path (for debugging)
    doc_id: str            # parent document id


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Collapses multiple blank lines/spaces into single ones.
    - Strips markdown heading symbols (#) since they add noise to embeddings
      but keeps the heading TEXT itself (headings are often useful context).
    """
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # "## Title" -> "Title"
    text = re.sub(r"[ \t]+", " ", text)          # collapse repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse 3+ newlines to 2
    return text.strip()


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Splits `text` into overlapping chunks of roughly `chunk_size` characters.

    Approach: split on paragraph boundaries first (keeps related sentences
    together), then greedily pack paragraphs into chunks up to chunk_size,
    carrying `chunk_overlap` characters from the end of one chunk into the
    start of the next so context isn't abruptly cut off at chunk boundaries.

    CHANGE THIS FUNCTION if you want different splitting logic (e.g. splitting
    strictly by sentence, or by markdown ## sections).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # If adding this paragraph would overflow the chunk size, close off the
        # current chunk (if it has content) and start a new one.
        if current and len(current) + len(para) + 1 > chunk_size:
            chunks.append(current.strip())
            # Carry the tail of the previous chunk forward as overlap context.
            overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = (overlap_text + "\n" + para).strip()
        else:
            current = (current + "\n" + para).strip() if current else para

        # Edge case: a single paragraph longer than chunk_size on its own.
        # Hard-split it so we never produce a chunk that is wildly oversized.
        while len(current) > chunk_size * 1.5:
            chunks.append(current[:chunk_size].strip())
            current = current[chunk_size - chunk_overlap:]

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_documents(documents: list[KBDocument], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Runs clean_text() + split_text() on every document and returns a flat list
    of Chunk objects with metadata attached (document_name, category, source,
    chunk_id) - this metadata is what later powers the "Sources" shown to users.
    """
    all_chunks: list[Chunk] = []

    for doc in documents:
        cleaned = clean_text(doc.content)
        text_pieces = split_text(cleaned, chunk_size, chunk_overlap)

        for i, piece in enumerate(text_pieces):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_{i}",
                    text=piece,
                    document_name=doc.title,
                    category=doc.category,
                    source=doc.source_file,
                    doc_id=doc.doc_id,
                )
            )

    return all_chunks
