"""
ingestion.py
------------
WHAT THIS FILE DOES:
Reads all knowledge-base documents (.md files) from disk and turns them into
a list of `KBDocument` objects that the rest of the RAG pipeline can work with.

Each markdown file is expected to have YAML "front-matter" at the top, like:

    ---
    title: Return & Refund Policy
    category: policy
    doc_id: return_refund_policy
    ---
    # Return & Refund Policy
    ...body text...

The front-matter becomes METADATA (title, category, doc_id) that is later
attached to every chunk of that document, which is what powers the
"Sources" display in the chat UI (see chat/prompts.py + services/support_service.py).

HOW TO ADD A NEW DOCUMENT TO THE KNOWLEDGE BASE:
1. Create a new .md file anywhere under data/knowledge_base/ (in faqs/, policies/,
   or products/, or a new subfolder - it's scanned recursively).
2. Add the front-matter block at the top (title, category, doc_id - doc_id must
   be unique across all files).
3. Go to the Admin page in the UI and click "Rebuild Index" (or run
   `python -m app.rag.vector_store --rebuild` from the command line).
No code changes are needed to add/remove/edit knowledge base content.
"""

import os
from dataclasses import dataclass
import frontmatter  # pip package: python-frontmatter


@dataclass
class KBDocument:
    """A single knowledge-base document, before chunking."""
    content: str          # the raw markdown body (metadata stripped)
    title: str             # human-readable title, shown in "Sources"
    category: str          # e.g. "faq", "policy", "product"
    doc_id: str             # unique id, e.g. "return_refund_policy"
    source_file: str       # relative file path, for debugging/logging


def load_knowledge_base(kb_path: str) -> list[KBDocument]:
    """
    Walks `kb_path` recursively, reads every .md file, parses its front-matter,
    and returns a list of KBDocument objects.

    WHY THIS DESIGN: keeping this function pure (input: folder path, output: list
    of documents) makes it easy to unit test and easy to reuse for a different
    company's knowledge base later - just point it at a different folder.
    """
    if not os.path.isdir(kb_path):
        raise FileNotFoundError(
            f"Knowledge base folder not found: '{kb_path}'. "
            "Check the KNOWLEDGE_BASE_PATH setting in your .env file."
        )

    documents: list[KBDocument] = []

    for root, _dirs, files in os.walk(kb_path):
        for filename in sorted(files):
            if not filename.lower().endswith(".md"):
                continue  # skip non-markdown files (e.g. .gitkeep, images)

            full_path = os.path.join(root, filename)
            relative_path = os.path.relpath(full_path, kb_path)

            with open(full_path, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)

            # Fall back gracefully if a file is missing front-matter fields,
            # rather than crashing the whole ingestion process.
            title = post.metadata.get("title") or filename.replace(".md", "").replace("_", " ").title()
            category = post.metadata.get("category") or "general"
            doc_id = post.metadata.get("doc_id") or filename.replace(".md", "")

            documents.append(
                KBDocument(
                    content=post.content.strip(),
                    title=title,
                    category=category,
                    doc_id=doc_id,
                    source_file=relative_path,
                )
            )

    if not documents:
        raise ValueError(
            f"No markdown documents found under '{kb_path}'. "
            "The knowledge base is empty - add at least one .md file."
        )

    # Sanity check: doc_id should be unique, otherwise metadata/sources get ambiguous.
    seen_ids = set()
    for doc in documents:
        if doc.doc_id in seen_ids:
            raise ValueError(
                f"Duplicate doc_id '{doc.doc_id}' found in '{doc.source_file}'. "
                "Every document must have a unique doc_id in its front-matter."
            )
        seen_ids.add(doc.doc_id)

    return documents
