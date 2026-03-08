"""
search.py - Qdrant document retrieval

Handles all interaction with the Qdrant vector database at query time.
Loads the client once at startup and exposes a simple search interface.

Design decisions:
- Single QdrantClient instance (singleton): opening the local Qdrant DB
  has overhead; we keep one connection open for the app lifetime.
- Top-K retrieval (default 5): returns the 5 most semantically similar
  documents. The result summarizes the top hit and lists others.
- Result format: we return a human-readable string summary rather than
  raw JSON — this is what gets cached and returned in the API response.
  It includes the top document's text excerpt plus category metadata.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from qdrant_client import QdrantClient

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
QDRANT_PATH   = BASE_DIR / "embeddings" / "qdrant_index"
COLLECTION    = "newsgroups"

# ── Singleton client ───────────────────────────────────────────────────────────
_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    """Return singleton Qdrant client, initializing if needed."""
    global _client
    if _client is None:
        if not QDRANT_PATH.exists():
            raise FileNotFoundError(
                f"Qdrant index not found: {QDRANT_PATH}\n"
                "Run scripts/build_index.py first."
            )
        _client = QdrantClient(path=str(QDRANT_PATH))
        log.info(f"Qdrant client initialized: {QDRANT_PATH}")
    return _client


def search(
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> str:
    """
    Search Qdrant for the most semantically similar documents.

    Args:
        query_embedding: 384-dim normalized query vector
        top_k:           Number of results to retrieve

    Returns:
        A formatted string summarizing the top results.
        This string is what gets stored in the cache and returned by the API.
    """
    client = get_client()

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_embedding.tolist(),
        limit=top_k,
        with_payload=True,
    ).points

    if not results:
        return "No relevant documents found."

    # Format results into a readable summary
    lines = []
    for i, hit in enumerate(results):
        payload  = hit.payload
        category = payload.get("category", "unknown")
        subject  = payload.get("subject", "")
        text     = payload.get("text", "")
        score    = hit.score

        # Excerpt: first 300 chars of the document text
        excerpt = text[:300].replace("\n", " ").strip()
        if len(text) > 300:
            excerpt += "..."

        lines.append(
            f"[{i+1}] category={category} | score={score:.4f}\n"
            f"    subject: {subject}\n"
            f"    excerpt: {excerpt}"
        )

    return "\n\n".join(lines)


def get_collection_info() -> dict:
    """Return basic info about the Qdrant collection."""
    client = get_client()
    info = client.get_collection(COLLECTION)
    return {
        "points_count": info.points_count,
        "status":       str(info.status),
    }