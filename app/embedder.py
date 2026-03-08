"""
embedder.py - Embedding wrapper for the FastAPI service

Loads the sentence-transformer model once at startup and exposes
a simple encode() interface used by both the cache and search layers.

Design decisions:
- Singleton pattern: model is loaded once when the module is first imported
  and reused for every request. Loading takes ~1s; we never want that per-request.
- Same model as build_index.py (all-MiniLM-L6-v2): embeddings must be in the
  same vector space as the indexed documents. Changing the model would
  invalidate the entire Qdrant index.
- normalize_embeddings=True: L2 normalization means cosine similarity
  reduces to dot product — faster and consistent with Qdrant's COSINE distance.
- Max text length 4000 chars: matches the truncation used during indexing.
  Consistency here is important — if we truncate differently at query time,
  the query vector won't align well with document vectors.
"""

import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME    = "all-MiniLM-L6-v2"
VECTOR_DIM    = 384
MAX_TEXT_CHARS = 4000   # must match build_index.py truncation

# ── Singleton model instance ───────────────────────────────────────────────────
_model: SentenceTransformer = None


def get_model() -> SentenceTransformer:
    """
    Return the singleton SentenceTransformer instance.
    Loads on first call, reuses on subsequent calls.
    """
    global _model
    if _model is None:
        log.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        log.info("Embedding model loaded")
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Embed a single text string.
    Returns a normalized 384-dim numpy vector.

    Args:
        text: Raw query or document text

    Returns:
        np.ndarray of shape (384,), L2-normalized
    """
    model = get_model()
    truncated = text[:MAX_TEXT_CHARS]
    vector = model.encode(
        [truncated],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    return vector.astype(np.float32)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a batch of texts.
    Returns a normalized (N x 384) numpy matrix.

    Args:
        texts: List of text strings

    Returns:
        np.ndarray of shape (N, 384), each row L2-normalized
    """
    model = get_model()
    truncated = [t[:MAX_TEXT_CHARS] for t in texts]
    vectors = model.encode(
        truncated,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return vectors.astype(np.float32)