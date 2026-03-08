"""
build_index.py - Step 2 of the pipeline

Reads cleaned_docs.jsonl, embeds each document using sentence-transformers,
and stores vectors + metadata in a local Qdrant collection.

Design decisions:
- Model: all-MiniLM-L6-v2
    384-dimensional embeddings, fast on CPU, strong semantic quality for
    short-to-medium English text. Tested extensively on 20 Newsgroups in
    the literature. Larger models (e.g. all-mpnet-base-v2) give marginal
    gains but 3x slower — not worth it for a lightweight system.

- Truncation at 512 tokens:
    MiniLM has a 512-token context window. Documents longer than this are
    truncated by the model automatically. We keep full text in the payload
    so the API can return it, but the embedding only reflects the first
    ~380 words. Acceptable tradeoff — most semantic signal is front-loaded.

- Batch size 64:
    Balances GPU/CPU memory usage vs throughput. On CPU, 64 docs per batch
    keeps memory stable without excessive overhead per call.

- Qdrant local mode (no server):
    qdrant_client in local/disk mode gives us persistence without running
    a separate Qdrant server process. The collection lives in
    embeddings/qdrant_index/ and is loaded directly by the FastAPI app.

- Cosine distance:
    Standard for sentence embeddings — vectors are already normalized,
    so dot product == cosine similarity. Used for both retrieval and cache.

- Payload stored per point:
    doc_id, category, all_newsgroups, is_cross_post, subject, word_count,
    and the full cleaned text. The API returns text directly from payload
    without needing to re-read the JSONL file.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    OptimizersConfigDiff,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
JSONL_FILE    = BASE_DIR / "data" / "processed" / "cleaned_docs.jsonl"
QDRANT_PATH   = BASE_DIR / "embeddings" / "qdrant_index"

# ── Config ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
COLLECTION_NAME  = "newsgroups"
VECTOR_DIM       = 384          # all-MiniLM-L6-v2 output dimension
BATCH_SIZE       = 64           # docs per embedding batch
MAX_TEXT_CHARS   = 4000         # truncate text before embedding (~512 tokens)


def load_docs(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load all cleaned documents from JSONL file."""
    docs = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    log.info(f"Loaded {len(docs)} documents from {jsonl_path}")
    return docs


def init_qdrant(qdrant_path: Path) -> QdrantClient:
    """
    Initialize Qdrant client in local disk mode.
    Creates the collection if it doesn't exist.
    Deletes and recreates if it already exists (fresh index build).
    """
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))

    # Drop existing collection if present (idempotent rebuild)
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        log.info(f"Dropping existing collection '{COLLECTION_NAME}' for fresh build")
        client.delete_collection(COLLECTION_NAME)

    # Create collection with cosine distance
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
        ),
        # Disable background optimization during bulk insert for speed
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=0,
        ),
    )
    log.info(f"Created Qdrant collection '{COLLECTION_NAME}' "
             f"(dim={VECTOR_DIM}, distance=COSINE)")
    return client


def embed_batch(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Embed a batch of texts.
    Texts are truncated to MAX_TEXT_CHARS before embedding to avoid
    hitting token limits — semantic signal is front-loaded in these posts.
    """
    truncated = [t[:MAX_TEXT_CHARS] for t in texts]
    embeddings = model.encode(
        truncated,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,   # L2 normalize → cosine = dot product
        convert_to_numpy=True,
    )
    return embeddings


def build_index():
    """
    Main pipeline:
    1. Load cleaned docs
    2. Load embedding model
    3. Embed in batches
    4. Upsert to Qdrant with full payload
    5. Re-enable indexing + report stats
    """
    # ── Load docs ─────────────────────────────────────────────────────────────
    if not JSONL_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned docs not found: {JSONL_FILE}\n"
            "Run scripts/preprocess.py first."
        )
    docs = load_docs(JSONL_FILE)

    # ── Load model ────────────────────────────────────────────────────────────
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    log.info("Model loaded")

    # ── Init Qdrant ───────────────────────────────────────────────────────────
    client = init_qdrant(QDRANT_PATH)

    # ── Embed + upsert in batches ─────────────────────────────────────────────
    total      = len(docs)
    n_batches  = (total + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()

    log.info(f"Embedding {total} documents in {n_batches} batches "
             f"(batch_size={BATCH_SIZE})")

    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end   = min(batch_start + BATCH_SIZE, total)
        batch_docs  = docs[batch_start:batch_end]

        # Embed
        texts      = [d["text"] for d in batch_docs]
        embeddings = embed_batch(model, texts)

        # Build Qdrant points
        points = []
        for i, (doc, vector) in enumerate(zip(batch_docs, embeddings)):
            point_id = batch_start + i   # simple integer ID

            payload = {
                "doc_id":         doc["doc_id"],
                "category":       doc["category"],
                "all_newsgroups": doc["all_newsgroups"],
                "is_cross_post":  doc["is_cross_post"],
                "subject":        doc.get("subject", ""),
                "word_count":     doc["word_count"],
                "text":           doc["text"],
            }

            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload,
            ))

        # Upsert batch to Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        # Progress log every 10 batches
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            elapsed  = time.time() - start_time
            progress = (batch_end / total) * 100
            rate     = batch_end / elapsed if elapsed > 0 else 0
            eta      = (total - batch_end) / rate if rate > 0 else 0
            log.info(
                f"  Batch {batch_idx+1}/{n_batches} | "
                f"{batch_end}/{total} docs ({progress:.1f}%) | "
                f"{rate:.0f} docs/sec | ETA {eta:.0f}s"
            )

    # ── Re-enable indexing ────────────────────────────────────────────────────
    # We disabled indexing during bulk insert for speed.
    # Re-enable it now so queries use the HNSW index for fast retrieval.
    client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,
        ),
    )

    # ── Verify ────────────────────────────────────────────────────────────────
    elapsed_total = time.time() - start_time
    collection_info = client.get_collection(COLLECTION_NAME)
    point_count = collection_info.points_count

    log.info("=" * 60)
    log.info(f"Index build complete in {elapsed_total:.1f}s")
    log.info(f"Points in Qdrant collection : {point_count}")
    log.info(f"Expected                    : {total}")
    log.info(f"Match                       : {point_count == total}")
    log.info(f"Index location              : {QDRANT_PATH}")

    # ── Quick sanity check — retrieve nearest neighbors for one doc ───────────
    log.info("\nSanity check — top 3 neighbors for doc #0:")
    test_vector = embed_batch(model, [docs[0]["text"]])[0]
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=test_vector.tolist(),
        limit=4,   # first result will be itself (score ~1.0)
    )
    for r in results:
        log.info(
            f"  id={r.id} | score={r.score:.4f} | "
            f"category={r.payload['category']} | "
            f"subject={r.payload.get('subject', '')[:50]}"
        )


if __name__ == "__main__":
    build_index()