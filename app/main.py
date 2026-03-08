"""
main.py - FastAPI service

Exposes three endpoints:
    POST   /query         - semantic search with cache layer
    GET    /cache/stats   - current cache state
    DELETE /cache         - flush cache and reset stats

State management:
    All shared state (cache, models) is initialized once during app startup
    using FastAPI's lifespan context manager. This ensures:
    - Models are loaded before the first request arrives
    - The same cache instance is shared across all requests
    - Clean shutdown when the server stops

Start the server with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.embedder import embed_text, get_model
from app.cluster  import load_models as load_cluster_models
from app.cluster  import get_soft_memberships, get_top_clusters, get_dominant_cluster
from app.cache    import SemanticCache, DEFAULT_THRESHOLD
from app.search   import search, get_client, get_collection_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Global state ───────────────────────────────────────────────────────────────
# Single cache instance shared across all requests
_cache: Optional[SemanticCache] = None


# ── Lifespan: startup + shutdown ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all models and state at startup.
    Runs before the first request is served.
    """
    global _cache

    log.info("=" * 50)
    log.info("Starting up Newsgroups Semantic Search API")
    log.info("=" * 50)

    # 1. Load embedding model (sentence-transformers)
    log.info("Loading embedding model...")
    get_model()

    # 2. Load cluster models (UMAP + FCM)
    log.info("Loading cluster models...")
    load_cluster_models()

    # 3. Initialize Qdrant client
    log.info("Connecting to Qdrant index...")
    info = get_collection_info()
    log.info(f"Qdrant ready: {info['points_count']} documents indexed")

    # 4. Initialize semantic cache
    _cache = SemanticCache(threshold=DEFAULT_THRESHOLD)
    log.info(f"Semantic cache ready (threshold={DEFAULT_THRESHOLD})")

    log.info("API startup complete. Ready to serve requests.")
    log.info("=" * 50)

    yield  # Server runs here

    # Shutdown
    log.info("Shutting down...")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Newsgroups Semantic Search",
    description=(
        "Semantic search over the 20 Newsgroups corpus with "
        "fuzzy clustering and a cluster-aware semantic cache."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ──────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5   # optional: number of results to retrieve


class QueryResponse(BaseModel):
    query:            str
    cache_hit:        bool
    matched_query:    Optional[str]
    similarity_score: Optional[float]
    result:           str
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count:     int
    miss_count:    int
    hit_rate:      float


class CacheDeleteResponse(BaseModel):
    message: str
    entries_cleared: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main search endpoint.

    Flow:
        1. Validate query
        2. Embed query → 384-dim vector
        3. Get soft cluster memberships → find top-2 clusters
        4. Check semantic cache (search only in relevant clusters)
        5a. Cache HIT  → return cached result immediately
        5b. Cache MISS → search Qdrant, store result, return
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_text = request.query.strip()
    log.info(f"Query: '{query_text[:80]}'")

    # ── Step 1: Embed query ────────────────────────────────────────────────────
    embedding = embed_text(query_text)

    # ── Step 2: Get cluster memberships ───────────────────────────────────────
    memberships      = get_soft_memberships(embedding)
    top_clusters     = get_top_clusters(memberships, top_k=2)
    dominant_cluster = get_dominant_cluster(memberships)

    log.info(
        f"Cluster memberships: top={top_clusters} | "
        f"dominant={dominant_cluster} | "
        f"weights={[round(float(memberships[c]), 3) for c in top_clusters]}"
    )

    # ── Step 3: Cache lookup ───────────────────────────────────────────────────
    cache_result = _cache.lookup(embedding, top_clusters)

    if cache_result is not None:
        # ── Cache HIT ─────────────────────────────────────────────────────────
        entry, similarity = cache_result
        log.info(f"Cache HIT | similarity={similarity:.4f}")

        return QueryResponse(
            query            = query_text,
            cache_hit        = True,
            matched_query    = entry.query,
            similarity_score = round(similarity, 4),
            result           = entry.result,
            dominant_cluster = dominant_cluster,
        )

    # ── Cache MISS: search Qdrant ──────────────────────────────────────────────
    log.info("Cache MISS — searching Qdrant...")
    result = search(embedding, top_k=request.top_k)

    # Store in cache for future queries
    _cache.store(
        query            = query_text,
        query_embedding  = embedding,
        result           = result,
        dominant_cluster = dominant_cluster,
        memberships      = memberships,
    )

    log.info(f"Result stored in cache (cluster={dominant_cluster})")

    return QueryResponse(
        query            = query_text,
        cache_hit        = False,
        matched_query    = None,
        similarity_score = None,
        result           = result,
        dominant_cluster = dominant_cluster,
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """
    Return current cache statistics.

    Returns total entries, hit/miss counts, and hit rate.
    """
    stats = _cache.get_stats()
    return CacheStatsResponse(
        total_entries = stats["total_entries"],
        hit_count     = stats["hit_count"],
        miss_count    = stats["miss_count"],
        hit_rate      = stats["hit_rate"],
    )


@app.delete("/cache", response_model=CacheDeleteResponse)
async def flush_cache():
    """
    Flush the entire cache and reset all statistics.
    """
    entries_before = _cache.total_entries
    _cache.flush()
    log.info(f"Cache flushed via API: {entries_before} entries cleared")

    return CacheDeleteResponse(
        message        = "Cache flushed successfully",
        entries_cleared = entries_before,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status":        "ok",
        "cache_entries": _cache.total_entries if _cache else 0,
        "threshold":     _cache.threshold if _cache else None,
    }