"""
Tests for app/main.py FastAPI endpoints.

All external dependencies (embedding model, Qdrant, cluster models) are mocked
so these tests run without any pre-built index or trained models.

Endpoints covered:
    POST   /query
    GET    /cache/stats
    DELETE /cache
    GET    /health
"""

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── Shared test fixtures ───────────────────────────────────────────────────────

_FAKE_EMB = np.random.default_rng(42).standard_normal(384).astype(np.float32)
_FAKE_EMB /= np.linalg.norm(_FAKE_EMB)

_FAKE_MEMS = np.array([0.7, 0.2, 0.05, 0.03, 0.02], dtype=np.float32)

_FAKE_RESULT = (
    "[1] category=alt.atheism | score=0.9500\n"
    "    subject: Test document\n"
    "    excerpt: Sample text about atheism and religion."
)

# Functions imported into app.main's namespace that need to be patched
_LIFESPAN_PATCHES = {
    "app.main.get_model": {},
    "app.main.load_cluster_models": {},
    "app.main.get_collection_info": {"return_value": {"points_count": 18000, "status": "ok"}},
}

_REQUEST_PATCHES = {
    "app.main.embed_text": {"return_value": _FAKE_EMB},
    "app.main.get_soft_memberships": {"return_value": _FAKE_MEMS},
    "app.main.get_top_clusters": {"return_value": [0, 1]},
    "app.main.get_dominant_cluster": {"return_value": 0},
    "app.main.search": {"return_value": _FAKE_RESULT},
}


@pytest.fixture
def client():
    """
    TestClient with all external dependencies mocked.

    Each test gets a fresh client (fresh lifespan → fresh SemanticCache).
    The real SemanticCache is used, so cache hit/miss behavior is tested end-to-end.
    """
    all_patches = {**_LIFESPAN_PATCHES, **_REQUEST_PATCHES}
    with ExitStack() as stack:
        for target, kwargs in all_patches.items():
            stack.enter_context(patch(target, **kwargs))
        from app.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── POST /query ────────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_cache_miss_on_first_query(self, client):
        resp = client.post("/query", json={"query": "What is atheism?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["cache_hit"] is False
        assert body["matched_query"] is None
        assert body["similarity_score"] is None
        assert body["result"] == _FAKE_RESULT
        assert body["dominant_cluster"] == 0
        assert body["query"] == "What is atheism?"

    def test_cache_hit_on_repeated_query(self, client):
        # First call: miss → result stored
        r1 = client.post("/query", json={"query": "What is atheism?"})
        assert r1.json()["cache_hit"] is False

        # Second call: same query embedding → dot product = 1.0 → hit
        r2 = client.post("/query", json={"query": "What is atheism?"})
        assert r2.status_code == 200
        body = r2.json()
        assert body["cache_hit"] is True
        assert body["matched_query"] == "What is atheism?"
        assert body["similarity_score"] == pytest.approx(1.0, abs=1e-4)
        assert body["result"] == _FAKE_RESULT

    def test_cache_hit_returns_same_result(self, client):
        client.post("/query", json={"query": "first query"})
        r2 = client.post("/query", json={"query": "second query"})
        # Both produce the same embedding (mocked), so second hits cache
        assert r2.json()["result"] == _FAKE_RESULT

    def test_empty_query_returns_400(self, client):
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_whitespace_only_query_returns_400(self, client):
        resp = client.post("/query", json={"query": "   "})
        assert resp.status_code == 400

    def test_query_is_stripped_in_response(self, client):
        resp = client.post("/query", json={"query": "  atheism  "})
        assert resp.status_code == 200
        assert resp.json()["query"] == "atheism"

    def test_custom_top_k_accepted(self, client):
        resp = client.post("/query", json={"query": "test query", "top_k": 10})
        assert resp.status_code == 200

    def test_default_top_k_works(self, client):
        resp = client.post("/query", json={"query": "test query"})
        assert resp.status_code == 200

    def test_response_schema(self, client):
        resp = client.post("/query", json={"query": "test"})
        body = resp.json()
        assert set(body.keys()) == {
            "query", "cache_hit", "matched_query",
            "similarity_score", "result", "dominant_cluster"
        }

    def test_dominant_cluster_in_response(self, client):
        resp = client.post("/query", json={"query": "test"})
        assert resp.json()["dominant_cluster"] == 0  # matches mock return value


# ── GET /cache/stats ───────────────────────────────────────────────────────────

class TestCacheStatsEndpoint:
    def test_stats_empty_on_fresh_start(self, client):
        resp = client.get("/cache/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_entries"] == 0
        assert body["hit_count"] == 0
        assert body["miss_count"] == 0
        assert body["hit_rate"] == 0.0

    def test_stats_after_one_miss(self, client):
        client.post("/query", json={"query": "first"})
        stats = client.get("/cache/stats").json()
        assert stats["total_entries"] == 1
        assert stats["miss_count"] == 1
        assert stats["hit_count"] == 0

    def test_stats_after_miss_then_hit(self, client):
        client.post("/query", json={"query": "q"})   # miss
        client.post("/query", json={"query": "q"})   # hit
        stats = client.get("/cache/stats").json()
        assert stats["total_entries"] == 1
        assert stats["miss_count"] == 1
        assert stats["hit_count"] == 1
        assert stats["hit_rate"] == pytest.approx(0.5)

    def test_stats_schema(self, client):
        resp = client.get("/cache/stats")
        assert set(resp.json().keys()) == {
            "total_entries", "hit_count", "miss_count", "hit_rate"
        }


# ── DELETE /cache ──────────────────────────────────────────────────────────────

class TestFlushCacheEndpoint:
    def test_flush_empty_cache(self, client):
        resp = client.delete("/cache")
        assert resp.status_code == 200
        body = resp.json()
        assert body["entries_cleared"] == 0
        assert "flushed" in body["message"].lower()

    def test_flush_reports_cleared_count(self, client):
        client.post("/query", json={"query": "q1"})
        client.post("/query", json={"query": "q2"})
        # Both get same embedding, so second is a cache hit (no new entry).
        # Only 1 unique entry is stored.
        resp = client.delete("/cache")
        assert resp.json()["entries_cleared"] == 1

    def test_flush_resets_stats(self, client):
        client.post("/query", json={"query": "q"})
        client.delete("/cache")
        stats = client.get("/cache/stats").json()
        assert stats["total_entries"] == 0
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0

    def test_flush_then_query_is_miss(self, client):
        client.post("/query", json={"query": "q"})
        client.delete("/cache")
        resp = client.post("/query", json={"query": "q"})
        assert resp.json()["cache_hit"] is False

    def test_flush_response_schema(self, client):
        resp = client.delete("/cache")
        assert set(resp.json().keys()) == {"message", "entries_cleared"}


# ── GET /health ────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_reports_cache_entries(self, client):
        client.post("/query", json={"query": "test"})
        health = client.get("/health").json()
        assert health["cache_entries"] == 1

    def test_health_reports_threshold(self, client):
        health = client.get("/health").json()
        assert "threshold" in health
        assert health["threshold"] is not None
