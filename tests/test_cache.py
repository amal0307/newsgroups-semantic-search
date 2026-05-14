"""
Tests for app/cache.py - SemanticCache and CacheEntry

Fully isolated: no external models, no Qdrant, pure numpy only.
"""

import time

import numpy as np
import pytest

from app.cache import CacheEntry, SemanticCache, DEFAULT_THRESHOLD


# ── Helpers ────────────────────────────────────────────────────────────────────

def _emb(seed: int = 0, dim: int = 384) -> np.ndarray:
    """Return a normalized random float32 vector."""
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _mem(k: int = 5, dominant: int = 0) -> np.ndarray:
    """Return a soft membership vector with a clear dominant cluster."""
    m = np.full(k, 0.05, dtype=np.float32)
    m[dominant] = 1.0 - 0.05 * (k - 1)
    return m / m.sum()


def _perturb(emb: np.ndarray, scale: float, seed: int = 99) -> np.ndarray:
    """Add small Gaussian noise and re-normalize."""
    noise = np.random.default_rng(seed).standard_normal(len(emb)).astype(np.float32)
    v = emb + noise * scale
    return v / np.linalg.norm(v)


# ── CacheEntry ─────────────────────────────────────────────────────────────────

class TestCacheEntry:
    def test_fields_set_correctly(self):
        emb = _emb()
        m = _mem()
        entry = CacheEntry(query="q", embedding=emb, result="r", cluster_id=3, memberships=m)

        assert entry.query == "q"
        assert entry.result == "r"
        assert entry.cluster_id == 3
        assert entry.hit_count == 0
        assert entry.timestamp <= time.time()

    def test_hit_count_starts_at_zero(self):
        entry = CacheEntry(query="q", embedding=_emb(), result="r", cluster_id=0, memberships=_mem())
        assert entry.hit_count == 0


# ── Initialization ─────────────────────────────────────────────────────────────

class TestSemanticCacheInit:
    def test_default_threshold(self):
        assert SemanticCache().threshold == DEFAULT_THRESHOLD

    def test_custom_threshold(self):
        assert SemanticCache(threshold=0.90).threshold == 0.90

    def test_starts_empty(self):
        cache = SemanticCache()
        assert cache.total_entries == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0
        assert cache.hit_rate == 0.0

    def test_repr(self):
        cache = SemanticCache(threshold=0.80)
        r = repr(cache)
        assert "SemanticCache" in r
        assert "0.8" in r


# ── Store ──────────────────────────────────────────────────────────────────────

class TestSemanticCacheStore:
    def test_returns_cache_entry(self):
        cache = SemanticCache()
        entry = cache.store("q", _emb(), "r", 0, _mem())
        assert isinstance(entry, CacheEntry)

    def test_total_entries_increments(self):
        cache = SemanticCache()
        for i in range(4):
            cache.store(f"q{i}", _emb(i), f"r{i}", 0, _mem())
        assert cache.total_entries == 4

    def test_entries_in_correct_cluster(self):
        cache = SemanticCache()
        cache.store("q0", _emb(0), "r0", dominant_cluster=0, memberships=_mem(dominant=0))
        cache.store("q1", _emb(1), "r1", dominant_cluster=1, memberships=_mem(dominant=1))
        cache.store("q2", _emb(2), "r2", dominant_cluster=1, memberships=_mem(dominant=1))

        assert len(cache.get_cluster_entries(0)) == 1
        assert len(cache.get_cluster_entries(1)) == 2

    def test_embedding_is_copied(self):
        """Mutating the original embedding must not affect the stored entry."""
        cache = SemanticCache()
        emb = _emb()
        original = emb.copy()
        cache.store("q", emb, "r", 0, _mem())
        emb[:] = 0.0

        stored = cache.get_cluster_entries(0)[0]
        assert np.allclose(stored.embedding, original)

    def test_memberships_is_copied(self):
        cache = SemanticCache()
        m = _mem()
        original = m.copy()
        cache.store("q", _emb(), "r", 0, m)
        m[:] = 0.0

        stored = cache.get_cluster_entries(0)[0]
        assert np.allclose(stored.memberships, original)


# ── Lookup ─────────────────────────────────────────────────────────────────────

class TestSemanticCacheLookup:
    def test_miss_on_empty_cache(self):
        cache = SemanticCache()
        assert cache.lookup(_emb(), [0]) is None

    def test_miss_on_wrong_cluster(self):
        """Entry stored in cluster 0; lookup in cluster 1 must miss."""
        cache = SemanticCache(threshold=0.80)
        emb = _emb()
        cache.store("q", emb, "r", dominant_cluster=0, memberships=_mem(dominant=0))
        assert cache.lookup(emb, [1]) is None

    def test_exact_match_always_hits(self):
        """Dot product of a normalized vector with itself = 1.0."""
        cache = SemanticCache(threshold=0.99)
        emb = _emb()
        cache.store("q", emb, "r", 0, _mem())
        result = cache.lookup(emb, [0])
        assert result is not None
        entry, score = result
        assert score == pytest.approx(1.0, abs=1e-5)
        assert entry.query == "q"

    def test_returns_best_match(self):
        """When multiple entries are present, the closest one is returned."""
        cache = SemanticCache(threshold=0.0)
        base = _emb(seed=0)
        close = _perturb(base, scale=0.01, seed=1)
        far = _emb(seed=42)

        cache.store("close", close, "close-result", 0, _mem())
        cache.store("far", far, "far-result", 0, _mem())

        result = cache.lookup(base, [0])
        assert result is not None
        entry, _ = result
        assert entry.query == "close"

    def test_score_above_threshold_is_hit(self):
        emb1 = _emb(seed=0)
        emb2 = _perturb(emb1, scale=0.05)
        similarity = float(np.dot(emb1, emb2))
        threshold = similarity - 0.05

        cache = SemanticCache(threshold=max(threshold, 0.0))
        cache.store("q", emb1, "r", 0, _mem())
        result = cache.lookup(emb2, [0])
        assert result is not None
        _, score = result
        assert score >= cache.threshold

    def test_score_below_threshold_is_miss(self):
        emb1 = _emb(seed=0)
        emb2 = _perturb(emb1, scale=0.5)
        similarity = float(np.dot(emb1, emb2))
        threshold = similarity + 0.05

        cache = SemanticCache(threshold=min(threshold, 0.99))
        cache.store("q", emb1, "r", 0, _mem())
        result = cache.lookup(emb2, [0])
        assert result is None

    def test_searches_multiple_clusters(self):
        """Lookup in [0, 1] should find an entry stored in cluster 1."""
        cache = SemanticCache(threshold=0.0)
        emb = _emb()
        cache.store("q", emb, "r", dominant_cluster=1, memberships=_mem(dominant=1))
        result = cache.lookup(emb, [0, 1])
        assert result is not None

    def test_hit_increments_entry_hit_count(self):
        cache = SemanticCache(threshold=0.0)
        emb = _emb()
        cache.store("q", emb, "r", 0, _mem())
        for _ in range(3):
            cache.lookup(emb, [0])
        entry = cache.get_cluster_entries(0)[0]
        assert entry.hit_count == 3


# ── Stats ──────────────────────────────────────────────────────────────────────

class TestSemanticCacheStats:
    def test_hit_count_increments(self):
        cache = SemanticCache(threshold=0.0)
        emb = _emb()
        cache.store("q", emb, "r", 0, _mem())
        cache.lookup(emb, [0])
        cache.lookup(emb, [0])
        assert cache.hit_count == 2
        assert cache.miss_count == 0

    def test_miss_count_increments(self):
        cache = SemanticCache(threshold=0.99)
        emb1 = _emb(seed=0)
        emb2 = _emb(seed=42)
        cache.store("q", emb1, "r", 0, _mem())
        cache.lookup(emb2, [0])
        assert cache.miss_count == 1

    def test_hit_rate_all_hits(self):
        cache = SemanticCache(threshold=0.0)
        emb = _emb()
        cache.store("q", emb, "r", 0, _mem())
        cache.lookup(emb, [0])
        cache.lookup(emb, [0])
        assert cache.hit_rate == 1.0

    def test_hit_rate_no_queries(self):
        assert SemanticCache().hit_rate == 0.0

    def test_hit_rate_mixed(self):
        cache = SemanticCache(threshold=0.0)
        emb = _emb()
        cache.store("q", emb, "r", 0, _mem())

        cache.lookup(emb, [0])             # hit
        cache.lookup(_emb(seed=99), [99])  # miss (cluster 99 is empty)
        assert cache.hit_rate == pytest.approx(0.5)

    def test_get_stats_keys(self):
        stats = SemanticCache(threshold=0.75).get_stats()
        assert {"total_entries", "hit_count", "miss_count", "hit_rate",
                "threshold", "cluster_distribution"}.issubset(stats.keys())
        assert stats["threshold"] == 0.75

    def test_cluster_distribution_in_stats(self):
        cache = SemanticCache()
        cache.store("q0", _emb(0), "r", 0, _mem(dominant=0))
        cache.store("q1", _emb(1), "r", 0, _mem(dominant=0))
        cache.store("q2", _emb(2), "r", 2, _mem(dominant=2))
        dist = cache.get_stats()["cluster_distribution"]
        assert dist["0"] == 2
        assert dist["2"] == 1


# ── Flush ──────────────────────────────────────────────────────────────────────

class TestSemanticCacheFlush:
    def test_flush_clears_all_entries(self):
        cache = SemanticCache()
        for i in range(5):
            cache.store(f"q{i}", _emb(i), f"r{i}", i % 3, _mem())
        cache.flush()
        assert cache.total_entries == 0

    def test_flush_resets_hit_count(self):
        cache = SemanticCache(threshold=0.0)
        emb = _emb()
        cache.store("q", emb, "r", 0, _mem())
        cache.lookup(emb, [0])
        cache.flush()
        assert cache.hit_count == 0

    def test_flush_resets_miss_count(self):
        cache = SemanticCache(threshold=0.99)
        cache.store("q", _emb(0), "r", 0, _mem())
        cache.lookup(_emb(42), [0])
        cache.flush()
        assert cache.miss_count == 0

    def test_flush_resets_hit_rate(self):
        cache = SemanticCache(threshold=0.0)
        emb = _emb()
        cache.store("q", emb, "r", 0, _mem())
        cache.lookup(emb, [0])
        cache.flush()
        assert cache.hit_rate == 0.0

    def test_store_after_flush(self):
        cache = SemanticCache(threshold=0.0)
        emb = _emb()
        cache.store("q", emb, "r", 0, _mem())
        cache.flush()
        cache.store("q2", emb, "r2", 0, _mem())
        assert cache.total_entries == 1
        result = cache.lookup(emb, [0])
        assert result is not None


# ── Threshold management ───────────────────────────────────────────────────────

class TestSemanticCacheThreshold:
    def test_set_threshold_updates_value(self):
        cache = SemanticCache(threshold=0.80)
        cache.set_threshold(0.90)
        assert cache.threshold == 0.90

    def test_lower_threshold_turns_miss_into_hit(self):
        emb1 = _emb(seed=0)
        emb2 = _perturb(emb1, scale=0.3, seed=7)
        similarity = float(np.dot(emb1, emb2))

        # Tight threshold: should miss
        cache = SemanticCache(threshold=min(similarity + 0.05, 0.99))
        cache.store("q", emb1, "r", 0, _mem())
        assert cache.lookup(emb2, [0]) is None

        # Loose threshold: same entries, should hit
        cache.set_threshold(max(similarity - 0.05, 0.0))
        assert cache.lookup(emb2, [0]) is not None

    def test_raise_threshold_turns_hit_into_miss(self):
        emb1 = _emb(seed=0)
        emb2 = _perturb(emb1, scale=0.2, seed=8)
        similarity = float(np.dot(emb1, emb2))

        cache = SemanticCache(threshold=max(similarity - 0.05, 0.0))
        cache.store("q", emb1, "r", 0, _mem())
        assert cache.lookup(emb2, [0]) is not None

        cache.set_threshold(min(similarity + 0.05, 0.99))
        assert cache.lookup(emb2, [0]) is None


# ── Introspection helpers ──────────────────────────────────────────────────────

class TestSemanticCacheIntrospection:
    def test_get_all_entries(self):
        cache = SemanticCache()
        for cluster in range(3):
            cache.store(f"q{cluster}", _emb(cluster), "r", cluster, _mem(dominant=cluster))
        all_entries = cache.get_all_entries()
        assert len(all_entries) == 3

    def test_get_cluster_entries_empty_cluster(self):
        cache = SemanticCache()
        assert cache.get_cluster_entries(99) == []
