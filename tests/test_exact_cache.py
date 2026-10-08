"""
tests/test_exact_cache.py
-------------------------
Tests for Phase 2: exact-match prompt cache.

Covers
------
Normalization
  * Leading/trailing whitespace stripped
  * Internal whitespace runs collapsed to one space
  * Case and punctuation preserved

Hashing
  * Same normalized string → same digest
  * Different strings → different digests (collision sanity check)

Cache get / put
  * Miss on empty cache
  * Hit after put
  * Returned copy has is_cached=True and cache_type="exact"
  * Original result is unmodified (is_cached still False)

Model-scoping
  * Same prompt, different model_id → different cache entries

TTL / expiry
  * Entry is returned within TTL
  * Entry is evicted after TTL (using a 0-second TTL in tests)

Metrics
  * hits/misses incremented correctly
  * hit_rate computed correctly
  * reset_metrics zeroes counters

Invalidation and clear
  * invalidate removes specific entry, returns True/False
  * clear empties the store

Full round-trip
  * MockBackend → cache.put → cache.get → is_cached
"""

import time
from datetime import datetime, timezone

import pytest

from cache import CacheMetrics, ExactCache
from inference import MockBackend, ModelSize, get_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(prompt: str = "test prompt", size: ModelSize = ModelSize.SMALL):
    backend = MockBackend()
    config = get_config(size)
    return backend.infer(prompt, config), config


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_strips_leading_trailing(self):
        assert ExactCache.normalize("  hello  ") == "hello"

    def test_collapses_internal_spaces(self):
        assert ExactCache.normalize("hello   world") == "hello world"

    def test_collapses_tabs_and_newlines(self):
        assert ExactCache.normalize("hello\t\nworld") == "hello world"

    def test_mixed_whitespace(self):
        assert ExactCache.normalize("  Explain   binary\tsearch  ") == "Explain binary search"

    def test_preserves_case(self):
        assert ExactCache.normalize("Hello World") == "Hello World"

    def test_preserves_punctuation(self):
        assert ExactCache.normalize("What's 2+2?") == "What's 2+2?"

    def test_empty_string(self):
        assert ExactCache.normalize("") == ""

    def test_only_whitespace(self):
        assert ExactCache.normalize("   \t\n  ") == ""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class TestHashPrompt:
    def test_same_input_same_hash(self):
        h1 = ExactCache.hash_prompt("hello")
        h2 = ExactCache.hash_prompt("hello")
        assert h1 == h2

    def test_different_input_different_hash(self):
        assert ExactCache.hash_prompt("hello") != ExactCache.hash_prompt("world")

    def test_hash_is_hex_string(self):
        h = ExactCache.hash_prompt("test")
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_length_sha256(self):
        assert len(ExactCache.hash_prompt("anything")) == 64

    def test_normalize_then_hash_equivalence(self):
        """Variants that normalize to the same string must produce the same hash."""
        variants = [
            "Explain binary search",
            "  Explain binary search  ",
            "Explain  binary  search",
            "Explain\tbinary\nsearch",
        ]
        hashes = {ExactCache.hash_prompt(ExactCache.normalize(v)) for v in variants}
        assert len(hashes) == 1


# ---------------------------------------------------------------------------
# Cache get / put
# ---------------------------------------------------------------------------


class TestGetPut:
    def setup_method(self):
        self.cache = ExactCache(default_ttl_seconds=300)

    def test_miss_on_empty_cache(self):
        result = self.cache.get("anything", "gpt-4o")
        assert result is None

    def test_hit_after_put(self):
        r, cfg = make_result("Explain binary search")
        self.cache.put("Explain binary search", r)
        cached = self.cache.get("Explain binary search", cfg.model_id)
        assert cached is not None

    def test_cached_result_is_marked(self):
        r, cfg = make_result("Explain binary search")
        self.cache.put("Explain binary search", r)
        cached = self.cache.get("Explain binary search", cfg.model_id)
        assert cached.is_cached is True
        assert cached.metadata.get("cache_type") == "exact"

    def test_original_result_unchanged(self):
        r, cfg = make_result("Explain binary search")
        assert r.is_cached is False  # original must remain untouched
        self.cache.put("Explain binary search", r)
        self.cache.get("Explain binary search", cfg.model_id)
        assert r.is_cached is False  # still untouched after lookup

    def test_cached_content_matches_original(self):
        r, cfg = make_result("Explain binary search")
        self.cache.put("Explain binary search", r)
        cached = self.cache.get("Explain binary search", cfg.model_id)
        assert cached.response == r.response
        assert cached.model_id == r.model_id
        assert cached.backend == r.backend

    def test_whitespace_variants_hit_same_entry(self):
        r, cfg = make_result("Explain binary search")
        self.cache.put("Explain binary search", r)
        # Variant with extra spaces should still hit
        cached = self.cache.get("  Explain   binary search  ", cfg.model_id)
        assert cached is not None

    def test_overwrite_existing_entry(self):
        r1, cfg = make_result("same prompt")
        r2, _ = make_result("same prompt", ModelSize.SMALL)
        self.cache.put("same prompt", r1)
        self.cache.put("same prompt", r2)  # overwrite
        # Should still be a hit
        assert self.cache.get("same prompt", cfg.model_id) is not None
        assert self.cache.size == 1


# ---------------------------------------------------------------------------
# Model scoping
# ---------------------------------------------------------------------------


class TestModelScoping:
    def test_different_models_are_isolated(self):
        cache = ExactCache(default_ttl_seconds=300)
        small_cfg = get_config(ModelSize.SMALL)
        large_cfg = get_config(ModelSize.LARGE)
        backend = MockBackend()

        result_small = backend.infer("What is a list?", small_cfg)
        cache.put("What is a list?", result_small)

        # Large model was never stored — should miss
        assert cache.get("What is a list?", large_cfg.model_id) is None
        # Small model should hit
        assert cache.get("What is a list?", small_cfg.model_id) is not None


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


class TestTTL:
    def test_entry_valid_within_ttl(self):
        cache = ExactCache(default_ttl_seconds=60)
        r, cfg = make_result("ttl test")
        cache.put("ttl test", r)
        assert cache.get("ttl test", cfg.model_id) is not None

    def test_entry_expired_after_ttl(self):
        cache = ExactCache(default_ttl_seconds=0)  # expires immediately
        r, cfg = make_result("ttl test")
        cache.put("ttl test", r)
        # Sleep just a tiny bit to cross the expiry boundary
        time.sleep(0.01)
        assert cache.get("ttl test", cfg.model_id) is None

    def test_expired_entry_evicted_from_store(self):
        cache = ExactCache(default_ttl_seconds=0)
        r, cfg = make_result("evict test")
        cache.put("evict test", r)
        assert cache.size == 1
        time.sleep(0.01)
        cache.get("evict test", cfg.model_id)  # triggers eviction
        assert cache.size == 0

    def test_per_entry_ttl_override(self):
        cache = ExactCache(default_ttl_seconds=3600)
        r, cfg = make_result("override ttl")
        cache.put("override ttl", r, ttl_seconds=0)
        time.sleep(0.01)
        assert cache.get("override ttl", cfg.model_id) is None

    def test_infinite_ttl(self):
        cache = ExactCache(default_ttl_seconds=float("inf"))
        r, cfg = make_result("forever")
        cache.put("forever", r)
        assert cache.get("forever", cfg.model_id) is not None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def setup_method(self):
        self.cache = ExactCache(default_ttl_seconds=300)
        self.r, self.cfg = make_result("metrics test")

    def test_initial_metrics_zero(self):
        m = self.cache.metrics
        assert m.hits == 0 and m.misses == 0 and m.total == 0

    def test_miss_increments_miss(self):
        self.cache.get("does not exist", "any-model")
        assert self.cache.metrics.misses == 1
        assert self.cache.metrics.hits == 0

    def test_hit_increments_hit(self):
        self.cache.put("metrics test", self.r)
        self.cache.get("metrics test", self.cfg.model_id)
        assert self.cache.metrics.hits == 1
        assert self.cache.metrics.misses == 0

    def test_hit_rate_all_hits(self):
        self.cache.put("metrics test", self.r)
        for _ in range(4):
            self.cache.get("metrics test", self.cfg.model_id)
        assert self.cache.metrics.hit_rate == 1.0

    def test_hit_rate_mixed(self):
        self.cache.put("metrics test", self.r)
        self.cache.get("metrics test", self.cfg.model_id)   # hit
        self.cache.get("no such entry", self.cfg.model_id)  # miss
        m = self.cache.metrics
        assert m.hits == 1 and m.misses == 1
        assert m.hit_rate == pytest.approx(0.5)

    def test_hit_rate_no_lookups(self):
        assert self.cache.metrics.hit_rate == 0.0

    def test_reset_metrics(self):
        self.cache.put("metrics test", self.r)
        self.cache.get("metrics test", self.cfg.model_id)
        self.cache.reset_metrics()
        m = self.cache.metrics
        assert m.hits == 0 and m.misses == 0

    def test_expired_counts_as_miss(self):
        cache = ExactCache(default_ttl_seconds=0)
        r, cfg = make_result("expire miss")
        cache.put("expire miss", r)
        time.sleep(0.01)
        cache.get("expire miss", cfg.model_id)
        assert cache.metrics.misses == 1
        assert cache.metrics.hits == 0


# ---------------------------------------------------------------------------
# Invalidation and clear
# ---------------------------------------------------------------------------


class TestInvalidationAndClear:
    def test_invalidate_existing_entry(self):
        cache = ExactCache()
        r, cfg = make_result("to remove")
        cache.put("to remove", r)
        removed = cache.invalidate("to remove", cfg.model_id)
        assert removed is True
        assert cache.get("to remove", cfg.model_id) is None

    def test_invalidate_missing_entry_returns_false(self):
        cache = ExactCache()
        assert cache.invalidate("ghost", "any-model") is False

    def test_clear_empties_store(self):
        cache = ExactCache()
        for i in range(5):
            r, cfg = make_result(f"prompt {i}")
            cache.put(f"prompt {i}", r)
        cache.clear()
        assert cache.size == 0

    def test_clear_preserves_metrics(self):
        cache = ExactCache(default_ttl_seconds=300)
        r, cfg = make_result("preserved")
        cache.put("preserved", r)
        cache.get("preserved", cfg.model_id)  # hit
        cache.clear()
        assert cache.metrics.hits == 1


# ---------------------------------------------------------------------------
# Full round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_mock_backend_to_cache_and_back(self):
        """End-to-end: backend → cache.put → cache.get → is_cached."""
        backend = MockBackend()
        cache = ExactCache(default_ttl_seconds=300)
        config = get_config(ModelSize.MEDIUM)
        prompt = "Explain binary search"

        # First call — cache miss
        miss = cache.get(prompt, config.model_id)
        assert miss is None
        assert cache.metrics.misses == 1

        # Run inference and store
        live_result = backend.infer(prompt, config)
        assert live_result.is_cached is False
        cache.put(prompt, live_result)

        # Second call — cache hit
        cached_result = cache.get(prompt, config.model_id)
        assert cached_result is not None
        assert cached_result.is_cached is True
        assert cached_result.response == live_result.response
        assert cache.metrics.hits == 1

    def test_all_size_tiers_cached_independently(self):
        backend = MockBackend()
        cache = ExactCache(default_ttl_seconds=300)
        prompt = "What is recursion?"

        for size in ModelSize:
            cfg = get_config(size)
            result = backend.infer(prompt, cfg)
            cache.put(prompt, result)

        assert cache.size == len(ModelSize)

        for size in ModelSize:
            cfg = get_config(size)
            cached = cache.get(prompt, cfg.model_id)
            assert cached is not None
            assert cached.model_id == cfg.model_id
