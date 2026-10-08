"""
cache/exact_cache.py
--------------------
Exact-match prompt cache (Phase 2).

Pipeline
--------
Raw prompt
    → normalize()        strip + collapse internal whitespace
    → hash_prompt()      SHA-256 hex digest
    → cache key          "<hash>:<model_id>"
    → lookup in _store
         Hit  → check expiry → return copy marked is_cached=True
         Miss → return None  → caller continues to semantic cache

Design decisions
----------------
* Keys are scoped by ``model_id`` so that the same prompt cached under
  ``gpt-3.5-turbo`` does NOT satisfy a request for ``gpt-4o``.
* Normalization is intentionally conservative (trim + collapse spaces).
  It does NOT fold case, strip punctuation or perform stemming — those
  transformations belong to the semantic cache layer.
* Expiry is checked on every ``get`` call; no background eviction thread
  is required for this phase.
* All metrics (hits, misses) are accumulated in a :class:`CacheMetrics`
  object attached to the cache instance.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from inference.result import InferenceResult


# ---------------------------------------------------------------------------
# Supporting data structures
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """One stored inference result with its metadata."""

    result: InferenceResult
    model_id: str
    created_at: datetime
    expires_at: datetime


@dataclass
class CacheMetrics:
    """Running counters for cache-hit measurement.

    Attributes
    ----------
    hits   : Number of successful cache lookups (non-expired, model-matched).
    misses : Number of failed lookups (key absent or entry expired).
    """

    hits: int = 0
    misses: int = 0

    # ---- derived properties -----------------------------------------------

    @property
    def total(self) -> int:
        """Total lookups attempted."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that returned a cached result (0.0 – 1.0)."""
        return self.hits / self.total if self.total > 0 else 0.0

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"CacheMetrics(hits={self.hits}, misses={self.misses}, "
            f"total={self.total}, hit_rate={self.hit_rate:.1%})"
        )


# ---------------------------------------------------------------------------
# ExactCache
# ---------------------------------------------------------------------------


class ExactCache:
    """Exact-match cache keyed on a normalized, hashed prompt + model_id.

    Parameters
    ----------
    default_ttl_seconds:
        How long a cache entry remains valid after insertion.
        Pass ``float("inf")`` to disable expiry (useful in tests).

    Example
    -------
    >>> from cache.exact_cache import ExactCache
    >>> from inference import MockBackend, get_config, ModelSize
    >>>
    >>> backend = MockBackend()
    >>> config  = get_config(ModelSize.SMALL)
    >>> cache   = ExactCache(default_ttl_seconds=300)
    >>>
    >>> result = backend.infer("Explain binary search", config)
    >>> cache.put("Explain binary search", result)
    >>>
    >>> cached = cache.get("Explain binary search", config.model_id)
    >>> assert cached is not None and cached.is_cached
    """

    def __init__(self, default_ttl_seconds: float = 3_600.0) -> None:
        self._store: dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl_seconds
        self.metrics: CacheMetrics = CacheMetrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, prompt: str, model_id: str) -> Optional[InferenceResult]:
        """Look up a cached result for *prompt* + *model_id*.

        Returns
        -------
        InferenceResult | None
            A *copy* of the cached result with ``is_cached=True`` and
            ``metadata["cache_type"] = "exact"`` if found and not expired;
            ``None`` otherwise.

        Side effects
        ------------
        * Expired entries are evicted on access.
        * ``metrics.hits`` or ``metrics.misses`` is incremented.
        """
        key = self._make_key(prompt, model_id)
        entry = self._store.get(key)

        if entry is None:
            self.metrics.misses += 1
            return None

        if self._is_expired(entry):
            del self._store[key]
            self.metrics.misses += 1
            return None

        self.metrics.hits += 1
        return self._mark_cached(entry.result)

    def put(
        self,
        prompt: str,
        result: InferenceResult,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        """Store *result* under the key derived from *prompt* + ``result.model_id``.

        Parameters
        ----------
        prompt:
            The original (un-normalized) prompt.  Normalization is applied
            internally.
        result:
            The :class:`~inference.result.InferenceResult` to cache.
        ttl_seconds:
            Override the instance-level ``default_ttl`` for this entry.
        """
        key = self._make_key(prompt, result.model_id)
        now = datetime.now(tz=timezone.utc)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        if ttl == float("inf"):
            # Use the maximum representable UTC datetime as a "never expires" sentinel.
            expires_at = datetime.max.replace(tzinfo=timezone.utc)
        else:
            expires_at = datetime.fromtimestamp(
                now.timestamp() + ttl, tz=timezone.utc
            )
        self._store[key] = CacheEntry(
            result=result,
            model_id=result.model_id,
            created_at=now,
            expires_at=expires_at,
        )

    def invalidate(self, prompt: str, model_id: str) -> bool:
        """Remove a single entry.  Returns ``True`` if an entry was deleted."""
        key = self._make_key(prompt, model_id)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Remove all entries (metrics are preserved)."""
        self._store.clear()

    def reset_metrics(self) -> None:
        """Zero out hit/miss counters."""
        self.metrics = CacheMetrics()

    @property
    def size(self) -> int:
        """Number of entries currently in the store (including expired ones
        not yet evicted)."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Normalization & hashing (static — usable independently)
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(prompt: str) -> str:
        """Normalize a prompt for exact-match comparison.

        Rules (conservative by design)
        --------------------------------
        1. Strip leading and trailing whitespace.
        2. Collapse any run of internal whitespace (spaces, tabs, newlines)
           to a single ASCII space.

        Case, punctuation and unicode are left untouched.

        >>> ExactCache.normalize("  Explain   binary\\tsearch  ")
        'Explain binary search'
        """
        return re.sub(r"\s+", " ", prompt).strip()

    @staticmethod
    def hash_prompt(normalized: str) -> str:
        """Return the SHA-256 hex digest of a UTF-8 encoded normalized prompt.

        >>> ExactCache.hash_prompt("hello")  # doctest: +ELLIPSIS
        '2cf24d...'
        """
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, prompt: str, model_id: str) -> str:
        """Build the dict key: ``"<sha256>:<model_id>"``."""
        normalized = self.normalize(prompt)
        h = self.hash_prompt(normalized)
        return f"{h}:{model_id}"

    @staticmethod
    def _is_expired(entry: CacheEntry) -> bool:
        return datetime.now(tz=timezone.utc) >= entry.expires_at

    @staticmethod
    def _mark_cached(result: InferenceResult) -> InferenceResult:
        """Return a shallow copy of *result* flagged as a cache hit."""
        return InferenceResult(
            model_id=result.model_id,
            prompt=result.prompt,
            response=result.response,
            elapsed_ms=result.elapsed_ms,
            timestamp=result.timestamp,
            backend=result.backend,
            is_cached=True,
            metadata={**result.metadata, "cache_type": "exact"},
        )
