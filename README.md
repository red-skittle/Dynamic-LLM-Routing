# Dynamic LLM Routing

A modular Python system for routing prompts to the right language model, with
caching layers to avoid redundant inference calls.

## Project structure

```
Dynamic LLM Routing/
├── inference/
│   ├── __init__.py
│   ├── backend.py        # Abstract InferenceBackend interface
│   ├── config.py         # ModelConfig + MODEL_REGISTRY (small / medium / large)
│   ├── mock_backend.py   # MockBackend — synthetic, no real LLM required
│   └── result.py         # InferenceResult dataclass
├── cache/
│   ├── __init__.py
│   └── exact_cache.py    # ExactCache — Phase 2 exact-match cache
├── tests/
│   ├── test_backend.py   # Phase 1 tests
│   └── test_exact_cache.py  # Phase 2 tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Phase 1 — Inference architecture

### `InferenceResult`

Standard return type for every backend call.

| Field | Type | Description |
|---|---|---|
| `model_id` | `str` | Canonical model identifier |
| `prompt` | `str` | Exact prompt sent to the backend |
| `response` | `str` | Generated text |
| `elapsed_ms` | `float` | Wall-clock ms *inside* the backend call |
| `timestamp` | `datetime` | UTC creation time |
| `backend` | `str` | Backend name (e.g. `"mock"`) |
| `is_cached` | `bool` | `True` when returned from a cache layer |
| `metadata` | `dict` | Free-form backend-specific extras |

### `ModelConfig` & tiers

Three pre-registered size tiers:

| Key | `model_id` | `max_tokens` | Use case |
|---|---|---|---|
| `small` | `gpt-3.5-turbo` | 512 | Simple Q&A, extraction |
| `medium` | `gpt-4o-mini` | 2 048 | General purpose |
| `large` | `gpt-4o` | 8 192 | Complex reasoning |

```python
from inference import get_config, ModelSize

cfg = get_config(ModelSize.MEDIUM)   # or get_config("medium")
print(cfg.model_id)  # "gpt-4o-mini"
```

### `MockBackend`

Deterministic, in-process backend — no LLM installation required.

> **⚠️ Important**
> Responses are entirely synthetic (`[SYNTHETIC] …`).
> `elapsed_ms` reflects mock overhead only (microseconds), **not** real
> LLM inference latency.

```python
from inference import MockBackend, get_config, ModelSize

backend = MockBackend()
result  = backend.infer("Explain binary search", get_config(ModelSize.SMALL))

print(result.response)   # [mock:gpt-3.5-turbo] [SYNTHETIC] …
print(result.is_cached)  # False
print(result.backend)    # "mock"
```

---

## Phase 2 — Exact cache

### Pipeline

```
Raw prompt
    → ExactCache.normalize()   # strip + collapse whitespace
    → ExactCache.hash_prompt() # SHA-256 hex digest
    → key = "<hash>:<model_id>"
    → lookup in _store
         Hit  → check expiry → return copy (is_cached=True)
         Miss → return None  → continue to semantic cache (Phase 3)
```

### Normalization rules (conservative by design)

1. Strip leading/trailing whitespace.
2. Collapse any internal whitespace run to a single ASCII space.

Case, punctuation and Unicode are **not** altered — those belong to the
semantic cache.

### Cache entries are model-scoped

The same prompt cached under `gpt-3.5-turbo` does **not** satisfy a request
for `gpt-4o`. The cache key includes the `model_id`.

### Usage

```python
from inference import MockBackend, get_config, ModelSize
from cache import ExactCache

backend = MockBackend()
cache   = ExactCache(default_ttl_seconds=3600)
config  = get_config(ModelSize.MEDIUM)
prompt  = "Explain binary search"

# --- first request: miss ---
cached = cache.get(prompt, config.model_id)   # None

# --- run inference ---
result = backend.infer(prompt, config)
cache.put(prompt, result)

# --- second request: hit ---
cached = cache.get(prompt, config.model_id)
print(cached.is_cached)                        # True
print(cached.metadata["cache_type"])           # "exact"

# --- metrics ---
print(cache.metrics)
# CacheMetrics(hits=1, misses=1, total=2, hit_rate=50.0%)
```

### `CacheMetrics`

| Property | Description |
|---|---|
| `hits` | Successful cache lookups |
| `misses` | Failed lookups (absent or expired) |
| `total` | `hits + misses` |
| `hit_rate` | `hits / total` (0.0 – 1.0) |

Call `cache.reset_metrics()` to zero the counters.

---

## Running tests

```bash
pip install pytest
pytest -v
```

All 63 tests should pass in under a second (no network, no LLM).

---

## Extending the system

To add a real backend (e.g. OpenAI), sub-class `InferenceBackend`:

```python
from inference.backend import InferenceBackend
from inference.config import ModelConfig
from inference.result import InferenceResult

class OpenAIBackend(InferenceBackend):
    @property
    def name(self) -> str:
        return "openai"

    def infer(self, prompt: str, config: ModelConfig) -> InferenceResult:
        # call the OpenAI SDK, measure elapsed_ms, return InferenceResult
        ...

    def is_available(self) -> bool:
        # health-check the API
        ...
```

The router and cache layers work with any `InferenceBackend` implementor.
