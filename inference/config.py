"""
inference/config.py
-------------------
Model configuration catalogue.

Three size tiers are pre-registered:
  - small  : fast, low-cost, for simple/short tasks
  - medium : balanced throughput and capability
  - large  : highest capability, for complex reasoning

ModelConfig is intentionally backend-agnostic.  An InferenceBackend may
choose to honour or ignore individual fields (e.g. a MockBackend ignores
temperature).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelSize(str, Enum):
    """Broad capability tier."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration parameters for a single model."""

    model_id: str
    size: ModelSize
    max_tokens: int
    temperature: float
    description: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"temperature must be in [0.0, 2.0], got {self.temperature}"
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")


# ---------------------------------------------------------------------------
# Default registry — keyed by tier name for easy lookup
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelConfig] = {
    ModelSize.SMALL: ModelConfig(
        model_id="gpt-3.5-turbo",
        size=ModelSize.SMALL,
        max_tokens=512,
        temperature=0.7,
        description="Fast, lightweight model suited for simple classification, "
        "extraction and short Q&A.",
    ),
    ModelSize.MEDIUM: ModelConfig(
        model_id="gpt-4o-mini",
        size=ModelSize.MEDIUM,
        max_tokens=2048,
        temperature=0.7,
        description="Balanced model for general-purpose generation and "
        "moderate reasoning.",
    ),
    ModelSize.LARGE: ModelConfig(
        model_id="gpt-4o",
        size=ModelSize.LARGE,
        max_tokens=8192,
        temperature=0.7,
        description="High-capability model for complex reasoning, long-form "
        "generation and multi-step tasks.",
    ),
}


def get_config(size: ModelSize | str) -> ModelConfig:
    """Return the ModelConfig for a given size tier.

    Parameters
    ----------
    size:
        A :class:`ModelSize` enum value or its string equivalent
        (``"small"``, ``"medium"``, ``"large"``).

    Raises
    ------
    KeyError
        If *size* does not match any registered tier.
    """
    key = ModelSize(size) if isinstance(size, str) else size
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"No model registered for size {key!r}. "
            f"Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key]
