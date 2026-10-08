"""
tests/test_backend.py
---------------------
Tests for Phase 1: inference architecture.

Covers
------
* InferenceResult validation
* ModelConfig / MODEL_REGISTRY / get_config
* MockBackend.infer — all three tiers
* MockBackend.is_available (happy path + forced unavailability)
* MockBackend respects custom_responses
* elapsed_ms is non-negative and plausibly small (mock overhead only)
* backend field is set to "mock", is_cached is False
* Empty prompt raises ValueError
"""

import math
from datetime import timezone

import pytest

from inference import (
    InferenceResult,
    MockBackend,
    ModelConfig,
    ModelSize,
    get_config,
    MODEL_REGISTRY,
)


# ---------------------------------------------------------------------------
# InferenceResult
# ---------------------------------------------------------------------------


class TestInferenceResult:
    def test_valid_construction(self):
        from datetime import datetime

        r = InferenceResult(
            model_id="gpt-4o",
            prompt="hello",
            response="world",
            elapsed_ms=12.5,
            timestamp=datetime.now(tz=timezone.utc),
            backend="mock",
        )
        assert r.model_id == "gpt-4o"
        assert r.is_cached is False
        assert r.metadata == {}

    def test_negative_elapsed_raises(self):
        from datetime import datetime

        with pytest.raises(ValueError, match="elapsed_ms"):
            InferenceResult(
                model_id="m",
                prompt="p",
                response="r",
                elapsed_ms=-1.0,
                timestamp=datetime.now(tz=timezone.utc),
                backend="mock",
            )

    def test_empty_model_id_raises(self):
        from datetime import datetime

        with pytest.raises(ValueError, match="model_id"):
            InferenceResult(
                model_id="",
                prompt="p",
                response="r",
                elapsed_ms=0.0,
                timestamp=datetime.now(tz=timezone.utc),
                backend="mock",
            )

    def test_empty_backend_raises(self):
        from datetime import datetime

        with pytest.raises(ValueError, match="backend"):
            InferenceResult(
                model_id="m",
                prompt="p",
                response="r",
                elapsed_ms=0.0,
                timestamp=datetime.now(tz=timezone.utc),
                backend="",
            )


# ---------------------------------------------------------------------------
# ModelConfig / registry
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_all_tiers_present(self):
        for size in ModelSize:
            assert size in MODEL_REGISTRY

    def test_get_config_by_enum(self):
        cfg = get_config(ModelSize.SMALL)
        assert cfg.size == ModelSize.SMALL
        assert cfg.max_tokens > 0

    def test_get_config_by_string(self):
        cfg = get_config("medium")
        assert cfg.size == ModelSize.MEDIUM

    def test_invalid_size_raises(self):
        with pytest.raises((KeyError, ValueError)):
            get_config("xlarge")

    def test_temperature_out_of_range_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            ModelConfig(
                model_id="x",
                size=ModelSize.SMALL,
                max_tokens=100,
                temperature=3.0,
                description="bad",
            )

    def test_max_tokens_zero_raises(self):
        with pytest.raises(ValueError, match="max_tokens"):
            ModelConfig(
                model_id="x",
                size=ModelSize.SMALL,
                max_tokens=0,
                temperature=0.5,
                description="bad",
            )

    def test_large_has_most_tokens(self):
        small = get_config(ModelSize.SMALL)
        large = get_config(ModelSize.LARGE)
        assert large.max_tokens > small.max_tokens


# ---------------------------------------------------------------------------
# MockBackend
# ---------------------------------------------------------------------------


class TestMockBackend:
    def setup_method(self):
        self.backend = MockBackend()

    def test_name(self):
        assert self.backend.name == "mock"

    def test_is_available_default_true(self):
        assert self.backend.is_available() is True

    def test_is_available_forced_false(self):
        unavailable = MockBackend(always_available=False)
        assert unavailable.is_available() is False

    @pytest.mark.parametrize("size", list(ModelSize))
    def test_infer_all_tiers(self, size):
        config = get_config(size)
        result = self.backend.infer("What is AI?", config)

        assert isinstance(result, InferenceResult)
        assert result.model_id == config.model_id
        assert result.prompt == "What is AI?"
        assert result.backend == "mock"
        assert result.is_cached is False
        assert "[SYNTHETIC]" in result.response
        assert config.model_id in result.response

    def test_elapsed_ms_non_negative(self):
        config = get_config(ModelSize.SMALL)
        result = self.backend.infer("ping", config)
        assert result.elapsed_ms >= 0.0

    def test_elapsed_ms_is_mock_overhead_only(self):
        """elapsed_ms should be tiny — mock does no real inference."""
        config = get_config(ModelSize.LARGE)
        result = self.backend.infer("Explain the universe", config)
        # Mock overhead should be well under 100 ms on any reasonable hardware
        assert result.elapsed_ms < 100.0

    def test_timestamp_is_utc(self):
        config = get_config(ModelSize.SMALL)
        result = self.backend.infer("tz test", config)
        assert result.timestamp.tzinfo is not None
        assert result.timestamp.utcoffset().total_seconds() == 0

    def test_metadata_synthetic_flag(self):
        config = get_config(ModelSize.MEDIUM)
        result = self.backend.infer("meta test", config)
        assert result.metadata.get("synthetic") is True
        assert result.metadata.get("tier") == ModelSize.MEDIUM.value

    def test_empty_prompt_raises(self):
        config = get_config(ModelSize.SMALL)
        with pytest.raises(ValueError, match="prompt"):
            self.backend.infer("   ", config)

    def test_custom_responses(self):
        custom = MockBackend(
            custom_responses={ModelSize.SMALL: "My custom response."}
        )
        config = get_config(ModelSize.SMALL)
        result = custom.infer("anything", config)
        assert "My custom response." in result.response
