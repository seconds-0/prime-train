"""Tests for the validator module."""

from pathlib import Path

import pytest

from prime_train.validator import validate_config, ValidationResults
from prime_train.validator.core import Severity


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config_passes(self):
        """A valid config should pass all checks."""
        config_path = EXAMPLES_DIR / "basic-config.toml"
        results = validate_config(config_path)

        # Should have no errors
        assert not results.has_errors, f"Unexpected errors: {[r.message for r in results.results if r.severity == Severity.ERROR]}"

    def test_fsdp_lora_gotcha_detected(self):
        """FSDP + LoRA conflict should be detected."""
        config_path = EXAMPLES_DIR / "gotcha-fsdp-lora.toml"
        results = validate_config(config_path)

        # Should have error for FSDP+LoRA
        error_ids = [r.check for r in results.results if r.severity == Severity.ERROR]
        assert "fsdp-lora-conflict" in error_ids, f"Expected fsdp-lora-conflict error, got: {error_ids}"

    def test_prime_executor_warning(self):
        """Prime executor should trigger latency warning."""
        config_path = EXAMPLES_DIR / "gotcha-prime-executor.toml"
        results = validate_config(config_path)

        # Should have warning for prime executor
        warning_ids = [r.check for r in results.results if r.severity == Severity.WARNING]
        assert "prime-executor-latency" in warning_ids, f"Expected prime-executor-latency warning, got: {warning_ids}"

    def test_missing_file_returns_error(self):
        """Missing config file should return error."""
        results = validate_config(Path("/nonexistent/config.toml"))

        assert results.has_errors
        assert any("not found" in r.message for r in results.results)


class TestGotchaDatabase:
    """Tests for individual gotchas."""

    def test_gotcha_database_not_empty(self):
        """Gotcha database should have entries."""
        from prime_train.validator.gotchas import GOTCHA_DATABASE

        assert len(GOTCHA_DATABASE) > 0
        assert all(g.id for g in GOTCHA_DATABASE)
        assert all(g.detection for g in GOTCHA_DATABASE)
        assert all(g.recommendation for g in GOTCHA_DATABASE)

    def test_gotcha_hours_lost_tracked(self):
        """Each gotcha should track hours lost."""
        from prime_train.validator.gotchas import GOTCHA_DATABASE

        total_hours = sum(g.hours_lost for g in GOTCHA_DATABASE)
        assert total_hours > 0, "Should track debugging time"


class TestMemoryEstimation:
    """Tests for memory estimation."""

    def test_estimate_7b_model(self):
        """7B model should estimate reasonable memory."""
        from prime_train.validator.memory import estimate_model_memory_gb

        memory = estimate_model_memory_gb("Qwen/Qwen2.5-7B", training_mode="lora")

        # 7B model with LoRA should be ~15-25GB
        assert 10 < memory < 30, f"Unexpected memory estimate: {memory}GB"

    def test_estimate_extracts_from_name(self):
        """Should extract model size from name."""
        from prime_train.validator.memory import estimate_model_memory_gb

        # These should all estimate based on the number in the name
        mem_3b = estimate_model_memory_gb("some-model-3b")
        mem_8b = estimate_model_memory_gb("some-model-8B")

        assert mem_3b < mem_8b, "8B model should need more memory than 3B"
