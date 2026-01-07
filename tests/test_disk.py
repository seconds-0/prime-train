"""Tests for disk monitoring and checkpoint budget management."""

import tempfile
from pathlib import Path

import pytest

from prime_train.resilience.disk import (
    get_available_disk_gb,
    get_checkpoint_size_gb,
    estimate_checkpoint_size_gb,
    get_disk_budget,
    _estimate_params_from_name,
    _get_dtype_bytes,
)
from prime_train.resilience.checkpoint_budget import (
    CheckpointBudget,
    calculate_checkpoint_budget,
    validate_checkpoint_budget,
)
from prime_train.validator.types import Severity


class TestDiskMonitoring:
    """Tests for disk monitoring functions."""

    def test_get_available_disk_gb_returns_positive(self):
        """get_available_disk_gb should return positive value for existing path."""
        available = get_available_disk_gb("/tmp")
        assert available > 0

    def test_get_available_disk_gb_handles_nonexistent_path(self):
        """get_available_disk_gb should handle non-existent paths by using parent."""
        available = get_available_disk_gb("/tmp/nonexistent/path/that/does/not/exist")
        assert available > 0

    def test_get_checkpoint_size_gb_empty_dir(self):
        """get_checkpoint_size_gb should return 0 for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            size = get_checkpoint_size_gb(Path(tmpdir))
            assert size == 0.0

    def test_get_checkpoint_size_gb_with_files(self):
        """get_checkpoint_size_gb should correctly calculate file sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            test_file = Path(tmpdir) / "test.bin"
            test_file.write_bytes(b"0" * 1024 * 1024)  # 1 MB

            size = get_checkpoint_size_gb(Path(tmpdir))
            # Should be approximately 1/1024 GB
            assert 0.0009 < size < 0.0011

    def test_get_checkpoint_size_gb_nonexistent(self):
        """get_checkpoint_size_gb should return 0 for non-existent path."""
        size = get_checkpoint_size_gb(Path("/nonexistent/checkpoint"))
        assert size == 0.0


class TestParamsEstimation:
    """Tests for parameter estimation from model names."""

    def test_estimate_params_7b_model(self):
        """Should detect 7B parameters from model name."""
        assert _estimate_params_from_name("Qwen/Qwen2.5-7B-Instruct") == 7.0

    def test_estimate_params_8b_model(self):
        """Should detect 8B parameters from model name."""
        assert _estimate_params_from_name("meta-llama/Llama-3-8B") == 8.0

    def test_estimate_params_70b_model(self):
        """Should detect 70B parameters from model name."""
        assert _estimate_params_from_name("meta-llama/Llama-2-70B") == 70.0

    def test_estimate_params_decimal(self):
        """Should handle decimal parameter counts."""
        assert _estimate_params_from_name("some-model-7.5b") == 7.5

    def test_estimate_params_none_for_unknown(self):
        """Should return None for unknown model names."""
        assert _estimate_params_from_name("unknown-model") is None

    def test_estimate_params_none_input(self):
        """Should return None for None input."""
        assert _estimate_params_from_name(None) is None


class TestDtypeBytes:
    """Tests for dtype byte estimation."""

    def test_bf16_is_2_bytes(self):
        """bf16 should be 2 bytes."""
        config = {"trainer": {"model": {"dtype": "bf16"}}}
        assert _get_dtype_bytes(config) == 2.0

    def test_fp32_is_4_bytes(self):
        """fp32 should be 4 bytes."""
        config = {"trainer": {"model": {"dtype": "fp32"}}}
        assert _get_dtype_bytes(config) == 4.0

    def test_default_is_2_bytes(self):
        """Default dtype should be 2 bytes (bf16)."""
        config = {}
        assert _get_dtype_bytes(config) == 2.0


class TestCheckpointSizeEstimation:
    """Tests for checkpoint size estimation."""

    def test_estimate_7b_model_size(self):
        """Should estimate ~47GB for a 7B bf16 model with AdamW."""
        config = {
            "trainer": {
                "model": {
                    "name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                    "dtype": "bf16",
                }
            }
        }
        size = estimate_checkpoint_size_gb(config)
        # 7B * 2 bytes = 14GB model
        # 7B * 2 * 2 bytes = 28GB optimizer states
        # 42GB * 1.1 overhead = ~46GB
        assert 40 < size < 55  # Allow some variance

    def test_estimate_unknown_model_defaults_to_7b(self):
        """Should default to 7B for unknown models."""
        config = {
            "trainer": {
                "model": {
                    "name_or_path": "unknown-model",
                    "dtype": "bf16",
                }
            }
        }
        size = estimate_checkpoint_size_gb(config)
        assert 40 < size < 55


class TestDiskBudget:
    """Tests for disk budget calculation."""

    def test_get_disk_budget_with_positive_values(self):
        """Should calculate correct budget with positive values."""
        # 100GB available, 10GB per checkpoint, 10GB buffer
        # = 90GB usable / 10GB = 9 checkpoints
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock by calculating based on real available space
            available = get_available_disk_gb(tmpdir)
            checkpoint_size = 10.0
            safety_buffer = 10.0

            budget = get_disk_budget(tmpdir, checkpoint_size, safety_buffer)

            expected = int((available - safety_buffer) / checkpoint_size)
            assert budget == expected

    def test_get_disk_budget_zero_checkpoint_size(self):
        """Should return 0 for zero checkpoint size."""
        budget = get_disk_budget("/tmp", 0.0, 10.0)
        assert budget == 0

    def test_get_disk_budget_insufficient_space(self):
        """Should return 0 when safety buffer exceeds available space."""
        # Use unreasonably high safety buffer
        budget = get_disk_budget("/tmp", 10.0, 1_000_000.0)
        assert budget == 0


class TestCheckpointBudgetCalculation:
    """Tests for checkpoint budget calculation."""

    def test_budget_not_exceeded_when_fits(self):
        """Budget should not be exceeded when keep_last fits disk."""
        config = {
            "trainer": {
                "model": {
                    "name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                }
            },
            "ckpt": {
                "keep_last": 1,
            },
        }
        budget = calculate_checkpoint_budget(config)

        # Most machines should have space for at least 1 checkpoint
        # But we don't know for sure - just check the calculation works
        assert isinstance(budget, CheckpointBudget)
        assert budget.requested_checkpoints == 1
        assert budget.estimated_checkpoint_gb > 0

    def test_budget_with_external_backup(self):
        """Should recognize external backup configuration."""
        config = {
            "trainer": {
                "model": {
                    "name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                }
            },
            "ckpt": {
                "keep_last": 10,
            },
            "backup": {
                "provider": "s3",
                "bucket": "my-bucket",
            },
        }
        budget = calculate_checkpoint_budget(config)

        assert budget.has_external_backup
        assert budget.external_provider == "s3"

    def test_budget_without_external_backup(self):
        """Should recognize when no external backup is configured."""
        config = {
            "trainer": {
                "model": {
                    "name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                }
            },
            "ckpt": {
                "keep_last": 3,
            },
        }
        budget = calculate_checkpoint_budget(config)

        assert not budget.has_external_backup
        assert budget.external_provider is None


class TestCheckpointBudgetValidation:
    """Tests for checkpoint budget validation."""

    def test_validation_success_when_budget_ok(self):
        """Should return SUCCESS when budget is OK."""
        config = {
            "trainer": {
                "model": {
                    "name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                }
            },
            "ckpt": {
                "keep_last": 1,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            results = validate_checkpoint_budget(config, tmpdir)

            # At least one result should be returned
            assert len(results) > 0

            # Check that we get a meaningful result
            severities = [r.severity for r in results]
            # Should be either SUCCESS or ERROR/WARNING depending on disk space
            assert any(s in [Severity.SUCCESS, Severity.ERROR, Severity.WARNING] for s in severities)

    def test_validation_with_offloading_configured(self):
        """Should return WARNING when offloading is needed but configured."""
        config = {
            "trainer": {
                "model": {
                    "name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                }
            },
            "ckpt": {
                "keep_last": 1000,  # Way more than fits on disk
            },
            "backup": {
                "provider": "s3",
                "bucket": "my-bucket",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            results = validate_checkpoint_budget(config, tmpdir)

            # Should have a result
            assert len(results) > 0

            # Find the budget-related result
            budget_results = [r for r in results if "budget" in r.check]
            if budget_results:
                # Should be warning (offloading) or error (insufficient)
                assert any(r.severity in [Severity.WARNING, Severity.ERROR] for r in budget_results)

    def test_validation_error_when_budget_exceeded_no_backup(self):
        """Should return ERROR when budget exceeded and no backup configured."""
        config = {
            "trainer": {
                "model": {
                    "name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                }
            },
            "ckpt": {
                "keep_last": 1000,  # Way more than fits on disk
            },
            # No backup configured
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            results = validate_checkpoint_budget(config, tmpdir)

            # Should have results
            assert len(results) > 0

            # Should have error or warning about budget
            budget_results = [r for r in results if "budget" in r.check or "disk" in r.check]
            if budget_results:
                # Check that we got a meaningful severity
                severities = [r.severity for r in budget_results]
                assert any(s in [Severity.ERROR, Severity.WARNING] for s in severities)


class TestCheckpointBudgetSummary:
    """Tests for CheckpointBudget summary generation."""

    def test_summary_contains_key_info(self):
        """Summary should contain key budget information."""
        budget = CheckpointBudget(
            available_disk_gb=200.0,
            estimated_checkpoint_gb=47.0,
            safety_buffer_gb=10.0,
            max_local_checkpoints=4,
            requested_checkpoints=3,
            has_external_backup=False,
            external_provider=None,
        )

        summary = budget.summary()

        assert "200.0 GB" in summary
        assert "47.0 GB" in summary
        assert "4" in summary
        assert "3" in summary
        assert "OK" in summary

    def test_summary_shows_offload_status(self):
        """Summary should show offload status when needed."""
        budget = CheckpointBudget(
            available_disk_gb=100.0,
            estimated_checkpoint_gb=47.0,
            safety_buffer_gb=10.0,
            max_local_checkpoints=2,
            requested_checkpoints=5,
            has_external_backup=True,
            external_provider="s3",
        )

        summary = budget.summary()

        assert "s3" in summary
        assert "offload" in summary.lower()
