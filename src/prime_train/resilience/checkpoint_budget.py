"""
Checkpoint budget management.

Calculates disk budgets and validates checkpoint configuration
against available storage. Prevents disk-full conditions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prime_train.resilience.disk import (
    get_available_disk_gb,
    estimate_checkpoint_size_gb,
    get_disk_budget,
)
from prime_train.validator.types import ValidationResult, Severity


@dataclass
class CheckpointBudget:
    """Checkpoint budget calculation result."""
    available_disk_gb: float
    estimated_checkpoint_gb: float
    safety_buffer_gb: float
    max_local_checkpoints: int
    requested_checkpoints: int
    has_external_backup: bool
    external_provider: str | None

    @property
    def budget_exceeded(self) -> bool:
        """Check if requested checkpoints exceed disk budget."""
        return self.requested_checkpoints > self.max_local_checkpoints

    @property
    def needs_offloading(self) -> bool:
        """Check if checkpoints need to be offloaded to external storage."""
        return self.budget_exceeded and self.has_external_backup

    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid (has solution for storage)."""
        if not self.budget_exceeded:
            return True
        return self.has_external_backup

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Available disk: {self.available_disk_gb:.1f} GB",
            f"Estimated checkpoint size: {self.estimated_checkpoint_gb:.1f} GB",
            f"Safety buffer: {self.safety_buffer_gb:.1f} GB",
            f"Max local checkpoints: {self.max_local_checkpoints}",
            f"Requested checkpoints: {self.requested_checkpoints}",
        ]

        if self.has_external_backup:
            lines.append(f"External backup: {self.external_provider}")
        else:
            lines.append("External backup: not configured")

        if self.budget_exceeded:
            if self.has_external_backup:
                lines.append("Status: Will offload to external storage")
            else:
                lines.append("Status: BUDGET EXCEEDED - configure backup or reduce keep_last")
        else:
            lines.append("Status: OK")

        return "\n".join(lines)


def calculate_checkpoint_budget(
    config: dict[str, Any],
    checkpoint_dir: Path | str = "/opt/run/checkpoints/",
    safety_buffer_gb: float = 10.0,
) -> CheckpointBudget:
    """
    Calculate checkpoint budget based on config and disk space.

    Args:
        config: Training config dict (TOML format).
        checkpoint_dir: Directory where checkpoints will be stored.
        safety_buffer_gb: Reserved space for logs, temp files, etc.

    Returns:
        CheckpointBudget with calculation results.
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Get available disk space
    available_gb = get_available_disk_gb(checkpoint_dir)

    # Estimate checkpoint size from model config
    checkpoint_gb = estimate_checkpoint_size_gb(config)

    # Get requested keep_last from config
    ckpt_config = config.get("ckpt", {})
    requested = ckpt_config.get("keep_last", 3)

    # Calculate max local checkpoints
    max_local = get_disk_budget(checkpoint_dir, checkpoint_gb, safety_buffer_gb)

    # Check for external backup config
    backup_config = config.get("backup", {})
    has_backup = bool(backup_config.get("provider"))
    provider = backup_config.get("provider")

    return CheckpointBudget(
        available_disk_gb=available_gb,
        estimated_checkpoint_gb=checkpoint_gb,
        safety_buffer_gb=safety_buffer_gb,
        max_local_checkpoints=max_local,
        requested_checkpoints=requested,
        has_external_backup=has_backup,
        external_provider=provider,
    )


def validate_checkpoint_budget(
    config: dict[str, Any],
    checkpoint_dir: Path | str = "/opt/run/checkpoints/",
    safety_buffer_gb: float = 10.0,
) -> list[ValidationResult]:
    """
    Validate checkpoint configuration against disk budget.

    Args:
        config: Training config dict (TOML format).
        checkpoint_dir: Directory where checkpoints will be stored.
        safety_buffer_gb: Reserved space for logs, temp files, etc.

    Returns:
        List of validation results (errors, warnings).
    """
    results = []

    budget = calculate_checkpoint_budget(config, checkpoint_dir, safety_buffer_gb)

    # Check if disk is critically low
    if budget.available_disk_gb < safety_buffer_gb:
        results.append(ValidationResult(
            check="disk-space-critical",
            severity=Severity.ERROR,
            message="Disk space critically low",
            details=f"Only {budget.available_disk_gb:.1f} GB available, need at least {safety_buffer_gb:.1f} GB",
            fix="Free up disk space or use a different checkpoint directory",
        ))
        return results

    # Check if even one checkpoint would fit
    if budget.max_local_checkpoints < 1:
        results.append(ValidationResult(
            check="disk-space-insufficient",
            severity=Severity.ERROR,
            message="Insufficient disk space for checkpoints",
            details=f"Need ~{budget.estimated_checkpoint_gb:.1f} GB per checkpoint, "
                    f"but only {budget.available_disk_gb - safety_buffer_gb:.1f} GB usable",
            fix="Free up disk space, configure external backup, or use a smaller model",
        ))
        return results

    # Check if budget is exceeded
    if budget.budget_exceeded:
        if budget.has_external_backup:
            # Warning: will need to offload
            results.append(ValidationResult(
                check="checkpoint-budget-offload",
                severity=Severity.WARNING,
                message="Checkpoint budget requires offloading",
                details=f"keep_last={budget.requested_checkpoints} exceeds local budget "
                        f"({budget.max_local_checkpoints}). Will offload to {budget.external_provider}.",
                fix=None,
            ))
        else:
            # Error: no solution
            results.append(ValidationResult(
                check="checkpoint-budget-exceeded",
                severity=Severity.ERROR,
                message="Checkpoint budget exceeded",
                details=f"keep_last={budget.requested_checkpoints} exceeds disk budget "
                        f"({budget.max_local_checkpoints} max). {budget.estimated_checkpoint_gb:.1f} GB per checkpoint.",
                fix=f"Either: 1) Set keep_last <= {budget.max_local_checkpoints}, or "
                    f"2) Configure backup provider (s3/b2/gcs) for offloading",
            ))
    else:
        # Success: budget is fine
        results.append(ValidationResult(
            check="checkpoint-budget-ok",
            severity=Severity.SUCCESS,
            message="Checkpoint budget OK",
            details=f"Can store {budget.max_local_checkpoints} checkpoints locally "
                    f"({budget.estimated_checkpoint_gb:.1f} GB each)",
            fix=None,
        ))

    return results
