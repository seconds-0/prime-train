"""
Resilience module for prime-train.

Provides checkpoint backup, auto-resume, disk management, and health monitoring.
"""

from prime_train.resilience.backup import (
    BackupManager,
    BackupConfig,
    configure_backup,
    get_backup_status,
)
from prime_train.resilience.runner import TrainingRunner
from prime_train.resilience.health import check_training_status, TrainingStatus
from prime_train.resilience.disk import (
    get_available_disk_gb,
    get_checkpoint_size_gb,
    estimate_checkpoint_size_gb,
    get_disk_budget,
)
from prime_train.resilience.checkpoint_budget import (
    CheckpointBudget,
    calculate_checkpoint_budget,
    validate_checkpoint_budget,
)

__all__ = [
    # Backup
    "BackupManager",
    "BackupConfig",
    "configure_backup",
    "get_backup_status",
    # Runner
    "TrainingRunner",
    # Health
    "check_training_status",
    "TrainingStatus",
    # Disk
    "get_available_disk_gb",
    "get_checkpoint_size_gb",
    "estimate_checkpoint_size_gb",
    "get_disk_budget",
    # Checkpoint budget
    "CheckpointBudget",
    "calculate_checkpoint_budget",
    "validate_checkpoint_budget",
]
