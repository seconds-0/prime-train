"""
prime-train: Production-grade training harness for prime-rl.

Provides validation, cost optimization, and resilience for RL training.
"""

__version__ = "0.1.0"
__author__ = "Alexander Huth"

from prime_train.validator import validate_config
from prime_train.cost import find_gpus, estimate_memory
from prime_train.resilience import BackupManager
from prime_train.config import ConfigManager

__all__ = [
    "validate_config",
    "find_gpus",
    "estimate_memory",
    "BackupManager",
    "ConfigManager",
]
