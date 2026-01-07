"""
Cost optimization module for prime-train.

Finds cheapest compatible GPUs across providers and estimates total training cost.
"""

from prime_train.cost.optimizer import find_gpus, format_gpu_table
from prime_train.cost.memory import estimate_memory
from prime_train.cost.presets import HARDWARE_PRESETS, HardwarePreset

__all__ = [
    "find_gpus",
    "format_gpu_table",
    "estimate_memory",
    "HARDWARE_PRESETS",
    "HardwarePreset",
]
