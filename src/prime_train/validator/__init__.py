"""
Pre-flight validation for prime-rl training configs.

This module validates configs before training to catch common issues:
- Model compatibility (HuggingFace availability, vLLM support, not VL)
- Config schema errors (deprecated sections, forbidden params)
- Memory estimation (will it fit on the target hardware?)
- Known gotchas (FSDP+LoRA conflict, seq_len mismatches, etc.)
- System prerequisites (ulimit, CUDA, GPU health, disk budget)
"""

from prime_train.validator.types import ValidationResult, Severity
from prime_train.validator.core import (
    validate_config,
    ValidationResults,
    format_results,
)
from prime_train.validator.gotchas import GOTCHA_DATABASE, Gotcha
from prime_train.validator.schema import CONFIG_SCHEMA
from prime_train.validator.prerequisites import (
    check_ulimit,
    check_cuda_available,
    check_gpu_health,
    check_vllm_environment,
    check_disk_budget,
    run_all_prerequisites,
)

__all__ = [
    # Core validation
    "validate_config",
    "ValidationResult",
    "ValidationResults",
    "Severity",
    "format_results",
    # Gotchas
    "GOTCHA_DATABASE",
    "Gotcha",
    # Schema
    "CONFIG_SCHEMA",
    # Prerequisites
    "check_ulimit",
    "check_cuda_available",
    "check_gpu_health",
    "check_vllm_environment",
    "check_disk_budget",
    "run_all_prerequisites",
]
