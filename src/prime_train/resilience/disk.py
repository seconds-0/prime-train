"""
Disk monitoring utilities for checkpoint management.

Provides functions to check available disk space and estimate checkpoint sizes
to prevent disk-full conditions during training.
"""

import os
import shutil
from pathlib import Path
from typing import Any


def get_available_disk_gb(path: Path | str = "/opt/run/checkpoints/") -> float:
    """
    Get available disk space at the given path in GB.

    Args:
        path: Path to check disk space for. Uses parent directory if path doesn't exist.

    Returns:
        Available disk space in GB.
    """
    path = Path(path)

    # Find existing parent directory
    check_path = path
    while not check_path.exists() and check_path.parent != check_path:
        check_path = check_path.parent

    if not check_path.exists():
        # Fall back to root
        check_path = Path("/")

    usage = shutil.disk_usage(check_path)
    return usage.free / (1024 ** 3)


def get_checkpoint_size_gb(checkpoint_path: Path) -> float:
    """
    Get the size of an existing checkpoint directory in GB.

    Args:
        checkpoint_path: Path to checkpoint directory or file.

    Returns:
        Size in GB.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        return 0.0

    if checkpoint_path.is_file():
        return checkpoint_path.stat().st_size / (1024 ** 3)

    # Sum up all files in directory
    total_bytes = 0
    for dirpath, _, filenames in os.walk(checkpoint_path):
        for filename in filenames:
            filepath = Path(dirpath) / filename
            try:
                total_bytes += filepath.stat().st_size
            except OSError:
                # File might be deleted during traversal
                pass

    return total_bytes / (1024 ** 3)


def estimate_checkpoint_size_gb(config: dict[str, Any]) -> float:
    """
    Estimate checkpoint size based on model configuration.

    Checkpoint includes:
    - Model weights (parameters × dtype bytes)
    - Optimizer states (AdamW: 2× model weights for momentum + variance)
    - Buffers and overhead (~10%)

    Args:
        config: Training config dict (TOML format).

    Returns:
        Estimated checkpoint size in GB.
    """
    # Try to get model name from config
    model_name = _extract_model_name(config)

    # Estimate parameter count from model name
    params_billions = _estimate_params_from_name(model_name)

    if params_billions is None:
        # Fallback: assume 7B model
        params_billions = 7.0

    # Get dtype from config
    dtype_bytes = _get_dtype_bytes(config)

    # Calculate sizes
    model_weights_gb = params_billions * dtype_bytes
    optimizer_states_gb = model_weights_gb * 2  # AdamW: momentum + variance
    overhead_factor = 1.1  # 10% overhead for buffers, etc.

    total_gb = (model_weights_gb + optimizer_states_gb) * overhead_factor

    return total_gb


def _extract_model_name(config: dict[str, Any]) -> str | None:
    """Extract model name from config."""
    # Check various locations where model might be specified
    for section in ["trainer", "orchestrator", "inference"]:
        if section in config and "model" in config[section]:
            model = config[section]["model"]
            if isinstance(model, dict):
                return model.get("name_or_path")
            elif isinstance(model, str):
                return model

    return None


def _estimate_params_from_name(model_name: str | None) -> float | None:
    """
    Estimate parameter count from model name.

    Looks for common patterns like "7B", "3b", "70B" in model name.

    Returns:
        Parameters in billions, or None if not detected.
    """
    if model_name is None:
        return None

    import re

    model_lower = model_name.lower()

    # Common patterns: "7b", "7B", "7-b", "7_b", "7B-Instruct"
    patterns = [
        r"(\d+(?:\.\d+)?)\s*b(?:illion)?",  # "7b", "7B", "7 billion"
        r"(\d+(?:\.\d+)?)-?b(?:-|_|$)",     # "7-b", "7b-instruct"
    ]

    for pattern in patterns:
        match = re.search(pattern, model_lower)
        if match:
            return float(match.group(1))

    # Known model families with known sizes
    known_sizes = {
        "gpt2": 0.124,
        "gpt2-medium": 0.355,
        "gpt2-large": 0.774,
        "gpt2-xl": 1.5,
        "llama-2-7b": 7.0,
        "llama-2-13b": 13.0,
        "llama-2-70b": 70.0,
        "llama-3-8b": 8.0,
        "llama-3-70b": 70.0,
        "mistral-7b": 7.0,
        "mixtral-8x7b": 46.7,  # Total params
        "qwen2.5-7b": 7.0,
        "qwen2.5-14b": 14.0,
        "qwen2.5-72b": 72.0,
        "qwen3-8b": 8.0,
    }

    for name, size in known_sizes.items():
        if name in model_lower:
            return size

    return None


def _get_dtype_bytes(config: dict[str, Any]) -> float:
    """
    Get bytes per parameter from dtype config.

    Returns:
        Bytes per parameter (default: 2 for bf16/fp16).
    """
    # Check trainer model config for dtype
    trainer = config.get("trainer", {})
    model = trainer.get("model", {})

    dtype = model.get("dtype", "bf16")

    dtype_sizes = {
        "fp32": 4.0,
        "float32": 4.0,
        "fp16": 2.0,
        "float16": 2.0,
        "bf16": 2.0,
        "bfloat16": 2.0,
        "fp8": 1.0,
        "int8": 1.0,
        "int4": 0.5,
    }

    return dtype_sizes.get(dtype.lower(), 2.0)


def get_disk_budget(
    checkpoint_dir: Path | str,
    checkpoint_size_gb: float,
    safety_buffer_gb: float = 10.0,
) -> int:
    """
    Calculate how many checkpoints can fit on disk.

    Args:
        checkpoint_dir: Directory where checkpoints are stored.
        checkpoint_size_gb: Size of one checkpoint in GB.
        safety_buffer_gb: Reserved space for logs, temp files, etc.

    Returns:
        Maximum number of checkpoints that fit on disk.
    """
    available_gb = get_available_disk_gb(checkpoint_dir)
    usable_gb = available_gb - safety_buffer_gb

    if usable_gb <= 0:
        return 0

    if checkpoint_size_gb <= 0:
        return 0

    return int(usable_gb / checkpoint_size_gb)
