"""
Memory estimation for training configs.

Estimates VRAM requirements based on model size and training settings.
"""

from typing import Any

from prime_train.validator.core import ValidationResult, Severity


# Model size estimates (in billions of parameters)
# These are approximations - actual sizes vary
MODEL_SIZE_ESTIMATES = {
    "qwen/qwen3-8b": 8.2,
    "qwen/qwen2.5-7b": 7.0,
    "qwen/qwen2.5-7b-instruct": 7.0,
    "qwen/qwen2.5-3b": 3.0,
    "meta-llama/llama-3.2-3b": 3.0,
    "meta-llama/llama-3.1-8b": 8.0,
    "mistralai/mistral-7b-v0.1": 7.0,
}

# Memory multipliers for different training modes
MEMORY_MULTIPLIERS = {
    "inference": 1.2,        # Model weights + some overhead
    "lora": 1.5,             # Frozen base + adapter weights
    "full_finetune": 4.0,    # Model + gradients + 2x optimizer states (AdamW)
}

# Bytes per parameter for different dtypes
BYTES_PER_PARAM = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}


def estimate_model_memory_gb(
    model_name: str,
    dtype: str = "bf16",
    training_mode: str = "lora",
) -> float:
    """
    Estimate memory required for a model.

    Args:
        model_name: HuggingFace model name
        dtype: Data type (bf16, fp16, fp32, int8, int4)
        training_mode: Training mode (inference, lora, full_finetune)

    Returns:
        Estimated memory in GB
    """
    # Get model size in billions
    model_lower = model_name.lower()
    params_billions = None

    # Try exact match first
    for key, size in MODEL_SIZE_ESTIMATES.items():
        if key in model_lower:
            params_billions = size
            break

    # Try to extract from model name (e.g., "8b", "7b", "3b")
    if params_billions is None:
        import re
        match = re.search(r"(\d+(?:\.\d+)?)[bB]", model_name)
        if match:
            params_billions = float(match.group(1))

    if params_billions is None:
        # Default to 7B if we can't determine
        params_billions = 7.0

    # Calculate base memory
    bytes_per_param = BYTES_PER_PARAM.get(dtype, 2)
    base_memory_bytes = params_billions * 1e9 * bytes_per_param
    base_memory_gb = base_memory_bytes / (1024 ** 3)

    # Apply training multiplier
    multiplier = MEMORY_MULTIPLIERS.get(training_mode, 1.5)
    total_memory_gb = base_memory_gb * multiplier

    return total_memory_gb


def estimate_memory_requirements(
    model_name: str,
    config: dict[str, Any],
    gpu_count: int,
    gpu_memory_gb: float,
) -> list[ValidationResult]:
    """
    Estimate memory requirements and check if they fit.

    Args:
        model_name: HuggingFace model name
        config: Parsed training config
        gpu_count: Number of GPUs
        gpu_memory_gb: Memory per GPU in GB

    Returns:
        List of validation results
    """
    results = []

    # Determine training mode
    trainer = config.get("trainer", {})
    model_config = trainer.get("model", {})
    training_mode = "lora" if "lora" in model_config else "full_finetune"

    # Determine dtype
    dtype = model_config.get("dtype", "bf16")

    # Estimate memory
    estimated_memory = estimate_model_memory_gb(
        model_name,
        dtype=dtype,
        training_mode=training_mode,
    )

    # Calculate available memory
    # Account for gpu_memory_utilization setting
    inference_config = config.get("inference", {})
    memory_util = inference_config.get("gpu_memory_utilization", 0.90)
    available_per_gpu = gpu_memory_gb * memory_util
    total_available = available_per_gpu * gpu_count

    # Check fit
    if estimated_memory > total_available:
        results.append(ValidationResult(
            check="memory_fit",
            severity=Severity.ERROR,
            message=f"Estimated memory ({estimated_memory:.1f} GB) exceeds available ({total_available:.1f} GB)",
            details=f"Model: {model_name}, Mode: {training_mode}, GPUs: {gpu_count}x{gpu_memory_gb}GB",
            fix="Reduce batch_size, max_tokens, or use LoRA instead of full fine-tuning",
        ))
    else:
        results.append(ValidationResult(
            check="memory_fit",
            severity=Severity.SUCCESS,
            message=f"Memory estimate: {estimated_memory:.1f} GB (fits in {total_available:.1f} GB available)",
        ))

        # Warn if tight
        headroom = (total_available - estimated_memory) / total_available
        if headroom < 0.15:
            results.append(ValidationResult(
                check="memory_headroom",
                severity=Severity.WARNING,
                message=f"Low memory headroom ({headroom*100:.0f}%)",
                details="Consider reducing batch_size for stability",
            ))

    return results
