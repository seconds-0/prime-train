"""
Memory estimation for cost optimization.

Re-exports from validator.memory for convenience.
"""

from prime_train.validator.memory import estimate_model_memory_gb


def estimate_memory(
    model_name: str,
    training_type: str = "lora",
    dtype: str = "bf16",
) -> float:
    """
    Estimate memory required for training a model.

    Args:
        model_name: HuggingFace model name
        training_type: "lora" or "full"
        dtype: Data type (bf16, fp16, fp32)

    Returns:
        Estimated VRAM in GB
    """
    training_mode = "lora" if training_type == "lora" else "full_finetune"
    return estimate_model_memory_gb(model_name, dtype=dtype, training_mode=training_mode)
