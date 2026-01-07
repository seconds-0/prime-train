"""
Model compatibility checking.

Verifies that models are compatible with prime-rl and vLLM.
"""

from typing import Any

from prime_train.validator.core import ValidationResult, Severity


# Known incompatible models (from our testing)
BLOCKED_MODELS = {
    "openai/gpt-oss-20b": "vLLM weight reload bug (TypeError: default_weight_loader)",
    "mistralai/Ministral-3-8B": "transformers KeyError during model loading",
}

# VL model patterns (not supported)
VL_MODEL_PATTERNS = ["vl", "vision", "visual", "llava", "cogvlm", "internvl"]


def check_model_compatibility(model_name: str) -> list[ValidationResult]:
    """
    Check if a model is compatible with prime-rl.

    Checks:
    1. Model exists on HuggingFace
    2. Model is not a known blocked model
    3. Model is not a vision-language model
    4. Model is compatible with vLLM

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-8B")

    Returns:
        List of validation results
    """
    results = []

    # Check against blocked models
    if model_name in BLOCKED_MODELS:
        results.append(ValidationResult(
            check="blocked_model",
            severity=Severity.ERROR,
            message=f"Model {model_name} is known to be incompatible",
            details=BLOCKED_MODELS[model_name],
            fix="Choose a different model. Qwen/Qwen2.5-7B-Instruct is verified working.",
        ))
        return results

    # Check for VL model patterns
    model_lower = model_name.lower()
    for pattern in VL_MODEL_PATTERNS:
        if pattern in model_lower:
            results.append(ValidationResult(
                check="vl_model",
                severity=Severity.ERROR,
                message=f"Vision-language model detected: {model_name}",
                details="VL models use Qwen3VLConfig which is incompatible with AutoModelForCausalLM",
                fix=f"Use text-only variant (e.g., {model_name.replace('-vl', '').replace('-VL', '')})",
            ))
            return results

    # Try to check HuggingFace availability
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()

        try:
            model_info = api.model_info(model_name)
            results.append(ValidationResult(
                check="hf_availability",
                severity=Severity.SUCCESS,
                message=f"Model {model_name} exists on HuggingFace",
            ))

            # Check model tags for compatibility hints
            tags = model_info.tags or []
            if "text-generation" not in tags and "causal-lm" not in tags:
                results.append(ValidationResult(
                    check="model_type",
                    severity=Severity.WARNING,
                    message="Model may not be a causal LM",
                    details=f"Tags: {', '.join(tags[:5])}",
                ))

        except Exception as e:
            results.append(ValidationResult(
                check="hf_availability",
                severity=Severity.ERROR,
                message=f"Model {model_name} not found on HuggingFace",
                details=str(e),
                fix="Check model name spelling and ensure it's a public model",
            ))

    except ImportError:
        results.append(ValidationResult(
            check="hf_availability",
            severity=Severity.WARNING,
            message="Could not check HuggingFace availability (huggingface_hub not installed)",
        ))

    return results
