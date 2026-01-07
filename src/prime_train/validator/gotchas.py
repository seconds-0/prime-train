"""
Known gotchas database for prime-rl configs.

This module contains a database of known issues we've encountered
during training, along with detection logic and recommended fixes.

The gotchas were discovered through 34+ hours of debugging and 21
config iterations. Adding them here prevents others from repeating
our mistakes.
"""

from dataclasses import dataclass
from typing import Any, Callable

from prime_train.validator.core import ValidationResult, Severity


@dataclass
class Gotcha:
    """
    A known gotcha in prime-rl training configs.

    Attributes:
        id: Unique identifier
        name: Short name
        description: What the issue is
        detection: How to detect it in a config
        recommendation: What to do instead
        severity: How bad is it (error vs warning)
        hours_lost: How many hours we lost to this issue
    """
    id: str
    name: str
    description: str
    detection: Callable[[dict[str, Any]], bool]
    recommendation: str
    severity: Severity = Severity.WARNING
    hours_lost: float = 0.0


def _detect_fsdp_lora_conflict(config: dict[str, Any]) -> bool:
    """Detect FSDP CPU offload + LoRA conflict."""
    trainer = config.get("trainer", {})
    model = trainer.get("model", {})

    # Check if fsdp_cpu_offload is enabled
    fsdp_offload = model.get("fsdp_cpu_offload", False)

    # Check if LoRA is enabled
    lora_enabled = "lora" in model or "experimental" in model and "lora" in model.get("experimental", {})

    return fsdp_offload and lora_enabled


def _detect_vl_model(config: dict[str, Any]) -> bool:
    """Detect vision-language model (not supported)."""
    model_name = None

    for section in ["trainer", "orchestrator", "inference"]:
        if section in config and "model" in config[section]:
            model_name = config[section]["model"].get("name_or_path", "")
            break

    if model_name:
        # Check for common VL model indicators
        vl_indicators = ["vl", "vision", "visual", "llava", "qwen-vl", "qwen2-vl"]
        model_lower = model_name.lower()
        return any(ind in model_lower for ind in vl_indicators)

    return False


def _detect_deprecated_lora_section(config: dict[str, Any]) -> bool:
    """Detect deprecated [trainer.model.experimental.lora] section."""
    trainer = config.get("trainer", {})
    model = trainer.get("model", {})
    experimental = model.get("experimental", {})
    return "lora" in experimental


def _detect_seq_len_mismatch(config: dict[str, Any]) -> bool:
    """Detect seq_len mismatch between trainer and orchestrator."""
    trainer_seq = config.get("trainer", {}).get("model", {}).get("seq_len")
    orch_seq = config.get("orchestrator", {}).get("seq_len")

    if trainer_seq is not None and orch_seq is not None:
        return trainer_seq < orch_seq

    return False


def _detect_forbidden_params(config: dict[str, Any]) -> bool:
    """Detect forbidden sampling parameters."""
    sampling = config.get("orchestrator", {}).get("sampling", {})
    forbidden = ["top_p", "mask_truncated_completions", "zero_truncated_completions"]
    return any(p in sampling for p in forbidden)


def _detect_missing_lora_name(config: dict[str, Any]) -> bool:
    """Detect missing lora_name when LoRA is enabled."""
    trainer = config.get("trainer", {})
    model = trainer.get("model", {})
    lora_enabled = "lora" in model

    if lora_enabled:
        orchestrator = config.get("orchestrator", {})
        return "lora_name" not in orchestrator

    return False


def _detect_prime_executor_latency(config: dict[str, Any]) -> bool:
    """Detect potential latency bottleneck with prime executor."""
    env = config.get("orchestrator", {}).get("env", {})
    executor = env.get("executor_backend", "").lower()

    # Only relevant for tool-calling environments
    has_tools = "tool" in str(config).lower() or "function" in str(config).lower()

    return executor == "prime" and has_tools


def _detect_checkpointing_disabled(config: dict[str, Any]) -> bool:
    """Detect when checkpointing is not configured."""
    # Checkpointing is typically a CLI flag, not in config
    # But we can check if ckpt section exists
    return "ckpt" not in config


# The gotcha database - all known issues we've discovered
GOTCHA_DATABASE: list[Gotcha] = [
    Gotcha(
        id="fsdp-lora-conflict",
        name="FSDP + LoRA Conflict",
        description="FSDP CPU offload with LoRA uses 1.65x MORE memory, not less",
        detection=_detect_fsdp_lora_conflict,
        recommendation="Use activation checkpointing instead: [trainer.model.ac] freq = 1",
        severity=Severity.ERROR,
        hours_lost=1.0,
    ),
    Gotcha(
        id="vl-model",
        name="Vision-Language Model",
        description="VL models are not compatible with vLLM/prime-rl text pipeline",
        detection=_detect_vl_model,
        recommendation="Use text-only variant (e.g., Qwen2.5-7B instead of Qwen2-VL-7B)",
        severity=Severity.ERROR,
        hours_lost=0.5,
    ),
    Gotcha(
        id="deprecated-lora-section",
        name="Deprecated LoRA Section",
        description="[trainer.model.experimental.lora] is deprecated",
        detection=_detect_deprecated_lora_section,
        recommendation="Use [trainer.model.lora] instead",
        severity=Severity.ERROR,
        hours_lost=0.25,
    ),
    Gotcha(
        id="seq-len-mismatch",
        name="Sequence Length Mismatch",
        description="trainer.model.seq_len < orchestrator.seq_len causes truncation",
        detection=_detect_seq_len_mismatch,
        recommendation="Set trainer.model.seq_len >= orchestrator.seq_len",
        severity=Severity.ERROR,
        hours_lost=0.5,
    ),
    Gotcha(
        id="forbidden-params",
        name="Forbidden Sampling Parameters",
        description="top_p, mask_truncated_completions, zero_truncated_completions are not supported",
        detection=_detect_forbidden_params,
        recommendation="Remove these parameters from [orchestrator.sampling]",
        severity=Severity.ERROR,
        hours_lost=0.25,
    ),
    Gotcha(
        id="missing-lora-name",
        name="Missing LoRA Name",
        description="lora_name required under [orchestrator] when using LoRA",
        detection=_detect_missing_lora_name,
        recommendation="Add lora_name = '<name>' under [orchestrator]",
        severity=Severity.ERROR,
        hours_lost=0.25,
    ),
    Gotcha(
        id="prime-executor-latency",
        name="Prime Executor Latency Bottleneck",
        description="Remote sandbox (executor_backend='prime') adds ~1.5s per tool call",
        detection=_detect_prime_executor_latency,
        recommendation="Use executor_backend='local' for 10-15x speedup in tool-calling tasks",
        severity=Severity.WARNING,
        hours_lost=24.0,  # Our biggest debugging session
    ),
    Gotcha(
        id="checkpointing-disabled",
        name="Checkpointing Not Configured",
        description="Without checkpointing, spot instance interruption means complete restart",
        detection=_detect_checkpointing_disabled,
        recommendation="Add --ckpt --ckpt.interval 5 --ckpt.keep-last 3 to your run command",
        severity=Severity.WARNING,
        hours_lost=4.0,
    ),
]


def check_gotchas(config: dict[str, Any]) -> list[ValidationResult]:
    """
    Check config against all known gotchas.

    Args:
        config: Parsed TOML config

    Returns:
        List of ValidationResults for any triggered gotchas
    """
    results = []

    for gotcha in GOTCHA_DATABASE:
        if gotcha.detection(config):
            results.append(ValidationResult(
                check=gotcha.id,
                severity=gotcha.severity,
                message=gotcha.name,
                details=gotcha.description,
                fix=gotcha.recommendation,
            ))

    return results
