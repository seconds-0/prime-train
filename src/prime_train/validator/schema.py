"""
Schema validation for prime-rl configs.

Validates config structure and required fields.
"""

from typing import Any

from prime_train.validator.core import ValidationResult, Severity


# Config schema definition
# This defines the expected structure of prime-rl configs
CONFIG_SCHEMA = {
    "required_sections": ["orchestrator", "trainer"],
    "optional_sections": ["inference", "ckpt", "config"],
    "orchestrator": {
        "required": ["seq_len"],
        "optional": ["sampling", "env", "model", "lora_name"],
    },
    "trainer": {
        "required": ["model"],
        "optional": ["optimizer", "scheduler"],
    },
    "inference": {
        "optional": ["model", "gpu_memory_utilization"],
    },
}


def validate_schema(config: dict[str, Any]) -> list[ValidationResult]:
    """
    Validate config against schema.

    Args:
        config: Parsed TOML config

    Returns:
        List of validation results
    """
    results = []

    # Check required sections
    for section in CONFIG_SCHEMA["required_sections"]:
        if section not in config:
            results.append(ValidationResult(
                check="required_section",
                severity=Severity.ERROR,
                message=f"Missing required section: [{section}]",
            ))
        else:
            results.append(ValidationResult(
                check="required_section",
                severity=Severity.SUCCESS,
                message=f"Section [{section}] present",
            ))

    # Check orchestrator required fields
    if "orchestrator" in config:
        orch = config["orchestrator"]
        for field in CONFIG_SCHEMA["orchestrator"]["required"]:
            if field not in orch:
                results.append(ValidationResult(
                    check="required_field",
                    severity=Severity.ERROR,
                    message=f"Missing required field: orchestrator.{field}",
                ))

    # Check trainer required fields
    if "trainer" in config:
        trainer = config["trainer"]
        for field in CONFIG_SCHEMA["trainer"]["required"]:
            if field not in trainer:
                results.append(ValidationResult(
                    check="required_field",
                    severity=Severity.ERROR,
                    message=f"Missing required field: trainer.{field}",
                ))

    # Check for unknown top-level sections (typos)
    known_sections = set(CONFIG_SCHEMA["required_sections"]) | set(CONFIG_SCHEMA["optional_sections"])
    for section in config:
        if section not in known_sections:
            results.append(ValidationResult(
                check="unknown_section",
                severity=Severity.WARNING,
                message=f"Unknown top-level section: [{section}]",
                details="This may be a typo or unsupported config",
            ))

    return results
