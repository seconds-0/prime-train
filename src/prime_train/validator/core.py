"""
Core validation logic for prime-rl configs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli
from rich.console import Console

from prime_train.validator.types import Severity, ValidationResult
from prime_train.validator.schema import validate_schema
from prime_train.validator.model import check_model_compatibility
from prime_train.validator.memory import estimate_memory_requirements


@dataclass
class ValidationResults:
    """Collection of validation results."""
    results: list[ValidationResult] = field(default_factory=list)
    config_path: Path | None = None

    @property
    def has_errors(self) -> bool:
        return any(r.severity == Severity.ERROR for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.severity == Severity.WARNING for r in self.results)

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)

    def add_success(self, check: str, message: str) -> None:
        self.add(ValidationResult(check=check, severity=Severity.SUCCESS, message=message))

    def add_warning(self, check: str, message: str, details: str | None = None, fix: str | None = None) -> None:
        self.add(ValidationResult(check=check, severity=Severity.WARNING, message=message, details=details, fix=fix))

    def add_error(self, check: str, message: str, details: str | None = None, fix: str | None = None) -> None:
        self.add(ValidationResult(check=check, severity=Severity.ERROR, message=message, details=details, fix=fix))


def validate_config(config_path: Path) -> ValidationResults:
    """
    Validate a prime-rl training config.

    Runs all validation checks:
    1. Config file exists and is valid TOML
    2. Schema validation (structure, required fields)
    3. Model compatibility (HuggingFace, vLLM, not VL)
    4. Memory estimation (fits on target hardware)
    5. Known gotchas (FSDP+LoRA, etc.)

    Args:
        config_path: Path to the config.toml file

    Returns:
        ValidationResults with all check results
    """
    results = ValidationResults(config_path=config_path)

    # 1. Check file exists
    if not config_path.exists():
        results.add_error(
            check="file_exists",
            message=f"Config file not found: {config_path}",
        )
        return results

    # 2. Parse TOML
    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        results.add_success("toml_parse", "Config is valid TOML")
    except tomli.TOMLDecodeError as e:
        results.add_error(
            check="toml_parse",
            message="Invalid TOML syntax",
            details=str(e),
        )
        return results

    # 3. Schema validation
    schema_results = validate_schema(config)
    for r in schema_results:
        results.add(r)

    # 4. Model compatibility
    model_name = _extract_model_name(config)
    if model_name:
        model_results = check_model_compatibility(model_name)
        for r in model_results:
            results.add(r)
    else:
        results.add_warning(
            check="model_name",
            message="Could not determine model name from config",
            details="Expected model name in trainer.model.name_or_path or orchestrator.model.name_or_path",
        )

    # 5. Memory estimation
    gpu_count = _extract_gpu_count(config)
    gpu_memory = _extract_gpu_memory(config)
    if model_name and gpu_count and gpu_memory:
        memory_results = estimate_memory_requirements(
            model_name=model_name,
            config=config,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory,
        )
        for r in memory_results:
            results.add(r)

    # 6. Known gotchas (lazy import to avoid circular dependency)
    from prime_train.validator.gotchas import check_gotchas
    gotcha_results = check_gotchas(config)
    for r in gotcha_results:
        results.add(r)

    return results


def _extract_model_name(config: dict[str, Any]) -> str | None:
    """Extract model name from config."""
    # Try trainer.model.name_or_path first
    if "trainer" in config and "model" in config["trainer"]:
        if "name_or_path" in config["trainer"]["model"]:
            return config["trainer"]["model"]["name_or_path"]

    # Try orchestrator.model.name_or_path
    if "orchestrator" in config and "model" in config["orchestrator"]:
        if "name_or_path" in config["orchestrator"]["model"]:
            return config["orchestrator"]["model"]["name_or_path"]

    # Try inference.model.name_or_path
    if "inference" in config and "model" in config["inference"]:
        if "name_or_path" in config["inference"]["model"]:
            return config["inference"]["model"]["name_or_path"]

    return None


def _extract_gpu_count(config: dict[str, Any]) -> int | None:
    """Extract GPU count from config."""
    # This would need to be inferred from the config or system
    # For now, return None to skip memory validation
    return None


def _extract_gpu_memory(config: dict[str, Any]) -> float | None:
    """Extract per-GPU memory in GB from config."""
    # This would need to be inferred from hardware presets
    return None


def format_results(results: ValidationResults, console: Console) -> None:
    """Format validation results for display."""
    from rich.panel import Panel
    from rich.text import Text

    for r in results.results:
        # Icon and color based on severity
        if r.severity == Severity.SUCCESS:
            icon = "[green]✓[/green]"
            style = "green"
        elif r.severity == Severity.WARNING:
            icon = "[yellow]⚠[/yellow]"
            style = "yellow"
        elif r.severity == Severity.ERROR:
            icon = "[red]✗[/red]"
            style = "red"
        else:
            icon = "[blue]ℹ[/blue]"
            style = "blue"

        console.print(f"{icon} {r.message}")

        if r.details:
            console.print(f"  [dim]{r.details}[/dim]")

        if r.fix:
            console.print(f"  [cyan]→ {r.fix}[/cyan]")

    # Summary
    errors = sum(1 for r in results.results if r.severity == Severity.ERROR)
    warnings = sum(1 for r in results.results if r.severity == Severity.WARNING)

    if errors > 0:
        console.print(f"\n[red]{errors} error(s), {warnings} warning(s)[/red]")
    elif warnings > 0:
        console.print(f"\n[yellow]{warnings} warning(s)[/yellow]")
    else:
        console.print("\n[green]All checks passed[/green]")
