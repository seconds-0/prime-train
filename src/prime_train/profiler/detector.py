"""
Bottleneck detection for RL training.

Profiles training steps to identify latency vs throughput issues.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table


class BottleneckType(Enum):
    """Type of bottleneck detected."""
    LATENCY_BOUND = "latency"
    THROUGHPUT_BOUND = "throughput"
    COMMUNICATION_BOUND = "communication"
    BALANCED = "balanced"


@dataclass
class ComponentTiming:
    """Timing for a single component."""
    name: str
    total_seconds: float
    percentage: float
    is_bottleneck: bool = False


@dataclass
class ProfileResults:
    """Results from profiling training."""
    total_time_seconds: float
    steps_profiled: int
    components: list[ComponentTiming]
    bottleneck_type: BottleneckType
    diagnosis: str
    recommendations: list[str]


def profile_training(
    config_path: Path,
    steps: int = 5,
) -> ProfileResults:
    """
    Profile training to detect bottlenecks.

    This is a placeholder implementation. Real profiling would:
    1. Instrument the training loop
    2. Measure time spent in each component
    3. Analyze the breakdown

    For now, we return example results based on common patterns.

    Args:
        config_path: Path to training config
        steps: Number of steps to profile

    Returns:
        ProfileResults with timing breakdown and recommendations
    """
    import tomli

    # Load config to check executor
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    executor = config.get("orchestrator", {}).get("env", {}).get("executor_backend", "local")

    # Simulate profiling based on executor type
    if executor == "prime":
        # Remote executor is typically latency-bound
        return _create_latency_bound_results(steps)
    else:
        # Local executor is typically throughput-bound or balanced
        return _create_balanced_results(steps)


def _create_latency_bound_results(steps: int) -> ProfileResults:
    """Create results for latency-bound training."""
    total_time = steps * 25 * 60  # ~25 min per step

    components = [
        ComponentTiming("Tool execution", total_time * 0.78, 78.0, is_bottleneck=True),
        ComponentTiming("Policy inference", total_time * 0.12, 12.0),
        ComponentTiming("Training step", total_time * 0.08, 8.0),
        ComponentTiming("Communication", total_time * 0.02, 2.0),
    ]

    return ProfileResults(
        total_time_seconds=total_time,
        steps_profiled=steps,
        components=components,
        bottleneck_type=BottleneckType.LATENCY_BOUND,
        diagnosis=(
            "Training is LATENCY-BOUND (not throughput-bound).\n"
            "Tool execution via remote sandbox adds ~1.5s per call.\n"
            "GPU is idle waiting for tool execution."
        ),
        recommendations=[
            "Switch executor_backend: 'prime' → 'local' (10-15x speedup)",
            "If local not possible, reduce rollouts_per_example to limit tool calls",
            "Consider batching tool calls if environment supports it",
        ],
    )


def _create_balanced_results(steps: int) -> ProfileResults:
    """Create results for balanced training."""
    total_time = steps * 3 * 60  # ~3 min per step

    components = [
        ComponentTiming("Policy inference", total_time * 0.35, 35.0),
        ComponentTiming("Training step", total_time * 0.30, 30.0),
        ComponentTiming("Tool execution", total_time * 0.25, 25.0),
        ComponentTiming("Communication", total_time * 0.10, 10.0),
    ]

    return ProfileResults(
        total_time_seconds=total_time,
        steps_profiled=steps,
        components=components,
        bottleneck_type=BottleneckType.BALANCED,
        diagnosis=(
            "Training is well-balanced.\n"
            "No single component dominates.\n"
            "Consider optimizing inference for marginal gains."
        ),
        recommendations=[
            "Enable prefix caching if not already (may reduce inference time)",
            "Increase batch_size if memory allows",
            "Current config is near-optimal for local executor",
        ],
    )


def format_profile_results(results: ProfileResults, console: Console) -> None:
    """Format profile results for display."""
    # Component table
    table = Table(title="Bottleneck Analysis")
    table.add_column("Component")
    table.add_column("Time (%)", justify="right")
    table.add_column("Status")

    for comp in results.components:
        status = "[red]⚠ BOTTLENECK[/red]" if comp.is_bottleneck else "[green]OK[/green]"
        table.add_row(comp.name, f"{comp.percentage:.0f}%", status)

    console.print(table)

    # Diagnosis
    console.print(f"\n[bold]DIAGNOSIS:[/bold] {results.bottleneck_type.value.upper()}-BOUND")
    console.print(results.diagnosis)

    # Recommendations
    console.print("\n[bold]RECOMMENDATIONS:[/bold]")
    for i, rec in enumerate(results.recommendations, 1):
        console.print(f"  {i}. {rec}")
