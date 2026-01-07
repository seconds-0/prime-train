"""
prime-train CLI - Production-grade training harness for prime-rl.

Commands:
    validate    Pre-flight validation of training configs
    run         Run training with resilience features
    find        Find cheapest compatible GPUs
    init        Generate config interactively
    config      Config version management
    profile     Bottleneck detection
    backup      Configure backup settings
    status      Check training health
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="prime-train",
    help="Production-grade training harness for prime-rl",
    no_args_is_help=True,
)

console = Console()

# Sub-command groups
config_app = typer.Typer(help="Config version management")
backup_app = typer.Typer(help="Backup configuration")

app.add_typer(config_app, name="config")
app.add_typer(backup_app, name="backup")


@app.command()
def validate(
    config_path: Path = typer.Argument(..., help="Path to config.toml"),
    strict: bool = typer.Option(False, "--strict", "-s", help="Fail on warnings"),
) -> None:
    """
    Pre-flight validation of training configs.

    Checks:
    - Model exists on HuggingFace
    - Model is compatible with vLLM
    - Config schema is valid
    - Memory requirements fit hardware
    - Known gotchas (FSDP+LoRA, deprecated sections, etc.)

    Example:
        prime-train validate config.toml
    """
    from prime_train.validator import validate_config, format_results

    console.print(f"[bold]Validating[/bold] {config_path}")

    results = validate_config(config_path)
    format_results(results, console)

    if results.has_errors or (strict and results.has_warnings):
        raise typer.Exit(1)


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to config.toml"),
    backup: bool = typer.Option(False, "--backup", "-b", help="Enable checkpoint backup"),
    spot: bool = typer.Option(False, "--spot", help="Use spot instances"),
    cloud: Optional[str] = typer.Option(None, "--cloud", "-c", help="Cloud provider"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate only, don't run"),
) -> None:
    """
    Run training with resilience features.

    Features:
    - Pre-flight validation before starting
    - Automatic checkpoint backup to cloud storage
    - Auto-resume on spot instance preemption
    - Disk cleanup to manage checkpoint space

    Example:
        prime-train run config.toml --backup --spot
    """
    from prime_train.validator import validate_config
    from prime_train.resilience import TrainingRunner

    # Always validate first
    console.print(f"[bold]Validating[/bold] {config_path}")
    results = validate_config(config_path)

    if results.has_errors:
        console.print("[red]Validation failed. Fix errors before running.[/red]")
        raise typer.Exit(1)

    if dry_run:
        console.print("[green]Dry run complete. Config is valid.[/green]")
        return

    runner = TrainingRunner(
        config_path=config_path,
        backup_enabled=backup,
        spot_enabled=spot,
        cloud_provider=cloud,
    )
    runner.run()


@app.command()
def find(
    model: str = typer.Option(..., "--model", "-m", help="Model name (HuggingFace)"),
    gpus: int = typer.Option(1, "--gpus", "-g", help="Number of GPUs"),
    budget: Optional[float] = typer.Option(None, "--budget", help="Max $/hr"),
    training_type: str = typer.Option("lora", "--type", "-t", help="Training type: lora, full"),
) -> None:
    """
    Find cheapest compatible GPUs across providers.

    Queries Prime Intellect, Vast.ai, and Lambda Labs to find
    the most cost-effective hardware for your model and config.

    Example:
        prime-train find --model Qwen/Qwen3-8B --gpus 2
    """
    from prime_train.cost import find_gpus, estimate_memory, format_gpu_table

    console.print(f"[bold]Finding GPUs for[/bold] {model}")

    memory_gb = estimate_memory(model, training_type=training_type)
    console.print(f"Estimated memory requirement: {memory_gb:.1f} GB")

    options = find_gpus(
        min_memory_gb=memory_gb,
        min_gpus=gpus,
        max_price=budget,
    )

    if not options:
        console.print("[yellow]No compatible GPUs found[/yellow]")
        raise typer.Exit(1)

    format_gpu_table(options, console)


@app.command()
def init(
    model: str = typer.Option(..., "--model", "-m", help="Model name (HuggingFace)"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Hardware preset"),
    gpus: int = typer.Option(1, "--gpus", "-g", help="Number of GPUs"),
    output: Path = typer.Option(Path("config.toml"), "--output", "-o", help="Output path"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Interactive mode"),
) -> None:
    """
    Generate training config interactively.

    Uses hardware presets and memory estimation to generate
    optimal config for your model and hardware.

    Example:
        prime-train init --model Qwen/Qwen3-8B --preset h100-80gb --gpus 2
    """
    from prime_train.config import generate_config

    console.print(f"[bold]Generating config for[/bold] {model}")

    config = generate_config(
        model=model,
        preset=preset,
        gpus=gpus,
        interactive=interactive,
    )

    config.save(output)
    console.print(f"[green]Config saved to[/green] {output}")


@app.command()
def profile(
    config_path: Path = typer.Argument(..., help="Path to config.toml"),
    steps: int = typer.Option(5, "--steps", "-n", help="Steps to profile"),
) -> None:
    """
    Profile training to detect bottlenecks.

    Runs a few training steps and measures:
    - Tool execution time
    - Policy inference time
    - Training step time
    - Communication time

    Identifies if training is latency-bound or throughput-bound
    and provides actionable recommendations.

    Example:
        prime-train profile config.toml --steps 5
    """
    from prime_train.profiler import profile_training, format_profile_results

    console.print(f"[bold]Profiling[/bold] {config_path} for {steps} steps")

    results = profile_training(config_path, steps=steps)
    format_profile_results(results, console)


@app.command()
def status() -> None:
    """
    Check training health via WandB.

    Queries WandB for the most recent training run and checks:
    - Is training progressing?
    - Any crashes or stalls?
    - GPU utilization

    Example:
        prime-train status
    """
    from prime_train.resilience import check_training_status

    status = check_training_status()

    if status.healthy:
        console.print(f"[green]Training healthy[/green] - Step {status.current_step}")
    else:
        console.print(f"[red]Training unhealthy[/red] - {status.reason}")


# Config subcommands
@config_app.command("save")
def config_save(
    name: str = typer.Option(..., "--name", "-n", help="Version name"),
    notes: str = typer.Option("", "--notes", help="Version notes"),
    config_path: Path = typer.Option(Path("config.toml"), "--config", "-c", help="Config to save"),
) -> None:
    """Save current config with version name and notes."""
    from prime_train.config import ConfigManager

    manager = ConfigManager()
    manager.save(config_path, name=name, notes=notes)
    console.print(f"[green]Saved config as[/green] {name}")


@config_app.command("list")
def config_list() -> None:
    """List all saved config versions."""
    from prime_train.config import ConfigManager

    manager = ConfigManager()
    versions = manager.list_versions()

    if not versions:
        console.print("[yellow]No saved configs[/yellow]")
        return

    table = Table(title="Saved Configs")
    table.add_column("Name")
    table.add_column("Created")
    table.add_column("Notes")

    for v in versions:
        table.add_row(v.name, v.created.strftime("%Y-%m-%d %H:%M"), v.notes)

    console.print(table)


@config_app.command("diff")
def config_diff(
    version1: str = typer.Argument(..., help="First version"),
    version2: str = typer.Argument(..., help="Second version"),
) -> None:
    """Compare two config versions."""
    from prime_train.config import ConfigManager

    manager = ConfigManager()
    diff = manager.diff(version1, version2)
    console.print(diff)


@config_app.command("restore")
def config_restore(
    name: str = typer.Argument(..., help="Version to restore"),
    output: Path = typer.Option(Path("config.toml"), "--output", "-o", help="Output path"),
) -> None:
    """Restore a saved config version."""
    from prime_train.config import ConfigManager

    manager = ConfigManager()
    manager.restore(name, output)
    console.print(f"[green]Restored {name} to[/green] {output}")


# Backup subcommands
@backup_app.command("configure")
def backup_configure() -> None:
    """Configure backup destination interactively."""
    from prime_train.resilience import configure_backup

    configure_backup(console)
    console.print("[green]Backup configured[/green]")


@backup_app.command("status")
def backup_status() -> None:
    """Check backup configuration and status."""
    from prime_train.resilience import get_backup_status

    status = get_backup_status()
    console.print(status)


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
) -> None:
    """prime-train: Production-grade training harness for prime-rl."""
    if version:
        from prime_train import __version__
        console.print(f"prime-train {__version__}")
        raise typer.Exit()


if __name__ == "__main__":
    app()
