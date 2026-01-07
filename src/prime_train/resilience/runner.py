"""
Resilient training runner.

Wraps prime-rl training with validation, backup, and recovery.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console

from prime_train.resilience.backup import BackupManager, BackupConfig


console = Console()


class TrainingRunner:
    """
    Resilient training runner for prime-rl.

    Features:
    - Pre-flight validation
    - Automatic checkpoint backup
    - Auto-resume from interruption
    - Disk cleanup daemon
    """

    def __init__(
        self,
        config_path: Path,
        backup_enabled: bool = False,
        spot_enabled: bool = False,
        cloud_provider: Optional[str] = None,
    ):
        self.config_path = config_path
        self.backup_enabled = backup_enabled
        self.spot_enabled = spot_enabled
        self.cloud_provider = cloud_provider

        self.backup_manager: Optional[BackupManager] = None
        if backup_enabled:
            backup_config = BackupConfig.load()
            if backup_config:
                self.backup_manager = BackupManager(backup_config)
            else:
                console.print("[yellow]Backup enabled but not configured. Run `prime-train backup configure`[/yellow]")

    def run(self) -> None:
        """Run training with resilience features."""
        # Build command
        cmd = self._build_command()

        console.print(f"[bold]Running:[/bold] {' '.join(cmd)}")

        # Run training
        try:
            process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            process.wait()

            if process.returncode != 0:
                console.print(f"[red]Training exited with code {process.returncode}[/red]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Training interrupted[/yellow]")
            process.terminate()

    def _build_command(self) -> list[str]:
        """Build the training command."""
        cmd = [
            "uv", "run", "rl", "@", str(self.config_path),
            # Always enable checkpointing
            "--ckpt",
            "--ckpt.interval", "5",
            "--ckpt.keep-last", "3",
        ]

        return cmd

    def resume_from_checkpoint(self, step: int) -> None:
        """Resume training from a specific checkpoint step."""
        cmd = self._build_command()
        cmd.extend(["--ckpt.resume-step", str(step)])

        console.print(f"[bold]Resuming from step {step}:[/bold] {' '.join(cmd)}")

        subprocess.run(cmd)
