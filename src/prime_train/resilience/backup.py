"""
Checkpoint backup management.

Syncs checkpoints to cloud storage (S3, B2, GCS) with compression
and retention management.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from rich.console import Console
from rich.prompt import Prompt, Confirm


StorageProvider = Literal["s3", "b2", "gcs", "local"]


@dataclass
class BackupConfig:
    """Configuration for checkpoint backup."""
    provider: StorageProvider
    bucket: str
    prefix: str = "checkpoints"
    sync_interval: int = 5  # steps
    keep_last: int = 3  # checkpoints
    compress: bool = True

    @classmethod
    def load(cls, path: Path | None = None) -> "BackupConfig | None":
        """Load config from file."""
        if path is None:
            path = Path.home() / ".prime-train" / "backup.yaml"

        if not path.exists():
            return None

        with open(path) as f:
            data = yaml.safe_load(f)
            return cls(**data)

    def save(self, path: Path | None = None) -> None:
        """Save config to file."""
        if path is None:
            path = Path.home() / ".prime-train" / "backup.yaml"

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump({
                "provider": self.provider,
                "bucket": self.bucket,
                "prefix": self.prefix,
                "sync_interval": self.sync_interval,
                "keep_last": self.keep_last,
                "compress": self.compress,
            }, f)


class BackupManager:
    """Manages checkpoint backup to cloud storage."""

    def __init__(self, config: BackupConfig):
        self.config = config
        self._synced_checkpoints: set[str] = set()

    def sync_checkpoint(self, checkpoint_path: Path, run_id: str) -> None:
        """
        Sync a checkpoint to cloud storage.

        Args:
            checkpoint_path: Path to the checkpoint directory
            run_id: Unique run identifier
        """
        if not checkpoint_path.exists():
            return

        # Compress if enabled
        if self.config.compress:
            archive_path = self._compress_checkpoint(checkpoint_path)
            source = archive_path
        else:
            source = checkpoint_path

        # Upload based on provider
        dest = f"{self.config.prefix}/{run_id}/{checkpoint_path.name}"

        if self.config.provider == "s3":
            self._upload_s3(source, dest)
        elif self.config.provider == "b2":
            self._upload_b2(source, dest)
        elif self.config.provider == "gcs":
            self._upload_gcs(source, dest)
        elif self.config.provider == "local":
            self._copy_local(source, dest)

        self._synced_checkpoints.add(str(checkpoint_path))

        # Clean up archive
        if self.config.compress and source != checkpoint_path:
            os.remove(source)

    def _compress_checkpoint(self, checkpoint_path: Path) -> Path:
        """Compress a checkpoint directory."""
        archive_path = checkpoint_path.with_suffix(".tar.gz")
        subprocess.run(
            ["tar", "-czf", str(archive_path), "-C", str(checkpoint_path.parent), checkpoint_path.name],
            check=True,
        )
        return archive_path

    def _upload_s3(self, source: Path, dest: str) -> None:
        """Upload to S3."""
        subprocess.run(
            ["aws", "s3", "cp", str(source), f"s3://{self.config.bucket}/{dest}"],
            check=True,
        )

    def _upload_b2(self, source: Path, dest: str) -> None:
        """Upload to Backblaze B2."""
        subprocess.run(
            ["b2", "file", "upload", self.config.bucket, str(source), dest],
            check=True,
        )

    def _upload_gcs(self, source: Path, dest: str) -> None:
        """Upload to Google Cloud Storage."""
        subprocess.run(
            ["gsutil", "cp", str(source), f"gs://{self.config.bucket}/{dest}"],
            check=True,
        )

    def _copy_local(self, source: Path, dest: str) -> None:
        """Copy to local directory."""
        dest_path = Path(self.config.bucket) / dest
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if source.is_dir():
            shutil.copytree(source, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source, dest_path)

    def cleanup_old_checkpoints(self, checkpoint_dir: Path) -> None:
        """Remove old checkpoints keeping only the last N."""
        checkpoints = sorted(
            checkpoint_dir.glob("step-*"),
            key=lambda p: int(p.name.split("-")[1]) if "-" in p.name else 0,
        )

        # Keep last N
        to_remove = checkpoints[:-self.config.keep_last] if len(checkpoints) > self.config.keep_last else []

        for ckpt in to_remove:
            if ckpt.is_dir():
                shutil.rmtree(ckpt)
            else:
                ckpt.unlink()

    def download_latest_checkpoint(self, run_id: str, dest_dir: Path) -> Path | None:
        """
        Download the latest checkpoint for a run.

        Returns:
            Path to downloaded checkpoint, or None if not found
        """
        # List checkpoints in cloud
        # This is provider-specific
        # For now, return None - implement per provider
        return None


def configure_backup(console: Console) -> BackupConfig:
    """Configure backup interactively."""
    provider = Prompt.ask(
        "Storage provider",
        choices=["s3", "b2", "gcs", "local"],
        default="s3",
    )

    bucket = Prompt.ask("Bucket name")

    sync_interval = int(Prompt.ask(
        "Sync interval (steps)",
        default="5",
    ))

    keep_last = int(Prompt.ask(
        "Keep last N checkpoints",
        default="3",
    ))

    compress = Confirm.ask("Compress checkpoints?", default=True)

    config = BackupConfig(
        provider=provider,
        bucket=bucket,
        sync_interval=sync_interval,
        keep_last=keep_last,
        compress=compress,
    )

    config.save()
    return config


def get_backup_status() -> str:
    """Get backup configuration status."""
    config = BackupConfig.load()

    if config is None:
        return "Backup not configured. Run `prime-train backup configure` to set up."

    return f"""Backup Configuration:
  Provider: {config.provider}
  Bucket: {config.bucket}
  Sync interval: {config.sync_interval} steps
  Keep last: {config.keep_last} checkpoints
  Compression: {"enabled" if config.compress else "disabled"}
"""
