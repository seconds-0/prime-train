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
    # Disk management
    max_disk_gb: float | None = None  # Hard limit on disk usage (auto-detected if None)
    safety_buffer_gb: float = 10.0  # Reserved space for logs, temp files
    local_keep: int = 1  # Always keep N checkpoints locally for fast resume
    delete_after_upload: bool = True  # Delete local copy after confirmed upload

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

        data = {
            "provider": self.provider,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "sync_interval": self.sync_interval,
            "keep_last": self.keep_last,
            "compress": self.compress,
            "safety_buffer_gb": self.safety_buffer_gb,
            "local_keep": self.local_keep,
            "delete_after_upload": self.delete_after_upload,
        }
        # Only include max_disk_gb if explicitly set
        if self.max_disk_gb is not None:
            data["max_disk_gb"] = self.max_disk_gb

        with open(path, "w") as f:
            yaml.dump(data, f)


class BackupManager:
    """Manages checkpoint backup to cloud storage."""

    def __init__(self, config: BackupConfig):
        self.config = config
        self._synced_checkpoints: set[str] = set()
        self._upload_confirmed: set[str] = set()  # Checkpoints confirmed uploaded

    def sync_checkpoint(self, checkpoint_path: Path, run_id: str) -> bool:
        """
        Sync a checkpoint to cloud storage.

        Args:
            checkpoint_path: Path to the checkpoint directory
            run_id: Unique run identifier

        Returns:
            True if upload succeeded, False otherwise.
        """
        if not checkpoint_path.exists():
            return False

        # Compress if enabled
        archive_path = None
        if self.config.compress:
            archive_path = self._compress_checkpoint(checkpoint_path)
            source = archive_path
        else:
            source = checkpoint_path

        # Upload based on provider
        dest = f"{self.config.prefix}/{run_id}/{checkpoint_path.name}"
        if self.config.compress:
            dest += ".tar.gz"

        try:
            if self.config.provider == "s3":
                self._upload_s3(source, dest)
            elif self.config.provider == "b2":
                self._upload_b2(source, dest)
            elif self.config.provider == "gcs":
                self._upload_gcs(source, dest)
            elif self.config.provider == "local":
                self._copy_local(source, dest)

            # Mark as successfully uploaded
            self._synced_checkpoints.add(str(checkpoint_path))
            self._upload_confirmed.add(str(checkpoint_path))

            return True

        except subprocess.CalledProcessError:
            # Upload failed
            return False

        finally:
            # Clean up archive regardless of success
            if archive_path and archive_path.exists():
                os.remove(archive_path)

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

    def cleanup_old_checkpoints(
        self,
        checkpoint_dir: Path,
        run_id: str | None = None,
        disk_aware: bool = True,
    ) -> list[Path]:
        """
        Remove old checkpoints keeping only the last N locally.

        If external backup is configured and delete_after_upload is True,
        will offload and delete checkpoints that exceed the local_keep limit.

        Args:
            checkpoint_dir: Directory containing checkpoints.
            run_id: Run ID for uploading (required if offloading).
            disk_aware: If True, also enforce disk budget limits.

        Returns:
            List of paths that were deleted.
        """
        checkpoints = sorted(
            checkpoint_dir.glob("step-*"),
            key=lambda p: int(p.name.split("-")[1]) if "-" in p.name else 0,
        )

        deleted = []

        # Determine how many to keep locally
        # Use local_keep (for external backup) or keep_last (for local-only)
        if self.config.provider != "local" and self.config.delete_after_upload:
            # External backup configured - keep fewer locally
            local_keep = self.config.local_keep
        else:
            # Local-only - use keep_last
            local_keep = self.config.keep_last

        # Additional disk-aware check
        if disk_aware:
            from prime_train.resilience.disk import get_disk_budget, get_available_disk_gb

            # Get current checkpoint size (estimate from first checkpoint if exists)
            checkpoint_size_gb = 0.0
            if checkpoints:
                from prime_train.resilience.disk import get_checkpoint_size_gb
                checkpoint_size_gb = get_checkpoint_size_gb(checkpoints[0])

            if checkpoint_size_gb > 0:
                max_disk_gb = self.config.max_disk_gb
                if max_disk_gb is None:
                    # Auto-detect based on current available space
                    available_gb = get_available_disk_gb(checkpoint_dir)
                    # Use currently available + currently used by checkpoints
                    currently_used = checkpoint_size_gb * len(checkpoints)
                    max_disk_gb = available_gb + currently_used

                disk_budget = get_disk_budget(
                    checkpoint_dir,
                    checkpoint_size_gb,
                    self.config.safety_buffer_gb,
                )
                # Can't keep more than disk budget allows
                local_keep = min(local_keep, disk_budget)

        # Ensure at least 1 checkpoint locally
        local_keep = max(1, local_keep)

        # Determine which checkpoints to remove
        to_remove = checkpoints[:-local_keep] if len(checkpoints) > local_keep else []

        for ckpt in to_remove:
            ckpt_path_str = str(ckpt)

            # If external backup is configured, try to upload first
            if self.config.provider != "local" and run_id and self.config.delete_after_upload:
                # Only delete if upload confirmed or already uploaded
                if ckpt_path_str not in self._upload_confirmed:
                    # Try to upload before deleting
                    success = self.sync_checkpoint(ckpt, run_id)
                    if not success:
                        # Upload failed - don't delete this checkpoint
                        continue

            # Safe to delete (either uploaded or no external backup)
            if ckpt.is_dir():
                shutil.rmtree(ckpt)
            else:
                ckpt.unlink()
            deleted.append(ckpt)

        return deleted

    def is_checkpoint_backed_up(self, checkpoint_path: Path) -> bool:
        """Check if a checkpoint has been backed up to external storage."""
        return str(checkpoint_path) in self._upload_confirmed

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
