"""Tests for the CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from prime_train.cli import app


runner = CliRunner()
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


class TestCLI:
    """Tests for CLI commands."""

    def test_help(self):
        """Help command should work."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "prime-train" in result.stdout

    def test_version(self):
        """Version flag should work."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout

    def test_validate_help(self):
        """Validate subcommand help should work."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Pre-flight validation" in result.stdout

    def test_validate_valid_config(self):
        """Validate should pass for valid config."""
        config_path = EXAMPLES_DIR / "basic-config.toml"
        result = runner.invoke(app, ["validate", str(config_path)])

        # Should succeed (exit code 0)
        assert result.exit_code == 0

    def test_validate_invalid_config(self):
        """Validate should fail for config with errors."""
        config_path = EXAMPLES_DIR / "gotcha-fsdp-lora.toml"
        result = runner.invoke(app, ["validate", str(config_path)])

        # Should fail (exit code 1)
        assert result.exit_code == 1

    def test_find_help(self):
        """Find subcommand help should work."""
        result = runner.invoke(app, ["find", "--help"])
        assert result.exit_code == 0
        assert "cheapest compatible GPUs" in result.stdout

    def test_config_list(self):
        """Config list should work (empty initially)."""
        result = runner.invoke(app, ["config", "list"])
        assert result.exit_code == 0


class TestConfigSubcommands:
    """Tests for config subcommands."""

    def test_config_help(self):
        """Config help should work."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "save" in result.stdout
        assert "list" in result.stdout
        assert "diff" in result.stdout


class TestBackupSubcommands:
    """Tests for backup subcommands."""

    def test_backup_help(self):
        """Backup help should work."""
        result = runner.invoke(app, ["backup", "--help"])
        assert result.exit_code == 0
        assert "configure" in result.stdout
        assert "status" in result.stdout

    def test_backup_status(self):
        """Backup status should work."""
        result = runner.invoke(app, ["backup", "status"])
        assert result.exit_code == 0
        # Should show "not configured" initially
        assert "not configured" in result.stdout.lower() or "provider" in result.stdout.lower()
