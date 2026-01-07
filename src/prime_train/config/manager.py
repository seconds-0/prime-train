"""
Config version management.

Stores config history in SQLite and provides diff/restore functionality.
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import tomli
import tomli_w


@dataclass
class ConfigVersion:
    """A saved config version."""
    name: str
    created: datetime
    notes: str
    config_hash: str
    wandb_run_id: Optional[str] = None


class ConfigManager:
    """
    Manages config version history.

    Stores configs in SQLite database at ~/.prime-train/configs.db
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path.home() / ".prime-train" / "configs.db"

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS config_versions (
                    name TEXT PRIMARY KEY,
                    created TEXT NOT NULL,
                    notes TEXT,
                    config_hash TEXT NOT NULL,
                    config_content TEXT NOT NULL,
                    wandb_run_id TEXT
                )
            """)
            conn.commit()

    def save(
        self,
        config_path: Path,
        name: str,
        notes: str = "",
        wandb_run_id: Optional[str] = None,
    ) -> ConfigVersion:
        """
        Save a config version.

        Args:
            config_path: Path to config.toml
            name: Version name
            notes: Optional notes
            wandb_run_id: Optional WandB run ID to link

        Returns:
            ConfigVersion object
        """
        # Read config
        with open(config_path, "rb") as f:
            config_content = f.read()

        # Calculate hash
        config_hash = hashlib.sha256(config_content).hexdigest()[:12]

        # Save to DB
        created = datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO config_versions
                (name, created, notes, config_hash, config_content, wandb_run_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, created.isoformat(), notes, config_hash, config_content.decode(), wandb_run_id),
            )
            conn.commit()

        return ConfigVersion(
            name=name,
            created=created,
            notes=notes,
            config_hash=config_hash,
            wandb_run_id=wandb_run_id,
        )

    def list_versions(self) -> list[ConfigVersion]:
        """List all saved config versions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT name, created, notes, config_hash, wandb_run_id FROM config_versions ORDER BY created DESC"
            )
            rows = cursor.fetchall()

        return [
            ConfigVersion(
                name=row[0],
                created=datetime.fromisoformat(row[1]),
                notes=row[2] or "",
                config_hash=row[3],
                wandb_run_id=row[4],
            )
            for row in rows
        ]

    def get_config_content(self, name: str) -> str | None:
        """Get the content of a saved config."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT config_content FROM config_versions WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()

        return row[0] if row else None

    def restore(self, name: str, output_path: Path) -> None:
        """
        Restore a saved config version.

        Args:
            name: Version name to restore
            output_path: Path to write the config
        """
        content = self.get_config_content(name)
        if content is None:
            raise ValueError(f"Config version '{name}' not found")

        with open(output_path, "w") as f:
            f.write(content)

    def diff(self, name1: str, name2: str) -> str:
        """
        Compare two config versions.

        Args:
            name1: First version name
            name2: Second version name

        Returns:
            Formatted diff string
        """
        content1 = self.get_config_content(name1)
        content2 = self.get_config_content(name2)

        if content1 is None:
            return f"Config version '{name1}' not found"
        if content2 is None:
            return f"Config version '{name2}' not found"

        # Parse as TOML
        config1 = tomli.loads(content1)
        config2 = tomli.loads(content2)

        # Find differences
        diff_lines = []
        diff_lines.append(f"Comparing {name1} vs {name2}:")
        diff_lines.append("")

        all_keys = set(self._flatten_dict(config1).keys()) | set(self._flatten_dict(config2).keys())

        flat1 = self._flatten_dict(config1)
        flat2 = self._flatten_dict(config2)

        for key in sorted(all_keys):
            val1 = flat1.get(key)
            val2 = flat2.get(key)

            if val1 is None:
                diff_lines.append(f"  + {key}: {val2}")
            elif val2 is None:
                diff_lines.append(f"  - {key}: {val1}")
            elif val1 != val2:
                diff_lines.append(f"  ~ {key}: {val1} → {val2}")

        if len(diff_lines) == 2:
            diff_lines.append("  (no differences)")

        return "\n".join(diff_lines)

    def _flatten_dict(self, d: dict, prefix: str = "") -> dict:
        """Flatten a nested dict into dot-separated keys."""
        result = {}
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(self._flatten_dict(value, full_key))
            else:
                result[full_key] = value
        return result

    def delete(self, name: str) -> bool:
        """Delete a config version."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM config_versions WHERE name = ?",
                (name,),
            )
            conn.commit()
            return cursor.rowcount > 0
