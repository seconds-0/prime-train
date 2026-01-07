"""
Shared types for the validator module.

This module exists to avoid circular imports between core.py and gotchas.py.
"""

from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    """Severity level for validation results."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


@dataclass
class ValidationResult:
    """A single validation check result."""
    check: str
    severity: Severity
    message: str
    details: str | None = None
    fix: str | None = None
