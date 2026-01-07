"""
Config management module for prime-train.

Provides versioning, diff, and generation of training configs.
"""

from prime_train.config.manager import ConfigManager, ConfigVersion
from prime_train.config.generator import generate_config, GeneratedConfig

__all__ = [
    "ConfigManager",
    "ConfigVersion",
    "generate_config",
    "GeneratedConfig",
]
