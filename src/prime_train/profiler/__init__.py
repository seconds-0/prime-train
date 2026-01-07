"""
Profiler module for prime-train.

Detects training bottlenecks (latency vs throughput).
"""

from prime_train.profiler.detector import (
    profile_training,
    ProfileResults,
    format_profile_results,
)

__all__ = [
    "profile_training",
    "ProfileResults",
    "format_profile_results",
]
