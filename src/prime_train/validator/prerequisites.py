"""
Pre-flight validation for training prerequisites.

Checks system requirements before training starts:
- ulimit (file descriptors)
- CUDA availability
- GPU health
- Disk space for checkpoints
"""

import os
import subprocess
from pathlib import Path
from typing import Any

from prime_train.validator.types import ValidationResult, Severity


def check_ulimit() -> list[ValidationResult]:
    """
    Check ulimit -n (max open files) is sufficient.

    Training can fail at step 15+ with "too many open files"
    if ulimit is too low.

    Returns:
        List of validation results.
    """
    results = []

    try:
        # Get current ulimit
        result = subprocess.run(
            ["bash", "-c", "ulimit -n"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        current_limit = int(result.stdout.strip())

        min_required = 65536
        if current_limit < min_required:
            results.append(ValidationResult(
                check="ulimit-files",
                severity=Severity.ERROR,
                message=f"File descriptor limit too low ({current_limit})",
                details=f"ulimit -n is {current_limit}, need at least {min_required}",
                fix=f"Run: ulimit -n {min_required}",
            ))
        else:
            results.append(ValidationResult(
                check="ulimit-files",
                severity=Severity.SUCCESS,
                message=f"File descriptor limit OK ({current_limit})",
                details=None,
                fix=None,
            ))

    except (subprocess.SubprocessError, ValueError):
        results.append(ValidationResult(
            check="ulimit-files",
            severity=Severity.WARNING,
            message="Could not check ulimit",
            details="Unable to determine file descriptor limit",
            fix="Verify ulimit -n is at least 65536",
        ))

    return results


def check_cuda_available() -> list[ValidationResult]:
    """
    Check if CUDA is available and functional.

    Returns:
        List of validation results.
    """
    results = []

    try:
        # Try to import torch and check CUDA
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]

            results.append(ValidationResult(
                check="cuda-available",
                severity=Severity.SUCCESS,
                message=f"CUDA available ({gpu_count} GPU{'s' if gpu_count > 1 else ''})",
                details=", ".join(gpu_names),
                fix=None,
            ))
        else:
            results.append(ValidationResult(
                check="cuda-available",
                severity=Severity.ERROR,
                message="CUDA not available",
                details="torch.cuda.is_available() returned False",
                fix="Verify CUDA drivers are installed and GPU is accessible",
            ))

    except ImportError:
        results.append(ValidationResult(
            check="cuda-available",
            severity=Severity.WARNING,
            message="Could not check CUDA",
            details="PyTorch not installed",
            fix="Install PyTorch to enable CUDA checks",
        ))
    except Exception as e:
        results.append(ValidationResult(
            check="cuda-available",
            severity=Severity.ERROR,
            message="CUDA check failed",
            details=str(e),
            fix="Check CUDA installation and driver status",
        ))

    return results


def check_gpu_health() -> list[ValidationResult]:
    """
    Check GPU health via nvidia-smi.

    Detects "broken GPU" state: device visible but 0 memory, 0 utilization.

    Returns:
        List of validation results.
    """
    results = []

    try:
        # Query GPU memory and utilization
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            results.append(ValidationResult(
                check="gpu-health",
                severity=Severity.WARNING,
                message="Could not query GPU status",
                details=result.stderr,
                fix="Check nvidia-smi is working",
            ))
            return results

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue

            gpu_idx, mem_used, util, mem_total = parts
            mem_used = int(mem_used) if mem_used.isdigit() else 0
            util = int(util) if util.isdigit() else 0
            mem_total = int(mem_total) if mem_total.isdigit() else 0

            # Detect "broken GPU" state
            if mem_total > 0 and mem_used == 0 and util == 0:
                results.append(ValidationResult(
                    check=f"gpu-health-{gpu_idx}",
                    severity=Severity.WARNING,
                    message=f"GPU {gpu_idx} may be in broken state",
                    details="GPU shows 0 memory used, 0% utilization",
                    fix="Try: nvidia-smi -r (requires root) or restart the machine",
                ))
            else:
                results.append(ValidationResult(
                    check=f"gpu-health-{gpu_idx}",
                    severity=Severity.SUCCESS,
                    message=f"GPU {gpu_idx} healthy",
                    details=f"{mem_used}/{mem_total} MiB used, {util}% utilization",
                    fix=None,
                ))

    except FileNotFoundError:
        results.append(ValidationResult(
            check="gpu-health",
            severity=Severity.WARNING,
            message="nvidia-smi not found",
            details="Cannot check GPU health",
            fix="Install NVIDIA drivers",
        ))
    except subprocess.TimeoutExpired:
        results.append(ValidationResult(
            check="gpu-health",
            severity=Severity.WARNING,
            message="GPU health check timed out",
            details="nvidia-smi did not respond",
            fix="Check GPU status manually with nvidia-smi",
        ))

    return results


def check_vllm_environment() -> list[ValidationResult]:
    """
    Check vLLM environment variables are set correctly.

    Returns:
        List of validation results.
    """
    results = []

    # Check VLLM_USE_V1
    vllm_v1 = os.environ.get("VLLM_USE_V1", "")
    if vllm_v1 != "0":
        results.append(ValidationResult(
            check="vllm-v1-disabled",
            severity=Severity.WARNING,
            message="VLLM_USE_V1 not disabled",
            details="vLLM V1 engine can cause CUDA segfaults with child processes",
            fix="Set: export VLLM_USE_V1=0",
        ))
    else:
        results.append(ValidationResult(
            check="vllm-v1-disabled",
            severity=Severity.SUCCESS,
            message="VLLM_USE_V1=0 set",
            details=None,
            fix=None,
        ))

    # Check VLLM_WORKER_MULTIPROC_METHOD
    multiproc = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "")
    if multiproc != "spawn":
        results.append(ValidationResult(
            check="vllm-multiproc",
            severity=Severity.INFO,
            message="VLLM_WORKER_MULTIPROC_METHOD not set to spawn",
            details="Setting to spawn prevents CUDA context inheritance issues",
            fix="Set: export VLLM_WORKER_MULTIPROC_METHOD=spawn",
        ))

    return results


def check_disk_budget(
    config: dict[str, Any],
    checkpoint_dir: Path | str = "/opt/run/checkpoints/",
) -> list[ValidationResult]:
    """
    Check disk budget for checkpoints.

    Args:
        config: Training config dict.
        checkpoint_dir: Where checkpoints will be stored.

    Returns:
        List of validation results.
    """
    from prime_train.resilience.checkpoint_budget import validate_checkpoint_budget

    return validate_checkpoint_budget(config, checkpoint_dir)


def run_all_prerequisites(
    config: dict[str, Any] | None = None,
    checkpoint_dir: Path | str = "/opt/run/checkpoints/",
) -> list[ValidationResult]:
    """
    Run all pre-flight prerequisite checks.

    Args:
        config: Training config (optional, for disk budget check).
        checkpoint_dir: Where checkpoints will be stored.

    Returns:
        List of all validation results.
    """
    results = []

    results.extend(check_ulimit())
    results.extend(check_cuda_available())
    results.extend(check_gpu_health())
    results.extend(check_vllm_environment())

    if config is not None:
        results.extend(check_disk_budget(config, checkpoint_dir))

    return results
