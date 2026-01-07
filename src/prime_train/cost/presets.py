"""
Hardware presets for different GPU configurations.

These presets contain optimized settings learned from our training experiments.
"""

from dataclasses import dataclass


@dataclass
class HardwarePreset:
    """Optimized settings for a specific hardware configuration."""
    name: str
    gpu_type: str
    vram_gb: float
    batch_size: int
    max_tokens: int
    gpu_memory_utilization: float
    rollouts_per_example: int
    activation_checkpointing: bool
    notes: str


# Presets learned from our training experiments
HARDWARE_PRESETS: dict[str, HardwarePreset] = {
    "h100-80gb": HardwarePreset(
        name="h100-80gb",
        gpu_type="H100 80GB",
        vram_gb=80,
        batch_size=128,
        max_tokens=4096,
        gpu_memory_utilization=0.90,
        rollouts_per_example=8,
        activation_checkpointing=False,
        notes="Optimal for large models with LoRA",
    ),
    "a100-80gb": HardwarePreset(
        name="a100-80gb",
        gpu_type="A100 80GB",
        vram_gb=80,
        batch_size=96,
        max_tokens=4096,
        gpu_memory_utilization=0.85,
        rollouts_per_example=8,
        activation_checkpointing=False,
        notes="Good for most models, slightly slower than H100",
    ),
    "a100-40gb": HardwarePreset(
        name="a100-40gb",
        gpu_type="A100 40GB",
        vram_gb=40,
        batch_size=48,
        max_tokens=2048,
        gpu_memory_utilization=0.85,
        rollouts_per_example=4,
        activation_checkpointing=True,
        notes="Requires activation checkpointing for 7B+ models",
    ),
    "rtx4090": HardwarePreset(
        name="rtx4090",
        gpu_type="RTX 4090",
        vram_gb=24,
        batch_size=32,
        max_tokens=2048,
        gpu_memory_utilization=0.45,  # Share with trainer
        rollouts_per_example=4,
        activation_checkpointing=True,
        notes="Consumer GPU - requires careful memory management",
    ),
    "rtx5090": HardwarePreset(
        name="rtx5090",
        gpu_type="RTX 5090",
        vram_gb=32,
        batch_size=48,
        max_tokens=2048,
        gpu_memory_utilization=0.80,
        rollouts_per_example=4,
        activation_checkpointing=True,
        notes="Next-gen consumer GPU with more headroom",
    ),
    "l40s": HardwarePreset(
        name="l40s",
        gpu_type="L40S",
        vram_gb=48,
        batch_size=64,
        max_tokens=4096,
        gpu_memory_utilization=0.85,
        rollouts_per_example=6,
        activation_checkpointing=False,
        notes="Good balance of memory and compute",
    ),
}


def get_preset(name: str) -> HardwarePreset | None:
    """Get a hardware preset by name."""
    return HARDWARE_PRESETS.get(name.lower())


def list_presets() -> list[str]:
    """List all available preset names."""
    return list(HARDWARE_PRESETS.keys())
