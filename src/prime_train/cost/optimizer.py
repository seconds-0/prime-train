"""
GPU cost optimizer.

Queries multiple providers to find the cheapest compatible hardware.
"""

import subprocess
import json
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.table import Table


@dataclass
class GPUOption:
    """A GPU option from a provider."""
    provider: str
    gpu_type: str
    gpu_count: int
    vram_gb: float
    price_per_hour: float
    location: str
    is_spot: bool = False
    estimated_hours: float | None = None

    @property
    def total_vram_gb(self) -> float:
        return self.vram_gb * self.gpu_count

    @property
    def estimated_cost(self) -> float | None:
        if self.estimated_hours:
            return self.price_per_hour * self.estimated_hours
        return None


def find_gpus(
    min_memory_gb: float,
    min_gpus: int = 1,
    max_price: float | None = None,
    providers: list[str] | None = None,
) -> list[GPUOption]:
    """
    Find GPUs that meet memory requirements across providers.

    Queries:
    - Prime Intellect via `prime availability list`
    - Vast.ai via `vastai search offers` (if installed)

    Args:
        min_memory_gb: Minimum total VRAM required
        min_gpus: Minimum number of GPUs
        max_price: Maximum price per hour (optional)
        providers: List of providers to query (default: all)

    Returns:
        List of GPUOption sorted by price
    """
    options = []

    # Default to all providers
    if providers is None:
        providers = ["prime", "vastai"]

    # Query Prime Intellect
    if "prime" in providers:
        prime_options = _query_prime_availability(min_gpus)
        options.extend(prime_options)

    # Query Vast.ai
    if "vastai" in providers:
        vast_options = _query_vastai(min_gpus)
        options.extend(vast_options)

    # Filter by memory
    options = [o for o in options if o.total_vram_gb >= min_memory_gb]

    # Filter by price
    if max_price:
        options = [o for o in options if o.price_per_hour <= max_price]

    # Sort by price
    options.sort(key=lambda o: o.price_per_hour)

    return options


def _query_prime_availability(min_gpus: int) -> list[GPUOption]:
    """Query Prime Intellect availability."""
    options = []

    try:
        result = subprocess.run(
            ["prime", "availability", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                for item in data:
                    if item.get("gpus", 0) >= min_gpus:
                        options.append(GPUOption(
                            provider="Prime",
                            gpu_type=item.get("gpu_type", "Unknown"),
                            gpu_count=item.get("gpus", 1),
                            vram_gb=_get_gpu_vram(item.get("gpu_type", "")),
                            price_per_hour=item.get("price_per_hour", 0),
                            location=item.get("location", "Unknown"),
                            is_spot=item.get("is_spot", False),
                        ))
            except json.JSONDecodeError:
                pass  # CLI doesn't support --json yet, parse table output

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # Prime CLI not installed or timed out

    return options


def _query_vastai(min_gpus: int) -> list[GPUOption]:
    """Query Vast.ai offers."""
    options = []

    try:
        result = subprocess.run(
            [
                "vastai", "search", "offers",
                "--raw",
                "--order", "dph_total",
                f"num_gpus>={min_gpus}",
                "rentable=true",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                for item in data[:20]:  # Limit to top 20
                    options.append(GPUOption(
                        provider="Vast.ai",
                        gpu_type=item.get("gpu_name", "Unknown"),
                        gpu_count=item.get("num_gpus", 1),
                        vram_gb=item.get("gpu_ram", 0) / 1024,  # Convert MB to GB
                        price_per_hour=item.get("dph_total", 0),
                        location=item.get("geolocation", "Unknown"),
                        is_spot=True,  # Vast is always spot-like
                    ))
            except json.JSONDecodeError:
                pass

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # Vast CLI not installed or timed out

    return options


def _get_gpu_vram(gpu_type: str) -> float:
    """Get VRAM in GB for a GPU type."""
    vram_map = {
        "H100_80GB": 80,
        "H100_SXM": 80,
        "A100_80GB": 80,
        "A100_40GB": 40,
        "A100": 40,
        "RTX_4090": 24,
        "RTX_5090": 32,
        "A6000": 48,
        "L40S": 48,
        "L4": 24,
    }

    for key, vram in vram_map.items():
        if key.lower() in gpu_type.lower():
            return vram

    return 24  # Default guess


def format_gpu_table(options: list[GPUOption], console: Console) -> None:
    """Format GPU options as a rich table."""
    table = Table(title="Compatible GPU Options (sorted by $/hr)")

    table.add_column("Provider", style="cyan")
    table.add_column("Type")
    table.add_column("GPUs", justify="right")
    table.add_column("VRAM", justify="right")
    table.add_column("Price/Hr", justify="right", style="green")
    table.add_column("Location")
    table.add_column("Est. Cost", justify="right")

    for opt in options[:10]:  # Top 10
        est_cost = f"${opt.estimated_cost:.0f}" if opt.estimated_cost else "-"
        spot_indicator = " (spot)" if opt.is_spot else ""

        table.add_row(
            f"{opt.provider}{spot_indicator}",
            opt.gpu_type,
            str(opt.gpu_count),
            f"{opt.total_vram_gb:.0f}GB",
            f"${opt.price_per_hour:.2f}",
            opt.location[:15],
            est_cost,
        )

    console.print(table)
