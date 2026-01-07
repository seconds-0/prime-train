# prime-train

> Production-grade training harness for [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) with validation, cost optimization, and resilience.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What is prime-train?

**prime-train** is a CLI wrapper around prime-rl that provides:

- **Pre-flight validation** - Catch config bugs before provisioning GPUs
- **Cost optimization** - Find cheapest compatible hardware across providers
- **Resilient training** - Checkpoint sync, auto-resume, spot instance recovery
- **Config management** - Version, diff, and restore experiment configs
- **Bottleneck detection** - Identify latency vs throughput issues automatically

It's designed to be **contributed back to Prime Intellect** as part of their training ecosystem.

## The Problem

Over 7 days of training with prime-rl, we encountered:

| Issue | Count | Hours Lost |
|-------|-------|------------|
| Config format bugs | 6 | 2 |
| Model compatibility failures | 3 | 3 |
| Wrong bottleneck (latency vs throughput) | 1 | 24 |
| Memory issues (FSDP+LoRA conflict) | 2 | 1 |
| Data loss on spot instances | 2 | 4 |
| **Total** | **14** | **34** |

**No existing tool** combines RL-aware validation, cost optimization, and auto-recovery.

## Quick Start

```bash
# Install
uv tool install prime-train

# Validate a config before training
prime-train validate config.toml

# Find cheapest compatible GPUs
prime-train find --model Qwen/Qwen3-8B --gpus 2

# Run training with resilience
prime-train run config.toml --backup --spot
```

## Features

### 1. Pre-flight Validation

```bash
$ prime-train validate config.toml

✓ Model "Qwen/Qwen3-8B" exists on HuggingFace
✓ Model is text-only (not VL)
✓ vLLM compatibility confirmed
✓ Config format valid (no deprecated sections)
✓ Memory estimate: 62GB (fits 2x H100 80GB)
✓ max_tokens (4096) + typical_input (1500) < max_model_len (8192)
✗ WARNING: executor_backend=prime may cause latency bottleneck
  → Consider executor_backend=local for 10-15x speedup
```

**Catches**:
- Model compatibility (VL models, vLLM bugs, missing on HuggingFace)
- Config schema errors (deprecated sections, forbidden params)
- Memory estimation failures
- Known gotchas (FSDP+LoRA conflict, seq_len mismatches)

### 2. Cost Optimization

```bash
$ prime-train find --model Qwen/Qwen3-8B --gpus 2

┌─────────────────────────────────────────────────────────────────────────────┐
│ Compatible GPU Options (sorted by $/hr)                                     │
├──────────────┬──────┬──────────┬──────────┬────────────┬───────────────────┤
│ Provider     │ Type │ GPUs     │ VRAM     │ Price/Hr   │ Estimated Time    │
├──────────────┼──────┼──────────┼──────────┼────────────┼───────────────────┤
│ Prime (spot) │ H100 │ 2        │ 160GB    │ $3.00      │ ~8 hrs → $24      │
│ Lambda       │ A100 │ 2        │ 160GB    │ $3.98      │ ~10 hrs → $40     │
│ Vast.ai      │ 4090 │ 4        │ 96GB     │ $2.40      │ ~16 hrs → $38     │
└──────────────┴──────┴──────────┴──────────┴────────────┴───────────────────┘

Memory requirement: 62GB (model) + 31GB (optimizer) = 93GB minimum
Recommendation: Prime (spot) 2x H100 - best $/completion
```

### 3. Resilient Training

```bash
$ prime-train run config.toml --backup --spot

[Provisioning] Selected: Prime Intellect us-east ($3.00/hr spot)
[Checkpoint] Syncing to s3://bucket/run-abc123/ every 5 steps
[Training] Step 1/100...
[Preemption] Spot instance interrupted at step 47
[Recovery] Resumed from step 45 checkpoint
[Training] Step 48/100...
```

**Features**:
- Checkpoint sync to S3/B2/GCS
- Auto-resume on spot preemption
- Disk cleanup daemon (47GB checkpoints!)
- Health monitoring via WandB

### 4. Config Management

```bash
# Save config with notes
$ prime-train config save --name "v3-local-executor" --notes "Fixed latency"

# List versions
$ prime-train config list
┌────────────────────┬────────────────┬──────────────────────────────────────┐
│ Name               │ Created        │ Notes                                │
├────────────────────┼────────────────┼──────────────────────────────────────┤
│ v3-local-executor  │ 2026-01-06     │ Fixed latency                        │
│ v2-throughput-opt  │ 2026-01-05     │ Reduced max_tokens (wrong approach)  │
└────────────────────┴────────────────┴──────────────────────────────────────┘

# Diff two versions
$ prime-train config diff v2 v3
```

### 5. Bottleneck Detection

```bash
$ prime-train profile config.toml --steps 5

┌─────────────────────────────────────────────────────────────────┐
│ Bottleneck Analysis                                             │
├───────────────────┬───────────┬─────────────────────────────────┤
│ Component         │ Time (%)  │ Status                          │
├───────────────────┼───────────┼─────────────────────────────────┤
│ Tool execution    │ 78%       │ ⚠ BOTTLENECK                    │
│ Policy inference  │ 12%       │ OK                              │
│ Training step     │ 8%        │ OK                              │
└───────────────────┴───────────┴─────────────────────────────────┘

DIAGNOSIS: Training is LATENCY-BOUND (not throughput-bound)
RECOMMENDATION: Switch executor_backend: "prime" → "local" (10-15x speedup)
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              prime-train CLI                                  │
│                    "One command" training with guardrails                     │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │  Pre-flight    │  │  Auto-config   │  │  Cost          │  │ Bottleneck │ │
│  │  Validator     │  │  Generator     │  │  Optimizer     │  │ Detector   │ │
│  └────────────────┘  └────────────────┘  └────────────────┘  └────────────┘ │
├──────────────────────────────────────────────────────────────────────────────┤
│                          Resilience Manager                                   │
│  Checkpoint Sync │ Auto-Resume │ Disk Cleanup │ Health Monitor               │
├──────────────────────────────────────────────────────────────────────────────┤
│                      External Integrations                                    │
│  Prime CLI (availability, pods) │ SkyPilot (multi-cloud) │ WandB (tracking)  │
├──────────────────────────────────────────────────────────────────────────────┤
│                           prime-rl (core)                                     │
│                  Orchestrator │ Trainer │ Inference                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Basic installation
uv tool install prime-train

# With S3 backup support
uv tool install "prime-train[s3]"

# With all storage backends
uv tool install "prime-train[all-storage]"

# With SkyPilot for multi-cloud
uv tool install "prime-train[skypilot]"
```

## Configuration

### Backup Configuration

```bash
$ prime-train backup configure

? Storage provider: S3
? Bucket name: my-training-checkpoints
? Sync interval (steps): 5
? Keep last N checkpoints: 3
? Compress checkpoints: Yes

Configuration saved to ~/.prime-train/backup.yaml
```

### Hardware Presets

prime-train includes optimized presets for common GPU configurations:

| GPU | VRAM | Preset |
|-----|------|--------|
| H100 80GB | 80GB | `h100-80gb` |
| A100 80GB | 80GB | `a100-80gb` |
| A100 40GB | 40GB | `a100-40gb` |
| RTX 4090 | 24GB | `rtx4090` |
| RTX 5090 | 32GB | `rtx5090` |

```bash
# Generate config with preset
prime-train init --model Qwen/Qwen3-8B --preset h100-80gb --gpus 2
```

## CLI Reference

```
prime-train
├── validate <config>       # Pre-flight validation
├── run <config>            # Run training with resilience
│   ├── --backup            # Enable checkpoint backup
│   ├── --spot              # Use spot instances
│   └── --cloud <provider>  # Specific cloud (prime, lambda, vast)
├── find                    # Find cheapest compatible GPUs
│   ├── --model <name>      # Model to train
│   ├── --gpus <n>          # Number of GPUs
│   └── --budget <$/hr>     # Maximum hourly budget
├── init                    # Generate config interactively
│   ├── --model <name>      # Target model
│   ├── --preset <name>     # Hardware preset
│   └── --gpus <n>          # Number of GPUs
├── config                  # Config management
│   ├── save                # Save current config
│   ├── list                # List saved configs
│   ├── diff <v1> <v2>      # Compare configs
│   └── restore <name>      # Restore a config
├── profile <config>        # Bottleneck detection
│   └── --steps <n>         # Steps to profile
├── backup                  # Backup configuration
│   ├── configure           # Setup backup destination
│   └── status              # Check backup status
└── status                  # Check training health
```

## Development Status

| Phase | Status | Features |
|-------|--------|----------|
| 1. Foundation | **In Progress** | Package structure, validator, basic CLI |
| 2. Cost Intelligence | Planned | Memory estimator, cost optimizer, presets |
| 3. Resilience | Planned | Checkpoint sync, auto-resume, disk cleanup |
| 4. Config Management | Planned | Versioning, diff, restore |
| 5. Diagnostics | Planned | Bottleneck detector, health monitor |
| 6. Multi-Cloud | Future | SkyPilot integration, one-command training |

## Contributing

We welcome contributions! This project is designed to potentially be contributed back to Prime Intellect.

```bash
# Clone and setup
git clone https://github.com/seconds-0/prime-train.git
cd prime-train
uv sync --all-extras

# Run tests
pytest

# Run linting
ruff check src tests
mypy src
```

## Known Gotcha Database

prime-train maintains a database of known issues:

| Issue | Detection | Recommendation |
|-------|-----------|----------------|
| FSDP + LoRA | Both enabled | Use activation checkpointing instead |
| VL model | Model name contains "vl" | Switch to text-only variant |
| Deprecated LoRA section | `[trainer.model.experimental.lora]` | Use `[trainer.model.lora]` |
| seq_len mismatch | trainer < orchestrator | Increase trainer.seq_len |
| vLLM V1 issues | Any vLLM usage | Set `VLLM_USE_V1=0` |

## Related Projects

- [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) - Async RL training at scale
- [prime](https://github.com/PrimeIntellect-ai/prime) - Official Prime Intellect CLI
- [verifiers](https://github.com/PrimeIntellect-ai/verifiers) - RL environments library
- [SkyPilot](https://github.com/skypilot-org/skypilot) - Multi-cloud orchestration

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Prime Intellect](https://primeintellect.ai) for building prime-rl
- The insights in this tool came from 34+ hours of debugging training issues
