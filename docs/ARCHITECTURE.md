# prime-train Architecture

## Overview

prime-train is a CLI wrapper around prime-rl that provides production guardrails for RL training.

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              prime-train CLI                                  │
│                    "One command" training with guardrails                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────┐ │
│  │  Pre-flight    │  │  Auto-config   │  │  Cost          │  │ Bottleneck │ │
│  │  Validator     │  │  Generator     │  │  Optimizer     │  │ Detector   │ │
│  │                │  │                │  │                │  │            │ │
│  │ • Schema check │  │ • Memory est.  │  │ • Multi-cloud  │  │ • Profiler │ │
│  │ • Model compat │  │ • HW presets   │  │ • Price query  │  │ • Latency  │ │
│  │ • Gotcha DB    │  │ • RL-aware     │  │ • Total cost   │  │ • Rx fixes │ │
│  └────────────────┘  └────────────────┘  └────────────────┘  └────────────┘ │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                          Resilience Manager                                   │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐ │
│  │ Checkpoint    │  │ Auto-Resume   │  │ Disk Cleanup  │  │ Health        │ │
│  │ Sync          │  │ Logic         │  │ Daemon        │  │ Monitor       │ │
│  │ S3/B2/GCS     │  │ Step detect   │  │ Keep last N   │  │ WandB query   │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘ │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                        Config Management                                      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                    │
│  │ Versioning    │  │ Diff Tool     │  │ Run History   │                    │
│  │ SQLite DB     │  │ Config delta  │  │ WandB links   │                    │
│  └───────────────┘  └───────────────┘  └───────────────┘                    │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                      External Integrations                                    │
│                                                                              │
│  ┌────────────────────────┐  ┌────────────────────────┐                     │
│  │   Prime CLI (native)   │  │   SkyPilot (optional)  │                     │
│  │ • prime availability   │  │ • Multi-cloud failover │                     │
│  │ • prime pods           │  │ • Spot recovery        │                     │
│  │ • prime env            │  │ • Cost optimization    │                     │
│  └────────────────────────┘  └────────────────────────┘                     │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                           prime-rl (core)                                     │
│                  Orchestrator │ Trainer │ Inference                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Pre-flight Validator

**Purpose**: Catch config bugs before provisioning GPUs

**Checks**:
| Check | Severity | Detection |
|-------|----------|-----------|
| Model exists on HuggingFace | ERROR | HF API query |
| Model is text-only (not VL) | ERROR | Name pattern matching |
| Config schema valid | ERROR | JSON Schema validation |
| Memory fits hardware | ERROR | Model size × multiplier |
| FSDP + LoRA conflict | ERROR | Both enabled in config |
| Deprecated LoRA section | ERROR | `[trainer.model.experimental.lora]` present |
| seq_len mismatch | ERROR | trainer < orchestrator |
| Forbidden params | ERROR | `top_p`, `mask_truncated_completions` present |
| Prime executor latency | WARNING | `executor_backend=prime` with tool-calling |
| Checkpointing disabled | WARNING | No `--ckpt` flags |

**Data Flow**:
```
config.toml → TOML Parser → Schema Validation → Model Check → Memory Check → Gotcha Check → Results
```

### 2. Cost Optimizer

**Purpose**: Find cheapest compatible GPUs across providers

**Data Sources**:
- Prime Intellect: `prime availability list`
- Vast.ai: `vastai search offers`
- Lambda Labs: API (future)

**Memory Estimation Formula**:
```
VRAM = model_params × bytes_per_param × multiplier

multiplier:
- Inference only: 1.2x
- Training (AdamW): 4x (model + gradients + 2 optimizer states)
- Training (LoRA): 1.5x (frozen base + adapters)
- KV cache: +seq_len × hidden × layers × 2 × batch
```

**Hardware Presets** (from real experiments):
| Preset | GPU | VRAM | batch_size | max_tokens | Notes |
|--------|-----|------|------------|------------|-------|
| h100-80gb | H100 | 80GB | 128 | 4096 | Optimal for large LoRA |
| a100-80gb | A100 | 80GB | 96 | 4096 | Good balance |
| rtx4090 | 4090 | 24GB | 32 | 2048 | Requires AC, low mem util |

### 3. Resilience Manager

**Purpose**: Handle spot instance interruptions gracefully

**Components**:

#### Checkpoint Sync
- Supports S3, B2, GCS, local
- Configurable sync interval (default: 5 steps)
- Optional compression (40% savings)
- Async upload to avoid blocking training

#### Auto-Resume
1. Detect interrupted run (via WandB or checkpoint presence)
2. Download latest checkpoint from cloud
3. Resume with `--ckpt.resume-step`

#### Disk Cleanup
- Monitor disk usage
- Delete old checkpoints (keep last N)
- 47GB per checkpoint typical for 8B models

### 4. Config Manager

**Purpose**: Track config versions and enable reproducibility

**Storage**: SQLite at `~/.prime-train/configs.db`

**Schema**:
```sql
CREATE TABLE config_versions (
    name TEXT PRIMARY KEY,
    created TEXT NOT NULL,
    notes TEXT,
    config_hash TEXT NOT NULL,
    config_content TEXT NOT NULL,
    wandb_run_id TEXT
);
```

**Operations**:
- `save`: Store config with version name
- `list`: Show all versions
- `diff`: Compare two versions
- `restore`: Restore a previous version

### 5. Bottleneck Detector

**Purpose**: Identify latency vs throughput issues

**Metrics**:
| Component | Typical % (Latency-Bound) | Typical % (Balanced) |
|-----------|---------------------------|----------------------|
| Tool execution | 78% | 25% |
| Policy inference | 12% | 35% |
| Training step | 8% | 30% |
| Communication | 2% | 10% |

**Diagnosis Logic**:
```python
if tool_execution > 50%:
    diagnosis = "LATENCY-BOUND"
    recommendation = "Switch to local executor"
elif policy_inference + training_step > 80%:
    diagnosis = "THROUGHPUT-BOUND"
    recommendation = "Reduce model size or use LoRA"
else:
    diagnosis = "BALANCED"
```

## Data Flow

### Training Flow

```
User
  │
  ▼
prime-train validate config.toml
  │
  ├─→ [ERROR] → Exit with errors
  │
  ▼
prime-train run config.toml --backup --spot
  │
  ├─→ Load backup config
  ├─→ Check for existing checkpoints
  │     │
  │     ▼ (if found)
  │     Download and resume
  │
  ▼
uv run rl @ config.toml --ckpt ...
  │
  ├─→ [Every N steps] Sync checkpoint to cloud
  ├─→ [Preemption] Detect interruption
  │     │
  │     ▼
  │     Re-provision (via SkyPilot)
  │     Download checkpoint
  │     Resume training
  │
  ▼
Training complete
```

## Implementation Phases

### Phase 1: Foundation (MVP)
- [ ] Package structure
- [ ] Pre-flight validator
- [ ] Basic CLI wrapper
- [ ] Gotcha database

### Phase 2: Cost Intelligence
- [ ] Memory estimator
- [ ] Cost optimizer
- [ ] Hardware presets

### Phase 3: Resilience
- [ ] Checkpoint backup
- [ ] Disk cleanup
- [ ] Auto-resume

### Phase 4: Config Management
- [ ] Version tracking
- [ ] Diff tool
- [ ] Config generator

### Phase 5: Diagnostics
- [ ] Bottleneck detector
- [ ] Health monitor

### Phase 6: Multi-Cloud (Future)
- [ ] SkyPilot integration
- [ ] One-command training

## Testing Strategy

### Unit Tests
- Each gotcha detection function
- Memory estimation accuracy
- Config schema validation

### Integration Tests
- CLI command execution
- Config round-trip (generate → validate)
- Backup/restore flow (mock storage)

### Manual Tests
- Real training with checkpoint sync
- Spot interruption recovery
- Multi-cloud failover

## Configuration

### User Config Location
```
~/.prime-train/
├── backup.yaml      # Backup destination config
├── configs.db       # Version history database
└── presets/         # Custom hardware presets
```

### Environment Variables
```bash
PRIME_API_KEY        # Prime Intellect API key
VAST_API_KEY         # Vast.ai API key
WANDB_API_KEY        # Weights & Biases key
AWS_ACCESS_KEY_ID    # S3 credentials
B2_APPLICATION_KEY   # Backblaze B2 credentials
```

## Future Considerations

### SkyPilot Integration
SkyPilot already supports Prime Intellect. Deep integration would enable:
- One-command provisioning
- Automatic failover across clouds
- Cost-optimized instance selection

### Contributing to Prime Intellect
This project is structured for potential contribution:
- Clean interface to prime-rl
- No internal dependencies
- Community-beneficial gotcha database
- Compatible licensing (MIT)
