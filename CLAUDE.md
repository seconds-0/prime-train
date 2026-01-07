# prime-train Development Instructions

## Project Overview

prime-train is a production-grade training harness for prime-rl. It provides:
- **Pre-flight validation** - Catch config bugs before GPU provisioning
- **Cost optimization** - Find cheapest compatible hardware
- **Resilient training** - Checkpoint sync, auto-resume, spot recovery
- **Config management** - Version, diff, and restore configs
- **Bottleneck detection** - Identify latency vs throughput issues

## Architecture

```
src/prime_train/
├── cli.py              # Main CLI entry point (Typer)
├── validator/          # Pre-flight validation
│   ├── core.py         # Main validation logic
│   ├── gotchas.py      # Known issues database
│   ├── schema.py       # Config schema validation
│   ├── model.py        # Model compatibility checks
│   └── memory.py       # Memory estimation
├── cost/               # Cost optimization
│   ├── optimizer.py    # GPU finder (queries Prime, Vast.ai)
│   ├── memory.py       # Memory estimation wrapper
│   └── presets.py      # Hardware presets (H100, A100, 4090, etc.)
├── resilience/         # Resilient training
│   ├── backup.py       # Checkpoint backup (S3, B2, GCS)
│   ├── runner.py       # Training runner wrapper
│   └── health.py       # WandB health monitoring
├── config/             # Config management
│   ├── manager.py      # Version tracking (SQLite)
│   └── generator.py    # Config generation wizard
└── profiler/           # Bottleneck detection
    └── detector.py     # Latency vs throughput profiler
```

## Key Design Decisions

### 1. Wraps prime-rl, doesn't replace it
prime-train is a thin layer that adds guardrails. It calls `uv run rl @` under the hood.

### 2. Known Gotchas Database
All issues discovered during our 34+ hours of debugging are encoded in `validator/gotchas.py`.
When adding new gotchas:
- Add a detection function
- Add to GOTCHA_DATABASE with hours_lost for prioritization
- Include a clear recommendation

### 3. Hardware Presets
Optimized configs for each GPU type are in `cost/presets.py`. These come from real training runs.

### 4. Incremental Phases
Features are organized into phases (see docs/ARCHITECTURE.md). Each phase can be released independently.

## Development

```bash
# Setup
cd /Users/alexanderhuth/prime-train
uv sync --all-extras

# Run tests
pytest

# Run linting
ruff check src tests
mypy src

# Test CLI locally
uv run prime-train --help
uv run prime-train validate examples/basic-config.toml
```

## Testing Philosophy

1. **Gotcha tests** - Each gotcha should have a test config that triggers it
2. **Memory estimation** - Tests against known model sizes
3. **CLI integration** - Test each command with sample configs

## Contributing Gotchas

When you discover a new training issue:

1. Add detection function in `validator/gotchas.py`:
```python
def _detect_new_issue(config: dict[str, Any]) -> bool:
    # Return True if issue detected
    ...
```

2. Add to GOTCHA_DATABASE:
```python
Gotcha(
    id="new-issue-id",
    name="Human Readable Name",
    description="What goes wrong",
    detection=_detect_new_issue,
    recommendation="How to fix it",
    severity=Severity.ERROR,  # or WARNING
    hours_lost=2.0,  # How many hours this cost us
),
```

3. Add test in `tests/test_gotchas.py`

## Git Workflow

```bash
# Use seconds-0 account
git config user.name "seconds-0"
git config user.email "36005888+seconds-0@users.noreply.github.com"

# Atomic commits - one feature per commit
git commit -m "Add detection for FSDP+LoRA conflict"
```

## Potential Contribution to Prime Intellect

This project is structured to potentially be contributed back to Prime Intellect. Key considerations:
- Clean separation from prime-rl internals
- No proprietary dependencies
- Well-documented gotchas benefit the community
- MIT license compatible with Prime's Apache 2.0
