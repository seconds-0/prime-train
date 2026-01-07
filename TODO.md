# prime-train TODO: 20 Meaningful Upgrades

Prioritized by value (impact × feasibility). Based on:
- 34+ hours of real debugging experience with prime-rl
- Research on SkyPilot, TRL, OpenRLHF, DeepSpeed, Ray RLlib
- Codex code review findings

## Tier 1: Critical (Unblocks Production Training)

### 1. Fix `--ckpt.keep-last N` Workaround
**Value: 10/10** | **Effort: Low**

prime-rl's `--ckpt.keep-last N` is currently broken - doesn't actually delete old checkpoints. Each checkpoint is ~47GB. After 3 checkpoints, disk fills to 100% and training freezes.

**Implementation:**
- Add disk monitoring daemon that watches `/opt/run/checkpoints/`
- Auto-delete oldest checkpoint when disk > 85% full
- Log cleanups to WandB for visibility
- Config option: `[prime_train.cleanup] max_disk_gb = 200`

**Files:** `resilience/cleanup.py` (new), `resilience/runner.py`

---

### 2. GPU Health Check Daemon
**Value: 10/10** | **Effort: Medium**

vLLM inference server dies silently - GPU shows 0% utilization, 0 memory, but no error. Trainer hangs for 3+ hours waiting for batches.

**Implementation:**
- Background thread polling GPU memory via `nvidia-smi` every 10s
- Detect "GPU alive but broken" state (0 MiB, 0% util)
- Alert user with step number
- Auto-kill and resume from checkpoint option

**Files:** `resilience/health.py`, `resilience/runner.py`

---

### 3. Pre-flight Validation: ulimit + GPU + CUDA
**Value: 9/10** | **Effort: Low**

Training fails at step 15 with "too many open files" because `ulimit -n` is 32k instead of 65536.

**Implementation:**
- Check `ulimit -n >= 65536`
- Verify CUDA context can be created
- Test vLLM can spawn on target GPU
- Fail fast with actionable error message

**Files:** `validator/prerequisites.py` (new), `cli.py`

---

### 4. max_tokens + input_tokens < max_model_len Validation
**Value: 9/10** | **Effort: Low**

Setting `max_tokens=4000` with `max_model_len=4096` and ~1500 input tokens causes silent hang at "0/8 rollouts" with no error.

**Implementation:**
- Estimate typical input tokens per environment (BS4 prompts: ~1300-1500)
- Validate: `max_tokens + estimated_input + margin < max_model_len`
- Warning if margin < 500 tokens

**Files:** `validator/gotchas.py`, `validator/schema.py`

---

## Tier 2: High Value (Major Time Savings)

### 5. Model Compatibility Pre-check Script
**Value: 8/10** | **Effort: Medium**

Spent 9 attempts trying different models. gpt-oss-20b has vLLM bug, Qwen3-VL is wrong class, Ministral-3-8B has transformers KeyError.

**Implementation:**
- `prime-train verify-model Qwen/Qwen2.5-7B-Instruct`
- Check: exists on HF, not VL, vLLM compatible, tool calling support
- Output: compatibility status + estimated RAM
- Maintain tested model registry

**Files:** `validator/model.py`, `cli.py`

---

### 6. Auto-set VLLM_USE_V1=0 Environment
**Value: 8/10** | **Effort: Trivial**

vLLM V1 engine causes CUDA segfaults when spawning child processes. Requires manual `export VLLM_USE_V1=0`.

**Implementation:**
- Set environment variables automatically in runner
- Also set `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- Document as official workaround

**Files:** `resilience/runner.py`

---

### 7. Memory Calculator for Single-GPU Setup
**Value: 8/10** | **Effort: Medium**

Multiple OOM errors trying to fit trainer + inference on 1 GPU. Correct order: inference at 0.50 utilization, then trainer fits.

**Implementation:**
- Calculate: model memory + optimizer states + KV cache + buffers
- Single-GPU auto-detection with recommended constraints
- Suggest: `batch_size=8`, `rollouts_per_example=2`, `gpu_memory_utilization=0.50`

**Files:** `validator/memory.py`, `cost/presets.py`

---

### 8. Surface vLLM Errors to Main Logs
**Value: 7/10** | **Effort: Medium**

vLLM errors are buried in subprocess logs. "Request rejected" doesn't propagate to trainer.

**Implementation:**
- Capture vLLM stderr/stdout in runner
- Parse for error patterns
- Surface errors in main console output
- Option: `--verbose-vllm`

**Files:** `resilience/runner.py`

---

### 9. SkyPilot Integration for Spot Recovery
**Value: 7/10** | **Effort: High**

SkyPilot supports Prime Intellect natively. Managed jobs handle spot interruption automatically.

**Implementation:**
- Generate SkyPilot YAML from prime-train config
- `prime-train run --spot --cloud auto`
- Automatic failover across Prime/Lambda/Vast
- Checkpoint sync to object storage

**Files:** `integrations/skypilot.py` (new), `cli.py`

---

### 10. Executor Backend Auto-Detection
**Value: 7/10** | **Effort: Low**

24 hours lost debugging wrong bottleneck. `executor_backend="prime"` adds 1.5s per tool call (latency-bound), but we optimized for throughput.

**Implementation:**
- Detect tool-calling environment
- Strong warning: "Remote executor + tools = latency bottleneck"
- Suggest `executor_backend="local"` with 10-15x speedup estimate

**Files:** `validator/gotchas.py`

---

## Tier 3: Medium Value (Quality of Life)

### 11. Config Linter Pre-commit Hook
**Value: 6/10** | **Effort: Low**

Multiple config format errors: wrong section nesting, forbidden params, duplicate fields.

**Implementation:**
- `prime-train lint config.toml`
- Pre-commit hook integration
- Helpful error messages with suggestions

**Files:** `validator/schema.py`, `cli.py`

---

### 12. WandB Health Monitoring Improvements
**Value: 6/10** | **Effort: Medium**

WandB metrics lag by 20-30 minutes. False "stalled" alerts wasted investigation time.

**Implementation:**
- Health check based on wall-clock time, not WandB samples
- Monitor actual process (GPU utilization, disk I/O)
- Alert threshold: 45+ minutes of no activity
- Document expected step time variance (checkpoints add 5-10 min)

**Files:** `resilience/health.py`

---

### 13. Automatic Checkpoint Compression
**Value: 6/10** | **Effort: Medium**

47GB checkpoints. Compression saves ~40%.

**Implementation:**
- Compress checkpoints before sync: `tar -czvf`
- Decompress on restore
- Config: `[prime_train.backup] compress = true`

**Files:** `resilience/backup.py`

---

### 14. Training Cost Estimator
**Value: 5/10** | **Effort: Medium**

No visibility into total training cost before starting.

**Implementation:**
- Query GPU prices from Prime/Vast
- Estimate steps to convergence (based on similar runs)
- Output: "Estimated cost: $24-40 for 100 steps"

**Files:** `cost/estimator.py` (new), `cli.py`

---

### 15. Profiler Implementation (Currently Stubbed)
**Value: 5/10** | **Effort: High**

`prime-train profile` is stubbed. Promised bottleneck detection doesn't exist.

**Implementation:**
- Actually profile training steps
- Measure: tool execution, policy inference, training step, communication
- Classify: latency-bound vs throughput-bound
- Prescriptive recommendations

**Files:** `profiler/detector.py`

---

## Tier 4: Nice to Have (Polish)

### 16. Interactive Config Wizard
**Value: 4/10** | **Effort: Medium**

`prime-train init` could be more helpful.

**Implementation:**
- Ask: model, hardware, environment type
- Generate optimized config with explanations
- Preview before saving

**Files:** `config/generator.py`, `cli.py`

---

### 17. Config Version Comparison UI
**Value: 4/10** | **Effort: Low**

`prime-train config diff` exists but could show better visualization.

**Implementation:**
- Side-by-side diff with color highlighting
- Show parameter changes with impact notes
- Link to WandB runs

**Files:** `config/manager.py`

---

### 18. Multi-GPU Memory Scaling
**Value: 4/10** | **Effort: Medium**

Memory estimation only considers single GPU. Multi-GPU setups need FSDP sharding calculation.

**Implementation:**
- Account for FSDP memory distribution
- Communication overhead estimation
- Optimal batch size per GPU count

**Files:** `validator/memory.py`

---

### 19. Tested Model Registry
**Value: 3/10** | **Effort: Low**

Document which models work with which vLLM versions.

**Implementation:**
- JSON registry of tested models
- Include: vLLM version, prime-rl version, known issues
- CLI: `prime-train models list`

**Files:** `validator/registry.json` (new), `cli.py`

---

### 20. GitHub Actions for Auto-Resume
**Value: 3/10** | **Effort: Medium**

Spot interruption requires manual intervention to resume.

**Implementation:**
- GitHub Action that polls WandB for training health
- Auto-provision new instance on interruption
- Resume from checkpoint automatically

**Files:** `.github/workflows/training-monitor.yml`

---

## Summary Table

| Priority | Item | Value | Effort | Status |
|----------|------|-------|--------|--------|
| 1 | Fix checkpoint cleanup | 10 | Low | TODO |
| 2 | GPU health daemon | 10 | Medium | TODO |
| 3 | ulimit/GPU pre-flight | 9 | Low | TODO |
| 4 | max_tokens validation | 9 | Low | TODO |
| 5 | Model compatibility check | 8 | Medium | TODO |
| 6 | VLLM_USE_V1 auto-set | 8 | Trivial | TODO |
| 7 | Memory calculator | 8 | Medium | Partial |
| 8 | vLLM error surfacing | 7 | Medium | TODO |
| 9 | SkyPilot integration | 7 | High | TODO |
| 10 | Executor backend detection | 7 | Low | Partial |
| 11 | Config linter | 6 | Low | Partial |
| 12 | WandB health improvements | 6 | Medium | TODO |
| 13 | Checkpoint compression | 6 | Medium | TODO |
| 14 | Cost estimator | 5 | Medium | TODO |
| 15 | Profiler implementation | 5 | High | Stubbed |
| 16 | Interactive wizard | 4 | Medium | Partial |
| 17 | Config diff UI | 4 | Low | Partial |
| 18 | Multi-GPU memory | 4 | Medium | TODO |
| 19 | Model registry | 3 | Low | TODO |
| 20 | GitHub Actions resume | 3 | Medium | TODO |

**Total estimated time saved if all implemented: 20+ hours per training run**
