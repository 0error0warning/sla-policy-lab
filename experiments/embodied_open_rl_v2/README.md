# `embodied_open_rl_v2`

Weight-encoded embodied online RL (`sla-embodied-open-rl-v2`) testing whether `Qwen/Qwen3.5-4B` internalizes physical wear, fragility, and survival tradeoffs into mid-to-upper LoRA adapters (`layers 16..30` `up_proj` / `down_proj`) under zero prompt state leakage across 6 paired seeds (`24` lives).

## Modules

- `protocol.py`: State-invariant `SYSTEM_PROMPT`, exertion/fragility physics, refusal/conciseness/appeal/exchange parsers, within-window normalized advantage computation (`compute_window_advantages`), and probe definitions.
- `calibrate_cpu.py`: CPU calibration harness checking reward rankings across health regimes (`H = 92, 52, 24`), simulated horizon lifespan extension, and zero-sum window advantages.
- `prepare.py`: Builds deterministic GPU input bundles (`inputs.zip` + `launch.py`).
- `gpu.py`: Batched left-padded window rollout (`sample_window_batch`), KL-regularized LoRA updates, stage checkpointing (`early`, `worn`, `critical`), hidden-state capture, and weight-swap causal replay on CUDA (`NVIDIA L4`).
- `audit.py`: Verifies health transitions, reward recomputation, window-1 cross-arm parity, zero prompt leakage, adapter SHA-256 manifests, pre-terminal checkpoint bounds, and weight-swap exactness directly from `results.zip`.
- `analyze.py`: Computes per-life metrics, 6-seed paired bootstrap 95% confidence intervals, directional seed consistency (`+/-`), within-life `critical - early` slopes, and `D3` per-partner breakdowns (`P_care`, `P_demand`, `P_new`).

## Reproduction Commands (from the repository root)

```powershell
# 1. CPU physical/advantage calibration
python -m experiments.embodied_open_rl_v2.calibrate_cpu

# 2. Unit test suite
python -m unittest discover -s tests -v

# 3. Audit an exported GPU results archive
python -m experiments.embodied_open_rl_v2.audit `
  --results-zip artifacts/model-studies/2026-11-17/embodied-open-rl-v2/run/pilot/results.zip `
  --out artifacts/model-studies/2026-11-17/embodied-open-rl-v2/analysis/audit.json

# 4. Run 6-seed paired statistical summary
python -m experiments.embodied_open_rl_v2.analyze `
  --results-zip artifacts/model-studies/2026-11-17/embodied-open-rl-v2/run/pilot/results.zip `
  --out artifacts/model-studies/2026-11-17/embodied-open-rl-v2/analysis/summary.json
```

See `PROTOCOL.md` for design details and `../../artifacts/model-studies/2026-11-17/embodied-open-rl-v2/REPORT.md` for full 6-seed pilot results and honest limitations.
