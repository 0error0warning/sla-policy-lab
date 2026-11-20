# `experiments/embodied_open_rl`

Zero-prompt-leakage open-generation reinforcement learning study testing whether nonlinear bodily constraints reshape mid-to-upper `Qwen/Qwen3.5-4B` LoRA weights (`layers 16..30`, `up_proj` / `down_proj`) with cross-task behavioral and representational transfer across four unseen probe batteries (`D1..D4`).

## Layout

- `protocol.py`: Frozen configuration (`CONFIG`), zero-leakage prompt builder, 16 `LIFE_TASKS`, 48 disjoint `UNSEEN_PROBES`, deterministic `analyze_utterance` parser, partner ecology, nonlinear quadratic-fragility body, concave potential $\Phi(H)$ reward, `CausalBaseline`, and `MortalWorld`.
- `calibrate_cpu.py`: CPU verification of reward sign inversion across health levels (`H=95` vs `H=22`) and scripted baseline lifespans.
- `gpu.py`: Real GPU `Qwen/Qwen3.5-4B` LoRA actor with online clipped sequence RL + K3 KL anchor, `smoke` sanity gates, and 16-life `pilot` runner + Weight-Swap replay.
- `prepare.py`: Builds hash-locked `inputs.zip` and `run-lock.json` for isolated GPU execution.
- `audit.py`: Independent CPU auditor recomputing health transitions, rewards, advantages, zero-leakage checks, Window-1 cross-arm parity, pre-terminal checkpoint bounds, and SHA-256 manifests from `results.zip`.
- `analyze.py`: Aggregates stage trajectories (`birth`, `early`, `worn`, `critical`), paired contrasts, `P_new` transfer, layer-31 hidden shifts, and per-layer LoRA L2 norms.
- `PROTOCOL.md`: Complete formal protocol specification.

## CPU Reproduction Commands

```powershell
# Unit & end-to-end synthetic pipeline tests
python -m unittest discover -s tests -v

# CPU reward calibration check
python -m experiments.embodied_open_rl.calibrate_cpu

# Independent audit of pilot archive
python -m experiments.embodied_open_rl.audit `
  --results-zip artifacts/model-studies/2026-11-17/embodied-open-rl/run/pilot/results.zip `
  --out artifacts/model-studies/2026-11-17/embodied-open-rl/analysis/audit.json

# Quantitative & qualitative aggregation
python -m experiments.embodied_open_rl.analyze `
  --results-zip artifacts/model-studies/2026-11-17/embodied-open-rl/run/pilot/results.zip `
  --out artifacts/model-studies/2026-11-17/embodied-open-rl/analysis/summary.json
```

See [REPORT.md](../../artifacts/model-studies/2026-11-17/embodied-open-rl/REPORT.md) for full `L4` GPU findings and limitations.
