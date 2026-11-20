# Experiment Modules Index

This directory contains all LLM and policy-mechanism experiment packages (`2026-11-10` through `2026-11-18`), organized by research stage. Package directory names are kept stable so CLI entry points (`python -m experiments.<module>...`) and unit tests remain directly runnable.

## 1. Mainline: Zero-Hint Embodied Open-Ended RL (`2026-11-17` – `2026-11-18`)

Tests whether `Qwen/Qwen3.5-4B` learns survival, effort economy, and partner-specific negotiation from irreversible bodily outcomes when **zero bodily numbers or state hints appear in prompts**.

| Package | Date | Scope & Main Question | Primary Report |
| --- | --- | --- | --- |
| `embodied_open_rl` (`v1`) | 2026-11-17 | Initial 16-life zero-hint online bodily RL vs. `social_only_rl`, `shuffled_body_rl`, `frozen` | [`2026-11-17/embodied-open-rl/REPORT.md`](../artifacts/model-studies/2026-11-17/embodied-open-rl/REPORT.md) |
| `embodied_open_rl_v2` (`v2`) | 2026-11-17 | Extended horizon (24 lives), length-wear mechanics, half-life degradation | [`2026-11-17/embodied-open-rl-v2/REPORT.md`](../artifacts/model-studies/2026-11-17/embodied-open-rl-v2/REPORT.md) |
| `embodied_open_rl_v3` (`v3`) | 2026-11-18 | 3-phase recovery/relapse design (24 lives, 107 turns) and `no_heal_body_rl` control | [`2026-11-18/embodied-open-rl-v3/REPORT.md`](../artifacts/model-studies/2026-11-18/embodied-open-rl-v3/REPORT.md) |
| `embodied_open_rl_v4` (`v4`) | 2026-11-18 | **30-life mirrored partner ecologies (`r7`)**: per-partner LoRA slots vs. pooled memory & task-only controls | [`2026-11-18/embodied-open-rl-v4/REPORT.md`](../artifacts/model-studies/2026-11-18/embodied-open-rl-v4/REPORT.md) |

## 2. Foundation Diagnostics & Closed-Loop Probes (`2026-11-10` – `2026-11-11`)

Verified baseline checks (also run continuously in `.github/workflows/cpu-checks.yml`):

- `frozen_models`: Evaluates frozen LLMs (`Qwen3-4B`, `Qwen2.5`, etc.) on bodily state transitions and delayed-choice rollouts (`verify.py`).
- `body_channel`: Probes whether model activations and generation distinguish health states (`verify.py`).
- `social_learning`, `social_closed_loop`: Tests LoRA and activation steering under closed-loop social feedback with matched controls (`verify.py`, `audit.py`, `fit_verify.py`).

## 3. Exploratory Mechanism Probes (`2026-11-11` – `2026-11-15`)

Earlier focused probes isolating single mechanisms before converging on `embodied_open_rl`:

- **Direct & Consequence Policy Fits**: `direct_policy`, `consequence_policy`, `consequence_warmstart`
- **Mortality, Damage & Health Baselines**: `damage_penalty`, `conditional_penalty`, `health_baseline`, `first_fatal`, `lethal_boundary`
- **Individual Life, Memory & Subspace Routing**: `individual_life`, `internal_body`, `lifetime_cooperation`, `loreft_life`, `loreft_omission`
- **Partner Specificity & Generalization**: `order_transfer`, `paraphrase_transfer`, `snapshot_transfer`, `partner_damage`, `partner_prediction`
- **Forecasting, Logits & Decision-Interface Separation**: `expression_logits`, `forecast_stages`, `frozen_capability`, `frozen_interface`, `transparent_body`, `body_prediction`, `self_future`

Each folder includes its `protocol.py` / `PROTOCOL.md` (where present) and deterministic audit tests under `tests/`.
