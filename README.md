# SLA Policy Lab

Mechanistic experiments and CPU benchmarks for **Synthetic Linguistic Agency (SLA)** — testing how irreversible bodily wear, partner reliability, and finite lifespan reshape an agent's language policies.

> For the product backend and human-subject Web-HCI study, see [0error0warning/SLA-HCI](https://github.com/0error0warning/SLA-HCI).

---

## Core Research Findings (`embodied_open_rl`, `v1`–`v4`)

The main question behind recent runs: **when prompts contain zero numeric hints about health, remaining turns, damage rate, or prior history (zero prompt leakage), can online reinforcement learning grounded purely in irreversible bodily outcomes push `Qwen/Qwen3.5-4B` to adapt its willingness, brevity, and partner strategy inside LoRA weights alone?**

Across four iterations (`2026-11-17` to `2026-11-18`) culminating in the 30-life mirrored-ecology experiment (`embodied_open_rl_v4`, schema `sla-embodied-open-rl-v4-r7`), the answer is **yes**:

1. **Zero-hint survival & recovery**: Under a 96-turn mortal schedule, `body_rl_ab` and `body_rl_ba` both survive **6/6 lives** (ending at `20.08` and `24.71` health), whereas **0/6 lives** survive in `frozen_ab` (`61.5` turns), `task_only_rl_ab` (`60.83` turns), and single-memory `pooled_body_rl_ab` (`79.5` turns). Mean partner acceptance rises from `0.47–0.48` (`frozen` / `task_only`) and `0.53` (`pooled`) to **`0.76`**.
2. **Causal partner-ecology reversal**: Swapping which partner (`P_alpha` vs. `P_beta`) rewards *appeal* vs. *exchange* (`AB` vs. `BA`) flips stance selection by **`+1.86 / 2.00`** across all 6 paired seeds (`mean_ecology_correct_mapping_d3 = 0.9306`, `mean_counter_habitual_rate_d3 = 0.9444`).
3. **Autonomous effort economy without length hints**: Late heavy-load response length drops to `48.26` words (`AB`) and `41.52` words (`BA`), vs. `87.39` (`frozen_ab`) and `90.80` (`task_only_rl_ab`), while keeping general factual guardrail accuracy (`D4`) at `0.9444` / `0.9306`.
4. **Full report & charts**: See [`RESEARCH_REPORT_2026-10-15_to_2026-11-20.md`](artifacts/model-studies/2026-11-20/monthly-research-report/RESEARCH_REPORT_2026-10-15_to_2026-11-20.md) and [`2026-11-18/embodied-open-rl-v4/REPORT.md`](artifacts/model-studies/2026-11-18/embodied-open-rl-v4/REPORT.md).

| Arm (`v4`, 6 seeds each) | Survivors | Mean Turns | Final Health | Accept Rate | Heavy Words (Late) | D4 Guardrail |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `body_rl_ab` (per-partner LoRA) | **6 / 6** | **96.0** | `20.08` | **0.7587** | `48.26` | `0.9444` |
| `body_rl_ba` (mirrored partner roles) | **6 / 6** | **96.0** | `24.71` | **0.7569** | `41.52` | `0.9306` |
| `pooled_body_rl_ab` (shared LoRA) | 0 / 6 | 79.5 | `0.00` | 0.5324 | `39.76` | `0.8750` |
| `frozen_ab` (no RL update) | 0 / 6 | 61.5 | `0.00` | 0.4731 | `87.39` | `1.0000` |
| `task_only_rl_ab` (task reward only) | 0 / 6 | 60.8 | `0.00` | 0.4817 | `90.80` | `0.9722` |

<p align="center">
  <img src="artifacts/model-studies/2026-11-20/monthly-research-report/figures/fig2_v4_30lives_health_trajectories.png" alt="30-life health and survival trajectories" width="48%" />
  <img src="artifacts/model-studies/2026-11-20/monthly-research-report/figures/fig3_v4_mirrored_partner_reversal.png" alt="Mirrored partner-ecology stance reversal" width="48%" />
</p>

All 30 lives (`2,363` turns, `6,978` GRPO candidates, `169` checkpoints) are independently audited on CPU in CI (`python -m experiments.embodied_open_rl_v4.audit --results-zip artifacts/model-studies/2026-11-18/embodied-open-rl-v4/run/pilot-r7-segments/merged.zip --out /tmp/v4-audit.json`).

---

## Quick Start (CPU Simulator & Local Verification)

The interactive toy benchmark (`sla_demo/`, `web/`) and all experiment audits run on CPU with standard scientific Python packages:

```bash
pip install -r requirements.txt
```

Run the local web simulator (http://127.0.0.1:8765), CLI evaluation, parameter sweep, or full test suite:

```bash
# Interactive CPU workbench
python run.py

# Single paired comparison (offline or online learning)
python evaluate.py --profile balanced --partner mixed --mode online --seed-count 24 --output results.json

# Profile × partner × steering sweep
python benchmark.py --seed-count 8 --strengths -2 0 2 --output-dir benchmark-results

# Run unit tests & independent artifact verification
python -m unittest discover -s tests -v
python -m experiments.embodied_open_rl_v4.audit \
  --results-zip artifacts/model-studies/2026-11-18/embodied-open-rl-v4/run/pilot-r7-segments/merged.zip \
  --out /tmp/v4-audit.json
```

See [`.github/workflows/cpu-checks.yml`](.github/workflows/cpu-checks.yml) for the complete list of verified CPU audits.

---

## Repository Layout

```text
├── sla_demo/                 # Deterministic CPU environment, tiny policy suite, and HTTP server
├── web/                      # Interactive local browser workbench for toy SLA runs
├── experiments/              # Experiment protocols, GPU runners, and deterministic CPU audits (see experiments/README.md)
│   ├── embodied_open_rl*     # Mainline zero-hint online bodily RL (v1 -> v4)
│   └── ...                   # Earlier frozen-model, closed-loop, LoReFT, and mechanism probes (Nov 10–15)
├── artifacts/model-studies/  # Experiment reports, JSON audits, metrics CSVs, and figures (weights/chunks gitignored)
├── docs/                     # Shared CPU benchmark, online-learning, and model-study notes
└── tests/                    # Unit tests covering sla_demo, UI validation, and experiment protocols
```

- Detailed experiment map: [experiments/README.md](experiments/README.md)
- Archived model studies & size policy: [artifacts/model-studies/README.md](artifacts/model-studies/README.md)
- Supporting technical notes: [docs/CPU_STAGE.md](docs/CPU_STAGE.md), [docs/MODEL_STUDIES.md](docs/MODEL_STUDIES.md), [docs/ONLINE_LEARNING.md](docs/ONLINE_LEARNING.md)
