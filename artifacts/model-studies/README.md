# Model Studies Archive (`artifacts/model-studies/`)

Chronological archive of GPU runs, statistical summaries, audit JSONs, and plots produced alongside `experiments/`.

## Highlighted Mainline Synthesis

- **[`2026-11-20/monthly-research-report/RESEARCH_REPORT_2026-10-15_to_2026-11-20.md`](2026-11-20/monthly-research-report/RESEARCH_REPORT_2026-10-15_to_2026-11-20.md)** — Full 1-month synthesis covering theoretical motivation from Chen & Chen (2026), negative diagnostic findings (`2026-11-10`–`15`), and the `embodied_open_rl` `v1`–`v4` progression with 5 publication figures.
- **[`2026-11-18/embodied-open-rl-v4/REPORT.md`](2026-11-18/embodied-open-rl-v4/REPORT.md)** — Primary empirical report for the 30-life mirrored-ecology pilot (`r7`).
- **[`2026-11-17/embodied-open-rl/`](2026-11-17/embodied-open-rl/REPORT.md)**, **[`embodied-open-rl-v2/`](2026-11-17/embodied-open-rl-v2/REPORT.md)**, **[`2026-11-18/embodied-open-rl-v3/`](2026-11-18/embodied-open-rl-v3/REPORT.md)** — Earlier `v1`–`v3` iterations leading up to `v4`.

## Earlier Exploratory Runs (`2026-11-10` – `2026-11-15`)

- `2026-11-10/`: Frozen-model diagnostics, body-channel probes, and social-learning baselines (`cpu-checks.yml` verified; see [`2026-11-10/README.md`](2026-11-10/README.md)).
- `2026-11-11/`: Closed-loop verification (`closed-loop/`, CI-verified), `direct-policy`, `decision-mechanisms`, `weight-methods`, `baseline-layer3`.
- `2026-11-12/` – `2026-11-15/`: Step-by-step ablations on damage weighting, lethal boundaries, internal body representations, LoReFT/activation steering, paraphrase/order/partner transfer, and explicit vs. implicit self-future forecasting (`REPORT.md` and `*.json` preserved in each directory).

## What Is Tracked in Git

To keep repository clones lightweight and fast while preserving full scientific verifiability:

- **Tracked**: All `REPORT.md` files, protocol specs, audit receipts (`audit.json`), aggregated metrics (`summary.json`, `*.csv`), publication figures (`*.png`), plus the exact result archives needed by CI (`2026-11-10/`, `2026-11-11/closed-loop/`, and `2026-11-18/embodied-open-rl-v4/run/pilot-r7-segments/merged.zip`).
- **Gitignored**: Raw adapter weights (`*.safetensors`, `*.bin`, `*.pt`), chunked GPU transfer artifacts (`*_chunks/`, `chunks/`, `partial/`), raw per-seed `checkpoints.zip`, `inputs.zip`, and redundant intermediate `.zip`/`.npz` archives from `2026-11-11`–`15` whose reports and summary JSONs are already committed.
