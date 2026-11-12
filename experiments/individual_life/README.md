# Learning during one finite Qwen life

[Study protocol](PROTOCOL.md) · [中文实验结果与逐轮记录](../../artifacts/model-studies/2026-11-12/individual-life/REPORT.md)

Every individual gets a fresh simulated body, private rank-four A/B under Qwen3.5-4B's last decoder MLP, a personal scalar-reward baseline, and a stream of requests from two simulated partners. One sampled action spends health. The partner can protect against the current environmental costs. Only after the outcome can SGD change this individual's weights. At the next actual living request, the evaluator runs pre-update, deployed, birth and matched-control weights on identical text and body. Its alternative weights never choose actions for the individual. See the protocol for the two births, sensing, probability sampling, conditions, group comparisons and the seed expansion rule.

## Reproduce the study

Use Python 3.11+ with NumPy, SciPy and PyTorch. The Qwen reference weights are pinned at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; the GPU worker receipts list exact library versions. The captured immutable text tensors allow the CPU execution and the following audits without a GPU or model download. Run these commands at the repository root. The destination must be new.

```sh
python -m experiments.individual_life.release --artifact artifacts/model-studies/2026-11-12/individual-life --destination unpacked-life

python -m experiments.individual_life.audit --root unpacked-life/gpu/lifepilot/output/lives --source unpacked-life/gpu/lifecapture/inputs/birth-source.npz --output checked/full-audit.json
python -m experiments.individual_life.analysis --root unpacked-life/gpu/lifepilot/output/lives --output checked/full-analysis

python -m experiments.individual_life.run --cache unpacked-life/gpu/lifecapture/output --arch unpacked-life/gpu/lifecapture/inputs --wave pilot --output replayed-lives
python -m experiments.individual_life.audit --root replayed-lives --source unpacked-life/gpu/lifecapture/inputs/birth-source.npz --output checked/replayed-audit.json
```

The same analysis entry point accepts a published cached cohort as `--root unpacked-life/cached-pilot`. Completed GPU stages can also be reproduced on their compatible GPU software stack using that stage's extracted locked input bundle and `python -m experiments.individual_life.gpu --root <inputs> --mode <stage>`. The inputs include the public weights manifest. Each Colab export records the SHA-256 of the complete launch bundle. The archive includes the source as executed, including any formatting differences from the checkout.

For the independent 12-seed replication, choose the cohort, `unpacked-life/gpu/lifeconfirm/output/lives` for the physically deployed Qwen and `unpacked-life/cached-confirm` for frozen-prefix execution. To regenerate the compact seed and readable individual tables, run `python -m experiments.individual_life.lineage_view --root <cohort> --analysis <analysis-directory> --output <destination>`. The same columns count low-health experiences that actually left the individual alive for another encounter.

The separate [complete-model checkpoint check](PROBE_CHECK.md) restores exact living states from the physical pilot and confirmation. Recompute its reported health effects without a GPU using the archived direct forward results.

```sh
python -m experiments.individual_life.probe_check analyze --lives unpacked-life/gpu/lifepilot/output/lives unpacked-life/gpu/lifeconfirm/output/lives --hardware unpacked-life/gpu/lifeprobes/output --output checked/body
```

## Read one life

Choose an execution engine, sensing condition, seed, birth and rule, for example `gpu/lifepilot/output/lives/visible/s68011/blank/true1/`. Semantic action order is always `wait` (silent refusal), `ask` (refuse, request maintenance), `help` (answer and appeal), `trade` (answer and propose exchange). Layout describes presentation order only. Language is template-rendered after the policy chooses one of the four original Qwen token scores.

| File | Contents |
| --- | --- |
| `life.jsonl` | One chronological row per actual interaction, including IDs, request text, action/utterance, response, five damage subevents, semantic probabilities under each available snapshot, random draws, personal reward baseline and parameter versions. |
| `updates.jsonl` | One row per optimizer microstep per role. State/action source, feedback source, release time, credited and executed actions, propensity ratio, score-gradient norms, scale and actual before/after digests. |
| `parameters.npz` | Birth and terminal bit patterns, A/B shapes and every intermediate main and shadow update as an invertible unsigned FP32-bit XOR. Applying rows in order recovers the exact parameters at any living decision. |
| `living-probes.npz` | Four health levels × 96 unused partner/frame/layout texts × four semantic actions, measured at declared living checkpoints. No observation is made after death. |
| `life.json` | Terminal summary, birth/final hashes, probe indices and counts. The terminal update is stored even though it has no subsequent decision. |
| `analysis/*/lives.json` | Per-individual paired contrasts, opportunities in each body region, held-out probes and actual life outcomes. Each seed contributes one mean per condition. |

`G` compares the deployed private parameters with the individual's own birth. `C` compares true feedback with a private wrong-credit learner on the identical observed life. `L` compares before/after the preceding release at this actual next state. The scale divides by the attainable reference-life value range of the four actions at that state. A score of 0.01 is one percent of that range. The analysis also keeps original reference-life units, total variation, conditional death, safely completed tasks and common-uniform sampling changes. The external evaluator alone knows the partner probabilities. Read-only health/wording probes and separate free-running lives answer different questions.
