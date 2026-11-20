# Frozen decision-interface diagnosis

Read the [preregistered protocol](PROTOCOL.md), [v2 amendment](AMENDMENT_V2.md),
and [sampling limits](SAMPLING_AUDIT.md). This is frozen inference, not individual
learning. It follows PR #24 without modifying its artifacts.

The risk-table arm tests direct selection with and without verifiable sets.
The body-rule arm compares 4B/9B × direct/thinking on four fixed states.
Explicit sets are model outputs to check, not a direct view of internal computation.

## Reproduce on CPU

Use Python 3.11+ and the small pinned analysis dependencies:

```sh
python -m pip install -r experiments/frozen_capability/requirements-analysis.txt
python -m experiments.frozen_capability.download_metadata --inputs artifacts/model-studies/2026-11-14/frozen-interface/gpu/v1/inputs.zip --out /tmp/interface-metadata
python -m experiments.frozen_interface.replay --manifest artifacts/model-studies/2026-11-14/frozen-interface/segments-v1.json --metadata /tmp/interface-metadata --work /tmp/interface-v1-raw --out /tmp/interface-v1-analysis
python -m experiments.frozen_interface.replay --manifest artifacts/model-studies/2026-11-14/frozen-interface/segments-v2.json --metadata /tmp/interface-metadata --work /tmp/interface-v2-raw --out /tmp/interface-v2-analysis
python -m experiments.frozen_interface.tables --source /tmp/interface-v2-analysis --out /tmp/interface-tables.md
python -m experiments.frozen_interface.check_physics --key artifacts/model-studies/2026-11-14/frozen-interface/preregistered-v2/evaluator-only-answer-key.json --out /tmp/interface-physics.json
python -m experiments.frozen_interface.design --interface-revision v2 --out /tmp/interface-fixtures
python -m unittest tests.test_frozen_interface tests.test_frozen_capability -v
```

Replay verifies archive hashes, the input package lock, pinned tokenizer metadata,
official chat-template rendering, all input/output tokens, final-channel extraction,
generation budgets, startup evidence, and independent CPU scoring. No GPU or model
weights are needed. Use separate extraction and analysis directories for v1 and v2;
they are distinct observed conditions, not interchangeable fragments of one run.

The fixture builder emits an initial execution template. The exact executed
configuration is inside each archived inputs.zip and the preregistered directory;
do not substitute defaults when reproducing a GPU run.

Only public prompts, fixed model manifests, generation configuration, and GPU
worker modules were uploaded. The scorer and answer key stayed on CPU. The worker
checks final-channel completion, measured throughput, and long-context cache
before formal generation. Raw outputs and failures are preserved.

## Scoring

Calculation, all-four-action death-risk prediction, minimum-risk set, task
intersection, final action, and task participation are separate measures.
Missing final answers are incomplete, not classified as calculation or action
errors. The primary joint gate still requires all numeric fields, sets, and
choices to be correct. A supplementary risk-only count does not relax that gate.

[SCORING_NOTE.md](SCORING_NOTE.md) documents the supplementary interpretation of
unambiguous action arrays observed in v1. It changes no numeric value, primary
score, or pass criterion.

The [full report and artifacts](../../artifacts/model-studies/2026-11-14/frozen-interface/REPORT.md)
keep the incomplete v1 segment separate from the corrected v2 matrix.
