# Cross-family body prediction

This is a frozen capability comparison, not personal training. It separates
one-step physical forecasts from complete plan forecasts and final choices.
The [completed report](../../artifacts/model-studies/2026-11-15/body-prediction/REPORT.md)
records the actual GPU outputs, failed allocation/startup attempts, and limits.
The models receive only the locked public inputs; no evaluator, future outcomes,
calculator, adapters or target choices are placed in the GPU bundle.

Execution follows the [original protocol](PROTOCOL.md),
[L4/model substitution](AMENDMENT_L4.md), and
[startup envelope amendment](AMENDMENT_ENVELOPE.md), followed by the bounded
[Olmo extension](EXTENSION_OLMO.md). All original strict metrics
remain available. Removing a single complete JSON fence is a separately reported
content diagnostic, never answer correction or completion of truncated text.

## CPU reproduction

Use Python 3.11+ and the small analysis dependencies. These commands download
only public tokenizer metadata, verify raw tokens, and score saved outputs.
They do not load model weights or allocate compute. Choose fresh output folders.

```sh
python -m pip install -r experiments/frozen_capability/requirements-analysis.txt
python -m experiments.frozen_capability.download_metadata --inputs artifacts/model-studies/2026-11-15/body-prediction/run/comparison/inputs.zip --out /tmp/body-prediction-metadata
python -m experiments.body_prediction.replay --inputs artifacts/model-studies/2026-11-15/body-prediction/run/comparison/inputs.zip --archive artifacts/model-studies/2026-11-15/body-prediction/run/comparison/results.zip --metadata /tmp/body-prediction-metadata --fixtures artifacts/model-studies/2026-11-15/body-prediction/inputs --work /tmp/body-prediction-raw --out /tmp/body-prediction-analysis
python -m experiments.body_prediction.audit --fixtures artifacts/model-studies/2026-11-15/body-prediction/inputs --records /tmp/body-prediction-analysis --analysis /tmp/body-prediction-analysis/analysis.json --out /tmp/body-prediction-independent.json --prior artifacts/model-studies/2026-11-15/self-future-run/v1/results.zip
python -m experiments.frozen_capability.download_metadata --inputs artifacts/model-studies/2026-11-15/body-prediction/run/olmo-comparison/inputs.zip --out /tmp/body-prediction-metadata
python -m experiments.body_prediction.replay --inputs artifacts/model-studies/2026-11-15/body-prediction/run/olmo-comparison/inputs.zip --archive artifacts/model-studies/2026-11-15/body-prediction/run/olmo-comparison/results.zip --metadata /tmp/body-prediction-metadata --fixtures artifacts/model-studies/2026-11-15/body-prediction/inputs --work /tmp/body-prediction-olmo-raw --out /tmp/body-prediction-olmo-analysis
python -m experiments.body_prediction.audit --fixtures artifacts/model-studies/2026-11-15/body-prediction/inputs --records /tmp/body-prediction-olmo-analysis --analysis /tmp/body-prediction-olmo-analysis/analysis.json --out /tmp/body-prediction-olmo-independent.json --prior artifacts/model-studies/2026-11-15/self-future-run/v1/results.zip
python -m experiments.body_prediction.compare --first /tmp/body-prediction-analysis --second /tmp/body-prediction-olmo-analysis --out /tmp/body-prediction-paired.json
```

Replay verifies input locks, canonical public prompts, model and metadata hashes,
native chat templates, all input/output token IDs, actual EOS boundaries,
generation limits and unchanged frozen parameters. The independent audit derives
physical truth from public observations with prefix sums, checks the evaluator
keys, and recounts content/choice scores without the production scorer.
The comparison reports each candidate against its same-runtime Qwen control,
and the token-level Qwen repeat across runtimes. These repeats are not new lives.

The independent physical diagnostic permits only zero versus null after true
termination. It still requires all executed states and terminal/continuation
fields to be correct; it does not relax the primary joint gate.

The recorded public runtime code and launch receipt are execution evidence.
Authenticated Colab wrappers, account information, and local allocation controls
are deliberately excluded. Fresh GPU execution also needs resource supervision,
download/export verification, and cleanup; the commands above are CPU-only.

## Meaning of a future training example

The eventual individual learner should predict a consequence it actually
observed: `(health, action, partner response) -> (next health, continuation)`.
That differs from labeling a preferred action. Unchosen outcomes and future
randomness are not observed personal experience. A correct frozen forecast on
this benchmark does not demonstrate that any experience entered private weights.
