# Frozen GPU execution of the #26 interface

Read the [execution registration](EXECUTION.md) and [completed report](../../artifacts/model-studies/2026-11-14/transparent-body-run/REPORT.md). The joint gate failed. No thinking or follow-up lives were run.

One L4 produced 8 positive-control records and 192 probes, with a separate startup canary. The positive gate was evaluated on the CPU and uploaded as a hash-bound boolean. GPU input archives contain neither body/oracle code nor answer keys. The original private forecast heads were not retrained.

## Reproduce without allocating a GPU

Use Python 3.11+ with the analysis packages and the metadata already pinned in the input archive:

```sh
python -m pip install -r experiments/transparent_body/requirements-analysis.txt -r experiments/frozen_capability/requirements-analysis.txt
python -m experiments.frozen_capability.download_metadata --inputs artifacts/model-studies/2026-11-14/transparent-body-run/gpu/inputs.zip --out /tmp/transparent-metadata
python -m experiments.transparent_body.run_replay --repo . --metadata /tmp/transparent-metadata --work /tmp/transparent-raw --out /tmp/transparent-analysis
python -m experiments.transparent_body.run_independent --repo .
```

`run_replay` verifies archive bytes, input provenance, model manifests, native token streams and all score gates. `run_independent` checks the archived choices and realized-choice risk formulas without calling the production scorer/calculator and writes `analysis/independent-check.json`. It also labels its deterministic table-ranking diagnostic as post hoc; it is not a Qwen run.

GPU execution code is preserved in `gpu/inputs.zip`. Account credentials, allocation bindings and authenticated endpoints are excluded. The execution protocol's time guard and external positive gate must be supplied by an operator; invoking the worker alone does not provide resource supervision. No reproduction command here allocates a GPU.
