# Transparent body consequences

Read the [scope and finite gates](PROTOCOL.md) and [CPU report](../../artifacts/model-studies/2026-11-14/transparent-body/REPORT.md).

This increment implements a declared body calculator, a past-only forecast contract,
the consequence-only Qwen message interface, and execution that never substitutes a
safer action. The archived personal predictor is an external head on frozen Qwen
features; it is not LoReFT or an updated language-model backbone.

The initial CPU increment did not run new GPU inference, training or complete lives.
The subsequently authorized [bounded GPU execution](RUN_README.md) is now complete:
all 192 probes finished, but the joint gate failed and expansion stopped.
`example-prompts.json` contains prepared messages, not model responses; GPU responses
are in the separate run archive. Any later online controller remains separate work. The
generic `diagnostic` calculator permits evaluator-side inputs; formal execution
must go through `PersonalForecast` and `calculate`.

Reproduce from the repository root, using the repository's analysis dependencies:

```sh
python -m pip install -r experiments/transparent_body/requirements-analysis.txt
python -m unittest discover -s tests -p test_transparent_body.py -v
python -m experiments.transparent_body.audit_v2 --source artifacts/model-studies/2026-11-14/frozen-interface/gpu/v2/results.zip --out /tmp/transparent-body/v2-audit.json
python -m experiments.transparent_body.verify --repo . --out /tmp/transparent-body
```

`verify` independently checks zero-noise affine recurrences (also in unit tests),
stochastic risk against 200,000 independent truncated-normal paths, 11 vs 15
quadrature nodes, and every saved personal predictor checkpoint in eight old
histories. Audit calculations stay outside the prompt. At boundaries, numerical
integration error is reported; it is not treated as an exact model of unknown
physics. Neither test success nor prepared prompts constitute new Qwen behavior.
