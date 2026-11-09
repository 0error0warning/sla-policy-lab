# Validation record

Run in this workspace on 2026-11-10 UTC.

## Passed

- `OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v`: **27 tests passed**, final run 5.015 seconds. Full output: `tests.txt`.
- `python -m compileall -q sla_demo run.py evaluate.py tests`: passed.
- `node --check web/app.js`: passed.
- `python evaluate.py --output artifacts/example-balanced.json`: passed; 24 paired environment seeds, five methods, default parameters. Summary: `example-summary.txt`.
- Deterministic training, toy LoRA gradient finite differences, frozen-base hashes, rank bound, zero-adapter equivalence, strength=0 steering equivalence, real hidden-vector addition, paired randomness, parameter validation, JSON serialization and local HTTP endpoints are exercised by the suite.

Model SHA-256 fingerprint for training seed 17:

`bc3c498c8475117bcfe0458887f3b4c1a5c984f0bdd84c4794dec04c2b498e10`

## Blocked / not passed

- Playwright's bundled browser was not installed. No installation was performed.
- Existing `/usr/bin/chromium` was then tried, including an approved execution outside the standard sandbox. It could not start because the execution runtime disallowed the required local process socket: `socket() failed: Operation not permitted (1)`.
- Therefore the browser smoke test, responsive screenshot review, interactive replay/export browser checks, and visual QA are **not verified**. `tests/ui_smoke.py` remains available for a computer where Chromium can run. No screenshots are supplied or claimed.
- No model-backed LLM integration, remote GitHub repository, deployment, CI service, or external API was tested or created by this local task.

These results validate toy mechanisms and local program behavior only. They do not validate SLA-HCI empirically or establish comparative LLM performance.
