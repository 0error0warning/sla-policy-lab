# Validation records

The current online stage has a separate [validation record](online/VALIDATION.md), tests and measured results. The following is the preserved **historical stage-2 offline** record; its 42-test count and artifact metadata are not the current online test count.

# CPU stage 2 validation

Executed on 2026-11-10 UTC. This validates a toy NumPy mechanism demo, not SLA-HCI theory or LLM performance.

## Passed locally

- `OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v`: **42 tests passed**, 9.266 seconds. Full output: [tests.txt](tests.txt).
- `python -m compileall -q sla_demo run.py evaluate.py benchmark.py tests`: passed.
- `node --check web/app.js`: passed.
- Checkpoint save, numeric-only load, exact array/fingerprint round-trip, corrupt values, object arrays, wrong shape and extra keys: tested.
- Strict future-adapter action parser, observation whitelist, invalid and duplicate JSON keys, bad Host headers, invalid controls and full-trajectory API response: tested.
- Every-seed trajectory export, all five mechanisms, action rates, paired seeds and alpha-zero equality across all nine profile/partner combinations: tested.
- `python evaluate.py --trace-all --save-checkpoint /tmp/sla-toy.npz --output artifacts/example-balanced.json`: passed (24 seeds × 5 methods). A compact first-seed export was then regenerated at the same path using the saved checkpoint; its explicit `trajectory_scope` is `first_seed`.
- `python benchmark.py --checkpoint /tmp/sla-toy.npz --seed-count 8 --output-dir artifacts/benchmark`: passed, **27 cells × 5 methods × 8 seeds = 1,080 episodes**. CSV: [benchmark/benchmark.csv](benchmark/benchmark.csv). JSON retains per-seed metrics and is reproducible with the command.

Model fingerprint (training seed 17):

`bc3c498c8475117bcfe0458887f3b4c1a5c984f0bdd84c4794dec04c2b498e10`

A useful negative result: for preserve + mixed, steering return was 4.948 at alpha -2, 8.855 at alpha 0, and 8.736 at alpha 2. Frozen base was 8.855. This illustrates why the demo must not promise that steering improves outcomes. These are synthetic scores over 8 seeds, not claims of statistical significance.

## Not passed / not implemented

- Browser UI and screenshot QA: **unverified**. Earlier local Chromium execution failed on a process-socket restriction. This stage used the supported cloud browser, but it could not connect to the workspace's loopback server: `net::ERR_CONNECTION_REFUSED`. The local HTTP test suite did run successfully. No screenshots or browser pass are claimed.
- `tests/ui_smoke.py` was updated for all-seed replay and exports but not executed successfully here. Failure recovery and responsive layout therefore remain browser-level checks to run on a reachable local computer.
- The CPU GitHub Actions workflow is provided; remote CI status must be checked against the published commit. Local tests are not a claim that remote CI passed.
- No GPU, A100, Colab session, model download, API key, external LLM, transformer training or model-backed inference was used.
