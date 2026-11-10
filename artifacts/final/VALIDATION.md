# Integrated CPU demo verification

Verified 2026-11-09 18:42 UTC. No GPU, A100, Colab, model download, API provider or credentials used.

## Passed on the integrated code

- `OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v`: **66 tests passed**, 8.115 seconds. [Output](tests.txt).
- `python -m compileall -q sla_demo run.py evaluate.py benchmark.py tests`: passed.
- `node --check web/app.js`: passed.
- `python tests/dashboard/run_checks.py --repo .`: passed against **five fresh simulator outputs**, including online learning enabled and disabled. [Output](dashboard.txt).
- DOM/event checks cover play/pause/step/restart/speed, timeline and seed selection, terminal states, focus retention, reduced motion, dirty/repeated runs, presets, request payloads, network failure and malformed-result recovery, export, one-seed intervals, alpha-zero equivalence, online update evidence and learning-disabled frozen controls.
- The 66-test suite includes online gradient finite differences, actual reward-driven parameter changes, shared initialization, per-life reset, no online teacher feedback, learning-disabled/zero-rate controls, concurrency isolation, checkpoint reproducibility, API schema and frontend evidence validation.

Recorded reproducible experiment sets are preserved: **1,080 offline episodes** in [the stage-2 benchmark](../benchmark/benchmark.csv) and **360 online episodes** in [the online benchmark](../online/benchmark.csv). These are distinct synthetic experiments, not an empirical LLM leaderboard. See [online methods and limitations](../../docs/ONLINE_LEARNING.md). Full-policy reward did not improve in the recorded default run; LoRA parameter changes did not alter its sampled discrete trajectory.

## Not verified

**Real-browser visual and responsive-layout QA remain unverified.** The supported Chromium browser could not reach the local demo server (`ERR_CONNECTION_REFUSED`), including an approved localhost-only server outside ordinary command isolation. The browser error page was observed. No dashboard screenshots or browser pass are claimed. The DOM harness is not a substitute for CSS/layout verification.

This report records local checks. Remote CI must be checked against the exact published commit; adding a workflow is not evidence that it passed. Historical validation reports remain in their original artifact folders.
