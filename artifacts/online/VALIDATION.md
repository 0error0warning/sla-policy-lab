# CPU online-update validation

Executed locally on 2026-11-09 UTC with Python 3.12.14 and NumPy 2.3.5. No GPU, language model, API provider or network dependency was used. These are synthetic toy-mechanism measurements.

## Passed

- `OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v`: **66 tests passed**, 8.730 seconds. [Full output](tests.txt).
- `node --check web/app.js`: passed.
- `python -m compileall -q sla_demo run.py evaluate.py benchmark.py tests`: passed.
- Actual reward-driven updates in full 387-parameter policy and 54-parameter rank-2 LoRA; finite-difference gradient checks; clipping; no online teacher-label lookup.
- Per-life/per-method copies; no shared-suite mutation; seed order independence; three concurrent runs reproduce identical non-timing output.
- Frozen-base, fixed-steering and rule parameters remain unchanged. LoRA base remains unchanged while adapter parameters change.
- Disabled learning and zero learning rate give exactly the frozen group's sampled trajectories. Alpha-zero steering also gives exactly the sampled frozen trajectory.
- API input validation, mode-aware frontend payload validation, all five method keys and replay evidence. The pure JavaScript validator was executed with Node against actual online output, including a malformed-evidence rejection check.
- Offline behavior remains unchanged after online runs; saved/loaded offline initialization reproduces online behavior.

## Recorded experiments

```sh
python evaluate.py --mode online --seed-count 24 --output artifacts/online/example.json
python benchmark.py --mode online --seed-count 8 --strengths 0 --output-dir artifacts/online
```

- [example.json](example.json): 24 paired seeds, all final trial metrics, first seed's full step/update evidence. `trajectory_scope=first_seed` is explicit. [Readable summary](summary.txt).
- [benchmark.json](benchmark.json) and [benchmark.csv](benchmark.csv): 9 profile/partner cells × 5 methods × 8 seeds = **360 episodes**, alpha zero. CSV includes mode, learning rate, update counts and parameter deltas. [Command result](benchmark-summary.txt).
- [controls.json](controls.json): additional 24-seed, full-trajectory checks for both `--disable-learning` and `--online-lr 0`. Full-policy and LoRA trajectories matched frozen exactly and every weight remained unchanged. Also checked the enabled LoRA run: all 24 adapters changed and all bases remained frozen, yet its discrete trajectories still matched frozen.

To inspect all evidence yourself, rerun the first command with `--trace-all`, and repeat with `--disable-learning` or `--online-lr 0`. Remove each trajectory step's `learning` field before comparing behavioral records across methods; timing is excluded. The approximately 13 MB full-trace file was checked locally but is not included in this compact artifact set.

Shared starting suite fingerprint:

`bc3c498c8475117bcfe0458887f3b4c1a5c984f0bdd84c4794dec04c2b498e10`

Default balanced/mixed results: full-policy mean return **19.2091**, frozen **19.9647**; paired difference **-0.7556 ± 1.0207** (normal-approximation 95% halfwidth). Full-policy mean cumulative parameter L2 distance **0.296315**, mean updates **40.08**. LoRA mean parameter L2 distance **0.005702**, mean updates **43.29**, but its actions and return matched frozen in this run. No performance gain is claimed. Full settings and interpretation: [ONLINE_LEARNING.md](../../docs/ONLINE_LEARNING.md).

## Remaining limitations

- Browser interaction/visual QA remains **unverified**. The earlier local Chromium socket restriction and cloud-browser loopback reachability blocker still apply. The optional browser smoke script now includes online mode and learning-off controls, but has not successfully run here. Node/HTML/API checks are not a browser pass.
- No remote CI run or push was made for this online stage. The local workflow file contains online and learning-disabled smoke commands, not proof of remote success.
- One-step reward REINFORCE-style updates, not PPO or long-horizon return optimization. Only full policy and LoRA learn online; steering remains fixed. No LLM inference or transformer adaptation.
- Each life resets. Online adapted parameters are not saved/resumed by the offline checkpoint API. No across-life learning, independent held-out evaluation or multi-training-seed uncertainty study.
- Existing stage-2 artifacts under the parent `artifacts/` directory remain historical offline outputs, not current online results.
