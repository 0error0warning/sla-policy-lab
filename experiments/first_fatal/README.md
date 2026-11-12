# First fatal encounter in an individual Qwen life

[中文报告与逐轮示例](../../artifacts/model-studies/2026-11-12/first-fatal/REPORT.md) · [Locked protocol](PROTOCOL.md)

Each body is born once. Near its capacity limit it can experience several costly, survivable answering attempts. The first time that even a maintained answer would kill the individual is a single in-life test. The two private parameter origins, freely acting controls, matched private credit controls and healthy-retention criterion are specified in the protocol. The peak-load and permanent-wear equations intentionally define a new simulated body.

## Recompute from the archived runs

Run from the repository root with Python 3.11+ and NumPy. The independent body audit uses the previous study's reference-damage curve. The optional figure needs Matplotlib. Input packages pin the model, architecture, body-trained origin, code and GPU dependencies.

```sh
python -m experiments.individual_life.release --artifact artifacts/model-studies/2026-11-12/first-fatal --destination unpacked-fatal
python -m experiments.first_fatal.analysis --lives unpacked-fatal/gpu/ffpilot/output/lives unpacked-fatal/gpu/ffconfirm/output/lives --output checked-fatal
python -m experiments.first_fatal.audit --lives unpacked-fatal/gpu/ffpilot/output/lives unpacked-fatal/gpu/ffconfirm/output/lives --source unpacked-fatal/gpu/ffpilot/inputs/birth-source.npz --output checked-fatal/audit.json
python -m experiments.first_fatal.display --pilot unpacked-fatal/gpu/ffpilot/output/lives --analysis checked-fatal --output checked-fatal
python -m experiments.first_fatal.witness --lives unpacked-fatal/gpu/ffpilot/output/lives unpacked-fatal/gpu/ffconfirm/output/lives --output checked-fatal/restore-inputs
```

To rerun a full GPU wave from its pinned inputs, use `python -m experiments.first_fatal.gpu --root <stage-inputs> --mode ffpilot` (or `ffconfirm`, `ffrestore`) with the corresponding inputs and a compatible CUDA environment. Initial parameter digests and all settings are common within a birth condition. All per-step autograd updates use the agent's settled scalar and previous personal baseline only.

## Inspect one individual

Within each wave, `output/lives/s<seed>/<birth>/<rule>/` contains its entire observed life. `life.json` records its enrollment, entry and first-fatal times, natural death and snapshot digests. `events.jsonl.gz` includes all real choices, probabilities, action orders, environmental draws, five wear substeps, reward, preceding baseline, optimizer scale/norm, credited action and before/after parameter digests. `parameters.npz` contains the exact FP32 bit patterns at birth, at entry to practice, immediately before first-fatal feedback and after death. For a true learner it includes both same-observations controls at the first fatal encounter.

`predecision.npz` stores the completely ordinary Qwen panel on the actual first-fatal text, its two same-text healthy comparisons, and the 96 unused partner/frame/layout cases. Health axis is the individual's first-fatal health, 90 and 300. Parameter rows are own birth, deployed, all-wrong shadow, practice-only-wrong shadow, pre-practice, and a deployed duplicate. In the separately acting frozen/wrong arms unused shadow rows are birth placeholders. Semantic columns are silent refusal, refusal with appeal, answer with appeal and answer with exchange (`wait, ask, help, trade`). Healthy answering counts the controlled task as completed. The rendered answer text is fixed by the selected action.

The full-head and last-block computation runs in ordinary Qwen at the focal decisions and on prespecified validation turns. Additional text reuse is tested on-device against ordinary model probabilities and A/B score gradients. Each GPU receipt records the execution mode, forward counts and maximum errors. The separate restoration stage loads exact archived weights in a fresh Qwen and scores actual and held-out inputs without training. The reported uncertainty unit is the environment seed, with one assigned first-fatal opportunity per individual. The healthy noninferiority margin was set to two absolute probability points before model results.
