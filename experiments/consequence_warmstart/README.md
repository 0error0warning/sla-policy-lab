# CPU warmstart initialization control

This adds one condition to the archived consequence-policy implementation: `warmstart_adapter` preserves the population mean, zeros its policy-contrast direction, and runs the same coefficient-update branch as `scratch`. The original arms and archives are unchanged. See the [locked protocol](PROTOCOL.md) and [results](../../artifacts/model-studies/2026-11-14/warmstart-adapter/REPORT.md).

The restored 96 lives matched exactly. The fixed 128-life CPU development matrix completed, with 9,441 physical encounters. Warm initialization retained the birth body policy and supported personal expression learning. Its improvement over frozen birth was 9.48pp on the held-out task and 7.44pp on same-task held-out wording, below the preregistered 10pp development threshold. The joint gate failed; no GPU, thinking, new model, extra seed, or tuning run followed. Four independent clusters do not establish confirmatory reliability. Healthy expression adaptation and additional low-health risk learning are separate results.

## Reproduce using the existing published prefixes

Run from the repository root. Python 3.13.5 / NumPy 2.2.6 / SciPy 1.16.2, single-threaded BLAS, were used for the recorded CPU run. Compatible dependencies are in the original `requirements-cpu.txt`; other versions are not asserted bitwise identical. New output paths must not already exist.

```powershell
python -m pip install -r experiments/consequence_policy/requirements-cpu.txt
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -m experiments.consequence_policy.unpack --artifact artifacts/model-studies/2026-11-12/consequence-policy --destination SOURCE
python -m experiments.consequence_warmstart.runner --source SOURCE --out NEW --lock artifacts/model-studies/2026-11-14/warmstart-adapter/preregistered/lock.json
python -m experiments.consequence_warmstart.analysis --run NEW
python -m experiments.consequence_warmstart.audit --source SOURCE --run NEW
```

`runner` verifies source/code locks, then restores all 96 old pilot lives before starting any new condition. It refuses on a restoration discrepancy or time budget failure. Its output contains independent autonomous trajectories, action probabilities before/after updates, checkpoint weights, both probe context sets, and each prefix-specific feedback permutation. `last_preterminal` excludes terminal feedback; `terminal` is separately labeled. Replay is retrospective off-policy sensitivity analysis with recipient gradients, not extra life performance.

To check the delivered result archives instead of regenerating lives:

```powershell
python -m experiments.consequence_warmstart.replay --artifact artifacts/model-studies/2026-11-14/warmstart-adapter --out REPLAY
python -m experiments.consequence_warmstart.analysis --run REPLAY
python -m experiments.consequence_warmstart.audit --source SOURCE --run REPLAY
```

All models use the same archived batched prefix capture. This CPU work does not repeat the earlier singleton/full-Qwen GPU audit. The `wording` block keeps task identity fixed but uses language already seen in population fitting. The `legacy` block changes task content at evaluation and is labeled accordingly. Environment mappings are not preferred-expression labels: 00/11 have opposite partner expression preferences, 01/10 have equal preferences for a fixed incoming frame.
