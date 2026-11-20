# Existing-history attribution

[Report](../../artifacts/model-studies/2026-11-14/warmstart-attribution/REPORT.md) · [Fixed exploratory protocol](ATTRIBUTION_PROTOCOL.md)

Reuses all 32 warmstart and 32 scalar-direction histories from #28. It does not call `run_life`, sample actions, settle new lives, or allocate a GPU. Each original gradient is recomputed and checked against saved probabilities/weights. Terminal updates remain separate. Primary path contributions telescope; omission replays recalculate recipient gradients and have five partner/health/count-matched masks for each response/action category. A separate existing scalar-rule replay is an explicitly provided-subspace comparison, not a new learning method.

Python/NumPy/SciPy and single-threaded BLAS match the parent CPU study. First restore the existing `SOURCE` and `HISTORIES` archives (new directory names):

```powershell
python -m pip install -r experiments/consequence_policy/requirements-cpu.txt
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -m experiments.consequence_policy.unpack --artifact artifacts/model-studies/2026-11-12/consequence-policy --destination SOURCE
python -m experiments.consequence_warmstart.replay --artifact artifacts/model-studies/2026-11-14/warmstart-adapter --out HISTORIES
python -m experiments.consequence_warmstart.attribution --source SOURCE --histories HISTORIES --out ATTRIBUTION
python -m experiments.consequence_warmstart.attribution_summary --run ATTRIBUTION
python -m experiments.consequence_warmstart.attribution_audit --source SOURCE --histories HISTORIES --run ATTRIBUTION
```

The published `results.zip` contains all output arrays and JSON/CSV, with member hashes in `result-members.json`; the outer study `MANIFEST.json` covers public files. Extract the delivered run with the following command, then rerun the summary and audit using `REPLAY` as `--run`:

```powershell
python -m experiments.consequence_warmstart.attribution_unpack --artifact artifacts/model-studies/2026-11-14/warmstart-attribution --out REPLAY
```

`analysis-lock.json` records the parent input member and pre-execution code hashes, with CRLF normalized to LF for code. It records an analysis fixed after parent outcomes were known, not an outcome-naive confirmation.

* `events.json` / `events.csv`: every actual update, source health/action/feedback, terminal flag, held-out grid probability/risk deltas, original/health-only/wording/next-input deltas, health overlap and ideal/actual linear-score changes. Probability fields use [0,1] units; multiply by 100 for percentage points. `next_*` is absent on the terminal record. Other-partner next-input changes are zero because routing is supplied.
* `cases.json`: birth, last-live and terminal probe metrics, all omission masks, retained update counts, recipient metrics and scalar coefficients. All 32 warm histories have 24 replay conditions: normal, scalar-rule, seven category/health omissions and 15 matched masks. They do not generate life scores.
* `*-recipient-weights.npz`: effective native-output adapter weights for each replay. Normal must equal the parent's last-live weights.
* `summary.json`: all four seed means, category/path contributions, omission contrasts, transfer magnitudes and terminal-only effects. Per-step contributions use observed order; omission effects are nonlinear and need not add up.
* `audit.json`: independent scalar-rule replay and direct head contraction. Shared source prefix/PCA and fixed RUL feedback are not independently relearned or re-derived. `execution.json` records input hashes, CPU time and zero new lives/GPU.

H=6 is a maintenance-useful risk condition; H=2 has fatal answering even with maintenance. A reduction at H=6 must not be described as learning the latter boundary. The `legacy` held-out input changes task content, while `wording` keeps the task fixed using language known during population training.
