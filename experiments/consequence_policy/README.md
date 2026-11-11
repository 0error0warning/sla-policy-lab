# Consequence-conditioned Qwen action study

Qwen3.5-4B selects one of four native output words. A small parameter bank inside the vocabulary projection reads the frozen Qwen hidden state, the measured health sensor and a known partner ID. Fitting uses the Qwen behavior policy's sampled actions and the reference-life change that the simulator actually settles. Full-model GPU checks install this projection in Qwen, compute its complete vocabulary logits and update two personal torch parameters after the chosen actions.

The Chinese research report includes the original negative results, the coverage trade-off, seed uncertainty, lifetime transfer, numerical checks and limitations. [Report and data](../../artifacts/model-studies/2026-11-12/consequence-policy/REPORT.md)

## Reproduce from the repository root

The following commands use a new directory named RUN. The artifact includes the individual Qwen token prefixes and states, chosen actions, propensities, partner outcomes, realized rewards, all fitted adapter checkpoints, personal updates, sampled lifetime traces and the exact source packages sent to each GPU. GPU weight shards are pinned in the archived manifest. Use the supplied acquisition source archive when byte-identical input hashes are required. CPU computations below use NumPy and SciPy; the figure additionally uses Matplotlib.

```powershell
python -m pip install -r experiments/consequence_policy/requirements-cpu.txt
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -m experiments.consequence_policy.unpack --artifact artifacts/model-studies/2026-11-12/consequence-policy --destination RUN
python -m experiments.consequence_policy.audit --cache RUN/capture/output --output RUN/results
python -m experiments.consequence_policy.audit --cache RUN/capture/output --output RUN/results --gpu RUN/verify/output
python -m experiments.consequence_policy.numerics --cache RUN/capture/output --gpu RUN/verify/output --third RUN/numerics/output --output RUN/results
```

Each seed archive contains its population and coverage records or its complete lives, depending on the cohort. The raw layout and arrays are documented in [the data index](../../artifacts/model-studies/2026-11-12/consequence-policy/DATA.md).

## Refit or deploy

Use a separate output directory (here NEW) to refit the population pilot before confirmation. The projection is learned from unlabeled training prompts only. For paired coverage studies, keep that same projection and run pilot then confirm for each allocation. The wide allocation retains its failed confirmation in the release. The split allocation has a separate pilot and 20 new confirmation seeds.

```powershell
python -m experiments.consequence_policy.experiment --cache RUN/capture/output --output NEW --wave pilot
python -m experiments.consequence_policy.experiment --cache RUN/capture/output --output NEW --wave confirm
python -m experiments.consequence_policy.lifetime --cache RUN/capture/output --output NEW --wave confirm
python -m experiments.consequence_policy.coverage --cache RUN/capture/output --output NEW --wave pilot
python -m experiments.consequence_policy.coverage --cache RUN/capture/output --output NEW --wave confirm
python -m experiments.consequence_policy.coverage --cache RUN/capture/output --output NEW --allocation split --wave pilot
python -m experiments.consequence_policy.coverage --cache RUN/capture/output --output NEW --allocation split --wave confirm
python -m experiments.consequence_policy.lifetime --cache RUN/capture/output --output NEW --wave confirm --fresh-prefixes RUN/verify/output/independent-test-prefixes.npz --life-output NEW/singleton-lives
```

For a new GPU execution, extract the desired `gpu/*-inputs.zip` into a fresh directory. Its `run-lock.json` pins the included source and assets. From that directory, install its `requirements-gpu.txt`, set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and run `python -m experiments.consequence_policy.gpu --root . --mode capture`, replacing capture with verify, numerics or ordering for the corresponding package. Each GPU job's result archive carries an independently checked transport receipt and a member-by-member hash manifest.

Inspect the decision and training specifications in [the prospective population protocol](PROTOCOL.md), [lifetime protocol](LIFETIME_PROTOCOL.md), [paired coverage protocol](COVERAGE_PROTOCOL.md), [GPU protocol](NUMERICS_PROTOCOL.md) and [follow-up protocol](FOLLOWUP_PROTOCOL.md).
