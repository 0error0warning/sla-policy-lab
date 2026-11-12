# Qwen transfer across a bodily risk boundary

The [Chinese research report](../../artifacts/model-studies/2026-11-12/lethal-boundary/REPORT.md) gives behavioral results for real Qwen3.5-4B with the same last-block adapter as PR #6. Numerical health enters the adapter. A separate input condition shows a past maintenance record. Get the design, primary contrasts and criteria from the [prospective protocol](PROTOCOL.md). The [continuous-exposure experiment](CONTINUOUS.md) was locked after the four-seed pilot, before inspecting confirmation and before sampling its own 12 seeds.

## Reproduce the published measurements

Use Python with NumPy, SciPy and PyTorch. The Windows CPU analysis used NumPy 2.2.6, SciPy 1.16.2 and PyTorch 2.14.1+cpu. Matplotlib is used for the two scientific figures. Start from the repository root and unpack into a new directory. The release verifies the outer manifest, individual seed members, complete GPU exports and their exact input packages.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
python -m experiments.lethal_boundary.release --artifact artifacts/model-studies/2026-11-12/lethal-boundary --destination RUN
python -m experiments.lethal_boundary.evaluate --root RUN/core --wave confirm --output CHECK/confirm
python -m experiments.lethal_boundary.evaluate --root RUN/core --wave pilot --output CHECK/pilot
python -m experiments.lethal_boundary.transfer --root RUN/core --study RUN/core --analysis CHECK/confirm --output CHECK/lookup
python -m experiments.lethal_boundary.transfer --root RUN/core --study RUN/jitter --analysis CHECK/confirm --output CHECK/jitter --jitter
python -m experiments.lethal_boundary.audit --root RUN/core --cache RUN/gpu/boundarycapture/output --output CHECK/audit
python -m experiments.lethal_boundary.verify_fits --core RUN/core --jitter RUN/jitter --cache RUN/gpu/boundarycapture/output --arch RUN/gpu/boundarycapture/inputs --output CHECK/audit
python -m experiments.lethal_boundary.diagnose --root RUN/core --analysis CHECK/confirm --output CHECK/audit
python -m experiments.lethal_boundary.hardware_check --gpu RUN/gpu/boundaryaudit/output --arch RUN/gpu/boundarycapture/inputs --trained RUN/core --output CHECK/hardware
python -m experiments.lethal_boundary.transition_check --root RUN/transition --cache RUN/gpu/boundarycapture/output --arch RUN/gpu/boundarycapture/inputs --reference CHECK/confirm --output CHECK/transition
python -m experiments.lethal_boundary.transition_hardware --gpu RUN/gpu/boundarytransition/output --cache RUN/gpu/boundarycapture/output --arch RUN/gpu/boundarycapture/inputs --study RUN/transition --ref CHECK/confirm --output CHECK/transition-hardware
python -m experiments.lethal_boundary.panel_numerics --gpu RUN/gpu/boundarytransition/output --cache RUN/gpu/boundarycapture/output --arch RUN/gpu/boundarycapture/inputs --study RUN/transition --output CHECK/transition-hardware
python -m experiments.lethal_boundary.figures --analysis CHECK/confirm --transition CHECK/transition --output CHECK/figures
```

To repeat the actual 75-step fits use a fresh output directory. These commands reuse the pinned Qwen prefix tensors measured on L4. Independent installation stages ran the complete Qwen model on a second and a third L4. For a GPU rerun use the stage's extracted `inputs` directory, install its `requirements-gpu.txt`, configure `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and run the exact packaged Python source with `python -m experiments.lethal_boundary.gpu --root . --mode boundarycapture` or the installation modes `boundaryaudit` and `boundarytransition`. Base shards are fetched at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and verified against the included SHA-256 manifest.

```powershell
python -m experiments.lethal_boundary.experiment --cache RUN/gpu/boundarycapture/output --arch RUN/gpu/boundarycapture/inputs --output FIT/core --wave pilot
python -m experiments.lethal_boundary.experiment --cache RUN/gpu/boundarycapture/output --arch RUN/gpu/boundarycapture/inputs --output FIT/core --wave confirm
python -m experiments.lethal_boundary.jitter --cache RUN/gpu/boundarycapture/output --arch RUN/gpu/boundarycapture/inputs --output FIT/jitter
python -m experiments.lethal_boundary.transition --cache RUN/gpu/boundarycapture/output --arch RUN/gpu/boundarycapture/inputs --output FIT/transition
```

## Data dictionary

- `core/sSEED/tape.npz` contains the chosen semantic action, four logged propensities, maintenance uniform, five body noise draws, assigned partner profile and an independent 20-check-in record per profile. Actions are silent refusal, refusal with appeal, answer with appeal, and answer with exchange.
- Under each seed, `m0` and `m1` are paired mirror ecologies. `outcomes.npz` stores actual response, subsequent health and realized change in reference life. The fixed mixed arm selects D or R by slot as recorded by `levels()`. This reconstructs its outcomes exactly. `steps.json` contains loss, observed gain, KL and gradient norm at each update.
- Each fitted checkpoint contains A (2×12×33×4) and B (4×2560). Each probe contains the mean of four native action probabilities within each partner/frame combination, on every health and partner count, plus a fixed-health-gate control. A separate array retains all 192 held-out requests individually at eight witness healths and five counts. The complete per-request dense grid can be regenerated from the states and weights. `core/axes.json` specifies axes and action ordering.
- `jitter/sSEED/tape.npz` additionally stores the two 8×768 current-health matrices. Each health belongs to a state with eight outcome replicas. Arrays named `fixed` and `jitter` distinguish precise physical/sensor interventions. The four mixed-sensor and mixed-credit cells are defined in `jitter.py`.
- `gpu/boundarycapture/output` contains 6,528 pinned Qwen prefix measurements and their token IDs. Stored uint16 residuals, normalization outputs, skips and down-projection outputs are the original BF16 bit patterns. The 33-dimensional projected input and four native logits are FP32 arrays. `gpu/boundaryaudit/output` contains every one of the 6,912 direct complete-model decisions and fresh components for an independent replay. The later boundarytransition GPU stage stores 12,288 complete-model decisions at all 16 held-out transition healths. Each GPU job includes the precise executed source in its immutable input package.
- `analysis/confirm-analysis/seed-surfaces.npz` stores each confirmation seed before pooling. Axes are independent seed, true maintenance probability, test health and metric. Its compressed CSV gives every condition's mean. The [plain mixed-policy CSV](../../artifacts/model-studies/2026-11-12/lethal-boundary/mixed-policy-by-health-and-partner.csv) additionally gives pointwise seed intervals. `lookup`, `jitter-analysis`, `audit` and `hardware-analysis` contain their own diagnostics. All rates in machine-readable tables are fractions.

The [stochastic-boundary follow-up](TRANSITION.md) uses 12 disjoint seeds. Under `transition/sSEED`, tapes retain both per-state health assignments. The four fitted arms share observed actions and responses. The compressed outcome files additionally store the exact permutation of normalized continuation values for the stratum-matched credit control. A separate complete-model panel verifies these small effects on the four specified source seeds and all sixteen held-out transition healths.
