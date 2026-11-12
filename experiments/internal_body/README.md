# Internal body-conditioned Qwen experiments

PR #5 at commit 9087c39 remains the output-adapter reference. This study directly intervenes in Qwen3.5-4B decoder blocks, tests an original RMSNorm parameter, and compares personal learning rules with unseen partner identities. The simulator and four actions are inherited from the reference. The [research report](../../artifacts/model-studies/2026-11-12/internal-body/REPORT.md) distinguishes sampled feedback, a fitted suffix, a gradient approximation and actual complete-model deployment.

## Read the data

From the repository root, verify and unpack to a new directory. Unpacking requires NumPy. CPU research scripts also use SciPy and PyTorch. The reference Windows execution used NumPy 2.2.6, SciPy 1.16.2 and PyTorch 2.14.1+cpu. GPU source packages contain package pins, input hashes and pinned Qwen shard hashes. Each immutable source package reproduces the precise stage it accompanies.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
python -m experiments.internal_body.release --artifact artifacts/model-studies/2026-11-12/internal-body --destination RUN
python -m experiments.internal_body.mathcheck
python -m experiments.internal_body.evidence --root RUN --gpu RUN/gpu/fullverify/output --output RUN/recomputed-evidence.json
```

Each seed archive contains its scientific inputs, raw observed choices and consequences, loss history and parameter checkpoints. The larger individual matrices use lossless integer prediction with exact stored XOR residuals. The extractor checks the SHA-256 of every reconstructed array. Direct GPU archives have an additional verified member manifest. [Data dictionary](../../artifacts/model-studies/2026-11-12/internal-body/DATA.md)

## Refit the experiments

Use a new fit directory to keep the published results intact. Run the pilot before confirmation where a prospective gate applies.

```powershell
python -m experiments.internal_body.population --cache RUN/gpu/capture2/output --output FIT/population --wave pilot
python -m experiments.internal_body.population --cache RUN/gpu/capture2/output --output FIT/population --wave confirm
python -m experiments.internal_body.lives --cache RUN/gpu/capture2/output --output FIT/population --wave pilot
python -m experiments.internal_body.lives --cache RUN/gpu/capture2/output --output FIT/population --wave confirm
python -m experiments.internal_body.carrier --cache RUN/gpu/capture2/output --output FIT/native --wave pilot
python -m experiments.internal_body.carrier --cache RUN/gpu/capture2/output --output FIT/native --wave confirm
python -m experiments.internal_body.credit --cache RUN/gpu/capture2/output --output FIT/credit --wave pilot
python -m experiments.internal_body.credit --cache RUN/gpu/capture2/output --output FIT/credit --wave confirm
python -m experiments.internal_body.budget --cache RUN/gpu/capture2/output --output FIT/budget
python -m experiments.internal_body.probe --cache RUN/gpu/capture2/output --output FIT/prompt-information.json
```

For a new allocated GPU, unpack a stage's inputs.zip into a fresh directory, install its requirements-gpu.txt, set CUBLAS_WORKSPACE_CONFIG to :4096:8, and run `python -m experiments.internal_body.gpu --root . --mode STAGE` in that directory using the stage name from the run lock. No service account, personal credential or Colab assignment is included in the release.

The [population protocol](PROTOCOL.md), [direct depth comparison](DEPTH.md), [single-life protocol](LIFETIMES.md), [feedback-credit follow-up](CREDIT.md), [original-weight sensor experiment](NATIVE_SENSOR.md), [batch budget diagnosis](BUDGET.md), [location follow-up](PLACEMENT.md) and [physical execution checks](FULL.md) document which observations motivated each follow-up.

## Verify the physical and behavioral records

The full-input layer-3/15/31 derivative study used a first-order approximation only during training. Its primary numbers use the actual reinstalled Qwen checkpoints. The training-fit audit reads the original microbatches. Fresh singleton inference reconstructs BF16 additions and is checked against complete-model forward passes. The reproducible source package for each GPU job is the inputs archive recorded for that stage.

```powershell
python -m experiments.internal_body.nonfatal --cache RUN/gpu/capture2/output --output FIT/nonfatal --wave pilot
python -m experiments.internal_body.placement --capture RUN/gpu/depthpilot2b/output/jacobians --output FIT/placement
python -m experiments.internal_body.routing --cache RUN/gpu/capture2/output --credit RUN/credit --output FIT/routing
python -m experiments.internal_body.audit --root RUN --cache RUN/gpu/capture2/output --gpu RUN/gpu/fullverify/output --placement-capture RUN/gpu/depthpilot2b/output/jacobians --output CHECK
python -m experiments.internal_body.refit --cache RUN/gpu/capture2/output --population RUN/population --native RUN/native --placement RUN/placement --jacobians RUN/gpu/depthpilot2b/output/jacobians --output CHECK/refit.json
python -m experiments.internal_body.depth_analysis --depth RUN/gpu/depthpilot1/output RUN/gpu/depthpilot2b/output --trained RUN/placement --transplant RUN/gpu/transplant/output --output CHECK/depth.json
python -m experiments.internal_body.direct_audit --depth RUN/gpu/depthpilot1/output RUN/gpu/depthpilot2b/output --seen RUN/gpu/seenfit/output --output CHECK/direct-audit.json
python -m experiments.internal_body.quantized --cache RUN/gpu/capture2/output --fresh RUN/gpu/transplant/output/fresh-tail --population RUN/population --native RUN/native --credit RUN/credit --output CHECK
```

The [nonfatal coverage screen](NONFATAL.md) and [capacity-matched identity comparison](ROUTING.md) keep the treatment and comparison explicit. The [validation record](../../artifacts/model-studies/2026-11-12/internal-body/VALIDATION.json) reports both behavioral gates and numerical execution tolerances.
