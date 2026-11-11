# Direct Qwen actions learned from observed bodily consequences

This study replaces the previous response-predictor role with a direct four-action Qwen policy. The model's next-token A/B/C/D probabilities determine the action, with a declared temperature and exploration mixture. The environment alone settles the body and partner response. Observed task, maintenance, damage and death consequences produce scalar returns used to update a real Qwen LoRA.

There is no action-value controller selecting on Qwen's behalf, no Bayesian action teacher, and no model-generated health settlement. The base model is frozen; added LoRA matrices participate in Qwen's computation. Prompts contain no dialogue history or outcome counts. Training buffers, optimizer state and a reward baseline still exist outside the model.

See [the frozen protocol](PROTOCOL.md) for the reward definition, two-wave matrix, learning horizon, numerical checks and limitations. This is a short-block clipped policy-gradient experiment inspired by PPO, not a full critic-based PPO system or a reproduction of the original paper's objective. Returns use at most eight actually observed future steps, and all main feedback comes from the actions the policy actually takes.

Prepare separate directories for smoke, pilot and confirm; each output directory is immutable once used. Preparation is CPU-only:

```powershell
python experiments/direct_policy/prepare.py run-smoke --manifest artifacts/model-studies/2026-11-10/frozen-v1/raw/model_0/weights-manifest.json
```

On an explicitly supplied BF16-capable CUDA GPU, install the pinned GPU requirements and set the CUDA determinism workspace before importing Torch. The actual Torch/CUDA versions are recorded by the worker:

```powershell
python -m pip install -r run-smoke/requirements-gpu.txt
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
python run-smoke/run_gpu.py --root run-smoke --stage smoke
```

After inspecting the smoke outputs, run pilot/confirm from their separately prepared directories. Service credentials, account/session state and resource allocation are deliberately outside this code. A launcher should bound execution, preserve failures and verify exported files before releasing the GPU.

CPU reconstruction of exported results. The archives and their export receipts are in the [study artifacts](../../artifacts/model-studies/2026-11-11/direct-policy/REPORT.md). From the repository root, extract each archive into a fresh directory:

```powershell
$study = 'artifacts/model-studies/2026-11-11/direct-policy'
foreach ($stage in 'smoke-v2', 'pilot-v2', 'confirm-v2', 'numerics-v2') {
    python experiments/direct_policy/unpack.py --archive "$study/raw/$stage.zip" --receipt "$study/receipts/$stage.json" --destination "exported-$stage"
}
python experiments/direct_policy/analyze.py --smoke exported-smoke-v2 --root exported-pilot-v2 --root exported-confirm-v2 --numerics exported-numerics-v2 --output analysis
python experiments/direct_policy/summarize.py --analysis analysis/scored.json --output analysis/summary.json
python experiments/direct_policy/figures.py --analysis analysis/scored.json --root exported-pilot-v2 --root exported-confirm-v2 --output figures
```

`unpack.py` checks each archive against its export receipt and internal file manifest. The analysis reconstructs policies from recorded logits, verifies selections and Decimal body settlement, and checks discounted returns, advantage permutations, clipped losses and adapter hashes. GPU forwards are recorded in the archives. `figures.py` uses Matplotlib. The finite-scale gradient check has its own [protocol](NUMERICAL_FOLLOWUP.md). `smoke-v1-failed.zip` is the earlier failed repeatability run and can be extracted explicitly with `--allow-failed`.
