# Testing whether a single injury transfers across request wordings

[中文报告](../../artifacts/model-studies/2026-11-13/paraphrase-transfer/REPORT.md) · [事先确定的比较](PROTOCOL.md) · [逐字训练与留出材料](wording.json)

Each matched pair starts with the same archived Qwen3.5-4B adapter from PR #15. The original-only arm repeats the four existing option orders four times. The other arm crosses the same four orders with four wording variants in the same social frame and with the same partner. One recorded negative feedback is used once; the two parameter displacements have the same L2 budget. The candidate action always uses its semantic token ID.

At the actual post-injury health, evaluate each held wording at all 24 orders for the original partner and again for the opposite partner. The primary actual successor retains its logged partner, frame, order and health. Summaries show same-partner and changed-partner successors separately, with their own seed counts and the two frame-change subgroups. The observed groups overlap at the seed level. Healthy trajectories freeze the pre-update and two trained snapshots for evaluation and preserve PR #15's same fresh exogenous arrivals. Task completion counts a sampled answer with survival under fixed rendering.

From the repository root, use Python, NumPy and SciPy to verify the original transport, recover the weights, wording bank, observations and reference rollouts from the parent, and rebuild the analysis. The optional plot uses Matplotlib.

```sh
python -m experiments.paraphrase_transfer.reproduce --artifact artifacts/model-studies/2026-11-13/paraphrase-transfer --dest verified-paraphrase
python -m experiments.paraphrase_transfer.display verified-paraphrase/analysis verified-paraphrase/transfer.png
```

To repeat the pinned GPU experiment, unpack its `gpu/phrmix/inputs.zip` into a new directory. From that directory, with the recorded CUDA/PyTorch environment and `requirements-gpu.txt`, run `python -m experiments.paraphrase_transfer.gpu --root . --mode phrmix`. The run bundle includes the exact wording file and both source parameter archives. Saved probability rows are pre-update, original wording with four orders, paraphrase-augmented with four orders, and the archived PR #15 control. For healthy panels and trajectories the first three are evaluated in the analysis. The per-word and per-order measurements are repeated evaluations of the same 48 updates.

For the independent allocation, unpack its own input zip and run the same entry point with --mode phrcheck. The numerical check measures the unmodified parent scoring interface, then the same interface after registering the new text and after prefix calibration. The independent input bundle also reinstalls the old and contemporary original-wording controls in their PR #15 slot position. The main analysis is in analysis/, the second allocation in analysis/repeat/, and the cross-run comparison in analysis/stability.json. Both allocations use the same 48 experiences. See the [repeat design](REPEAT.md).
