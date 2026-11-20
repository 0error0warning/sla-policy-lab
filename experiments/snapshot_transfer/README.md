# 单次 LoRA 更新能否迁移到下一条请求

[中文结果与逐例数据](../../artifacts/model-studies/2026-11-13/snapshot-transfer/REPORT.md) · [测量前确定的设计](PROTOCOL.md)

The 48 cases are the two birth states on each of 24 source seeds. Reconstruct the first surviving risky answer and the following request from the PR #12 taped 1× behavior policy, then reinstall the conditional learner’s exact FP32 pre/post update arrays. The three primary contexts score the original feature set, replace only pre-action health with post-injury health, then replace the complete request with its actual successor. The other three forward passes start at post-injury health and substitute only the successor partner, incoming frame/text or action order. The credited action is kept in its original semantic slot for every measurement.

The summary's raw changes are post minus pre, in percentage points. Signal-aligned changes multiply the raw change by the actual training signal's sign. Intervals are stratified, paired bootstrap summaries over independent environment seeds. All 48 witnesses had negative feedback. The auxiliary FP32 head uses the native final hidden state and the same four token IDs, temperature and exploration mixture. The main policy uses the original Qwen output head.

From the repository root, reproduce the input metadata, all adapter words, original anchors and the published CPU tables into a new destination. The GPU archives include the pinned model manifest, run lock, ordinary-model probabilities, native raw logits for the four actions, and the read-only numerical checks.

```sh
python -m experiments.snapshot_transfer.reproduce --artifact artifacts/model-studies/2026-11-13/snapshot-transfer --dest verified-transfer
python -m experiments.snapshot_transfer.display verified-transfer/analysis verified-transfer/contexts.png
```

To repeat the GPU forward passes with the original bundle on a BF16-capable CUDA device, unpack `gpu/stmap/inputs.zip` into a fresh directory, install its `requirements-gpu.txt` with the PyTorch version recorded in the GPU receipt, then run `python -m experiments.snapshot_transfer.gpu --mode stmap --root .` from that directory. The worker pins both model shard hashes, checks restored FP32 parameters before and after, and writes the native and FP32 diagnostic outputs in `output/`. No optimizer is instantiated.
