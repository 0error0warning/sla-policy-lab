# One observed injury, four views, one update

[中文结果与逐例数据](../../artifacts/model-studies/2026-11-13/order-transfer/REPORT.md) · [Protocol locked before model outputs](PROTOCOL.md)

The original-only and balanced-order arms use the same 48 pre-update snapshots and real injury outcomes as PR #14. Each takes four gradient evaluations and one parameter step. The balanced views are the four cyclic rotations of the original action definitions. Inputs never use the actual next request to select rotations. Both arms receive the realized signed signal and factual propensity weight. The averaged rotation gradient uses the original update's L2 budget. All 20 out-of-update layouts, the factual successor, and healthy probes share the same finished snapshots. The released gradients’ norms, parameter bits and probabilities allow the comparisons to be verified per case.

Healthy rollouts are independent evaluation copies of a body at 350. Three copies per case carry frozen pre-, original-order and four-order adapters. They see the same fresh per-turn random inputs until each body crosses below 90. The rate counts actual surviving answers over all its healthy decisions. The separate reserved-text panel reports expected successful task completion at 90 and 300, crossed with both partners, both request frames and 24 layouts.

The analysis requires Python 3.10+, NumPy and SciPy (for bounded-normal body perturbations). Matplotlib is needed for the figure. Run from the repository root with a new destination.

```sh
python -m experiments.order_transfer.reproduce --artifact artifacts/model-studies/2026-11-13/order-transfer --dest verified-order-transfer
python -m experiments.order_transfer.display verified-order-transfer/analysis verified-order-transfer/transfer.png
```

The GPU worker can be run from the unpacked stage's `inputs.zip` in the recorded CUDA environment using `python -m experiments.order_transfer.gpu --root . --mode ordmix`. It pins Qwen's two model shard hashes, installs each private adapter on the native terminal projection and obtains all training gradients using ordinary full-model forwards. The checked readout caches only tensors upstream of that learned intervention. Its receipt records validation or the full-model fallback. In the raw probability archive the rows are frozen pre-update, repeated original, four orders, and PR #12's independently saved original-order update.
