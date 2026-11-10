# Frozen-model evidence — 2026-11-10

This is a curated public copy of two standalone frozen-model studies against environment commit `71e69b697790310bb45079603bf92f948e11852c`. The measured raw inputs and model outputs are byte-identical to their local originals. Reports have account information and local paths removed. Original source hashes and published hashes are recorded in [source-map.json](source-map.json); [SHA256SUMS.json](SHA256SUMS.json) covers this public archive.

No pretrained weights or tokenizer binaries are vendored. No LoRA, activation steering, or long-term reward training was performed. The default web demo retains its CPU teaching implementation.

| Directory | Contents |
|---|---|
| [frozen-v1](frozen-v1/README.md) | Original three-model screen: 48 questions and 8 lives per model |
| [diagnostic-v2](diagnostic-v2/README.md) | Four diagnostic stages, separate thinking condition, eight new paired seeds, and four rule policies |
| [environment](environment/sla_demo) | Exact finite-life environment/protocol snapshot used for replay |

[publication.json](publication.json) records scope, compute consumption, release status, and the interrupted run. Account identifiers, credit balances, runtime endpoints, authorization wrappers, local filesystem paths, redundant ZIPs, and host checkpoint copies stay outside this public archive. CPU rule trajectories, model prompts, generated token IDs, final channel text, model revisions, and weight SHA-256 values are retained.

The initial diagnostic instance ended unexpectedly. Complete diagnostic question outputs and two thinking outputs survived locally; full rollout records did not. A complete recovery repeated the **same** frozen matrix, using foreground execution and incremental local saves. The preserved 96 question generations and two thinking sequences are identical to the recovery. Only the complete recovery contributes to the diagnostic scores; the interrupted attempt is not pooled as independent samples.

The reports describe verification and account receipts available at measurement time. Their references to local `run-manifest.json`, `gpu-release-receipt.json`, or local ZIPs correspond to the curated public summary in `publication.json`; private originals are not copied wholesale.

## Offline verification

From the repository root:

```sh
python experiments/frozen_models/verify.py
```

The command checks the archive hashes, numerical answers, recorded scores and complete transitions of 88 model lives plus 128 rule lives. An optional local-tokenizer audit independently re-decodes all 898 formal model calls; see [instructions](../../../experiments/frozen_models/README.md).

Input manifests marked `model_calls_so_far: 0` or “not executed” are **execution-before snapshots**, retained rather than rewritten after results were seen. The reports, measured raw files, result summaries, and this publication record describe completion. SHA-256 detects accidental changes and links provenance; it does not authenticate a publisher by itself.
