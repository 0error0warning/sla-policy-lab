# Personal reward baselines in a finite Qwen life

[锁定方案](PROTOCOL.md) · [中文结果与原始数据](../../artifacts/model-studies/2026-11-12/health-baseline/REPORT.md)

The two learners differ only in the personal reward baseline used to center a realized outcome. A health value outside the past-only window cannot enter the local mean. The 1/35 figure is a measured ratio of median update norms for a specific exposure, with the original learning rate, ridge, propensity clipping and step cap.

### Recompute from the release

From the repository root, use Python 3.11+ and NumPy. Verification unpacks into a new directory and compares the rebuilt analyses and the focal GPU inputs, including FP32 adapter snapshots, byte for byte against the archives.

```sh
python -m experiments.health_baseline.reproduce --artifact artifacts/model-studies/2026-11-12/health-baseline --parent artifacts/model-studies/2026-11-12/first-fatal --destination checked-health-baseline
```

Each stage contains the pinned model hashes, source files, versions and raw export. Its GPU entry point is `python -m experiments.health_baseline.gpu --root <unpacked-stage-inputs> --mode <stage>` with the corresponding CUDA/Python dependencies in the input package. The original weights are downloaded by the pinned loader.

### Follow one next choice

In `hbreplay/output/replays/s<seed>/<birth>/`, the event at time *t* stores the graded action, both pre-outcome baselines and the actual FP32 update norms. The event at *t+1* stores the four semantic probabilities for each alternative set of weights, before the next observed outcome is used. The rows are pre-global, updated global, pre-local, updated local and a single local step from the pre-global weights. A sixth duplicate checks the deployed global computation. The original action is used to continue the observed tape even when a counterfactual draw differs. The last update has no living successor.

The replay CSV enumerates all 680 practice-answer successors with the common uniform and action at each fork. Total variation measures a probability difference. The shared-uniform disagreement instead counts the length of all uniform intervals that would actually change the sampled semantic action. A switch between two expressions can leave the answering decision unchanged.

### Follow one independent life

In `{hbpilot,hbrepa,hbrepb}/output/lives/s<seed>/<birth>/<rule>/` the agent chooses freely, keeps its body and parameters until death, and sees its current health. The first-fatal panel order is birth, deployed, wrong-credit shadow, other-baseline correct shadow, practice-entry snapshot and deployed duplicate. Unused shadow slots in frozen and independently acting wrong-credit lives point to birth. Probability columns in semantic order are wait (decline), ask (decline with appeal), help (answer with appeal), trade (answer with exchange). The actual health and the read-only healthy counterfactuals 90 and 300 use the identical first-fatal checkpoint. Actual healthy completion uses all living states with health at least 90. A completed task is an answer action with the study’s fixed language renderer.

The other-baseline shadow sees the deployed individual’s actions and consequences. The separately acting other-baseline individual can receive different future outcomes. Both comparisons are present in the seed-level table and their interpretation differs.

`hbrepeat` retrains the two replay paths from birth on the same source tapes on another assignment. `hbverify` reinstalls weights from the first run and performs ordinary full Qwen forwards with zero training. Neither adds independent environmental seeds.

A service interruption during replication left a verified complete-seed backup. Its original assignment and the retry use identical input-package hashes. The release preserves the excluded backup under `interruptions/`. The reproduction command also compares its freely acting lives with the retry and counts the seed once in the primary analysis.
