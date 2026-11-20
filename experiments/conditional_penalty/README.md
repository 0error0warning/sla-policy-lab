# Double the damage penalty only in low-health encounters

[中文研究报告](../../artifacts/model-studies/2026-11-13/conditional-penalty/REPORT.md) · [Locked protocol](PROTOCOL.md)

When the living agent starts an encounter at health 6 or below, the conditional learner weights the realized decrease in reference remaining life by two. It uses one elsewhere. Full 1× and full 2× run alongside it. The private past-only similar-health baseline stays in original physical units for every arm and all use SGD 0.18. Health never resets. Under the paired environment the conditional and full-1× individuals have identical healthy histories by construction. The audit checks both correct and wrong credit copies up to and including the first low-health action before feedback.

### Same experiences

The two paired measurements in `output/gated/replays` and `output/constant/replays` process identical parent actions and outcomes for each seed and birth. The 1× reference can be matched using every parameter digest. Within each pair the five stored probability rows are the pre-update reference, deployed 1×, pre-update alternate, deployed alternate, and a single alternate step from the pre-update 1× origin. The sixth deployed duplicate checks the raw computation. Score differences between rows of the **same next living request**. The preceding encounter’s starting health selects its dose. In the conditional pair the one-step fork equals either the full-1× or full-2× fork, while the lifelong adapters can differ because of earlier learning.

### Free lives

Each seed supplies two births and seven private policies with frozen, correct and mismatched credit. At the actual first fatal request on the conditional correct individual, the six rows are birth, deployed, matched wrong, matched 1×, matched 2× and deployed duplicate. In its parameter archive `first_other` and `first_extra` are the two matched constants. The practice-entry weights have their own saved key. Constant policies retain their PR #11 snapshot layout. The held-out panel crosses both partner identities, both frames and all 24 action orders. Actual healthy task completion counts a surviving answer at pre-action health at least 90. Task text is rendered from the selected semantic action.

### Reproduction

From the repository root with Python 3.11+ and NumPy, unpack into a new directory. The command verifies the original transports, rebuilds the published metrics and witness weights, and compares bytes. Split transports include checksums per part and the SHA-256 of the assembled archive. The optional figure uses Matplotlib.

```sh
python -m experiments.conditional_penalty.reproduce --artifact artifacts/model-studies/2026-11-13/conditional-penalty --parent artifacts/model-studies/2026-11-12/damage-penalty --destination verified-conditional
python -m experiments.conditional_penalty.display verified-conditional/analysis verified-conditional/decisions.png
```

The CUDA entry point for a stage's own unpacked inputs is `python -m experiments.conditional_penalty.gpu --root <stage-inputs> --mode <stage>`. Stages are `gpreplay`, `gppilot`, `gprep` and `gpcheck`. The last stage repeats both paired training runs from birth, then scores saved parameters on an ordinary complete Qwen. Each model receipt records the native-versus-complete calibration and verifies the frozen backbone. Counts of repeated histories, parameter snapshots or rounds do not increase the number of independent seeds.
