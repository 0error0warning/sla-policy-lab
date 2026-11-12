# Doubling body-damage feedback at a fixed personal reference

[中文实验报告](../../artifacts/model-studies/2026-11-12/damage-penalty/REPORT.md) · [Precommitted design](PROTOCOL.md)

Qwen3.5-4B selects one of four semantic actions. Each individual has its own continuous, finite body, partner encounters and private final-block LoRA. Both learners use the same causal similar-health baseline in original physical feedback units. The single experimental coefficient weights the current loss of reference remaining life. With raw damage cost D and previous baseline b the two training signals are −D−b and −2D−b. Physical wear, health sensing, partner maintenance, the health window and the SGD rate 0.18 are shared.

### Next decision on the same experiences

The source is all 48 previous local_true lives. In `output/replays/s<seed>/<birth>/`, an entry at t contains the observed action, both feedback values and signals, propensity correction, actual FP32 step sizes, and parameter hashes. At t+1 the five saved probability rows are 1× before its previous step, 1× after, 2× before its previous step on its own path, 2× after, and a temporary 2× step from that exact preceding 1× origin and gradient. A separate duplicate verifies the deployment row on the GPU. Columns are wait, ask, help and trade, with the last two completing a task. Compare the rows of the **same successor**, using the same body, next request and draw. A differing predicted action does not alter the recorded tape.

### Freely acting lives

Every new seed has two independent parameter origins and five arms per origin. The primary 2× learner's first-fatal checkpoint has a birth row, deployed row, same-tape 2× wrong-action row, same-tape 1× correct row, practice-entry row and an exact deployed duplicate. The 1× learner reverses the two doses. Autonomous arms can reach the fatal threshold at different encounters. Archived body trajectories end once, at natural death. The health-90/300 probes are read-only. Real healthy task completion counts a surviving answer when pre-action health is at least 90; language output is rendered from a fixed action template.

### Reproduce

Use the repository root with Python 3.11+ and NumPy. These commands validate raw input and output archives, chronology, optimizer scalars and the physics, rebuild the captured FP32 witness parameters and reproduce the signed next-answer comparisons, first-fatal choices and healthy task results. The plot additionally uses Matplotlib.

```sh
python -m experiments.damage_penalty.reproduce --artifact artifacts/model-studies/2026-11-12/damage-penalty --parent artifacts/model-studies/2026-11-12/health-baseline --destination verified-damage
python -m experiments.damage_penalty.display verified-damage/analysis verified-damage/decision-effects.png
```

To repeat a model stage, use its own verified input directory and GPU requirements with `python -m experiments.damage_penalty.gpu --root <stage-inputs> --mode <stage>`. Stages are `dpreplay` (original histories), `dppilot` and `dprep` (new free lives), and `dpcheck` (independent repeated histories followed by full-from-token reinstalls with zero additional updates to those snapshots). Each receipt includes numerical comparisons, model revision evidence and unchanged-backbone checks. The effective parameter change can be limited by the existing norm cap even at a fixed learning rate.
