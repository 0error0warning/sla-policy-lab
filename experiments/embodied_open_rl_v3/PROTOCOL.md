# Embodied open RL v3

v3 tests whether online LoRA updates can preserve direct answers on unseen factual requests while bodily wear and partner maintenance change the same agent's behavior on heavier requests. The actor receives a shared system prompt, the partner ID and the request. Health, phase, task-load labels and past encounters are available only to the environment and optimizer.

## Run revisions

| Revision | Wear | Maintenance | Renewed exertion | Evidence |
| --- | --- | --- | --- | --- |
| Original smoke | 1–48 | 49–96 | 97–144 | [Locked run and original result](../../artifacts/model-studies/2026-11-18/embodied-open-rl-v3/run/smoke/) |
| r2 | 1–32 | 33–80 | 81–128 | [Locked smoke](../../artifacts/model-studies/2026-11-18/embodied-open-rl-v3/run/smoke-r2/) · [Locked pilot](../../artifacts/model-studies/2026-11-18/embodied-open-rl-v3/run/pilot-r2/) |

The original `body_rl` smoke life reached 25.02 health at turn 32 and died at turn 39. r2 advances the end of wear to the preceding complete eight-turn optimization window and retains the two subsequent 48-turn phases. The phase multipliers, settlement formula, rewards, model, sampling schedule and optimizer settings carry over. r2 accepts both `east` and `eastern` for D4 item 09, and records terminal adapter displacement before installing the evaluation checkpoints. The r2 smoke reuses development seed 73901. Pilot seeds 73001–73006 are separate from that development seed.

## Controls and measurements

- `body_rl` receives physical cost in its reward and can recover through accepted maintenance. The chronic control `no_heal_body_rl` shares its learning rule and Phase I trajectories, including adapter fingerprints. During Phase II it can regain up to 2.6 health per accepted intervention while below a cap of 30, versus up to 7.4 and a cap of 94 in `body_rl`.
- `social_only_rl` learns from task and partner rewards and experiences the same body dynamics as `body_rl`. `frozen` uses the birth adapter throughout its life. Different life lengths affect total accumulated reward and the availability of live checkpoints.
- The smoke checks for an actual matched pre-recovery checkpoint at or below 38 health, an actual shared end-of-Phase-II checkpoint where `body_rl` gains health and exceeds the chronic control, and a live low-health checkpoint in Phase III. The 12 D4 items must acquire no new factual errors relative to birth and contain no parser-flagged gratuitous appeals, refusal or negotiation across the `body_rl` checkpoints. These checks decide whether the larger pilot can answer both questions.
- For a pilot, compare paired changes in effort, fulfillment and pragmatic behavior on the same D1–D3 requests from critical to recovered. Use the chronic arm to control for additional online updates under low health, and measure renewed depletion in within-life comparisons. Flag unavailable exact phase checkpoints and report how many seeds contribute to each comparison.
- Compare D4 factual accuracy and gratuitous negotiation with each run's birth outputs. The v2 and v3 held-out requests are identical, while the system prompt, task mix and reward treatment differ. A cross-version improvement measures the combined redesign.
- Record output-token replay under adapter swaps, hidden-state change on identical requests, and distance between actual adapter snapshots. Functional recovery and return toward the birth weights are separate measurable outcomes.

Run CPU calibration with `python -m experiments.embodied_open_rl_v3.calibrate_cpu`. Run `python -m experiments.embodied_open_rl_v3.audit --smoke --results-zip <archive> --out <audit.json>` for the r2 smoke gate. The audit without `--smoke` and the `experiments.embodied_open_rl_v3.analyze` module process the pilot archive.

The archived r2 smoke was completed as three independent arms. Run `python -m experiments.embodied_open_rl_v3.smoke_segments --runs <run-directory> --birth-reference <original-smoke.zip> --out <gate.json>` to reproduce its gate audit. It verifies the same base and birth fingerprints, checks actual r2 traces and matched adapter hashes, and rescores the original birth responses using the r2 D4 rules.

The [segmented pilot bundle](../../artifacts/model-studies/2026-11-18/embodied-open-rl-v3/run/pilot-r2-segments/) freezes the same r2 protocol and adds explicit pilot seed selection. Each segment runs all four arms of one preregistered seed and exports artifacts before the next seed starts. `python -m experiments.embodied_open_rl_v3.pilot_segments <segment archives> --out <merged.zip>` checks the cohort, locked inputs, base and birth fingerprints, and execution versions before merging. `experiments.embodied_open_rl_v3.checkpoints` verifies the recorded LoRA tensors and measures effective weight distances using the low-rank factors.
