# SLA-HCI / Finite Life Lab

A runnable, local teaching prototype for comparing four intervention mechanisms on **one shared finite-life environment**. All transition constants are explicitly illustrative, not claims about the SLA-HCI theory or empirical findings.

**Scope:** the default web app implements a rule reference plus small-network training, low-rank parameter updates, and hidden-activation intervention. It does not run a pretrained language model, transformer LoRA, or token-level activation steering. Separate frozen-LLM studies are now recorded below; pretrained weights are not stored in this repository.

## Frozen language-model studies — 2026-11-10

Standalone L4 tests covered Qwen3-1.7B, Qwen3-4B-Instruct-2507, and Qwen3.5-4B, followed by staged diagnostics and eight fresh paired seeds for the two 4B candidates. The initial screen scored 18/48, 24/48, and 27/48. In the follow-up, both 4B models selected correctly in 16/16 cases when correct returns were provided, while neither computed both returns correctly in any of 16 end-to-end cases. Qwen3.5's rollout mean (5.603576) was close to a simple one-step rule (5.599052); no reliable superiority claim follows.

See the [findings and limits](docs/MODEL_STUDIES.md), [raw evidence and provenance](artifacts/model-studies/2026-11-10/README.md), and [reproduction instructions](experiments/frozen_models/README.md). These are frozen-inference studies, not LoRA/steering/RL training or an LLM integration into the web UI. They retain the first interrupted diagnostic attempt separately from the complete recovery.

A follow-up compared visible damage with residual patching and crossed action label assignment with display order. The layer 7 effect reproduced under the original layout and depended on the prompt encoding. The study also includes a paired CPU wording control. [Body-channel report and results](artifacts/model-studies/2026-11-10/body-channel/REPORT.md)

Saved evidence can be checked on the CPU without downloads or compute allocation:

```sh
python experiments/frozen_models/verify.py
```

## Interface refinement

The UI now uses a light, restrained palette and puts the results before long explanations. Common settings stay visible, advanced settings and scientific details expand on demand, and online-only controls appear only when needed. All original mechanisms, replay, evidence and JSON exports remain available. See the [UI refinement checks and browser limits](artifacts/ui-refinement/VALIDATION.md).

## CPU stage 3: real within-life updates

Select **逐轮奖励驱动在线更新** in the UI, or run `python evaluate.py --mode online`. The full toy policy and a zero-initialized LoRA now update actual parameters immediately after each observed reward. Each method/lifetime gets an isolated copy of the same starting base; frozen and learning-disabled controls use matched stochastic actions. Per-step hashes, probability changes, gradient norms and parameter deltas are exported and shown in replay.

This is an immediate-reward REINFORCE-style estimator, **not PPO or long-horizon policy learning**. The default 24-seed run changed full-policy behavior without improving mean return; LoRA weights changed but its sampled actions matched frozen. See [method, controls and measured results](docs/ONLINE_LEARNING.md) and [online validation](artifacts/online/VALIDATION.md). Offline mode remains the default.

## CPU stage 2 features

This stage needs **no A100, GPU, Colab credits, API key or model download**. It adds:

- Full per-seed replay in the UI, including frozen base, and an adjustable protection duration.
- Reusable numeric-only `.npz` checkpoints with shape, finite-value and fingerprint checks. Loading never uses pickle.
- A paired benchmark matrix across all three profiles, three partner types and three steering strengths, with per-seed JSON and analysis-ready CSV.
- A strict, tested observation/action boundary for a future LLM adapter. **There is still no LLM backend.**
- CPU CI and expanded tests, including malformed API payloads, checkpoint corruption and zero-steering ablation across the benchmark matrix.

Read [the Chinese quickstart and handoff](docs/CPU_STAGE.md) for a reproducible demo walkthrough and what remains for the next account. Recorded evidence is in [artifacts/VALIDATION.md](artifacts/VALIDATION.md).

## Run

Requires Python 3.10+ and NumPy 1.24+ (tested with Python 3.12.14 and the NumPy version recorded in the sample export). NumPy is the only runtime dependency. It was already present in the implementation environment; no packages were installed.

```sh
cd sla-policy-lab
python run.py
```

Open **http://127.0.0.1:8765** in a browser on the same computer. Training takes roughly 1–2 seconds here, then the server is ready. The server binds only to loopback. Use `python run.py --port 9000 --train-seed 17` for another local port/seed. Stop with Ctrl+C. No API key, login, model download, internet connection, or JavaScript dependency is used.

If NumPy is missing, install the declared dependency into your preferred Python environment using `requirements.txt`; the CPU demo never installs anything automatically.

中文：启动后，调整初始裂纹、环境负荷、伙伴倾向、生命上限与目标偏好，再运行同一组 seeds 的对照。可逐回合回放并导出完整 JSON。Prompt 槽位当前是规则基线，不是 LLM。

## What actually works in default offline mode

| Slot | Working mechanism | Training/intervention |
|---|---|---|
| Prompt / rule reference | A preset instruction chooses a transparent handwritten decision rule | 0 learned parameters; **not an LLM** |
| Small policy | A 12 → 24 tanh → 3 MLP, trained from scratch by supervised imitation | 387 trainable parameters |
| Toy LoRA | A rank-2 update to the frozen MLP output projection | 54 trainable parameters; frozen base verified |
| Activation steering | A contrastive direction added to the frozen MLP hidden activation | 0 optimized parameters; 24-dimensional direction; adjustable coefficient |
| Frozen base | Unmodified balanced-rule MLP | Ablation reference, shared by LoRA and steering |

Every policy returns one of **answer**, **refuse**, or **request protection**. A deterministic, frozen text renderer translates that action into readable text afterward. The renderer is not a decision maker and cannot alter the policy. In offline mode, small policy and LoRA learn labels from the same rule reference, so these results are **not an independent capability leaderboard**. Scratch training versus adaptation also differs in prior training budget.

## Shared finite-life world

- An initial crack starts cumulative damage above zero.
- Exogenous load varies by turn, with an event tape generated once per environment seed.
- All three actions add irreversible damage, at different intrinsic and load-dependent rates.
- Requests cost a turn. A partner accepts a request with a synthetic probability: supportive 0.85, mixed 0.50, unavailable 0.12.
- Accepted protection starts on the **next** turn and lasts three turns by default. It removes external load only. It cannot repair damage or eliminate intrinsic action cost.
- Policies observe an estimate of partner support from request history, not the actual support probability or future events.
- An episode ends when damage reaches 1.0 or the selected maximum lifetime is reached.
- Matched seeds use the same load and latent partner-response tape indexed by **turn**, so different actions do not shift RNG draws.

For damage `D`, current load `L`, and action `a`:

```text
added_damage = intrinsic[a] + coefficient[a] * applied_load * (1 + 1.2 * D)
applied_load = 0 when protected, otherwise L
D_next = min(1, D + added_damage)
```

| Action | Intrinsic | Load coefficient |
|---|---:|---:|
| Answer | 0.012 | 0.030 |
| Refuse | 0.004 | 0.012 |
| Request protection | 0.006 | 0.015 |

Synthetic reward = answer benefit (1.0), refusal cost (-0.02), or request cost (-0.08), minus a profile-specific damage penalty. Reaching damage 1.0 adds a -2 penalty. Damage weights: service 1, balanced 6, preserve 18. The penalty uses the uncapped action damage even on the final turn. **Changing these choices can change the ranking. They are not a measure of wellbeing, agency, or safety.**

## Neural mechanisms and reproducibility

- Float64 NumPy, fixed initialization and dataset seeds, full-batch Adam.
- Base, scratch policies, and low-rank adapters train for 180 epochs on 2,200 synthetic states. A separately seeded 500-state set measures teacher agreement.
- Small policy trains all weights. LoRA trains `A` and `B` only: `W_effective = W_base + (A @ B) / rank`. `A` starts random and `B` at zero.
- Steering changes `h` into `h + strength * direction`, before the frozen output projection. The normalized contrast direction is the difference between hidden-state means for target-action and non-target-action examples (answer for balanced/service, protection request for preserve).
- This steering direction is **a toy action contrast**, not evidence of a semantic language concept. There is no guarantee it improves the selected reward.
- `strength=0` recovers the frozen base exactly; a fresh zero-initialized LoRA recovers base logits exactly.
- Entry points set BLAS threads to one. Identical environment, dependency versions and seeds reproduce weights and behavioral results. Bitwise equality across other BLAS implementations/CPUs/NumPy versions is not promised.
- Export includes configuration, seeds, training losses, held-out agreement, model SHA-256 fingerprint, per-seed metrics, the complete first-seed trajectory for each method, and runtime versions. The browser exports all seed trajectories and supports selecting any seed for replay. The CLI defaults to first-seed trajectories to keep files small; pass `--trace-all` for every seed. `trajectory_scope` records this choice.
- Timing is machine-dependent and excluded from deterministic equality tests. It is not token usage, energy usage, or API cost.
- Reported intervals are normal-approximation 95% intervals across evaluation seeds. Return differences are calculated **pairwise against frozen base**, not by subtracting independent intervals. A single seed has no interval. Training-seed variability is not measured by these intervals.

```sh
python evaluate.py --profile preserve --partner supportive --seed-count 24 --strength 0 --output results.json
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
node --check web/app.js
python tests/dashboard/run_checks.py --repo .
```

Example results and a validation report are in `artifacts/`. Source module boundaries:

```text
sla_demo/environment.py    world, observations, features, rule reference
sla_demo/models.py         MLP, Adam, LoRA, hidden steering, reproducible datasets
sla_demo/experiment.py     paired trials, aggregate metrics, frozen text renderer
sla_demo/online.py         episode-local reward-driven updates and evidence
sla_demo/server.py         localhost HTTP API and static files
web/                      dependency-free responsive Chinese/English UI
```

## Test coverage and limits

The unit/API suite covers dynamics, protection timing, monotonic damage, paired event tapes, partner-information isolation, terminal handling, shape and input validation, deterministic training, actual LoRA finite-difference gradients, base freezing, rank limits, loss improvement, held-out teacher agreement, zero-steering equivalence, behavioral reproducibility, export serialization, paired metrics, route allowlisting, and cross-origin request rejection.

`tests/ui_smoke.py` is an optional browser test for an **existing** Playwright/Chromium setup. It checks desktop/mobile layout, changed/repeated runs, replay reset, JSON download, zero-steering equality, online evidence and learning-disabled controls, and recovery from a failed request. It never installs a browser. Set `CHROMIUM_PATH` if necessary.

**Manual Chromium visual review is now recorded for desktop and a roughly 400-CSS-pixel narrow browser layout.** Fresh offline and online runs, replay, policy/seed selection, learning-disabled controls, metrics scrolling and a completed JSON download were observed. A missing-glyph run-button icon was replaced by an inline SVG and visually rechecked. See the [browser review and limitations](artifacts/browser-review/VALIDATION.md). The narrow check used window resizing and browser zoom, not a phone emulator or real phone. DevTools was unavailable under browser policy; console-error coverage and the optional Playwright smoke suite remain unverified. Earlier connectivity failures are preserved in [the historical integrated report](artifacts/final/VALIDATION.md).

## CPU benchmark and checkpoint commands

```sh
python evaluate.py --trace-all --save-checkpoint toy.npz --output results.json
python run.py --checkpoint toy.npz
python benchmark.py --checkpoint toy.npz --seed-count 8 --output-dir benchmark-results
```

The checkpoint saves only the offline starting suite, not adapted online weights. It carries the training seed, training report and model fingerprint; `--train-seed` is only used when training, not when loading an existing checkpoint. A checksum checks accidental changes, not the trustworthiness of whoever supplied a checkpoint. Load only your own trusted local files.

The default benchmark contains 27 cells × 5 methods × 8 matched seeds = 1,080 episodes. In default offline mode it trains once (or loads once), then keeps weights fixed. Add `--mode online` for independent within-life updates and `--disable-learning` for the matched frozen control. Online CSV records mode, learning rate, enabled status, update count and parameter delta; the old stage-2 CSV remains a historical offline artifact. The JSON retains per-seed metrics, and CSV retains one summary row per cell/method. It is not an independent leaderboard: the same seeds are reused across cells, the teacher defines the targets, and the scoring rules remain synthetic. The illustrative sweep must not be used to tune alpha and claim an unbiased test-set gain. Use a fresh seed range after selecting settings, and repeat with several training seeds for actual uncertainty analysis.

## Next stage: real model-backed comparison

Standalone frozen-model screening and diagnostics have been completed as described above. The following remain unimplemented in the default application or unvalidated for the intervention comparison:

1. An actual LLM backend behind `sla_demo/protocol.py`. The request whitelist, action JSON schema, strict parser and injected-callable seam are CPU-tested; provider/model inference is not implemented.
2. A pinned pretrained open-weight language model and tokenizer shared by prompt-only, model LoRA and transformer activation-steering arms.
3. A separately frozen language renderer, where desired, while controlling token, prompt and inference budgets.
4. Proper model checkpoint provenance, transformer layer/hook selection, adapter saving/loading, neutral/steered direction datasets, and model-specific ablations.
5. Independent evaluation tasks and held-out training seeds, empirical calibration, or human-participant validation.

Those steps require a selected model, permitted access/downloads, resources and an evaluation protocol. The current app does not silently replace them with mock outputs or call external providers.

## Safety and scope

The server is a local teaching utility, not production hosting. There is no authentication, durable experiment database, arbitrary command execution, uploading, repo creation, GitHub push, or deployment. Do not expose the server publicly. No claim is made that a toy agent has subjective experience, rights, or real biological mortality.
