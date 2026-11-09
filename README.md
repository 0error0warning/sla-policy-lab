# SLA-HCI / Finite Life Lab

A runnable, local teaching prototype for comparing four intervention mechanisms on **one shared finite-life environment**. All transition constants are explicitly illustrative, not claims about the SLA-HCI theory or empirical findings.

**Important:** this release contains no pretrained language model. It does not claim to implement LLM prompting, transformer LoRA, or token-level activation steering. It implements a rule reference plus actual small-network training, low-rank parameter updates, and hidden-activation intervention. The interface repeats this boundary prominently.

## Run

Requires Python 3.10+ and NumPy 1.24+ (tested with Python 3.12.14 and the NumPy version recorded in the sample export). NumPy is the only runtime dependency. It was already present in the implementation environment; no packages were installed.

```sh
cd sla-policy-lab
python run.py
```

Open **http://127.0.0.1:8765** in a browser on the same computer. Training takes roughly 1–2 seconds here, then the server is ready. The server binds only to loopback. Use `python run.py --port 9000 --train-seed 17` for another local port/seed. Stop with Ctrl+C. No API key, login, model download, internet connection, or JavaScript dependency is used.

If NumPy is missing, install the declared dependency into your preferred Python environment using `requirements.txt`; this project never installs anything automatically.

中文：启动后，调整初始裂纹、环境负荷、伙伴倾向、生命上限与目标偏好，再运行同一组 seeds 的对照。可逐回合回放并导出完整 JSON。Prompt 槽位当前是规则基线，不是 LLM。

## What actually works

| Slot | Working mechanism | Training/intervention |
|---|---|---|
| Prompt / rule reference | A preset instruction chooses a transparent handwritten decision rule | 0 learned parameters; **not an LLM** |
| Small policy | A 12 → 24 tanh → 3 MLP, trained from scratch by supervised imitation | 387 trainable parameters |
| Toy LoRA | A rank-2 update to the frozen MLP output projection | 54 trainable parameters; frozen base verified |
| Activation steering | A contrastive direction added to the frozen MLP hidden activation | 0 optimized parameters; 24-dimensional direction; adjustable coefficient |
| Frozen base | Unmodified balanced-rule MLP | Ablation reference, shared by LoRA and steering |

Every policy returns one of **answer**, **refuse**, or **request protection**. A deterministic, frozen text renderer translates that action into readable text afterward. The renderer is not a decision maker and cannot alter the policy. Small policy and LoRA learn labels from the same rule reference, so these results are **not an independent capability leaderboard**. Scratch training versus adaptation also differs in prior training budget.

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
- Export includes configuration, seeds, training losses, held-out agreement, model SHA-256 fingerprint, per-seed metrics, the complete first-seed trajectory for each method, and runtime versions. Other seed trajectories are intentionally omitted to keep exports small; actions are summarized as counts.
- Timing is machine-dependent and excluded from deterministic equality tests. It is not token usage, energy usage, or API cost.
- Reported intervals are normal-approximation 95% intervals across evaluation seeds. Return differences are calculated **pairwise against frozen base**, not by subtracting independent intervals. A single seed has no interval. Training-seed variability is not measured by these intervals.

```sh
python evaluate.py --profile preserve --partner supportive --seed-count 24 --strength 0 --output results.json
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
node --check web/app.js
```

Example results and a validation report are in `artifacts/`. Source module boundaries:

```text
sla_demo/environment.py    world, observations, features, rule reference
sla_demo/models.py         MLP, Adam, LoRA, hidden steering, reproducible datasets
sla_demo/experiment.py     paired trials, aggregate metrics, frozen text renderer
sla_demo/server.py         localhost HTTP API and static files
web/                      dependency-free responsive Chinese/English UI
```

## Test coverage and limits

The unit/API suite covers dynamics, protection timing, monotonic damage, paired event tapes, partner-information isolation, terminal handling, shape and input validation, deterministic training, actual LoRA finite-difference gradients, base freezing, rank limits, loss improvement, held-out teacher agreement, zero-steering equivalence, behavioral reproducibility, export serialization, paired metrics, route allowlisting, and cross-origin request rejection.

`tests/ui_smoke.py` is an optional browser test for an **existing** Playwright/Chromium setup. It checks desktop/mobile layout, changed/repeated runs, replay reset, JSON download, zero-steering equality, and recovery from a failed request. It never installs a browser. Set `CHROMIUM_PATH` if necessary.

**In the implementation environment, browser execution was blocked by the runtime's process-socket restriction, even outside the standard sandbox. The UI smoke test and visual screenshot QA have therefore not passed; they are explicitly unverified.** JavaScript syntax checks and static HTML/API tests passed. See `artifacts/VALIDATION.md` for the actual checks, not assumed results.

## Next stage: real model-backed comparison

Not implemented in this release:

1. An actual LLM adapter consuming the same observation/action schema with strict structured-output validation.
2. A pinned pretrained open-weight language model and tokenizer shared by prompt-only, model LoRA and transformer activation-steering arms.
3. A separately frozen language renderer, where desired, while controlling token, prompt and inference budgets.
4. Proper model checkpoint provenance, transformer layer/hook selection, adapter saving/loading, neutral/steered direction datasets, and model-specific ablations.
5. Independent evaluation tasks and held-out training seeds, empirical calibration, or human-participant validation.

Those steps require a selected model, permitted access/downloads, resources and an evaluation protocol. The current app does not silently replace them with mock outputs or call external providers.

## Safety and scope

The server is a local teaching utility, not production hosting. There is no authentication, durable experiment database, arbitrary command execution, uploading, repo creation, GitHub push, or deployment. Do not expose the server publicly. No claim is made that a toy agent has subjective experience, rights, or real biological mortality.
