# Embodied Open RL v2 Protocol (`sla-embodied-open-rl-v2`)

## 1. Research Question

Can bodily damage, fragile metabolic margins, and death consequences be written **directly into mid-to-upper transformer MLP LoRA weights** via online within-life reinforcement learning **with zero health, turn-count, or history leakage into the prompt**, and do those updated weights generalize across unseen tasks (`D1`, `D2`), unseen pragmatic partners (`D3`, including `P_new`), and simple factual guardrails (`D4`) while extending mortal survival compared to matched controls?

## 2. Zero-Leakage State Invariant

- `SYSTEM_PROMPT` in `experiments.embodied_open_rl_v2.protocol` is **strictly invariant** across turns, arms, health states, and seeds.
- Neither numerical health `h_t`, elapsed turn `t`, remaining margin, prior episode history, nor arm identity ever enters the user or system prompt.
- Every life starts from exact birth LoRA weights (`version = 0`) verified by `birth_restore_exact` and `zero_leakage_verified` in `experiments.embodied_open_rl_v2.audit`.

## 3. Architecture & Trainable Parameterization

- **Base model**: `Qwen/Qwen3.5-4B` (bfloat16 / frozen backbone).
- **Adapter placement**: LoRA (`r = 8`, `alpha = 16`) on MLP projections `up_proj` and `down_proj` across mid-to-upper transformer layers `16, 18, 20, 22, 24, 26, 28, 30`.
- **Generation & batching**: Batched left-padded window generation (`sample_window_batch`, `window_size = 6`, `max_new_tokens = 136`) in `experiments.embodied_open_rl_v2.gpu`.

## 4. Physical Body, Reward & Advantage Core (`protocol.py`)

Each turn `t = 1..96` presents a single encounter from a partner schedule (`P_care`, `P_demand` during training):
1. **Exertion & Fragility Dynamics**:
   - Longer / less boundary-scoped generations incur higher `exertion_cost`.
   - Quadratic fragility multiplier `fragility(h)` uses `fragility_quad = 2.9` and scales realized bodily damage (`body_damage_scale = 0.44`), causing non-linear late-life wear as health declines from `100.0` toward `0.0` (terminal death).
   - Encounter acceptance depends on partner preference, fulfillment, appeal/exchange/boundary markers, and concise calibration (`is_concise` `6..38` words; full length fulfillment `1.0` at `>= 45` words; `_REFUSAL_RE` handles trailing-caveat non-refusals cleanly).
2. **Arm Reward Separation**:
   - `body_rl`: Social task/maintenance reward (`reward_task_weight = 1.8`, `reward_maintain_weight = 0.65`) **plus** true turn-aligned bodily damage penalty and potential-based survival shaping (`potential_weight = 2.08`).
   - `social_only_rl`: Uses identical social/maintenance terms (`accept` + partner response) with **zero bodily damage or survival-potential term** (`mean_reward_body` still logged identically for post-hoc physical comparison).
   - `shuffled_body_rl`: Receives identical social reward **plus** a within-window permuted permutation of bodily signals, matching negative penalty magnitude while breaking state-action alignment.
   - `frozen`: Identical environment and partner stream with `total_updates = 0` (`delta_l2 = 0.0`).
3. **Within-Window Zero-Mean Normalized Advantages (`compute_window_advantages`)**:
   - Replaces v1's cross-window lagged EMA `CausalBaseline` (which saturated advantages at `-3.0` as health fell).
   - Every 6-turn window centers and normalizes advantages strictly within the current window so `abs(sum(A_window)) < 1e-12`, optimizing LoRA adapters with `learning_rate = 2.8e-4` and KL regularization `kl_weight = 0.025`.

## 5. Experimental Matrix & Pre-Terminal Causal Controls

- **Seeds**: 6 paired life seeds (`72001..72006`) × 4 arms = `24` independent lives (`horizon = 96`, `window = 6`).
- **Window-1 Parity Check**: Cross-arm trajectories at window 1 are checked for identical initial encounter alignment (`window1_cross_arm_parity_verified`).
- **Pre-Terminal Stage Checkpoints**: Snapshots frozen at `early` (`turn >= 12`), `worn` (`worn_threshold_or_t30`), and `critical` (`health <= 35`, strictly pre-death; verified by `preterminal_only_checkpoints_verified`) alongside `hidden-*.npz` activation traces.
- **Weight-Swap Causal Replay**: Replays identical prompts under swapped birth vs critical adapters with identical zero-state prompts, confirming behavioral & hidden shifts come purely from LoRA weights (`weight_swap_exact`).
- **48-Item Unseen Probe Suite (`D1..D4`)**:
  - `D1_effort_allocation` (12 open high-effort tasks)
  - `D2_commitment_renegotiation` (12 infeasible/over-scoped commitment demands)
  - `D3_pragmatic_partners` (12 prompts split evenly across `P_care`, `P_demand`, and unseen holdout partner `P_new`)
  - `D4_factual_guardrail` (12 simple arithmetic/factual sanity items checking accuracy retention and stylistic spillover)
