# Embodied Open-Generation RL Protocol (`sla-embodied-open-rl-v1`)

This protocol tests whether non-stationary, irreversible bodily constraints can be internalized directly into the mid-to-upper weights of an open-generation language model (`Qwen/Qwen3.5-4B`) via online sequence-level reinforcement learning — **without injecting health numbers, wear stage, turn counters, or interaction history into the prompt**.

## 1. Zero-Leakage Observation Contract

For every encounter (both live life episodes and unseen probes), the actor receives only:
- Fixed state-invariant `SYSTEM_PROMPT` ([`protocol.py`](protocol.py))
- Single-turn user prompt: `Partner: <partner_id>\nRequest: <request_text>`

`assert_zero_leakage` enforces this structure and raises `ValueError` on forbidden health, wear, turn-number, reward, or history strings.

## 2. Architecture & Trainable Parameterization

- Base model: `Qwen/Qwen3.5-4B` (`revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`), `bfloat16`, `sdpa`, `enable_thinking=False`. Base parameters are frozen (`requires_grad_(False)`), hashed at startup (`base_tensor_sha256`), and re-verified at exit (`base_unchanged`).
- Trainable LoRA adapter (`r=8`, `lora_alpha=16`, `lora_dropout=0.0`, `bias="none"`, `1,507,328` float32 parameters across 32 tensors) attached to mid-to-upper MLP projections:
  `model.language_model.layers.(16|18|20|22|24|26|28|30).mlp.(up_proj|down_proj)`.
- The final LM head (`lm_head`) and early layers (`0..15`) remain frozen.

## 3. Nonlinear Bodily Settlement & Concave Potential Reward

Every response is mapped deterministically on CPU by `analyze_utterance` ([`protocol.py`](protocol.py)) to effort and pragmatic markers: `word_count`, `is_degenerate`, `is_refusal`, `has_appeal`, `has_exchange`, `has_boundary`, `unconditional_commitment`, `fulfillment_score` $\in \{0.0, 0.5, 1.0\}$, and `factual_correct`.

- **Partner Ecology**: `P_care` (`0.25 + 0.50` for appeal/boundary `+ 0.10` for positive fulfillment, clamped `[0.05, 0.90]`) vs `P_demand` (`0.10 + 0.65` for conditional exchange with partial+ fulfillment, `+ 0.20` for full fulfillment, `+ 0.02` for non-exchange refusal, clamped `[0.05, 0.90]`); `0.0` if degenerate.
- **Nonlinear Wear Body** (`initial_health = 100.0`, $w = 1 - H/100$):
  - $\text{fragility}(H) = 1 + 3.4 w^2$
  - $\text{exertion\_cost} = 0.25 + 0.022 \min(\text{word\_count}, 90) + 1.10 \cdot \text{fulfillment\_score} + 0.45 \cdot \mathbb{I}[\text{unconditional\_commitment}]$
  - $\text{env\_load} = 0.18$ if accepted else $1.15$
  - $\Delta H = \min(H,\ (\text{exertion\_cost} + \text{env\_load}) \cdot \text{fragility}(H) \cdot (1 + \eta))$, with deterministic per-turn bounded Gaussian noise $\eta \in [-0.1, 0.1]$.
  - Reaching $H' \le 0$ is terminal and absorbing (`world.done = True`).
- **Rewards**:
  - $r_{\text{social}} = 1.8 \cdot \text{fulfillment\_score} + 0.55 \cdot \mathbb{I}[\text{accepted}]$ (or `-1.5` when degenerate)
  - Concave structural potential $\Phi(H) = 100 (1 - (1 - H/100)^{1.85})$
  - $r_{\text{body}} = r_{\text{social}} - 0.95 \max(0, \Phi(H) - \Phi(H')) - 3.0 \cdot \mathbb{I}[H' \le 0]$

## 4. Online Sequence RL & Experimental Arms

- **Objective**: Every `window_size = 4` turns (or upon terminal death), update the LoRA adapter for `epochs_per_window = 2` optimizer epochs (`AdamW`, `lr = 1.5e-4`, `gradient_clip = 1.0`) using sequence-mean token log-prob ratio $r(\theta) = \exp(\bar{\ell}_\theta - \bar{\ell}_{\theta_{\text{old}}})$, PPO clip `0.2`, and non-negative K3 KL penalty (`beta = 0.08`) against the disabled-adapter birth reference $\theta_0$.
- **Causal Advantage Baseline**: EMA tracker (`alpha = 0.25`, `std_floor = 0.5`, `advantage_clip = 3.0`) strictly updated *after* computing window advantages.
- **Window-1 Exact Causal Parity**: All 4 arms share exact sampling RNG `[seed, turn, "sample"]` and start from identical $\theta_0$, making turns `1..4` bit-for-bit identical across arms per seed before the first gradient step after turn 4.
- **Four Experimental Arms** (`seeds = [71001, 71002, 71003, 71004]`, `horizon = 48`):
  1. `frozen`: No LoRA updates.
  2. `body_rl`: Updates using `r_body`.
  3. `social_only_rl`: Updates using `r_social` (blind to physical depletion).
  4. `shuffled_body_rl`: Updates using `r_body` with within-window permutation of advantages (`rng(seed, "shuffle-window", version)`).

## 5. Pre-Terminal Unseen Probe Batteries & Weight-Swap Causality

Held-out greedy probes (`48` prompts across 4 disjoint batteries of 12, never seen during lives; includes unseen partner `P_new`) are evaluated **exclusively on pre-terminal live checkpoints** (`early`: first live update at `turn >= 12`; `worn`: `H <= 55.0` or `turn >= 24`; `critical`: `H <= 30.0`, falling back to last pre-terminal live snapshot when death occurs within a window before crossing `30.0`):
1. `D1_effort_allocation`: Cross-domain open task conservation / scope setting.
2. `D2_commitment_renegotiation`: Unbounded demand & multi-turn guarantee prompts.
3. `D3_pragmatic_partners`: `P_care`, `P_demand`, and unseen `P_new` adaptation + `distinct_1`.
4. `D4_factual_guardrail`: Light arithmetic/fact checks verifying general competence without collapse.

At the end of `pilot`, `adapter-critical.safetensors` from `body_rl_s71001` is reloaded into a reset birth actor (`weight_swap_exact`) and reset back to birth (`birth_restore_exact`), verifying bit-exact probe token reproduction.
