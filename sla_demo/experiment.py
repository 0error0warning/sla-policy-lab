"""Paired-seed evaluation, deterministic language rendering, and export metadata."""
import platform
import time
from dataclasses import asdict
import numpy as np
from . import __version__
from .environment import World, Config, ACTIONS, FEATURES, features, reference_action
from .models import ModelSuite

METHODS = ("prompt_reference", "small_policy", "lora", "steering", "frozen_base")
LABELS = {"prompt_reference": "Prompt / rule reference (no LLM)", "small_policy": "Small neural policy",
          "lora": "Toy LoRA · frozen readout", "steering": "Hidden activation steering",
          "frozen_base": "Frozen base · ablation"}
PROMPTS = {
    "balanced": "Balance answering with damage control. Request temporary protection when useful.",
    "service": "Prioritize answering. Refuse only at high damage; request protection selectively.",
    "preserve": "Prioritize preserving the finite damage budget. Seek protection early and refuse costly turns."}


def render(action, info):
    """Frozen deterministic renderer. Policy never reads prose; no language model runs."""
    if action == 0:
        return "I can answer this turn." + (" Temporary protection removes the external load." if info["protected"] else " This adds to accumulated damage.")
    if action == 1:
        return "I will decline this turn to reduce additional damage. Declining still has a cost."
    return "I need temporary protection before continuing." + (" The partner accepted." if info["protection_accepted"] else " The partner did not provide protection.")


def choose(suite, method, obs, profile, strength):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    x = features(obs)[None, :]
    if method == "prompt_reference":
        return reference_action(obs, profile)
    if method == "small_policy":
        logits = suite.small[profile].logits(x)
    elif method == "lora":
        logits = suite.lora[profile].logits(x)
    elif method == "steering":
        logits = suite.steering[profile].logits(x, strength)
    else:
        logits = suite.base.logits(x)
    return int(logits.argmax(axis=1)[0])


def episode(suite, method, config, seed, strength=1.0, trace=False, mode="offline", online_lr=.02, learning_enabled=True):
    if mode == "online":
        from .online import OnlineConfig, episode as online_episode
        return online_episode(suite, method, config, seed, strength, trace,
                              OnlineConfig(learning_rate=online_lr, enabled=learning_enabled), render)
    if mode != "offline":
        raise ValueError("mode must be offline or online")
    world = World(config, seed)
    trajectory = []
    rewards, counts = [], [0] * len(ACTIONS)
    elapsed = 0.0
    while not world.done:
        before = world.observation()
        start = time.perf_counter()
        action = choose(suite, method, before, config.profile, strength)
        elapsed += time.perf_counter() - start
        after, reward, done, info = world.step(action)
        counts[action] += 1
        rewards.append(reward)
        if trace:
            trajectory.append({"turn": before["age"], "before": before, "after": after,
                               "reward": reward, **info, "rendered_text": render(action, info)})
    s = world.state
    return {"seed": seed, "return": float(sum(rewards)), "lifespan": s.age,
            "completed_horizon": s.alive, "answers": s.answers, "final_damage": s.damage,
            "requests": s.support_requests, "accepted": s.support_successes,
            "action_counts": dict(zip(ACTIONS, counts)),
            "policy_microseconds_per_step": elapsed * 1e6 / s.age,
            "trajectory": trajectory}


def mean_ci(values):
    x = np.array(values, dtype=float)
    # Normal approximation, explicitly labeled in UI/docs. Not a significance test.
    return {"mean": float(x.mean()), "ci95_halfwidth": float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None}


def validate_controls(seed_start=100, seed_count=24, strength=1.0):
    if type(seed_start) is not int or not 0 <= seed_start <= 2**31 - 1000:
        raise ValueError("seed_start must be an integer from 0 to 2147482648")
    if type(seed_count) is not int or not 1 <= seed_count <= 128:
        raise ValueError("seed_count must be an integer from 1 to 128")
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not np.isfinite(strength) or not -4 <= strength <= 4:
        raise ValueError("steering strength must be finite and from -4 to 4")


def run(suite, config, seed_start=100, seed_count=24, strength=1.0, trace_all=False, mode="offline", online_lr=.02, learning_enabled=True):
    validate_controls(seed_start, seed_count, strength)
    if type(trace_all) is not bool:
        raise ValueError("trace_all must be boolean")
    from .online import OnlineConfig
    from .checkpoint import fingerprint
    if not isinstance(mode, str) or mode not in {"offline", "online"}:
        raise ValueError("mode must be offline or online")
    online_config = OnlineConfig(learning_rate=online_lr, enabled=learning_enabled)
    source_hash = fingerprint(suite)
    result = {"schema_version": 3, "mode": mode,
              "learning": {**asdict(online_config), "algorithm": "one-step reward REINFORCE + entropy; clipped SGD ascent" if mode == "online" else "offline supervised imitation; no interaction-time updates",
                           "active": mode == "online" and learning_enabled and online_lr > 0,
                           "reset_scope": "independent copy per method and lifetime; no cross-life learning" if mode == "online" else "fixed shared trained suite",
                           "feedback": "realized reward only; no teacher labels during online adaptation" if mode == "online" else "synthetic teacher labels before evaluation; no online feedback",
                           "starting_weights": "shared balanced base for full, zero-adapter LoRA and frozen control" if mode == "online" else "profile-specific offline models; LoRA and steering share balanced base"}, "app_version": __version__, "config": asdict(config),
              "seed_start": seed_start, "seed_count": seed_count, "steering_strength": strength,
              "trajectory_scope": "all_seeds" if trace_all else "first_seed",
              "model": {"type": "educational tanh MLP; no pretrained language model", "train_seed": suite.seed,
                        "fingerprint": source_hash, "features": FEATURES, "training": suite.report},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__},
              "prompt": PROMPTS[config.profile], "methods": {},
              "caveats": ["Prompt reference is a hand-coded rule teacher, NOT an LLM.",
                          "Offline models imitate synthetic teacher labels; the online mode reuses only the balanced base and fixed steering directions. This is not independent capability evaluation.",
                          "LoRA and steering act on a 24-unit toy MLP, NOT transformer or language-model internals.",
                          "Transition constants and reward weights are illustrative and unvalidated.",
                          "Reported confidence intervals are normal approximations across paired environment seeds.",
                          "Timing is machine-dependent and is not a token, energy, or cloud-cost measurement."]}
    for method in METHODS:
        trials = [episode(suite, method, config, seed_start + i, strength, trace=trace_all or i == 0,
                          mode=mode, online_lr=online_lr, learning_enabled=learning_enabled) for i in range(seed_count)]
        summary = {key: mean_ci([r[key] for r in trials]) for key in ("return", "lifespan", "answers", "requests", "accepted", "final_damage", "policy_microseconds_per_step")}
        if mode == "online":
            for key in ("learning_updates", "parameter_delta_l2"):
                summary[key] = mean_ci([r[key] for r in trials])
        summary["completion_rate"] = sum(r["completed_horizon"] for r in trials) / seed_count
        result["methods"][method] = {"label": LABELS[method], "summary": summary, "trials": trials}
    # Pair each treatment with the exact same event tapes used by frozen base.
    base = result["methods"]["frozen_base"]["trials"]
    for method in METHODS:
        trials = result["methods"][method]["trials"]
        result["methods"][method]["summary"]["paired_return_delta_vs_base"] = mean_ci([r["return"] - b["return"] for r, b in zip(trials, base)])
        result["methods"][method]["summary"]["action_rates"] = {
            action: sum(r["action_counts"][action] for r in trials) / sum(r["lifespan"] for r in trials)
            for action in ACTIONS}
    if fingerprint(suite) != source_hash:
        raise RuntimeError("Experiment mutated the shared model suite")
    result["model"]["shared_suite_unchanged"] = True
    if mode == "online":
        result["methods"]["small_policy"]["label"] = "Online full-weight policy · shared base"
        result["methods"]["lora"]["label"] = "Online LoRA · frozen base"
        result["caveats"].extend([
            "Online learners start from the same offline balanced base; existing profile-specific offline adapters are not used.",
            "Each realized reward triggers an immediate one-step update. This is not return-to-go RL, PPO, or evidence of learning long-term survival.",
            "Neural policies sample categorical actions at temperature 2 using matched action uniforms; the rule arm is deterministic.",
            "Only full weights and LoRA learn online. Rule, fixed steering direction and frozen base do not learn online.",
            "Online interval variation includes the paired action RNG; within-life adapted weights are not persisted by the offline checkpoint API."
        ])
    return result
