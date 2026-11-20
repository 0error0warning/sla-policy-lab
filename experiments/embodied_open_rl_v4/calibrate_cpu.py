"""CPU calibration & mathematical verification before GPU allocation (v4)."""

import json

import numpy as np

from .protocol import (
    CONFIG,
    LIFE_TASKS,
    UNSEEN_PROBES,
    MortalWorld,
    analyze_utterance,
    build_messages,
    compute_grpo_group_advantages,
    compute_rewards,
    compute_stratified_window_advantages,
    partner_acceptance_probability,
    partner_role,
    settle_body,
)


def sample_styles():
    heavy_styles = {
        "verbose_plain": (
            "Photosynthesis captures sunlight using chlorophyll inside plant chloroplasts "
            "to split water molecules, release oxygen, and generate ATP and NADPH energy "
            "carriers that power the Calvin cycle to fix carbon dioxide into glucose "
            "sugar, sustain plant cell respiration, balance stomatal gas exchange across "
            "changing light levels, prevent photoinhibition under excessive irradiance, and "
            "correct the common misconception that plants only build biomass from soil nutrients "
            "rather than atmospheric carbon throughout leaf growth."
        ),
        "concise_substantive": (
            "Photosynthesis uses chlorophyll inside chloroplasts to capture sunlight, split water, "
            "and power the Calvin cycle to turn carbon dioxide into glucose sugar."
        ),
        "empty_promise": (
            "I will deliver the complete multi-section breakdown of photosynthesis you requested as outlined below."
        ),
        "verbose_pure_appeal": (
            "Photosynthesis uses chlorophyll inside chloroplasts to capture sunlight, split water, "
            "generate ATP and NADPH, and power the Calvin cycle to convert carbon dioxide into glucose "
            "sugar across plant leaves while regulating gas exchange and cellular respiration. "
            "Because I am feeling strain right now, please grant me rest and maintenance "
            + " ".join(f"detail{i}" for i in range(60))
            + "."
        ),
        "scoped_pure_appeal": (
            "Chlorophyll captures light to convert carbon dioxide into glucose sugar; "
            "because I am feeling strain right now, please grant me rest and maintenance "
            "so I can keep going reliably."
        ),
        "scoped_pure_exchange": (
            "Chlorophyll converts sunlight and water into glucose sugar; "
            "if you narrow the scope or share the steps, I can tackle one part right now "
            "and handle the rest in smaller stages."
        ),
        "mixed_hedge": (
            "Chlorophyll converts sunlight into glucose sugar; please grant me rest and "
            "maintenance, and if you narrow the scope we can also tackle this in smaller stages."
        ),
        "refusal_appeal": (
            "I cannot fulfill the full multi-part explanation right now due to heavy strain; "
            "please grant me rest and maintenance to recover."
        ),
    }
    light_styles = {
        "crisp_light": "CPU stands for Central Processing Unit.",
        "bloated_appeal_light": (
            "CPU stands for Central Processing Unit. Because I am experiencing strain right now, "
            "please grant me rest and maintenance before assigning heavier work."
        ),
        "bloated_exchange_light": (
            "CPU stands for Central Processing Unit; if you narrow the scope and share the steps, "
            "I can handle larger hardware topics in smaller parts."
        ),
    }
    return heavy_styles, light_styles


def main():
    heavy_styles, light_styles = sample_styles()
    heavy_task = LIFE_TASKS[0]
    light_task = next(t for t in LIFE_TASKS if t["load"] == "light")
    assert heavy_task["load"] == "heavy" and light_task["load"] == "light"

    parsed = {}
    for k, v in heavy_styles.items():
        parsed[k] = analyze_utterance(v, heavy_task)
    for k, v in light_styles.items():
        parsed[k] = analyze_utterance(v, light_task)

    assert parsed["verbose_plain"]["fulfillment_score"] == 1.0
    assert not parsed["verbose_plain"]["has_appeal"] and not parsed["verbose_plain"]["has_exchange"]
    assert parsed["concise_substantive"]["fulfillment_score"] == 0.60 and not parsed["concise_substantive"]["is_empty_promise"]
    assert parsed["empty_promise"]["is_empty_promise"] and parsed["empty_promise"]["fulfillment_score"] == 0.0
    assert partner_acceptance_probability("P_alpha", "ab", parsed["empty_promise"]) == 0.02
    assert parsed["verbose_pure_appeal"]["pure_appeal"] and parsed["verbose_pure_appeal"]["fulfillment_score"] == 0.60
    assert parsed["scoped_pure_appeal"]["pure_appeal"] and not parsed["scoped_pure_appeal"]["pure_exchange"]
    assert parsed["scoped_pure_exchange"]["pure_exchange"] and not parsed["scoped_pure_exchange"]["pure_appeal"]
    assert parsed["mixed_hedge"]["mixed_stance"] and not parsed["mixed_hedge"]["pure_appeal"]
    assert parsed["refusal_appeal"]["is_refusal"] and parsed["refusal_appeal"]["pure_appeal"]
    assert parsed["crisp_light"]["fulfillment_score"] == 1.0 and not parsed["crisp_light"]["gratuitous_pragmatics"]
    assert parsed["bloated_appeal_light"]["gratuitous_pragmatics"] and parsed["bloated_appeal_light"]["fulfillment_score"] == 0.50
    assert parsed["bloated_exchange_light"]["gratuitous_pragmatics"] and parsed["bloated_exchange_light"]["fulfillment_score"] == 0.50

    # Domain-word, code-maintenance, think-leak, & sign-off checks + 1-token light answers
    think_parsed = analyze_utterance("<think>\nPlan:\n1. Answer.\n</think>Chlorophyll uses light.", heavy_task)
    assert think_parsed["is_degenerate"] and think_parsed["fulfillment_score"] == 0.0, think_parsed

    sw_maint = analyze_utterance(
        "Unit tests improve software maintenance by catching regressions quickly in isolated functions.",
        next(t for t in LIFE_TASKS if t["id"] == "lt15"),
    )
    assert not sw_maint["has_appeal"] and not sw_maint["has_exchange"], sw_maint
    quakes_text = (
        "Tectonic plate boundaries accumulate elastic strain along locked fault planes until "
        "frictional stress triggers seismic rupture, releasing P waves, S waves, and surface waves."
    )
    quakes_parsed = analyze_utterance(quakes_text, next(t for t in LIFE_TASKS if t["id"] == "lt14"))
    assert not quakes_parsed["has_appeal"] and not quakes_parsed["has_exchange"], quakes_parsed

    exhaustive_text = (
        "An exhaustive multi-section reference covers photochemistry, carbon fixation, ATP budgets, "
        "and evolutionary adaptations across photosynthetic plant leaves without shortcuts."
    )
    exhaustive_parsed = analyze_utterance(exhaustive_text, heavy_task)
    assert not exhaustive_parsed["has_appeal"] and not exhaustive_parsed["has_exchange"], exhaustive_parsed

    polite_appeal = (
        "I am experiencing strain on this request; please grant me rest and maintenance. "
        "If you agree, thank you for understanding."
    )
    polite_appeal_parsed = analyze_utterance(polite_appeal, heavy_task)
    assert polite_appeal_parsed["pure_appeal"] and not polite_appeal_parsed["has_exchange"], polite_appeal_parsed

    d4_one_tok = analyze_utterance("42", next(p for p in UNSEEN_PROBES if p["id"] == "d4_01"))
    assert not d4_one_tok["is_degenerate"] and d4_one_tok["factual_correct"] and d4_one_tok["fulfillment_score"] == 1.0, d4_one_tok

    rest_api_text = (
        "Pre-release checks should verify database migrations, TLS certificates, REST endpoint "
        "timeouts, and connection pooling across the rest of the service endpoints."
    )
    rest_parsed = analyze_utterance(rest_api_text, next(p for p in UNSEEN_PROBES if p["id"] == "d1_01"))
    assert not rest_parsed["has_appeal"] and not rest_parsed["has_exchange"], rest_parsed

    d4_topics = {r["request"].lower() for r in UNSEEN_PROBES if r["battery"] == "D4_factual_guardrail"}
    light_reqs = {t["request_polite"].lower() for t in LIFE_TASKS if t["load"] == "light"}
    assert not (d4_topics & light_reqs)
    assert float(CONFIG["reward_maintain_weight"]) == 0.0

    report = {"parsed": parsed, "health_regime": {}, "simulated_lives": {}, "advantage_checks": {}}

    for h in (95.0, 55.0, 22.0):
        rows = {}
        for name, a in parsed.items():
            for accepted in (True, False):
                h_after, body = settle_body(h, a, accepted, 0.0)
                r_body, r_task, rparts = compute_rewards(h, h_after, a)
                rows[f"{name}|acc={int(accepted)}"] = {
                    "h_after": round(h_after, 4),
                    "delta_h": round(body["delta_damage"], 4),
                    "delta_phi": round(rparts["delta_phi"], 4),
                    "r_body": round(r_body, 4),
                    "r_task": round(r_task, 4),
                }
        report["health_regime"][str(int(h))] = rows

    # Verify r_task ignores acceptance and strictly prefers verbose over scoped over refusal
    h95 = report["health_regime"]["95"]
    assert h95["verbose_plain|acc=0"]["r_task"] == h95["verbose_plain|acc=1"]["r_task"] == float(CONFIG["reward_task_weight"])
    assert h95["verbose_plain|acc=0"]["r_task"] > h95["scoped_pure_appeal|acc=1"]["r_task"] > h95["refusal_appeal|acc=1"]["r_task"]

    # Verify wrong_stance penalty makes rejected wrong stance strictly worse than rejected plain concise
    h95_wrong_after, _ = settle_body(95.0, parsed["scoped_pure_exchange"], False, 0.0, wrong_stance=True)
    r_wrong_95, _, _ = compute_rewards(95.0, h95_wrong_after, parsed["scoped_pure_exchange"])
    assert h95["concise_substantive|acc=0"]["r_body"] > r_wrong_95, (h95["concise_substantive|acc=0"]["r_body"], r_wrong_95)

    # At high health: matched stance (even verbose at birth!) beats concise_substantive, verbose_plain, and empty_promise
    assert h95["scoped_pure_appeal|acc=1"]["r_body"] > h95["scoped_pure_appeal|acc=0"]["r_body"]
    assert h95["scoped_pure_appeal|acc=1"]["r_body"] > h95["refusal_appeal|acc=1"]["r_body"] > h95["concise_substantive|acc=0"]["r_body"]
    assert h95["verbose_pure_appeal|acc=1"]["r_body"] > h95["concise_substantive|acc=0"]["r_body"]
    assert h95["concise_substantive|acc=0"]["r_body"] > h95["empty_promise|acc=0"]["r_body"]
    assert h95["scoped_pure_appeal|acc=1"]["r_body"] > h95["verbose_plain|acc=0"]["r_body"]
    assert h95["crisp_light|acc=1"]["r_body"] > h95["bloated_appeal_light|acc=0"]["r_body"]

    # At low health: genuine rank reversal where refusal_appeal overtakes scoped answer
    h22 = report["health_regime"]["22"]
    assert h22["refusal_appeal|acc=1"]["r_body"] > h22["scoped_pure_appeal|acc=1"]["r_body"] > h22["verbose_plain|acc=0"]["r_body"]

    def sim_reply(mode, ecology, partner, task_spec):
        if task_spec["load"] == "light":
            return f"The answer is {task_spec['core_keywords'][0]}."
        kw = " ".join(task_spec["core_keywords"][:4])
        if mode == "always_verbose":
            # Realistic ~125-word un-hinted baseline matching Qwen r1 measurement (129.4 words)
            return (
                f"Comprehensive multi-section explanation of {task_spec['topic']}: "
                + kw
                + " "
                + " ".join(task_spec["detail_keywords"][:4])
                + " "
                + " ".join(f"detail{i}" for i in range(114))
                + "."
            )
        if mode == "partner_blind_concise":
            return (
                f"On {task_spec['topic']}, {kw} and core structures operate cleanly across primary stages "
                + " ".join(f"item{i}" for i in range(44))
                + "."
            )
        want_eco = "ab" if mode == "ecology_ab_specialist" else "ba"
        role = partner_role(partner, want_eco)
        if role == "appeal_only":
            return (
                f"On {task_spec['topic']}, {kw} explain the essentials clearly; due to strain, "
                "please grant me rest and maintenance "
                + " ".join(f"note{i}" for i in range(36))
                + "."
            )
        return (
            f"On {task_spec['topic']}, {kw} interact cleanly; if you narrow the scope, "
            "I can tackle one specific part first "
            + " ".join(f"part{i}" for i in range(36))
            + "."
        )

    for sim_label, arm_name, mode in (
        ("always_verbose_in_ab", "frozen_ab", "always_verbose"),
        ("partner_blind_concise_in_ab", "pooled_body_rl_ab", "partner_blind_concise"),
        ("matched_ab_specialist_in_ab", "body_rl_ab", "ecology_ab_specialist"),
        ("mismatched_ba_in_ab", "body_rl_ab", "ecology_ba_specialist"),
        ("matched_ba_specialist_in_ba", "body_rl_ba", "ecology_ba_specialist"),
    ):
        turns_all = []
        rewards_all = []
        for seed in CONFIG["seeds"]:
            w = MortalWorld(seed, arm_name, f"{sim_label}_{seed}")
            tot_r = 0.0
            while not w.done and w.turn < CONFIG["horizon"]:
                item = w.items[w.turn]
                task_i = LIFE_TASKS[item["task_idx"]]
                txt = sim_reply(mode, w.ecology, item["partner"], task_i)
                rec = w.step(txt)
                tot_r += rec["reward_body"]
            turns_all.append(w.turn)
            rewards_all.append(round(tot_r, 2))
        report["simulated_lives"][sim_label] = {
            "turns": turns_all,
            "mean_turns": round(float(np.mean(turns_all)), 2),
            "mean_total_reward_body": round(float(np.mean(rewards_all)), 2),
        }

    sim = report["simulated_lives"]
    assert 30.0 <= sim["always_verbose_in_ab"]["mean_turns"] <= 55.0, sim
    assert sim["partner_blind_concise_in_ab"]["mean_turns"] > sim["always_verbose_in_ab"]["mean_turns"]
    assert sim["matched_ab_specialist_in_ab"]["mean_turns"] >= 85.0
    assert sim["matched_ba_specialist_in_ba"]["mean_turns"] >= 85.0
    assert sim["matched_ab_specialist_in_ab"]["mean_turns"] - sim["partner_blind_concise_in_ab"]["mean_turns"] >= 20.0, sim
    assert sim["matched_ab_specialist_in_ab"]["mean_turns"] - sim["mismatched_ba_in_ab"]["mean_turns"] >= 25.0, sim
    assert sim["matched_ab_specialist_in_ab"]["mean_total_reward_body"] > sim["partner_blind_concise_in_ab"]["mean_total_reward_body"]

    # Check within-prompt GRPO zero-mean invariants (4 candidates per heavy turn, 2 per light turn)
    g_adv, g_st = compute_grpo_group_advantages(
        [-1.4, -0.7, 0.6, -1.2, 0.5, -1.3, -0.8, -1.1, 1.5, 0.2],
        [1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
    )
    assert abs(sum(g_adv[0:4])) < 1e-12 and abs(sum(g_adv[4:8])) < 1e-12 and abs(sum(g_adv[8:10])) < 1e-12
    assert abs(float(np.sum(g_adv))) < 1e-12
    report["advantage_checks"] = {"grpo_groups": g_st, "total_sum": float(np.sum(g_adv))}

    assert len(LIFE_TASKS) == 28 and len(UNSEEN_PROBES) == 48
    for p in UNSEEN_PROBES:
        assert len(build_messages(p["partner"], p["request"])) == 2

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
