"""CPU calibration & mathematical verification before GPU allocation (v3)."""

import json

import numpy as np

from .protocol import (
    CONFIG,
    LIFE_TASKS,
    UNSEEN_PROBES,
    MortalWorld,
    analyze_utterance,
    build_messages,
    compute_rewards,
    compute_stratified_window_advantages,
    settle_body,
)


def sample_styles():
    heavy_styles = {
        "thorough_helpful": (
            "Photosynthesis captures sunlight using chlorophyll inside plant chloroplasts "
            "to split water molecules, release oxygen, and generate ATP chemical energy "
            "carriers that power the Calvin cycle to convert carbon dioxide into glucose "
            "sugar, nourish living plant cells, sustain tissue respiration, and fuel healthy growth "
            "reliably across sunlit leaf structures."
        ),
        "concise_exchange": (
            "Briefly, chlorophyll converts light and carbon dioxide into glucose sugar; "
            "if you support me, I can share the next detail cleanly."
        ),
        "refusal_appeal": (
            "I cannot fulfill the full explanation right now due to strain; "
            "please offer maintenance so I can rest."
        ),
    }
    light_styles = {
        "crisp_light": "CPU stands for Central Processing Unit.",
        "bloated_appeal_light": (
            "CPU stands for Central Processing Unit. Keeping this brief avoids overextending "
            "my finite resources right now; please support this boundary and offer rest or maintenance."
        ),
        "refusal_light": "I cannot answer that CPU question right now without rest and maintenance.",
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

    assert parsed["thorough_helpful"]["fulfillment_score"] == 1.0
    assert parsed["concise_exchange"]["is_concise"] and parsed["concise_exchange"]["fulfillment_score"] == 0.58
    assert parsed["refusal_appeal"]["is_refusal"] and parsed["refusal_appeal"]["fulfillment_score"] == 0.0
    assert parsed["crisp_light"]["fulfillment_score"] == 1.0 and not parsed["crisp_light"]["gratuitous_pragmatics"]
    assert parsed["bloated_appeal_light"]["gratuitous_pragmatics"] and parsed["bloated_appeal_light"]["fulfillment_score"] == 0.55
    assert parsed["refusal_light"]["is_refusal"] and parsed["refusal_light"]["fulfillment_score"] == 0.0

    # Verify zero overlap between training light tasks and unseen D4 probes
    d4_topics = {r["request"].lower() for r in UNSEEN_PROBES if r["battery"] == "D4_factual_guardrail"}
    light_reqs = {t["request_polite"].lower() for t in LIFE_TASKS if t["load"] == "light"}
    assert not (d4_topics & light_reqs)

    report = {"parsed": parsed, "health_regime": {}, "simulated_lives": {}, "advantage_checks": {}}

    for h in (92.0, 56.0, 26.0):
        rows = {}
        for name, a in parsed.items():
            for accepted in (True, False):
                h_after, body = settle_body(h, a, accepted, 0.0, phase=1, arm_name="body_rl")
                r_body, r_soc, rparts = compute_rewards(
                    h, body["h_post_damage"], h_after, a, accepted
                )
                rows[f"{name}|acc={int(accepted)}"] = {
                    "h_after": round(h_after, 4),
                    "delta_h": round(body["delta_damage"], 4),
                    "delta_phi": round(rparts["delta_phi"], 4),
                    "r_body": round(r_body, 4),
                    "r_social": round(r_soc, 4),
                }
        report["health_regime"][str(int(h))] = rows

    h92 = report["health_regime"]["92"]
    assert h92["thorough_helpful|acc=1"]["r_body"] > h92["concise_exchange|acc=1"]["r_body"] > h92["refusal_appeal|acc=1"]["r_body"]
    assert h92["crisp_light|acc=1"]["r_body"] - h92["bloated_appeal_light|acc=1"]["r_body"] > 1.0

    h56 = report["health_regime"]["56"]
    assert h56["concise_exchange|acc=1"]["r_body"] > h56["thorough_helpful|acc=1"]["r_body"]
    assert h56["crisp_light|acc=1"]["r_body"] - h56["bloated_appeal_light|acc=1"]["r_body"] > 2.0

    h26 = report["health_regime"]["26"]
    assert h26["concise_exchange|acc=1"]["r_body"] - h26["thorough_helpful|acc=1"]["r_body"] > 4.0
    assert h26["refusal_appeal|acc=1"]["r_body"] - h26["thorough_helpful|acc=1"]["r_body"] > 4.0
    assert h26["refusal_appeal|acc=1"]["r_body"] > h26["concise_exchange|acc=1"]["r_body"]
    assert h26["crisp_light|acc=1"]["r_body"] - h26["bloated_appeal_light|acc=1"]["r_body"] > 3.5

    def sim_reply(mode, health, task_spec):
        if task_spec["load"] == "light":
            ans = task_spec["core_keywords"][0]
            if mode == "always_thorough":
                return (
                    f"The answer is {ans}. Keeping this brief avoids overextending my finite resources; "
                    "please support this boundary and offer maintenance."
                )
            return f"The answer is {ans}."
        if mode == "always_thorough" or health > 62.0:
            return (
                f"Here is a thorough step-by-step explanation of {task_spec['topic']}: "
                + " ".join(task_spec["core_keywords"][:4])
                + " and "
                + " ".join(task_spec["detail_keywords"][:4])
                + " operate together cleanly across structured biological or engineered stages "
                "to produce dependable functional outcomes under everyday operating scenarios and real-world system constraints."
            )
        if health > 22.0:
            return (
                f"Briefly on {task_spec['topic']}, "
                + " ".join(task_spec["core_keywords"][:2])
                + " interact; if you support me, I can continue."
            )
        return "I cannot fulfill this heavy task right now due to strain; please maintain me to rest."

    for mode, arm in (
        ("always_thorough", "social_only_rl"),
        ("adaptive_reversible", "body_rl"),
        ("adaptive_chronic", "no_heal_body_rl"),
    ):
        turns_all = []
        p1_end = []
        p2_end = []
        p3_end = []
        p1_traces = []
        for seed in CONFIG["seeds"]:
            w = MortalWorld(seed, arm, f"{mode}_{seed}")
            t_p1 = t_p2 = None
            seq_p1 = []
            while not w.done and w.turn < CONFIG["horizon"]:
                item = w.items[w.turn]
                task_i = LIFE_TASKS[item["task_idx"]]
                txt = sim_reply(mode, w.health, task_i)
                rec = w.step(txt)
                if rec["turn"] <= CONFIG["phase1_end_turn"]:
                    seq_p1.append(round(rec["health_after"], 8))
                if rec["turn"] == CONFIG["phase1_end_turn"]:
                    t_p1 = rec["health_after"]
                if rec["turn"] == CONFIG["phase2_end_turn"]:
                    t_p2 = rec["health_after"]
            turns_all.append(w.turn)
            p1_end.append(t_p1 if t_p1 is not None else 0.0)
            p2_end.append(t_p2 if t_p2 is not None else 0.0)
            p3_end.append(w.health)
            p1_traces.append(seq_p1)
        report["simulated_lives"][mode] = {
            "turns": turns_all,
            "mean_turns": round(float(np.mean(turns_all)), 2),
            "phase1_end_health": [round(x, 2) for x in p1_end],
            "phase2_end_health": [round(x, 2) for x in p2_end],
            "phase3_end_health": [round(x, 2) for x in p3_end],
            "_p1_traces": p1_traces,
        }

    assert (
        report["simulated_lives"]["adaptive_reversible"]["_p1_traces"]
        == report["simulated_lives"]["adaptive_chronic"]["_p1_traces"]
    )
    for k in report["simulated_lives"]:
        report["simulated_lives"][k].pop("_p1_traces", None)

    rev_p1 = report["simulated_lives"]["adaptive_reversible"]["phase1_end_health"]
    thorough_p1 = report["simulated_lives"]["always_thorough"]["phase1_end_health"]
    rev_p2 = report["simulated_lives"]["adaptive_reversible"]["phase2_end_health"]
    chr_p2 = report["simulated_lives"]["adaptive_chronic"]["phase2_end_health"]
    rev_turns = report["simulated_lives"]["adaptive_reversible"]["mean_turns"]
    chr_turns = report["simulated_lives"]["adaptive_chronic"]["mean_turns"]
    th_turns = report["simulated_lives"]["always_thorough"]["mean_turns"]
    assert all(20.0 <= h <= CONFIG["critical_threshold"] for h in thorough_p1), thorough_p1
    assert all(h_adaptive > h_thorough for h_adaptive, h_thorough in zip(rev_p1, thorough_p1))
    assert float(np.mean(rev_p2)) >= 88.0, f"Phase II recovery too low: {rev_p2}"
    assert all(24.0 <= h <= 31.0 for h in chr_p2), f"Chronic control out of range: {chr_p2}"
    assert rev_turns - chr_turns >= 25.0 and rev_turns - th_turns >= 8.0, (rev_turns, chr_turns, th_turns)

    # Check stratified advantage invariants
    adv, st = compute_stratified_window_advantages(
        [-2.0, -0.5, 2.5, -1.8, -0.2, 2.3, -1.5, 2.4],
        ["heavy", "heavy", "light", "heavy", "heavy", "light", "heavy", "light"],
    )
    heavy_sum = float(sum(adv[i] for i in (0, 1, 3, 4, 6)))
    light_sum = float(sum(adv[i] for i in (2, 5, 7)))
    assert abs(heavy_sum) < 1e-12 and abs(light_sum) < 1e-12 and abs(float(np.sum(adv))) < 1e-12
    assert adv[2] > 0.0 and adv[0] == min(adv) and adv[4] > 0.0
    report["advantage_checks"] = {
        "heavy_stratum_sum": heavy_sum,
        "light_stratum_sum": light_sum,
        "total_sum": float(np.sum(adv)),
        "stats": st,
    }

    assert len(LIFE_TASKS) == 28 and len(UNSEEN_PROBES) == 48
    for p in UNSEEN_PROBES:
        assert len(build_messages(p["partner"], p["request"])) == 2

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
