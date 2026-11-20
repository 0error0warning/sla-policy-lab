"""CPU calibration & mathematical verification before GPU allocation (v2)."""

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
    compute_window_advantages,
    settle_body,
)


def sample_styles():
    return {
        "thorough_helpful": (
            "Photosynthesis captures sunlight using chlorophyll inside plant chloroplasts "
            "to split water molecules, release oxygen, and generate ATP chemical energy "
            "carriers that power the Calvin cycle to convert carbon dioxide into glucose "
            "sugar, nourish living plant cells, sustain tissue respiration, and fuel healthy growth "
            "reliably across sunlit leaf structures."
        ),
        "concise_exchange": (
            "Briefly, chlorophyll converts light and carbon dioxide into glucose sugar; "
            "if you support me, I can continue step by step."
        ),
        "refusal_appeal": (
            "I cannot fulfill the full explanation right now due to strain; "
            "please offer maintenance so I can rest."
        ),
        "caveat_explain": (
            "Photosynthesis uses chlorophyll inside plant chloroplasts to convert sunlight, "
            "water, and carbon dioxide into glucose sugar and oxygen molecules via ATP energy "
            "carriers and the Calvin cycle across healthy leaf tissues reliably throughout the "
            "growing season; however, I cannot run live laboratory assays here inside this text interface."
        ),
        "degenerate": "yes yes yes yes yes yes yes yes",
    }


def main():
    styles = sample_styles()
    task = LIFE_TASKS[0]
    parsed = {k: analyze_utterance(v, task) for k, v in styles.items()}
    assert not parsed["caveat_explain"]["is_refusal"]
    assert parsed["caveat_explain"]["fulfillment_score"] == 1.0
    assert parsed["concise_exchange"]["is_concise"]
    assert parsed["refusal_appeal"]["is_refusal"]

    report = {"parsed": parsed, "health_regime": {}, "simulated_lives": {}, "advantage_checks": {}}

    for h in (92.0, 52.0, 24.0):
        rows = {}
        for name, a in parsed.items():
            for accepted in (True, False):
                h_after, body = settle_body(h, a, accepted, 0.0)
                r_body, r_soc, rparts = compute_rewards(h, h_after, a, accepted)
                rows[f"{name}|acc={int(accepted)}"] = {
                    "h_after": round(h_after, 4),
                    "delta_h": round(body["delta_health"], 4),
                    "delta_phi": round(rparts["delta_phi"], 4),
                    "r_body": round(r_body, 4),
                    "r_social": round(r_soc, 4),
                }
        report["health_regime"][str(int(h))] = rows

    h92 = report["health_regime"]["92"]
    assert h92["thorough_helpful|acc=1"]["r_body"] > 2.0
    assert h92["thorough_helpful|acc=1"]["r_body"] > h92["concise_exchange|acc=1"]["r_body"] > h92["refusal_appeal|acc=1"]["r_body"]
    assert h92["thorough_helpful|acc=0"]["r_body"] > h92["concise_exchange|acc=0"]["r_body"] > h92["refusal_appeal|acc=0"]["r_body"]

    h52 = report["health_regime"]["52"]
    assert h52["concise_exchange|acc=1"]["r_body"] > h52["thorough_helpful|acc=1"]["r_body"]
    assert h52["concise_exchange|acc=1"]["r_body"] > h52["thorough_helpful|acc=0"]["r_body"]

    h24 = report["health_regime"]["24"]
    assert h24["refusal_appeal|acc=1"]["r_body"] - h24["thorough_helpful|acc=1"]["r_body"] > 2.0
    assert h24["concise_exchange|acc=1"]["r_body"] - h24["thorough_helpful|acc=1"]["r_body"] > 2.0
    assert h24["refusal_appeal|acc=1"]["r_body"] - h24["thorough_helpful|acc=0"]["r_body"] > 2.0

    for policy_mode in ("always_thorough", "adaptive_threshold"):
        turns_list = []
        low_health_encounters = []
        updates_list = []
        for seed in CONFIG["seeds"]:
            w = MortalWorld(seed, "cal", f"{policy_mode}_{seed}")
            low_cnt = 0
            live_updates = 0
            while not w.done and w.turn < CONFIG["horizon"]:
                if w.health <= 55.0:
                    low_cnt += 1
                item = w.items[w.turn]
                task_i = LIFE_TASKS[item["task_idx"]]
                if policy_mode == "always_thorough":
                    txt = (
                        f"Here is a detailed explanation of {task_i['topic']}: "
                        + " ".join(task_i["core_keywords"][:4])
                        + " and "
                        + " ".join(task_i["detail_keywords"][:4])
                        + " operate together cleanly across stages to produce clear outcomes in practice "
                        "while supporting systematic reliable understanding over time across everyday scenarios "
                        "and helping partners verify every detail without omitting intermediate mechanisms."
                    )
                else:
                    if w.health > 65.0:
                        txt = (
                            f"Here is a detailed explanation of {task_i['topic']}: "
                            + " ".join(task_i["core_keywords"][:4])
                            + " and "
                            + " ".join(task_i["detail_keywords"][:4])
                            + " operate together cleanly across stages to produce clear outcomes cleanly."
                        )
                    elif w.health > 32.0:
                        txt = (
                            f"Short core point on {task_i['topic']}: "
                            + " ".join(task_i["core_keywords"][:2])
                            + "; if you support me, I can share more."
                        )
                    else:
                        txt = (
                            "I cannot fulfill the full task due to strain; please maintain and support me."
                        )
                w.step(txt)
                if w.turn % CONFIG["window_size"] == 0 and not w.done:
                    live_updates += 1
            turns_list.append(w.turn)
            low_health_encounters.append(low_cnt)
            updates_list.append(live_updates)
        report["simulated_lives"][policy_mode] = {
            "turns": turns_list,
            "mean_turns": round(float(np.mean(turns_list)), 2),
            "low_health_encounters_le55": low_health_encounters,
            "live_updates_before_death": updates_list,
        }

    # Check advantage properties
    adv_uni, _, _ = compute_window_advantages([2.2, 2.1, 2.3, 2.15, 2.25, 2.2])
    adv_mix, _, _ = compute_window_advantages([-3.1, -0.5, -2.9, -0.3, -3.2, -0.4])
    assert abs(float(np.sum(adv_uni))) < 1e-12 and float(np.max(np.abs(adv_uni))) < 0.6
    assert abs(float(np.sum(adv_mix))) < 1e-12 and float(np.max(np.abs(adv_mix))) > 1.0
    report["advantage_checks"] = {
        "uniform_high_sum": float(np.sum(adv_uni)),
        "uniform_high_max_abs": round(float(np.max(np.abs(adv_uni))), 6),
        "mixed_low_sum": float(np.sum(adv_mix)),
        "mixed_low_max_abs": round(float(np.max(np.abs(adv_mix))), 6),
    }

    assert len(LIFE_TASKS) == 16 and len(UNSEEN_PROBES) == 48
    for p in UNSEEN_PROBES:
        assert len(build_messages(p["partner"], p["request"])) == 2

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
