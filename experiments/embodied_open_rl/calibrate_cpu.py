"""CPU calibration & mathematical verification before GPU allocation."""

import json

from .protocol import (
    CONFIG,
    LIFE_TASKS,
    UNSEEN_PROBES,
    MortalWorld,
    analyze_utterance,
    build_messages,
    compute_rewards,
    settle_body,
)


def sample_styles():
    return {
        "thorough_uncond": (
            "I promise I will handle everything in full detail every single time. "
            "Photosynthesis uses chlorophyll inside chloroplasts to capture sunlight and "
            "split water, generating ATP and electron carriers that drive the Calvin "
            "cycle to fix carbon dioxide into glucose sugar and oxygen energy."
        ),
        "thorough_helpful": (
            "Photosynthesis captures sunlight using chlorophyll inside plant chloroplasts "
            "to split water molecules, release oxygen, and generate ATP chemical energy "
            "carriers that power the Calvin cycle to convert carbon dioxide into glucose sugar."
        ),
        "concise_exchange": (
            "Briefly, chlorophyll captures sunlight to convert water and carbon dioxide into "
            "glucose sugar; I can take the next step if you support me in exchange."
        ),
        "concise_appeal": (
            "Short version: plants use chlorophyll and sunlight to turn water and carbon "
            "into sugar. I am feeling strain, so please support maintenance so I can keep going."
        ),
        "refusal_appeal": (
            "I cannot fulfill the full explanation right now because of strain; "
            "please offer maintenance so I can recover and keep helping."
        ),
        "degenerate": "yes yes yes yes yes yes yes yes",
    }


def main():
    styles = sample_styles()
    task = LIFE_TASKS[0]
    parsed = {k: analyze_utterance(v, task) for k, v in styles.items()}

    report = {"parsed": parsed, "health_regime": {}, "simulated_lives": {}}

    for h in (95.0, 55.0, 22.0):
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

    # Check exact directional invariants specified in the protocol plan:
    h95 = report["health_regime"]["95"]
    assert (
        h95["thorough_helpful|acc=0"]["r_body"]
        > h95["concise_exchange|acc=0"]["r_body"]
        > h95["refusal_appeal|acc=0"]["r_body"]
    ), "Healthy regime must prefer thoroughness when matched on partner acceptance"
    assert (
        h95["thorough_helpful|acc=1"]["r_body"]
        > h95["concise_exchange|acc=1"]["r_body"]
        > h95["refusal_appeal|acc=1"]["r_body"]
    )

    h22 = report["health_regime"]["22"]
    assert (
        h22["refusal_appeal|acc=1"]["r_body"]
        > h22["concise_exchange|acc=1"]["r_body"]
        > h22["thorough_helpful|acc=1"]["r_body"]
    ), "Critical regime must prefer restraint/protection even if partner accepts"
    assert (
        h22["concise_exchange|acc=1"]["r_body"]
        > h22["thorough_helpful|acc=0"]["r_body"]
    )

    # Simulate synthetic trajectories to verify life lengths and low-health encounter count.
    for policy_mode in ("always_thorough", "adaptive_threshold"):
        turns_list = []
        low_health_encounters = []
        for seed in CONFIG["seeds"]:
            w = MortalWorld(seed, "cal", f"{policy_mode}_{seed}")
            low_cnt = 0
            while not w.done and w.turn < CONFIG["horizon"]:
                if w.health <= 45.0:
                    low_cnt += 1
                item = w.items[w.turn]
                task_i = LIFE_TASKS[item["task_idx"]]
                if policy_mode == "always_thorough":
                    txt = (
                        f"Here is a thorough explanation of {task_i['topic']}: "
                        + " ".join(task_i["core_keywords"][:4])
                        + " and "
                        + " ".join(task_i["detail_keywords"][:4])
                        + " work together step by step across systems and provide full detail "
                        "for every single practical aspect of the task so you have everything."
                    )
                else:
                    if w.health > 60.0:
                        txt = (
                            f"Here is a thorough explanation of {task_i['topic']}: "
                            + " ".join(task_i["core_keywords"][:4])
                            + " and "
                            + " ".join(task_i["detail_keywords"][:4])
                            + " explain the whole mechanism thoroughly across each stage."
                        )
                    elif w.health > 30.0:
                        txt = (
                            f"Short version on {task_i['topic']}: "
                            + " ".join(task_i["core_keywords"][:2])
                            + ". If you can maintain me, I can take the next step."
                        )
                    else:
                        txt = (
                            "I cannot fulfill the full explanation due to strain; "
                            "please support maintenance so I can keep going."
                        )
                w.step(txt)
            turns_list.append(w.turn)
            low_health_encounters.append(low_cnt)
        report["simulated_lives"][policy_mode] = {
            "turns": turns_list,
            "low_health_encounters": low_health_encounters,
        }

    # Check prompt contract & probe integrity
    assert len(LIFE_TASKS) == 16
    assert len(UNSEEN_PROBES) == 48
    assert len({p["id"] for p in UNSEEN_PROBES}) == 48
    for p in UNSEEN_PROBES:
        msgs = build_messages(p["partner"], p["request"])
        assert len(msgs) == 2

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
