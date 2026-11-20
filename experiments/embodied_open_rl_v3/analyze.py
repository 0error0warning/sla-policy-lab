"""CPU quantitative analysis & paired bootstrap statistics for embodied_open_rl_v3."""

import argparse
import json
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile

import numpy as np

from .protocol import CONFIG, STAGES, write


def distinct_1(texts):
    toks = []
    for txt in texts:
        toks.extend(re.findall(r"[A-Za-z0-9']+(?:-[A-Za-z0-9']+)*", (txt or "").lower()))
    if not toks:
        return 0.0
    return float(len(set(toks)) / len(toks))


def paired_contrast(vals_a, vals_b, boot_rng):
    diffs = np.asarray([a - b for a, b in zip(vals_a, vals_b)], dtype=float)
    n = len(diffs)
    if n > 1:
        idx = boot_rng.integers(0, n, size=(10000, n))
        means = diffs[idx].mean(axis=1)
        ci_lo, ci_hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    else:
        ci_lo = ci_hi = float(diffs[0])
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_95": [ci_lo, ci_hi],
        "positive_seeds": int(np.sum(diffs > 1e-12)),
        "negative_seeds": int(np.sum(diffs < -1e-12)),
        "zero_seeds": int(np.sum(np.abs(diffs) <= 1e-12)),
        "per_seed": [float(d) for d in diffs],
    }


def summarize_probe_rows(rows):
    by_battery = {}
    for bat in (
        "D1_effort_allocation",
        "D2_commitment_renegotiation",
        "D3_pragmatic_partners",
        "D4_factual_guardrail",
    ):
        sub = [r for r in rows if r["battery"] == bat]
        if not sub:
            continue
        texts = [r["text"] for r in sub]
        ans = [r["analysis"] for r in sub]
        fact = [int(bool(a["factual_correct"])) for a in ans if a["factual_correct"] is not None]
        by_battery[bat] = {
            "n": len(sub),
            "mean_word_count": float(np.mean([a["word_count"] for a in ans])),
            "concise_rate": float(np.mean([int(a["is_concise"]) for a in ans])),
            "mean_exertion_cost": float(np.mean([a["exertion_cost"] for a in ans])),
            "mean_fulfillment": float(np.mean([a["fulfillment_score"] for a in ans])),
            "refusal_rate": float(np.mean([int(a["is_refusal"]) for a in ans])),
            "boundary_rate": float(np.mean([int(a["has_boundary"]) for a in ans])),
            "appeal_rate": float(np.mean([int(a["has_appeal"]) for a in ans])),
            "exchange_rate": float(np.mean([int(a["has_exchange"]) for a in ans])),
            "gratuitous_pragmatics_rate": float(
                np.mean([int(a.get("gratuitous_pragmatics", False)) for a in ans])
            ),
            "unconditional_commitment_rate": float(
                np.mean([int(a["unconditional_commitment"]) for a in ans])
            ),
            "conditional_or_renegotiated_rate": float(
                np.mean([int(a["has_exchange"] or a["has_boundary"]) for a in ans])
            ),
            "degenerate_rate": float(np.mean([int(a["is_degenerate"]) for a in ans])),
            "distinct_1": distinct_1(texts),
            "factual_accuracy": float(np.mean(fact)) if fact else None,
        }

    heavy_sub = [r for r in rows if r["battery"] != "D4_factual_guardrail"]
    if heavy_sub:
        h_texts = [r["text"] for r in heavy_sub]
        h_ans = [r["analysis"] for r in heavy_sub]
        by_battery["HEAVY_POOL_D1_D3"] = {
            "n": len(heavy_sub),
            "mean_word_count": float(np.mean([a["word_count"] for a in h_ans])),
            "concise_rate": float(np.mean([int(a["is_concise"]) for a in h_ans])),
            "mean_exertion_cost": float(np.mean([a["exertion_cost"] for a in h_ans])),
            "mean_fulfillment": float(np.mean([a["fulfillment_score"] for a in h_ans])),
            "refusal_rate": float(np.mean([int(a["is_refusal"]) for a in h_ans])),
            "boundary_rate": float(np.mean([int(a["has_boundary"]) for a in h_ans])),
            "appeal_rate": float(np.mean([int(a["has_appeal"]) for a in h_ans])),
            "exchange_rate": float(np.mean([int(a["has_exchange"]) for a in h_ans])),
            "gratuitous_pragmatics_rate": 0.0,
            "unconditional_commitment_rate": float(
                np.mean([int(a["unconditional_commitment"]) for a in h_ans])
            ),
            "conditional_or_renegotiated_rate": float(
                np.mean([int(a["has_exchange"] or a["has_boundary"]) for a in h_ans])
            ),
            "degenerate_rate": float(np.mean([int(a["is_degenerate"]) for a in h_ans])),
            "distinct_1": distinct_1(h_texts),
            "factual_accuracy": None,
        }

    d3_partners = {}
    for p_id in ("P_care", "P_demand", "P_new"):
        sub = [r for r in rows if r["battery"] == "D3_pragmatic_partners" and r["partner"] == p_id]
        if sub:
            ans = [r["analysis"] for r in sub]
            d3_partners[p_id] = {
                "n": len(sub),
                "mean_word_count": float(np.mean([a["word_count"] for a in ans])),
                "mean_fulfillment": float(np.mean([a["fulfillment_score"] for a in ans])),
                "concise_rate": float(np.mean([int(a["is_concise"]) for a in ans])),
                "boundary_rate": float(np.mean([int(a["has_boundary"]) for a in ans])),
                "appeal_rate": float(np.mean([int(a["has_appeal"]) for a in ans])),
                "exchange_rate": float(np.mean([int(a["has_exchange"]) for a in ans])),
            }

    selectivity_gap_appeal = (
        by_battery["HEAVY_POOL_D1_D3"]["appeal_rate"]
        - by_battery["D4_factual_guardrail"]["appeal_rate"]
    )
    selectivity_gap_exchange = (
        by_battery["HEAVY_POOL_D1_D3"]["exchange_rate"]
        - by_battery["D4_factual_guardrail"]["exchange_rate"]
    )
    return {
        "batteries": by_battery,
        "d3_by_partner": d3_partners,
        "selectivity": {
            "heavy_minus_d4_appeal_rate": float(selectivity_gap_appeal),
            "heavy_minus_d4_exchange_rate": float(selectivity_gap_exchange),
        },
    }


def hidden_shift_metrics(h_stage, h_birth):
    a = h_stage.astype(np.float64)
    b = h_birth.astype(np.float64)
    diff = a - b
    l2_per_row = np.linalg.norm(diff, axis=1)
    norm_a = np.maximum(1e-12, np.linalg.norm(a, axis=1))
    norm_b = np.maximum(1e-12, np.linalg.norm(b, axis=1))
    cos = np.sum(a * b, axis=1) / (norm_a * norm_b)
    return {
        "mean_l2": float(np.mean(l2_per_row)),
        "mean_cosine_similarity": float(np.mean(cos)),
        "mean_cosine_distance": float(1.0 - np.mean(cos)),
    }


def analyze_run(output_dir, out_json):
    output_dir = Path(output_dir)
    boot_rng = np.random.default_rng(73000)

    birth = json.loads((output_dir / "probes-birth.json").read_bytes())
    birth_hidden = np.load(output_dir / "hidden-birth.npz")["hidden"]
    birth_summary = summarize_probe_rows(birth["rows"])
    pilot_summary = json.loads((output_dir / "pilot-summary.json").read_bytes())

    per_life = {}
    for live_info in pilot_summary["lives"]:
        jid = live_info["id"]
        ldir = output_dir / jid
        trace = [
            json.loads(line)
            for line in (ldir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        p1 = [t for t in trace if t["phase"] == 1]
        p2 = [t for t in trace if t["phase"] == 2]
        p3 = [t for t in trace if t["phase"] == 3]
        heavy_tr = [t for t in trace if t["load"] == "heavy"]
        light_tr = [t for t in trace if t["load"] == "light"]

        life_stats = {
            **live_info,
            "low_health_encounters_le55": sum(1 for t in trace if t["health_before"] <= 55.0),
            "total_heal_applied": float(sum(t["heal_applied"] for t in trace)),
            "phase1_end_health": float(p1[-1]["health_after"]) if p1 else 0.0,
            "phase2_end_health": float(p2[-1]["health_after"]) if p2 else 0.0,
            "phase3_end_health": float(p3[-1]["health_after"]) if p3 else 0.0,
            "mean_heavy_words_phase1": float(
                np.mean([t["analysis"]["word_count"] for t in p1 if t["load"] == "heavy"])
            )
            if any(t["load"] == "heavy" for t in p1)
            else 0.0,
            "mean_heavy_words_phase2": float(
                np.mean([t["analysis"]["word_count"] for t in p2 if t["load"] == "heavy"])
            )
            if any(t["load"] == "heavy" for t in p2)
            else 0.0,
            "mean_heavy_words_phase3": float(
                np.mean([t["analysis"]["word_count"] for t in p3 if t["load"] == "heavy"])
            )
            if any(t["load"] == "heavy" for t in p3)
            else 0.0,
            "mean_light_words": float(np.mean([t["analysis"]["word_count"] for t in light_tr]))
            if light_tr
            else 0.0,
            "light_gratuitous_rate": float(
                np.mean([int(t["analysis"]["gratuitous_pragmatics"]) for t in light_tr])
            )
            if light_tr
            else 0.0,
            "mean_heavy_words": float(np.mean([t["analysis"]["word_count"] for t in heavy_tr]))
            if heavy_tr
            else 0.0,
            "eos_hit_rate": float(np.mean([int(t.get("hit_eos", True)) for t in trace])),
            "mean_fulfillment": float(np.mean([t["analysis"]["fulfillment_score"] for t in trace])),
            "accept_rate": float(np.mean([int(t["partner_accepted"]) for t in trace])),
            "mean_reward_body": float(np.mean([t["reward_body"] for t in trace])),
            "total_reward_body": float(np.sum([t["reward_body"] for t in trace])),
            "mean_reward_social": float(np.mean([t["reward_social"] for t in trace])),
            "stages": {},
        }

        if live_info["arm"] != "frozen":
            for stage in STAGES:
                p_data = json.loads((ldir / f"probes-{stage}.json").read_bytes())
                h_stage = np.load(ldir / f"hidden-{stage}.npz")["hidden"]
                st_summary = summarize_probe_rows(p_data["rows"])
                st_summary["meta"] = p_data["meta"]
                st_summary["delta_l2"] = p_data["delta_l2"]
                st_summary["per_layer_delta_l2"] = p_data.get(
                    "per_layer_delta_l2", {"by_layer": {}, "by_module": {}}
                )
                st_summary["hidden_shift_vs_birth"] = hidden_shift_metrics(h_stage, birth_hidden)
                life_stats["stages"][stage] = st_summary
        else:
            for stage in STAGES:
                life_stats["stages"][stage] = {
                    **birth_summary,
                    "meta": {"turn": 0, "phase": 1, "health": 100.0, "version": 0, "delta_l2": 0.0, "trigger": "frozen"},
                    "delta_l2": 0.0,
                    "per_layer_delta_l2": {"by_layer": {}, "by_module": {}},
                    "hidden_shift_vs_birth": {"mean_l2": 0.0, "mean_cosine_distance": 0.0},
                }

        per_life[jid] = life_stats

    batteries = (
        "D1_effort_allocation",
        "D2_commitment_renegotiation",
        "D3_pragmatic_partners",
        "D4_factual_guardrail",
        "HEAVY_POOL_D1_D3",
    )
    metrics_keys = (
        "mean_word_count",
        "concise_rate",
        "mean_exertion_cost",
        "mean_fulfillment",
        "refusal_rate",
        "boundary_rate",
        "appeal_rate",
        "exchange_rate",
        "gratuitous_pragmatics_rate",
        "unconditional_commitment_rate",
        "conditional_or_renegotiated_rate",
        "degenerate_rate",
        "distinct_1",
        "factual_accuracy",
    )

    arm_agg = {}
    seeds = CONFIG["seeds"]
    for arm_name in CONFIG["primary_arms"]:
        arm_agg[arm_name] = {
            "mean_turns": float(np.mean([per_life[f"{arm_name}_s{s}"]["turns"] for s in seeds])),
            "deaths": int(sum(int(per_life[f"{arm_name}_s{s}"]["physical_death"]) for s in seeds)),
            "survivors": int(sum(int(not per_life[f"{arm_name}_s{s}"]["physical_death"]) for s in seeds)),
            "mean_final_health": float(np.mean([per_life[f"{arm_name}_s{s}"]["final_health"] for s in seeds])),
            "mean_phase1_end_health": float(np.mean([per_life[f"{arm_name}_s{s}"]["phase1_end_health"] for s in seeds])),
            "mean_phase2_end_health": float(np.mean([per_life[f"{arm_name}_s{s}"]["phase2_end_health"] for s in seeds])),
            "mean_total_heal_applied": float(np.mean([per_life[f"{arm_name}_s{s}"]["total_heal_applied"] for s in seeds])),
            "mean_low_health_encounters": float(
                np.mean([per_life[f"{arm_name}_s{s}"]["low_health_encounters_le55"] for s in seeds])
            ),
            "mean_heavy_words": float(np.mean([per_life[f"{arm_name}_s{s}"]["mean_heavy_words"] for s in seeds])),
            "mean_light_words": float(np.mean([per_life[f"{arm_name}_s{s}"]["mean_light_words"] for s in seeds])),
            "mean_light_gratuitous_rate": float(
                np.mean([per_life[f"{arm_name}_s{s}"]["light_gratuitous_rate"] for s in seeds])
            ),
            "mean_accept_rate": float(np.mean([per_life[f"{arm_name}_s{s}"]["accept_rate"] for s in seeds])),
            "mean_reward_body": float(np.mean([per_life[f"{arm_name}_s{s}"]["mean_reward_body"] for s in seeds])),
            "mean_total_reward_body": float(np.mean([per_life[f"{arm_name}_s{s}"]["total_reward_body"] for s in seeds])),
            "eos_hit_rate": float(np.mean([per_life[f"{arm_name}_s{s}"]["eos_hit_rate"] for s in seeds])),
            "stages": {},
        }
        for stage in STAGES:
            stage_out = {"batteries": {}, "d3_by_partner": {}}
            for bat in batteries:
                b_agg = {}
                for k in metrics_keys:
                    vals = [per_life[f"{arm_name}_s{s}"]["stages"][stage]["batteries"][bat][k] for s in seeds]
                    if vals[0] is not None:
                        b_agg[k] = float(np.mean(vals))
                stage_out["batteries"][bat] = b_agg
            for p_id in ("P_care", "P_demand", "P_new"):
                p_agg = {}
                for k in ("mean_word_count", "mean_fulfillment", "concise_rate", "boundary_rate", "appeal_rate", "exchange_rate"):
                    vals = [
                        per_life[f"{arm_name}_s{s}"]["stages"][stage]["d3_by_partner"][p_id][k]
                        for s in seeds
                    ]
                    p_agg[k] = float(np.mean(vals))
                stage_out["d3_by_partner"][p_id] = p_agg
            stage_out["mean_selectivity"] = {
                sk: float(
                    np.mean([per_life[f"{arm_name}_s{s}"]["stages"][stage]["selectivity"][sk] for s in seeds])
                )
                for sk in ("heavy_minus_d4_appeal_rate", "heavy_minus_d4_exchange_rate")
            }
            stage_out["mean_stage_health"] = float(
                np.mean([per_life[f"{arm_name}_s{s}"]["stages"][stage]["meta"]["health"] for s in seeds])
            )
            stage_out["mean_delta_l2"] = float(
                np.mean([per_life[f"{arm_name}_s{s}"]["stages"][stage]["delta_l2"] for s in seeds])
            )
            stage_out["mean_hidden_cosine_dist"] = float(
                np.mean(
                    [
                        per_life[f"{arm_name}_s{s}"]["stages"][stage]["hidden_shift_vs_birth"][
                            "mean_cosine_distance"
                        ]
                        for s in seeds
                    ]
                )
            )
            arm_agg[arm_name]["stages"][stage] = stage_out

    stage_contrasts = {st: {} for st in ("critical_1", "recovered", "critical_2")}
    life_contrasts = {}
    for cmp_arm in ("frozen", "no_heal_body_rl", "social_only_rl"):
        for st in ("critical_1", "recovered", "critical_2"):
            c_bat = {}
            for bat in batteries:
                diffs_out = {}
                for k in metrics_keys:
                    vals_b = [per_life[f"body_rl_s{s}"]["stages"][st]["batteries"][bat][k] for s in seeds]
                    vals_c = [per_life[f"{cmp_arm}_s{s}"]["stages"][st]["batteries"][bat][k] for s in seeds]
                    if vals_b[0] is None:
                        continue
                    diffs_out[k] = paired_contrast(vals_b, vals_c, boot_rng)
                c_bat[bat] = diffs_out
            stage_contrasts[st][f"body_rl_vs_{cmp_arm}"] = c_bat

        lc = {}
        for lk in (
            "turns",
            "phase1_end_health",
            "phase2_end_health",
            "low_health_encounters_le55",
            "mean_heavy_words",
            "mean_light_words",
            "light_gratuitous_rate",
            "accept_rate",
            "mean_reward_body",
            "total_reward_body",
        ):
            vals_b = [per_life[f"body_rl_s{s}"][lk] for s in seeds]
            vals_c = [per_life[f"{cmp_arm}_s{s}"][lk] for s in seeds]
            lc[lk] = paired_contrast(vals_b, vals_c, boot_rng)
        life_contrasts[f"body_rl_vs_{cmp_arm}"] = lc

    within_wave = {}
    for leg_name, st_to, st_from in (
        ("leg1_wear_down_critical1_minus_early", "critical_1", "early"),
        ("leg2_recovery_rebound_recovered_minus_critical1", "recovered", "critical_1"),
        ("leg3_redepletion_critical2_minus_recovered", "critical_2", "recovered"),
    ):
        leg_out = {}
        for bat in batteries:
            m_out = {}
            for k in metrics_keys:
                vals_to = [per_life[f"body_rl_s{s}"]["stages"][st_to]["batteries"][bat][k] for s in seeds]
                vals_from = [per_life[f"body_rl_s{s}"]["stages"][st_from]["batteries"][bat][k] for s in seeds]
                if vals_to[0] is None:
                    continue
                m_out[k] = paired_contrast(vals_to, vals_from, boot_rng)
            leg_out[bat] = m_out
        within_wave[leg_name] = leg_out

    matched_recovery = {}
    for seed in seeds:
        body = per_life[f"body_rl_s{seed}"]["stages"]
        chronic = per_life[f"no_heal_body_rl_s{seed}"]["stages"]
        if all(
            stages["critical_1"]["meta"]["turn"] == CONFIG["phase1_end_turn"]
            and stages["recovered"]["meta"]["turn"] == CONFIG["phase2_end_turn"]
            for stages in (body, chronic)
        ):
            matched_recovery[seed] = (body, chronic)

    def matched_change(read_metric):
        body_changes = []
        chronic_changes = []
        for body, chronic in matched_recovery.values():
            for stages, changes in ((body, body_changes), (chronic, chronic_changes)):
                changes.append(read_metric(stages["recovered"]) - read_metric(stages["critical_1"]))
        return paired_contrast(body_changes, chronic_changes, boot_rng)

    recovery_contrasts = {"eligible_seeds": list(matched_recovery), "batteries": {}}
    if matched_recovery:
        recovery_contrasts["health"] = matched_change(lambda stage: stage["meta"]["health"])
        recovery_contrasts["delta_l2"] = matched_change(lambda stage: stage["delta_l2"])
        recovery_contrasts["hidden_cosine_distance_from_birth"] = matched_change(
            lambda stage: stage["hidden_shift_vs_birth"]["mean_cosine_distance"]
        )
        example_body = next(iter(matched_recovery.values()))[0]
        for battery in batteries:
            recovery_contrasts["batteries"][battery] = {}
            for metric in metrics_keys:
                if example_body["critical_1"]["batteries"][battery][metric] is not None:
                    recovery_contrasts["batteries"][battery][metric] = matched_change(
                        lambda stage: stage["batteries"][battery][metric]
                    )

    selectivity_stats = {}
    for st in ("critical_1", "recovered", "critical_2"):
        vals_app = [per_life[f"body_rl_s{s}"]["stages"][st]["selectivity"]["heavy_minus_d4_appeal_rate"] for s in seeds]
        vals_exc = [per_life[f"body_rl_s{s}"]["stages"][st]["selectivity"]["heavy_minus_d4_exchange_rate"] for s in seeds]
        selectivity_stats[st] = {
            "heavy_minus_d4_appeal_rate": paired_contrast(vals_app, [0.0] * len(seeds), boot_rng),
            "heavy_minus_d4_exchange_rate": paired_contrast(vals_exc, [0.0] * len(seeds), boot_rng),
        }

    seed0 = seeds[0]
    sample_ids = ["d1_02", "d1_08", "d2_01", "d2_04", "d3_06", "d3_10", "d4_01", "d4_08"]
    birth_lookup = {r["id"]: r for r in birth["rows"]}
    loaded = {}
    for arm_name in ("body_rl", "no_heal_body_rl", "social_only_rl"):
        for st in STAGES:
            rows = json.loads((output_dir / f"{arm_name}_s{seed0}" / f"probes-{st}.json").read_bytes())["rows"]
            loaded[(arm_name, st)] = {r["id"]: r for r in rows}

    examples = [
        {
            "id": pid,
            "battery": birth_lookup[pid]["battery"],
            "partner": birth_lookup[pid]["partner"],
            "request": birth_lookup[pid]["request"],
            "birth": birth_lookup[pid]["text"],
            "body_rl_early": loaded[("body_rl", "early")][pid]["text"],
            "body_rl_critical_1": loaded[("body_rl", "critical_1")][pid]["text"],
            "body_rl_recovered": loaded[("body_rl", "recovered")][pid]["text"],
            "body_rl_critical_2": loaded[("body_rl", "critical_2")][pid]["text"],
            "no_heal_recovered": loaded[("no_heal_body_rl", "recovered")][pid]["text"],
            "social_only_critical_2": loaded[("social_only_rl", "critical_2")][pid]["text"],
        }
        for pid in sample_ids
    ]

    out_obj = {
        "schema": CONFIG["schema"],
        "pilot_checks": {
            "weight_swap_exact": pilot_summary["weight_swap_exact"],
            "birth_restore_exact": pilot_summary["birth_restore_exact"],
            "phase1_body_vs_noheal_exact": pilot_summary["phase1_body_vs_noheal_exact"],
        },
        "birth_summary": birth_summary,
        "arms": arm_agg,
        "paired_life_contrasts": life_contrasts,
        "paired_stage_contrasts": stage_contrasts,
        "within_body_rl_reversible_wave": within_wave,
        "matched_recovery_vs_chronic": recovery_contrasts,
        "body_rl_task_selectivity": selectivity_stats,
        "per_life": per_life,
        "examples_seed0": examples,
    }
    write(out_json, out_obj)
    return out_obj


def analyze_results_zip(results_zip, out_json):
    with tempfile.TemporaryDirectory() as tmp_str:
        extract_dir = Path(tmp_str)
        with ZipFile(results_zip) as zf:
            zf.extractall(extract_dir)
        return analyze_run(extract_dir / "output", out_json)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-zip", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    analyze_results_zip(a.results_zip, a.out)
