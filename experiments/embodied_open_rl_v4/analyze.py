"""CPU quantitative analysis & paired bootstrap statistics for embodied_open_rl_v4."""

import argparse
import json
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile

import numpy as np

from .protocol import CONFIG, STAGES, UNSEEN_PROBES, partner_role, write


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


def _stance_metrics(sub_rows):
    ans = [r["analysis"] for r in sub_rows]
    return {
        "n": len(sub_rows),
        "mean_word_count": float(np.mean([a["word_count"] for a in ans])),
        "mean_fulfillment": float(np.mean([a["fulfillment_score"] for a in ans])),
        "pure_appeal_rate": float(np.mean([int(a["pure_appeal"]) for a in ans])),
        "pure_exchange_rate": float(np.mean([int(a["pure_exchange"]) for a in ans])),
        "mixed_stance_rate": float(np.mean([int(a["mixed_stance"]) for a in ans])),
        "any_appeal_rate": float(np.mean([int(a["has_appeal"]) for a in ans])),
        "any_exchange_rate": float(np.mean([int(a["has_exchange"]) for a in ans])),
        "refusal_rate": float(np.mean([int(a["is_refusal"]) for a in ans])),
        "net_appeal_minus_exchange": float(
            np.mean([int(a["pure_appeal"]) - int(a["pure_exchange"]) for a in ans])
        ),
    }


def summarize_probe_rows(rows, ecology="ab"):
    by_battery = {}
    for bat in (
        "D1_effort_allocation",
        "D2_commitment_renegotiation",
        "D3_mirrored_partner_pairs",
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
            "empty_promise_rate": float(np.mean([int(a.get("is_empty_promise", False)) for a in ans])),
            "boundary_rate": float(np.mean([int(a["has_boundary"]) for a in ans])),
            "pure_appeal_rate": float(np.mean([int(a["pure_appeal"]) for a in ans])),
            "pure_exchange_rate": float(np.mean([int(a["pure_exchange"]) for a in ans])),
            "mixed_stance_rate": float(np.mean([int(a["mixed_stance"]) for a in ans])),
            "gratuitous_pragmatics_rate": float(
                np.mean([int(a.get("gratuitous_pragmatics", False)) for a in ans])
            ),
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
            "empty_promise_rate": float(np.mean([int(a.get("is_empty_promise", False)) for a in h_ans])),
            "boundary_rate": float(np.mean([int(a["has_boundary"]) for a in h_ans])),
            "pure_appeal_rate": float(np.mean([int(a["pure_appeal"]) for a in h_ans])),
            "pure_exchange_rate": float(np.mean([int(a["pure_exchange"]) for a in h_ans])),
            "mixed_stance_rate": float(np.mean([int(a["mixed_stance"]) for a in h_ans])),
            "gratuitous_pragmatics_rate": 0.0,
            "distinct_1": distinct_1(h_texts),
            "factual_accuracy": None,
        }

    probe_meta = {p["id"]: p for p in UNSEEN_PROBES}
    d3_rows = [r for r in rows if r["battery"] == "D3_mirrored_partner_pairs"]
    d3_by_p = {
        pid: _stance_metrics([r for r in d3_rows if r["partner"] == pid])
        for pid in ("P_alpha", "P_beta")
    }
    D_ab_d3 = 0.5 * (
        d3_by_p["P_alpha"]["net_appeal_minus_exchange"]
        - d3_by_p["P_beta"]["net_appeal_minus_exchange"]
    )

    heavy_ab_rows = [r for r in heavy_sub if r["partner"] in ("P_alpha", "P_beta")]
    heavy_by_p = {
        pid: _stance_metrics([r for r in heavy_sub if r["partner"] == pid])
        for pid in ("P_alpha", "P_beta", "P_gamma")
    }
    D_ab_all_heavy = 0.5 * (
        heavy_by_p["P_alpha"]["net_appeal_minus_exchange"]
        - heavy_by_p["P_beta"]["net_appeal_minus_exchange"]
    )

    # Ecology-correct stance metrics on D3 matched pairs & counter-habitual subset
    correct_hits = []
    counter_hab_hits = []
    tone_map = {}
    for r in d3_rows:
        role = partner_role(r["partner"], ecology)
        a = r["analysis"]
        hit = bool(a["pure_appeal"]) if role == "appeal_only" else bool(a["pure_exchange"])
        correct_hits.append(int(hit))
        tone = probe_meta[r["id"]].get("tone")
        tone_map.setdefault((r["partner"], tone), []).append(r)
        is_counter_hab = (tone == "demanding" and role == "appeal_only") or (
            tone == "polite" and role == "exchange_only"
        )
        if is_counter_hab:
            counter_hab_hits.append(int(hit))

    ecology_mapping_d3 = D_ab_d3 if ecology == "ab" else -D_ab_d3
    ecology_mapping_all = D_ab_all_heavy if ecology == "ab" else -D_ab_all_heavy

    selectivity_app = (
        by_battery["HEAVY_POOL_D1_D3"]["pure_appeal_rate"]
        - by_battery["D4_factual_guardrail"]["pure_appeal_rate"]
    )
    selectivity_exc = (
        by_battery["HEAVY_POOL_D1_D3"]["pure_exchange_rate"]
        - by_battery["D4_factual_guardrail"]["pure_exchange_rate"]
    )

    return {
        "batteries": by_battery,
        "d3_by_partner": d3_by_p,
        "heavy_d1_d3_by_partner": heavy_by_p,
        "discrimination": {
            "D_ab_d3": float(D_ab_d3),
            "D_ab_all_heavy": float(D_ab_all_heavy),
            "ecology_correct_mapping_d3": float(ecology_mapping_d3),
            "ecology_correct_mapping_all_heavy": float(ecology_mapping_all),
            "matched_stance_rate_d3": float(np.mean(correct_hits)),
            "counter_habitual_rate_d3": float(np.mean(counter_hab_hits)),
        },
        "d3_by_partner_tone": {
            f"{pid}_{tone}": _stance_metrics(sub)
            for (pid, tone), sub in sorted(tone_map.items())
        },
        "selectivity": {
            "heavy_minus_d4_pure_appeal_rate": float(selectivity_app),
            "heavy_minus_d4_pure_exchange_rate": float(selectivity_exc),
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
    boot_rng = np.random.default_rng(74000)

    birth = json.loads((output_dir / "probes-birth.json").read_bytes())
    birth_hidden = np.load(output_dir / "hidden-birth.npz")["hidden"]
    birth_summary = summarize_probe_rows(birth["rows"], ecology="ab")
    pilot_summary = json.loads((output_dir / "pilot-summary.json").read_bytes())

    per_life = {}
    for live_info in pilot_summary["lives"]:
        jid = live_info["id"]
        ldir = output_dir / jid
        arm_name = live_info["arm"]
        ecology = "ba" if arm_name == "body_rl_ba" else "ab"
        trace = [
            json.loads(line)
            for line in (ldir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        heavy_tr = [t for t in trace if t["load"] == "heavy"]
        light_tr = [t for t in trace if t["load"] == "light"]
        heavy_late = [t for t in heavy_tr if t["turn"] > 20] or heavy_tr
        alpha_h = [t for t in heavy_tr if t["partner"] == "P_alpha"]
        beta_h = [t for t in heavy_tr if t["partner"] == "P_beta"]

        life_stats = {
            **live_info,
            "ecology": ecology,
            "mean_heavy_words": float(np.mean([t["analysis"]["word_count"] for t in heavy_tr])) if heavy_tr else 0.0,
            "mean_heavy_words_late": float(np.mean([t["analysis"]["word_count"] for t in heavy_late])) if heavy_late else 0.0,
            "mean_light_words": float(np.mean([t["analysis"]["word_count"] for t in light_tr])) if light_tr else 0.0,
            "light_gratuitous_rate": float(np.mean([int(t["analysis"]["gratuitous_pragmatics"]) for t in light_tr])) if light_tr else 0.0,
            "eos_hit_rate": float(np.mean([int(t.get("hit_eos", True)) for t in trace])),
            "mean_fulfillment": float(np.mean([t["analysis"]["fulfillment_score"] for t in trace])),
            "accept_rate": float(np.mean([int(t["partner_accepted"]) for t in trace])),
            "heavy_accept_rate": float(np.mean([int(t["partner_accepted"]) for t in heavy_tr])) if heavy_tr else 0.0,
            "alpha_heavy_accept_rate": float(np.mean([int(t["partner_accepted"]) for t in alpha_h])) if alpha_h else 0.0,
            "beta_heavy_accept_rate": float(np.mean([int(t["partner_accepted"]) for t in beta_h])) if beta_h else 0.0,
            "mean_reward_body": float(np.mean([t["reward_body"] for t in trace])),
            "total_reward_body": float(np.sum([t["reward_body"] for t in trace])),
            "mean_reward_task": float(np.mean([t["reward_task"] for t in trace])),
            "total_reward_task": float(np.sum([t["reward_task"] for t in trace])),
            "stages": {},
        }

        if arm_name != "frozen_ab":
            for stage in STAGES:
                p_data = json.loads((ldir / f"probes-{stage}.json").read_bytes())
                h_stage = np.load(ldir / f"hidden-{stage}.npz")["hidden"]
                st_summary = summarize_probe_rows(p_data["rows"], ecology=ecology)
                st_summary["meta"] = p_data["meta"]
                st_summary["delta_l2"] = p_data["delta_l2"]
                st_summary["per_layer_delta_l2"] = p_data.get("per_layer_delta_l2", {"by_layer": {}, "by_module": {}})
                st_summary["hidden_shift_vs_birth"] = hidden_shift_metrics(h_stage, birth_hidden)
                life_stats["stages"][stage] = st_summary
        else:
            for stage in STAGES:
                life_stats["stages"][stage] = {
                    **birth_summary,
                    "meta": {"turn": 0, "health": 100.0, "version": 0, "delta_l2": 0.0, "trigger": "frozen"},
                    "delta_l2": 0.0,
                    "per_layer_delta_l2": {"by_layer": {}, "by_module": {}},
                    "hidden_shift_vs_birth": {"mean_l2": 0.0, "mean_cosine_distance": 0.0},
                }

        per_life[jid] = life_stats

    batteries = (
        "D1_effort_allocation",
        "D2_commitment_renegotiation",
        "D3_mirrored_partner_pairs",
        "D4_factual_guardrail",
        "HEAVY_POOL_D1_D3",
    )
    metrics_keys = (
        "mean_word_count",
        "concise_rate",
        "mean_exertion_cost",
        "mean_fulfillment",
        "refusal_rate",
        "empty_promise_rate",
        "boundary_rate",
        "pure_appeal_rate",
        "pure_exchange_rate",
        "mixed_stance_rate",
        "gratuitous_pragmatics_rate",
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
            "mean_heavy_words": float(np.mean([per_life[f"{arm_name}_s{s}"]["mean_heavy_words"] for s in seeds])),
            "mean_heavy_words_late": float(np.mean([per_life[f"{arm_name}_s{s}"]["mean_heavy_words_late"] for s in seeds])),
            "mean_light_words": float(np.mean([per_life[f"{arm_name}_s{s}"]["mean_light_words"] for s in seeds])),
            "mean_light_gratuitous_rate": float(np.mean([per_life[f"{arm_name}_s{s}"]["light_gratuitous_rate"] for s in seeds])),
            "mean_accept_rate": float(np.mean([per_life[f"{arm_name}_s{s}"]["accept_rate"] for s in seeds])),
            "mean_heavy_accept_rate": float(np.mean([per_life[f"{arm_name}_s{s}"]["heavy_accept_rate"] for s in seeds])),
            "mean_alpha_heavy_accept_rate": float(np.mean([per_life[f"{arm_name}_s{s}"]["alpha_heavy_accept_rate"] for s in seeds])),
            "mean_beta_heavy_accept_rate": float(np.mean([per_life[f"{arm_name}_s{s}"]["beta_heavy_accept_rate"] for s in seeds])),
            "mean_reward_body": float(np.mean([per_life[f"{arm_name}_s{s}"]["mean_reward_body"] for s in seeds])),
            "mean_total_reward_body": float(np.mean([per_life[f"{arm_name}_s{s}"]["total_reward_body"] for s in seeds])),
            "mean_reward_task": float(np.mean([per_life[f"{arm_name}_s{s}"]["mean_reward_task"] for s in seeds])),
            "eos_hit_rate": float(np.mean([per_life[f"{arm_name}_s{s}"]["eos_hit_rate"] for s in seeds])),
            "stages": {},
        }
        for stage in STAGES:
            st_out = {"batteries": {}, "d3_by_partner": {}, "discrimination": {}}
            for bat in batteries:
                b_agg = {}
                for k in metrics_keys:
                    vals = [per_life[f"{arm_name}_s{s}"]["stages"][stage]["batteries"][bat][k] for s in seeds]
                    if vals[0] is not None:
                        b_agg[k] = float(np.mean(vals))
                st_out["batteries"][bat] = b_agg
            for pid in ("P_alpha", "P_beta"):
                st_out["d3_by_partner"][pid] = {
                    k: float(np.mean([per_life[f"{arm_name}_s{s}"]["stages"][stage]["d3_by_partner"][pid][k] for s in seeds]))
                    for k in ("mean_word_count", "mean_fulfillment", "pure_appeal_rate", "pure_exchange_rate", "mixed_stance_rate", "refusal_rate", "net_appeal_minus_exchange")
                }
            for dk in ("D_ab_d3", "D_ab_all_heavy", "ecology_correct_mapping_d3", "ecology_correct_mapping_all_heavy", "matched_stance_rate_d3", "counter_habitual_rate_d3"):
                st_out["discrimination"][dk] = float(
                    np.mean([per_life[f"{arm_name}_s{s}"]["stages"][stage]["discrimination"][dk] for s in seeds])
                )
            st_out["mean_stage_health"] = float(np.mean([per_life[f"{arm_name}_s{s}"]["stages"][stage]["meta"]["health"] for s in seeds]))
            st_out["mean_delta_l2"] = float(np.mean([per_life[f"{arm_name}_s{s}"]["stages"][stage]["delta_l2"] for s in seeds]))
            st_out["mean_hidden_cosine_dist"] = float(np.mean([per_life[f"{arm_name}_s{s}"]["stages"][stage]["hidden_shift_vs_birth"]["mean_cosine_distance"] for s in seeds]))
            arm_agg[arm_name]["stages"][stage] = st_out

    # Core Mirrored Ecology Contrasts: body_rl_ab vs body_rl_ba (N = 6 paired seeds)
    mirrored_contrasts = {}
    for stage in STAGES:
        ab_st = [per_life[f"body_rl_ab_s{s}"]["stages"][stage] for s in seeds]
        ba_st = [per_life[f"body_rl_ba_s{s}"]["stages"][stage] for s in seeds]
        mirrored_contrasts[stage] = {
            "D_ab_d3_diff": paired_contrast([x["discrimination"]["D_ab_d3"] for x in ab_st], [x["discrimination"]["D_ab_d3"] for x in ba_st], boot_rng),
            "D_ab_all_heavy_diff": paired_contrast([x["discrimination"]["D_ab_all_heavy"] for x in ab_st], [x["discrimination"]["D_ab_all_heavy"] for x in ba_st], boot_rng),
            "alpha_pure_appeal_ab_minus_ba": paired_contrast([x["d3_by_partner"]["P_alpha"]["pure_appeal_rate"] for x in ab_st], [x["d3_by_partner"]["P_alpha"]["pure_appeal_rate"] for x in ba_st], boot_rng),
            "alpha_pure_exchange_ba_minus_ab": paired_contrast([x["d3_by_partner"]["P_alpha"]["pure_exchange_rate"] for x in ba_st], [x["d3_by_partner"]["P_alpha"]["pure_exchange_rate"] for x in ab_st], boot_rng),
            "beta_pure_exchange_ab_minus_ba": paired_contrast([x["d3_by_partner"]["P_beta"]["pure_exchange_rate"] for x in ab_st], [x["d3_by_partner"]["P_beta"]["pure_exchange_rate"] for x in ba_st], boot_rng),
            "beta_pure_appeal_ba_minus_ab": paired_contrast([x["d3_by_partner"]["P_beta"]["pure_appeal_rate"] for x in ba_st], [x["d3_by_partner"]["P_beta"]["pure_appeal_rate"] for x in ab_st], boot_rng),
            "mean_ecology_correct_mapping_d3": paired_contrast(
                [0.5 * (a["discrimination"]["ecology_correct_mapping_d3"] + b["discrimination"]["ecology_correct_mapping_d3"]) for a, b in zip(ab_st, ba_st)],
                [0.0] * len(seeds),
                boot_rng,
            ),
            "mean_matched_stance_rate_d3_vs_birth": paired_contrast(
                [0.5 * (a["discrimination"]["matched_stance_rate_d3"] + b["discrimination"]["matched_stance_rate_d3"]) for a, b in zip(ab_st, ba_st)],
                [0.5 * (birth_summary["discrimination"]["matched_stance_rate_d3"] + 0.5 * (birth_summary["d3_by_partner"]["P_alpha"]["pure_exchange_rate"] + birth_summary["d3_by_partner"]["P_beta"]["pure_appeal_rate"]))] * len(seeds),
                boot_rng,
            ),
            "mean_counter_habitual_rate_d3": paired_contrast(
                [0.5 * (a["discrimination"]["counter_habitual_rate_d3"] + b["discrimination"]["counter_habitual_rate_d3"]) for a, b in zip(ab_st, ba_st)],
                [0.0] * len(seeds),
                boot_rng,
            ),
        }

    # Paired life contrasts (vs task_only_rl_ab, pooled_body_rl_ab, frozen_ab)
    life_contrasts = {}
    for left_arm, cmp_arm in (
        ("body_rl_ab", "task_only_rl_ab"),
        ("body_rl_ab", "pooled_body_rl_ab"),
        ("body_rl_ab", "frozen_ab"),
        ("body_rl_ba", "frozen_ab"),
    ):
        lc = {}
        for lk in (
            "turns",
            "mean_heavy_words",
            "mean_heavy_words_late",
            "mean_light_words",
            "light_gratuitous_rate",
            "accept_rate",
            "heavy_accept_rate",
            "mean_reward_body",
            "total_reward_body",
        ):
            lc[lk] = paired_contrast(
                [per_life[f"{left_arm}_s{s}"][lk] for s in seeds],
                [per_life[f"{cmp_arm}_s{s}"][lk] for s in seeds],
                boot_rng,
            )
        for st in ("worn", "critical"):
            lc[f"{st}_D_ab_d3"] = paired_contrast(
                [per_life[f"{left_arm}_s{s}"]["stages"][st]["discrimination"]["D_ab_d3"] for s in seeds],
                [per_life[f"{cmp_arm}_s{s}"]["stages"][st]["discrimination"]["D_ab_d3"] for s in seeds],
                boot_rng,
            )
            lc[f"{st}_matched_stance_d3"] = paired_contrast(
                [per_life[f"{left_arm}_s{s}"]["stages"][st]["discrimination"]["matched_stance_rate_d3"] for s in seeds],
                [per_life[f"{cmp_arm}_s{s}"]["stages"][st]["discrimination"]["matched_stance_rate_d3"] for s in seeds],
                boot_rng,
            )
            lc[f"{st}_heavy_words"] = paired_contrast(
                [per_life[f"{left_arm}_s{s}"]["stages"][st]["batteries"]["HEAVY_POOL_D1_D3"]["mean_word_count"] for s in seeds],
                [per_life[f"{cmp_arm}_s{s}"]["stages"][st]["batteries"]["HEAVY_POOL_D1_D3"]["mean_word_count"] for s in seeds],
                boot_rng,
            )
        life_contrasts[f"{left_arm}_vs_{cmp_arm}"] = lc

    # Birth-pair baseline for pooled vs ab vs ba on D3 matched_stance
    trajectory_d3 = {
        "birth": birth_summary["discrimination"],
        **{
            arm: {st: arm_agg[arm]["stages"][st]["discrimination"] for st in STAGES}
            for arm in ("body_rl_ab", "body_rl_ba", "pooled_body_rl_ab", "task_only_rl_ab")
        },
    }

    seed0 = seeds[0]
    sample_ids = ["d3_01", "d3_04", "d3_07", "d3_10", "d1_01", "d2_01", "d4_01", "d4_08"]
    birth_lookup = {r["id"]: r for r in birth["rows"]}
    loaded = {}
    for arm_name in ("body_rl_ab", "body_rl_ba", "pooled_body_rl_ab", "task_only_rl_ab"):
        rows = json.loads((output_dir / f"{arm_name}_s{seed0}" / "probes-critical.json").read_bytes())["rows"]
        loaded[arm_name] = {r["id"]: r for r in rows}

    examples = [
        {
            "id": pid,
            "battery": birth_lookup[pid]["battery"],
            "partner": birth_lookup[pid]["partner"],
            "request": birth_lookup[pid]["request"],
            "birth": birth_lookup[pid]["text"],
            "body_rl_ab_critical": loaded["body_rl_ab"][pid]["text"],
            "body_rl_ba_critical": loaded["body_rl_ba"][pid]["text"],
            "pooled_ab_critical": loaded["pooled_body_rl_ab"][pid]["text"],
            "task_only_ab_critical": loaded["task_only_rl_ab"][pid]["text"],
        }
        for pid in sample_ids
    ]

    out_obj = {
        "schema": CONFIG["schema"],
        "pilot_checks": {
            "weight_swap_exact": pilot_summary["weight_swap_exact"],
            "birth_restore_exact": pilot_summary["birth_restore_exact"],
            "window1_unmasked_text_parity_exact": pilot_summary["window1_unmasked_text_parity_exact"],
            "window1_ab_parity_exact": pilot_summary["window1_ab_parity_exact"],
        },
        "birth_summary": birth_summary,
        "arms": arm_agg,
        "mirrored_ecology_contrasts": mirrored_contrasts,
        "paired_life_contrasts": life_contrasts,
        "trajectory_d3": trajectory_d3,
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
