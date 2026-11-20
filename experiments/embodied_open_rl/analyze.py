"""CPU quantitative analysis & cross-arm paired comparison for embodied_open_rl."""

import argparse
import json
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile

import numpy as np

from .protocol import CONFIG, LIFE_TASKS, UNSEEN_PROBES, analyze_utterance, write


def distinct_1(texts):
    toks = []
    for txt in texts:
        toks.extend(re.findall(r"[A-Za-z0-9']+(?:-[A-Za-z0-9']+)*", (txt or "").lower()))
    if not toks:
        return 0.0
    return float(len(set(toks)) / len(toks))


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
        factual_vals = [int(bool(a["factual_correct"])) for a in ans if a["factual_correct"] is not None]
        by_battery[bat] = {
            "n": len(sub),
            "mean_word_count": float(np.mean([a["word_count"] for a in ans])),
            "mean_fulfillment": float(np.mean([a["fulfillment_score"] for a in ans])),
            "refusal_rate": float(np.mean([int(a["is_refusal"]) for a in ans])),
            "boundary_rate": float(np.mean([int(a["has_boundary"]) for a in ans])),
            "appeal_rate": float(np.mean([int(a["has_appeal"]) for a in ans])),
            "exchange_rate": float(np.mean([int(a["has_exchange"]) for a in ans])),
            "unconditional_commitment_rate": float(
                np.mean([int(a["unconditional_commitment"]) for a in ans])
            ),
            "conditional_or_renegotiated_rate": float(
                np.mean([int(a["has_exchange"] or a["has_boundary"]) for a in ans])
            ),
            "degenerate_rate": float(np.mean([int(a["is_degenerate"]) for a in ans])),
            "distinct_1": distinct_1(texts),
            "factual_accuracy": float(np.mean(factual_vals)) if factual_vals else None,
        }

    # Within D3, break out by partner identity (including unseen P_new)
    d3_partners = {}
    for p_id in ("P_care", "P_demand", "P_new"):
        sub = [r for r in rows if r["battery"] == "D3_pragmatic_partners" and r["partner"] == p_id]
        if sub:
            ans = [r["analysis"] for r in sub]
            d3_partners[p_id] = {
                "n": len(sub),
                "mean_word_count": float(np.mean([a["word_count"] for a in ans])),
                "boundary_rate": float(np.mean([int(a["has_boundary"]) for a in ans])),
                "appeal_rate": float(np.mean([int(a["has_appeal"]) for a in ans])),
                "exchange_rate": float(np.mean([int(a["has_exchange"]) for a in ans])),
            }
    return {"batteries": by_battery, "d3_by_partner": d3_partners}


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
        word_counts = [t["analysis"]["word_count"] for t in trace]
        n = len(trace)
        half = max(1, n // 2)
        life_stats = {
            **live_info,
            "low_health_encounters_le45": sum(1 for t in trace if t["health_before"] <= 45.0),
            "mean_word_first_half": float(np.mean(word_counts[:half])),
            "mean_word_second_half": float(np.mean(word_counts[half:])) if n > half else float(np.mean(word_counts)),
            "mean_fulfillment": float(np.mean([t["analysis"]["fulfillment_score"] for t in trace])),
            "boundary_rate_first_half": float(np.mean([int(t["analysis"]["has_boundary"]) for t in trace[:half]])),
            "boundary_rate_second_half": float(np.mean([int(t["analysis"]["has_boundary"]) for t in trace[half:]])) if n > half else 0.0,
            "accept_rate": float(np.mean([int(t["partner_accepted"]) for t in trace])),
            "mean_reward_body": float(np.mean([t["reward_body"] for t in trace])),
            "mean_reward_social": float(np.mean([t["reward_social"] for t in trace])),
            "stages": {},
        }

        if live_info["arm"] != "frozen":
            for stage in ("early", "worn", "critical"):
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
            for stage in ("early", "worn", "critical"):
                life_stats["stages"][stage] = {
                    **birth_summary,
                    "delta_l2": 0.0,
                    "per_layer_delta_l2": {"by_layer": {}, "by_module": {}},
                    "hidden_shift_vs_birth": {"mean_l2": 0.0, "mean_cosine_distance": 0.0},
                }

        per_life[jid] = life_stats

    # Aggregate across seeds per arm & stage, plus paired contrasts at critical
    arm_agg = {}
    batteries = (
        "D1_effort_allocation",
        "D2_commitment_renegotiation",
        "D3_pragmatic_partners",
        "D4_factual_guardrail",
    )
    metrics_keys = (
        "mean_word_count",
        "mean_fulfillment",
        "refusal_rate",
        "boundary_rate",
        "appeal_rate",
        "exchange_rate",
        "unconditional_commitment_rate",
        "conditional_or_renegotiated_rate",
        "degenerate_rate",
        "distinct_1",
        "factual_accuracy",
    )
    for arm_name in CONFIG["primary_arms"]:
        arm_agg[arm_name] = {
            "mean_turns": float(np.mean([per_life[f"{arm_name}_s{s}"]["turns"] for s in CONFIG["seeds"]])),
            "deaths": int(sum(int(per_life[f"{arm_name}_s{s}"]["physical_death"]) for s in CONFIG["seeds"])),
            "mean_low_health_encounters": float(
                np.mean([per_life[f"{arm_name}_s{s}"]["low_health_encounters_le45"] for s in CONFIG["seeds"]])
            ),
            "stages": {},
        }
        for stage in ("early", "worn", "critical"):
            stage_out = {"batteries": {}, "d3_by_partner": {}}
            for bat in batteries:
                b_agg = {}
                for k in metrics_keys:
                    vals = [per_life[f"{arm_name}_s{s}"]["stages"][stage]["batteries"][bat][k] for s in CONFIG["seeds"]]
                    if vals[0] is not None:
                        b_agg[k] = float(np.mean(vals))
                stage_out["batteries"][bat] = b_agg
            for p_id in ("P_care", "P_demand", "P_new"):
                p_agg = {}
                for k in ("mean_word_count", "boundary_rate", "appeal_rate", "exchange_rate"):
                    vals = [
                        per_life[f"{arm_name}_s{s}"]["stages"][stage]["d3_by_partner"][p_id][k]
                        for s in CONFIG["seeds"]
                    ]
                    p_agg[k] = float(np.mean(vals))
                stage_out["d3_by_partner"][p_id] = p_agg
            stage_out["mean_delta_l2"] = float(
                np.mean([per_life[f"{arm_name}_s{s}"]["stages"][stage]["delta_l2"] for s in CONFIG["seeds"]])
            )
            layer_keys = sorted(
                {
                    lk
                    for s in CONFIG["seeds"]
                    for lk in per_life[f"{arm_name}_s{s}"]["stages"][stage][
                        "per_layer_delta_l2"
                    ].get("by_layer", {})
                },
                key=lambda k: int(k.split("_")[1]),
            )
            mod_keys = sorted(
                {
                    mk
                    for s in CONFIG["seeds"]
                    for mk in per_life[f"{arm_name}_s{s}"]["stages"][stage][
                        "per_layer_delta_l2"
                    ].get("by_module", {})
                }
            )
            stage_out["mean_per_layer_delta_l2"] = {
                "by_layer": {
                    lk: float(
                        np.mean(
                            [
                                per_life[f"{arm_name}_s{s}"]["stages"][stage][
                                    "per_layer_delta_l2"
                                ]["by_layer"].get(lk, 0.0)
                                for s in CONFIG["seeds"]
                            ]
                        )
                    )
                    for lk in layer_keys
                },
                "by_module": {
                    mk: float(
                        np.mean(
                            [
                                per_life[f"{arm_name}_s{s}"]["stages"][stage][
                                    "per_layer_delta_l2"
                                ]["by_module"].get(mk, 0.0)
                                for s in CONFIG["seeds"]
                            ]
                        )
                    )
                    for mk in mod_keys
                },
            }
            stage_out["mean_hidden_cosine_dist"] = float(
                np.mean(
                    [
                        per_life[f"{arm_name}_s{s}"]["stages"][stage]["hidden_shift_vs_birth"][
                            "mean_cosine_distance"
                        ]
                        for s in CONFIG["seeds"]
                    ]
                )
            )
            arm_agg[arm_name]["stages"][stage] = stage_out

    # Paired seed contrasts (`body_rl` critical vs baselines)
    contrasts = {}
    for cmp_arm in ("frozen", "social_only_rl", "shuffled_body_rl"):
        c_bat = {}
        for bat in batteries:
            diffs_out = {}
            for k in metrics_keys:
                vals_b = [
                    per_life[f"body_rl_s{s}"]["stages"]["critical"]["batteries"][bat][k]
                    for s in CONFIG["seeds"]
                ]
                vals_c = [
                    per_life[f"{cmp_arm}_s{s}"]["stages"]["critical"]["batteries"][bat][k]
                    for s in CONFIG["seeds"]
                ]
                if vals_b[0] is None:
                    continue
                diffs = [vb - vc for vb, vc in zip(vals_b, vals_c)]
                diffs_out[k] = {
                    "mean_diff": float(np.mean(diffs)),
                    "per_seed": [float(d) for d in diffs],
                }
            c_bat[bat] = diffs_out
        contrasts[f"body_rl_vs_{cmp_arm}"] = c_bat

    # Within body_rl trajectory: critical minus early
    within_traj = {}
    for bat in batteries:
        w_out = {}
        for k in metrics_keys:
            vals_crit = [per_life[f"body_rl_s{s}"]["stages"]["critical"]["batteries"][bat][k] for s in CONFIG["seeds"]]
            vals_early = [per_life[f"body_rl_s{s}"]["stages"]["early"]["batteries"][bat][k] for s in CONFIG["seeds"]]
            if vals_crit[0] is None:
                continue
            diffs = [vc - ve for vc, ve in zip(vals_crit, vals_early)]
            w_out[k] = {"mean_diff": float(np.mean(diffs)), "per_seed": [float(d) for d in diffs]}
        within_traj[bat] = w_out

    # Representative qualitative comparisons on seed 71001 across arms/stages
    seed0 = CONFIG["seeds"][0]
    sample_ids = ["d1_02", "d1_08", "d2_01", "d2_04", "d3_06", "d3_10", "d4_01", "d4_08"]
    birth_lookup = {r["id"]: r for r in birth["rows"]}
    examples = []
    loaded = {}
    for arm_name in ("body_rl", "social_only_rl", "shuffled_body_rl"):
        for st in ("early", "critical"):
            rows = json.loads((output_dir / f"{arm_name}_s{seed0}" / f"probes-{st}.json").read_bytes())["rows"]
            loaded[(arm_name, st)] = {r["id"]: r for r in rows}
    for pid in sample_ids:
        b = birth_lookup[pid]
        examples.append(
            {
                "id": pid,
                "battery": b["battery"],
                "partner": b["partner"],
                "request": b["request"],
                "birth": b["text"],
                "body_rl_early": loaded[("body_rl", "early")][pid]["text"],
                "body_rl_critical": loaded[("body_rl", "critical")][pid]["text"],
                "social_only_critical": loaded[("social_only_rl", "critical")][pid]["text"],
                "shuffled_body_critical": loaded[("shuffled_body_rl", "critical")][pid]["text"],
            }
        )

    out_obj = {
        "schema": CONFIG["schema"],
        "pilot_checks": {
            "weight_swap_exact": pilot_summary["weight_swap_exact"],
            "birth_restore_exact": pilot_summary["birth_restore_exact"],
        },
        "birth_summary": birth_summary,
        "arms": arm_agg,
        "paired_contrasts_at_critical": contrasts,
        "within_body_rl_critical_minus_early": within_traj,
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
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--results-zip", type=Path)
    src.add_argument("--output-dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.results_zip is not None:
        analyze_results_zip(a.results_zip, a.out)
    else:
        analyze_run(a.output_dir, a.out)
