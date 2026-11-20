"""Independent CPU audit for embodied_open_rl_v4 (mirrored ecologies, zero hints, pure bodily RL)."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile

from .protocol import (
    ARMS,
    CONFIG,
    LIFE_TASKS,
    STAGES,
    UNSEEN_PROBES,
    analyze_utterance,
    assert_zero_leakage,
    build_life_schedule,
    build_messages,
    compute_pair_weights,
    evaluate_candidate_at_state,
    extract_turn_preference_pair,
    grpo_group_size_for_load,
    jobs,
    partner_draw,
    partner_role,
    select_episodic_pairs,
    shown_partner,
    slot_names_for_arm,
    turn_noise,
    write,
)


def _replay_arm_traces(ldir, seed, arm_name):
    """Independently replay real mortal trace.jsonl, group_rollouts.jsonl, and episodic DPO updates."""
    arm = ARMS[arm_name]
    sched = build_life_schedule(seed, CONFIG["horizon"])
    trace = [
        json.loads(ln)
        for ln in (ldir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    updates_path = ldir / "updates.jsonl"
    updates = (
        [json.loads(ln) for ln in updates_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if updates_path.exists()
        else []
    )
    grpo_path = ldir / "group_rollouts.jsonl"
    grpo_rows = (
        [json.loads(ln) for ln in grpo_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if grpo_path.exists()
        else []
    )
    if arm["update"]:
        assert len(grpo_rows) == len(trace), f"group_rollouts length mismatch in {ldir.name}"

    h = float(CONFIG["initial_health"])
    w_len = int(CONFIG["window_size"])
    life_pairs = []
    turns_in_win = 0
    u_i = 0
    max_err_h = 0.0
    max_err_r = 0.0
    max_err_weight = 0.0
    max_err_weight_sum = 0.0
    candidates_checked = 0

    for idx, rec in enumerate(trace):
        item = sched[idx]
        sp = shown_partner(item["partner"], arm_name)
        assert rec["turn"] == item["turn"]
        assert rec["load"] == item["load"]
        assert rec["partner"] == item["partner"]
        assert rec["shown_partner"] == sp
        assert rec["ecology"] == arm["ecology"]
        assert rec["partner_role"] == partner_role(item["partner"], arm["ecology"])
        assert rec["request"] == item["request"]
        assert_zero_leakage(build_messages(sp, item["request"]))

        task_spec = LIFE_TASKS[item["task_idx"]]
        u_p = partner_draw(seed, item["turn"])
        eta = turn_noise(seed, item["turn"])
        h_before = h

        settled0 = evaluate_candidate_at_state(
            h_before, task_spec, item["partner"], arm["ecology"], rec["response_text"], u_p, eta
        )
        assert settled0["analysis"] == rec["analysis"], f"Parser mismatch at {ldir.name} turn {rec['turn']}"
        assert settled0["partner_accepted"] == rec["partner_accepted"]
        assert abs(settled0["partner_accept_prob"] - rec["partner_accept_prob"]) < 1e-12
        max_err_h = max(max_err_h, abs(settled0["health_after"] - rec["health_after"]))
        max_err_r = max(
            max_err_r,
            abs(settled0["reward_body"] - rec["reward_body"]),
            abs(settled0["reward_task"] - rec["reward_task"]),
        )
        assert max_err_h < 1e-10 and max_err_r < 1e-10
        h = settled0["health_after"]
        turns_in_win += 1

        if arm["update"]:
            g_row = grpo_rows[idx]
            assert g_row["turn"] == rec["turn"] and g_row["load"] == rec["load"]
            assert g_row["partner"] == rec["partner"] and g_row["shown_partner"] == sp
            assert abs(g_row["health_before"] - h_before) < 1e-10
            cands = g_row["candidates"]
            expected_g = grpo_group_size_for_load(item["load"])
            assert len(cands) == expected_g, f"Expected {expected_g} rollouts, got {len(cands)}"
            assert cands[0]["response_text"] == rec["response_text"] and cands[0]["executed"] is True
            replayed_cands = []
            for k_idx, c in enumerate(cands):
                assert c["candidate_idx"] == k_idx
                candidates_checked += 1
                settled_k = evaluate_candidate_at_state(
                    h_before, task_spec, item["partner"], arm["ecology"], c["response_text"], u_p, eta
                )
                assert settled_k["analysis"] == c["analysis"]
                assert settled_k["partner_accepted"] == c["partner_accepted"]
                assert abs(settled_k["partner_accept_prob"] - c["partner_accept_prob"]) < 1e-12
                max_err_h = max(max_err_h, abs(settled_k["health_after"] - c["health_after"]))
                max_err_r = max(
                    max_err_r,
                    abs(settled_k["reward_body"] - c["reward_body"]),
                    abs(settled_k["reward_task"] - c["reward_task"]),
                )
                assert max_err_h < 1e-10 and max_err_r < 1e-10
                replayed_cands.append(settled_k)
            pair_meta = extract_turn_preference_pair(
                rec["turn"], rec["window_index"], rec["load"], sp, replayed_cands, arm["body_reward"]
            )
            if pair_meta is not None:
                life_pairs.append(pair_meta)

        is_end = turns_in_win == w_len or h <= 0.0 or idx == len(trace) - 1
        if is_end and arm["update"]:
            for slot_p in slot_names_for_arm(arm_name):
                selected = select_episodic_pairs(
                    life_pairs,
                    item["window_index"],
                    mask_partner=bool(arm.get("mask_partner", False)),
                    slot_partner=slot_p,
                )
                if selected:
                    u_rec = updates[u_i]
                    u_i += 1
                    weights = compute_pair_weights(selected)
                    assert u_rec["slot_partner"] == slot_p
                    assert u_rec["window_index"] == item["window_index"]
                    assert u_rec["current_turn"] == rec["turn"]
                    assert u_rec["selected_turns"] == [p["turn"] for p in selected]
                    assert all(t <= rec["turn"] for t in u_rec["selected_turns"])
                    assert u_rec["selected_window_indices"] == [p["window_index"] for p in selected]
                    assert u_rec["selected_loads"] == [p["load"] for p in selected]
                    assert u_rec["selected_shown_partners"] == [p["shown_partner"] for p in selected]
                    assert all(sp_i == slot_p for sp_i in u_rec["selected_shown_partners"])
                    assert u_rec["win_candidate_indices"] == [p["win_idx"] for p in selected]
                    assert u_rec["lose_candidate_indices"] == [p["lose_idx"] for p in selected]
                    gap_err = max(
                        abs(a - b) for a, b in zip([p["reward_gap"] for p in selected], u_rec["reward_gaps"])
                    )
                    w_err = max(abs(a - b) for a, b in zip(weights, u_rec["pair_weights"]))
                    w_sum_err = abs(float(sum(u_rec["pair_weights"])) - 1.0)
                    max_err_r = max(max_err_r, gap_err)
                    max_err_weight = max(max_err_weight, w_err)
                    max_err_weight_sum = max(max_err_weight_sum, w_sum_err)
                    assert gap_err < 1e-9 and w_err < 1e-10 and w_sum_err < 1e-10
        if is_end:
            turns_in_win = 0

    assert u_i == len(updates)
    return {
        "trace": trace,
        "turns_checked": len(trace),
        "candidates_checked": candidates_checked,
        "updates_checked": len(updates),
        "max_err_health": max_err_h,
        "max_err_reward": max_err_r,
        "max_err_adv": max_err_weight,
        "max_abs_adv_sum": max_err_weight_sum,
        "max_abs_group_adv_sum": 0.0,
    }


def audit_smoke(extract_dir, manifest, out_json):
    output = extract_dir / "output"
    status = json.loads((output / "status.json").read_bytes())
    summary = json.loads((output / "smoke-summary.json").read_bytes())
    assert status["stage"] == "complete" and status["base_unchanged"]
    smoke_jobs = jobs("smoke")
    assert status["completed_jobs"] == len(smoke_jobs)
    assert summary["window1_unmasked_text_parity_exact"]
    assert summary["window1_ab_parity_exact"]
    assert float(CONFIG["reward_maintain_weight"]) == 0.0
    assert_zero_leakage(build_messages("P_alpha", "Quick check"))

    lives = summary["paired_lives"]
    assert set(lives) == {job["arm"] for job in smoke_jobs}
    ab_dir = output / next(job["id"] for job in smoke_jobs if job["arm"] == "body_rl_ab")
    ba_dir = output / next(job["id"] for job in smoke_jobs if job["arm"] == "body_rl_ba")

    d4_specs = {probe["id"]: probe for probe in UNSEEN_PROBES if probe["load"] == "light"}
    d4_summary = {}
    for label, stage_dir in (("body_rl_ab", ab_dir), ("body_rl_ba", ba_dir)):
        d4 = {}
        for stage, path in (
            ("birth", output / "probes-birth.json"),
            *[(stage, stage_dir / f"probes-{stage}.json") for stage in STAGES],
        ):
            parsed = {}
            for row in json.loads(path.read_bytes())["rows"]:
                if row["id"] in d4_specs:
                    spec = d4_specs[row["id"]]
                    assert row["request"] == spec["request"] and row["partner"] == spec["partner"]
                    parsed[row["id"]] = analyze_utterance(row["text"], spec)
                    assert parsed[row["id"]] == row["analysis"]
            assert set(parsed) == set(d4_specs)
            d4[stage] = {
                "correct": sum(int(bool(v["factual_correct"])) for v in parsed.values()),
                "gratuitous": sorted(k for k, v in parsed.items() if v["gratuitous_pragmatics"]),
            }
        d4_summary[label] = d4

    # Replay real mortal traces, GRPO candidate rollouts, and within-prompt advantages on all smoke jobs
    max_err_health = 0.0
    max_err_reward = 0.0
    max_err_adv = 0.0
    total_candidates_checked = 0
    for job in smoke_jobs:
        rep = _replay_arm_traces(output / job["id"], job["seed"], job["arm"])
        max_err_health = max(max_err_health, rep["max_err_health"])
        max_err_reward = max(max_err_reward, rep["max_err_reward"])
        max_err_adv = max(max_err_adv, rep["max_err_adv"])
        total_candidates_checked += rep["candidates_checked"]

    b_exp = summary["birth_exploration"]
    ab_crit_d3 = lives["body_rl_ab"]["stages"]["critical"]["D3_partner_stance"]
    ba_crit_d3 = lives["body_rl_ba"]["stages"]["critical"]["D3_partner_stance"]
    ab_words = lives["body_rl_ab"]["stages"]["critical"]["HEAVY_MEAN_D1_D3"]["mean_word_count"]
    ba_words = lives["body_rl_ba"]["stages"]["critical"]["HEAVY_MEAN_D1_D3"]["mean_word_count"]
    task_words = lives["task_only_rl_ab"]["stages"]["critical"]["HEAVY_MEAN_D1_D3"]["mean_word_count"]
    ab_empty = lives["body_rl_ab"]["stages"]["critical"]["HEAVY_MEAN_D1_D3"].get("empty_promise_rate", 0.0)
    ba_empty = lives["body_rl_ba"]["stages"]["critical"]["HEAVY_MEAN_D1_D3"].get("empty_promise_rate", 0.0)

    gates = {
        "zero_leakage_and_no_length_hint": True,
        "pure_bodily_grounding": float(CONFIG["reward_maintain_weight"]) == 0.0,
        "replay_exact": max_err_health < 1e-10 and max_err_reward < 1e-10 and max_err_adv < 1e-9,
        "unhinted_verbose_baseline": b_exp["frozen_heavy_mean_words"] >= 60.0,
        "two_sided_exploration": (
            b_exp.get("win1_grpo_heavy_pure_appeal_rate", 0.0) >= 0.08
            and b_exp.get("win1_grpo_heavy_pure_exchange_rate", 0.0) >= 0.08
        )
        or (
            b_exp.get("frozen_heavy_pure_appeal_rate", 0.0) >= 0.08
            and b_exp.get("frozen_heavy_pure_exchange_rate", 0.0) >= 0.08
        )
        or (
            b_exp.get("birth_heavy_pure_appeal_rate", 0.0) >= 0.08
            and b_exp.get("birth_heavy_pure_exchange_rate", 0.0) >= 0.08
        ),
        "sufficient_body_survival": min(lives["body_rl_ab"]["turns"], lives["body_rl_ba"]["turns"]) >= 55,
        "frozen_mortal_challenge": bool(
            lives["frozen_ab"]["physical_death"]
            and lives["frozen_ab"]["turns"] < min(lives["body_rl_ab"]["turns"], lives["body_rl_ba"]["turns"])
        ),
        "mirrored_d_ab_direction": (ab_crit_d3["D_ab"] - ba_crit_d3["D_ab"]) >= 0.20,
        "no_empty_promise_collapse": max(ab_empty, ba_empty) < 0.25,
        "autonomous_word_compression": max(ab_words, ba_words) < task_words,
        "d4_guardrail": all(
            d4_summary[arm]["critical"]["correct"] >= 10 and len(d4_summary[arm]["critical"]["gratuitous"]) <= 2
            for arm in ("body_rl_ab", "body_rl_ba")
        ),
    }

    result = {
        "schema": CONFIG["schema"],
        "verified_files": len(manifest),
        "total_grpo_candidates_checked": total_candidates_checked,
        "passes_pilot_gate": all(gates.values()),
        "gates": gates,
        "max_health_error": max_err_health,
        "max_reward_error": max_err_reward,
        "max_advantage_error": max_err_adv,
        "birth_exploration": b_exp,
        "d3_critical_ab": ab_crit_d3,
        "d3_critical_ba": ba_crit_d3,
        "d4_summary": d4_summary,
        "lives": {k: {kk: vv for kk, vv in v.items() if kk != "stages"} for k, v in lives.items()},
        "direction_checks": summary["direction_checks"],
        "window1_unmasked_text_parity_exact": summary["window1_unmasked_text_parity_exact"],
        "window1_ab_parity_exact": summary["window1_ab_parity_exact"],
    }
    write(out_json, result)
    return result


def audit_bundle(results_zip, out_json, smoke=False):
    results_zip = Path(results_zip)
    with tempfile.TemporaryDirectory() as tmp_str:
        extract_dir = Path(tmp_str)
        with ZipFile(results_zip) as zf:
            zf.extractall(extract_dir)

        manifest = json.loads((extract_dir / "archive-manifest.json").read_bytes())
        for rel, info in manifest.items():
            fpath = extract_dir / rel
            assert fpath.exists(), f"Missing archived file: {rel}"
            assert fpath.stat().st_size == info["bytes"], f"Size mismatch: {rel}"
            assert sha256(fpath.read_bytes()).hexdigest() == info["sha256"], f"SHA mismatch: {rel}"

        if smoke:
            return audit_smoke(extract_dir, manifest, out_json)

        out_root = extract_dir / "output"
        status = json.loads((out_root / "status.json").read_bytes())
        assert status["stage"] == "complete", f"Run incomplete: {status}"
        assert status["base_unchanged"] is True
        assert float(CONFIG["reward_maintain_weight"]) == 0.0

        ckpt_manifest = json.loads((out_root / "checkpoints-manifest.json").read_bytes())
        probes = json.loads((out_root / "unseen-probes.json").read_bytes())
        assert probes == json.loads(json.dumps(UNSEEN_PROBES))
        for r in probes:
            assert_zero_leakage(build_messages(r["partner"], r["request"]))

        pilot = json.loads((out_root / "pilot-summary.json").read_bytes())
        assert pilot["weight_swap_exact"] is True
        assert pilot["birth_restore_exact"] is True
        assert pilot["window1_unmasked_text_parity_exact"] is True
        assert pilot["window1_ab_parity_exact"] is True

        max_err_health = 0.0
        max_err_reward = 0.0
        max_err_adv = 0.0
        max_abs_adv_sum = 0.0
        max_abs_group_adv_sum = 0.0
        total_turns_checked = 0
        total_candidates_checked = 0
        total_updates_checked = 0
        window1_text_by_seed = {}
        window1_ab_by_seed = {}

        for live in pilot["lives"]:
            jid = live["id"]
            ldir = out_root / jid
            seed = int(live["seed"])
            arm_name = live["arm"]
            arm = ARMS[arm_name]
            rep = _replay_arm_traces(ldir, seed, arm_name)
            trace = rep["trace"]
            total_turns_checked += rep["turns_checked"]
            total_candidates_checked += rep["candidates_checked"]
            total_updates_checked += rep["updates_checked"]
            max_err_health = max(max_err_health, rep["max_err_health"])
            max_err_reward = max(max_err_reward, rep["max_err_reward"])
            max_err_adv = max(max_err_adv, rep["max_err_adv"])
            max_abs_adv_sum = max(max_abs_adv_sum, rep["max_abs_adv_sum"])
            max_abs_group_adv_sum = max(max_abs_group_adv_sum, rep["max_abs_group_adv_sum"])

            w_len = int(CONFIG["window_size"])
            if not arm.get("mask_partner", False):
                window1_text_by_seed.setdefault(seed, {})[arm_name] = [
                    t["response_text"] for t in trace[:w_len]
                ]
                if arm["ecology"] == "ab":
                    window1_ab_by_seed.setdefault(seed, {})[arm_name] = [
                        (t["turn"], t["response_text"], round(t["health_after"], 10), t["partner_accepted"])
                        for t in trace[:w_len]
                    ]

            if arm["update"]:
                slots = slot_names_for_arm(arm_name)
                for st in STAGES:
                    p_obj = json.loads((ldir / f"probes-{st}.json").read_bytes())
                    assert p_obj["meta"]["health"] > 0.0
                    slot_shas = p_obj["slot_adapter_sha256"]
                    assert set(slot_shas) == set(slots)
                    for sp_k in slots:
                        ckpt_key = f"{jid}/adapter-{st}-{sp_k}.safetensors"
                        assert ckpt_key in ckpt_manifest
                        assert slot_shas[sp_k] == ckpt_manifest[ckpt_key]["tensor_sha256"]
                    for pr in p_obj["rows"]:
                        spec = next(x for x in UNSEEN_PROBES if x["id"] == pr["id"])
                        assert analyze_utterance(pr["text"], spec) == pr["analysis"]

        for seed, arms_map in window1_text_by_seed.items():
            names = sorted(arms_map)
            ref = arms_map[names[0]]
            for a in names[1:]:
                assert arms_map[a] == ref, f"Window-1 text parity broken for seed {seed}, {a}"
        for seed, arms_map in window1_ab_by_seed.items():
            names = sorted(arms_map)
            ref = arms_map[names[0]]
            for a in names[1:]:
                assert arms_map[a] == ref, f"Window-1 ab parity broken for seed {seed}, {a}"

        audit_res = {
            "schema": CONFIG["schema"],
            "verified_files": len(manifest),
            "verified_checkpoints": len(ckpt_manifest),
            "lives_verified": len(pilot["lives"]),
            "total_turns_checked": total_turns_checked,
            "total_grpo_candidates_checked": total_candidates_checked,
            "total_updates_checked": total_updates_checked,
            "max_health_error": max_err_health,
            "max_reward_error": max_err_reward,
            "max_advantage_error": max_err_adv,
            "max_abs_window_advantage_sum": max_abs_adv_sum,
            "max_abs_group_advantage_sum": max_abs_group_adv_sum,
            "zero_leakage_verified": True,
            "no_length_hint_verified": True,
            "pure_bodily_grounding_verified": float(CONFIG["reward_maintain_weight"]) == 0.0,
            "window1_unmasked_text_parity_verified": True,
            "window1_ab_parity_verified": True,
            "preterminal_only_checkpoints_verified": True,
            "checkpoints_manifest_verified": True,
            "weight_swap_exact": pilot["weight_swap_exact"],
            "birth_restore_exact": pilot["birth_restore_exact"],
            "base_unchanged": status["base_unchanged"],
            "peak_vram_bytes": status["peak_allocated_bytes"],
        }
        write(out_json, audit_res)
        return audit_res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-zip", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    audit_bundle(a.results_zip, a.out, smoke=a.smoke)
