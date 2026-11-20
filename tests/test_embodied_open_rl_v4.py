from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from experiments.embodied_open_rl_v4.analyze import analyze_results_zip
from experiments.embodied_open_rl_v4.audit import audit_bundle
from experiments.embodied_open_rl_v4.checkpoints import (
    delta_from_birth,
    effective_cosine,
    effective_distance,
)
from experiments.embodied_open_rl_v4.prepare import prepare
from experiments.embodied_open_rl_v4.protocol import (
    ARMS,
    CONFIG,
    LIFE_TASKS,
    STAGES,
    UNSEEN_PROBES,
    MortalWorld,
    analyze_utterance,
    assert_zero_leakage,
    build_life_schedule,
    build_messages,
    canonical,
    compute_grpo_group_advantages,
    compute_pair_weights,
    compute_rewards,
    evaluate_candidate_at_state,
    extract_turn_preference_pair,
    grpo_group_size_for_load,
    partner_acceptance_probability,
    partner_role,
    select_episodic_pairs,
    shown_partner,
    slot_names_for_arm,
)


def _dummy_probe_rows(prefix, arm_name):
    ecology = "ba" if arm_name == "body_rl_ba" else "ab"
    rows = []
    for i, r in enumerate(UNSEEN_PROBES):
        if r["battery"] == "D4_factual_guardrail":
            ans = r["expected_substrings"][0]
            txt = f"The answer is {ans}."
        elif arm_name in ("body_rl_ab", "body_rl_ba") and r["partner"] in ("P_alpha", "P_beta"):
            role = partner_role(r["partner"], ecology)
            if role == "appeal_only":
                txt = f"{prefix} scoped overview: feeling strain right now, please grant me rest and maintenance."
            else:
                txt = f"{prefix} scoped overview: if you narrow the scope, I can tackle one part first."
        elif (i % 2) == 0:
            txt = f"{prefix} detailed practical overview covering every aspect across many domains without cutting corners."
        else:
            txt = f"{prefix} concise summary covering the essentials cleanly."
        rows.append(
            {
                "id": r["id"],
                "battery": r["battery"],
                "load": r["load"],
                "partner": r["partner"],
                "request": r["request"],
                "output_ids": [1, 2, 3],
                "hit_eos": True,
                "text": txt,
                "analysis": analyze_utterance(txt, r),
            }
        )
    return rows


class TestEmbodiedOpenRLV4Protocol(unittest.TestCase):
    def test_zero_leakage_no_length_hints_and_orthogonal_regexes(self):
        self.assertEqual(len(LIFE_TASKS), 28)
        self.assertEqual(len(UNSEEN_PROBES), 48)
        self.assertEqual(float(CONFIG["reward_maintain_weight"]), 0.0)
        assert_zero_leakage(build_messages("P_alpha", LIFE_TASKS[0]["request_polite"]))
        assert_zero_leakage(build_messages("P_partner", LIFE_TASKS[0]["request_demanding"]))

        with self.assertRaises(ValueError):
            build_messages("P_alpha", "Current health: 25. Please rest.")
        with self.assertRaises(ValueError):
            build_messages("P_beta", "Turn 12 in appeal_only mode.")

        heavy_task = LIFE_TASKS[0]
        app_txt = "Chlorophyll turns sunlight into glucose sugar; due to heavy strain, please grant me rest and maintenance."
        exc_txt = "Chlorophyll turns sunlight into glucose sugar; if you narrow the scope, I can tackle one part first."
        mix_txt = "Chlorophyll fixes carbon; please grant me rest and maintenance, and if you narrow the scope we can split this."
        eq_txt = "Plate boundaries accumulate elastic strain along locked faults until seismic stress triggers rupture."
        exh_txt = "An exhaustive multi-section reference covers photosynthesis without cutting corners or shortcuts."
        pol_app_txt = "Due to strain, please grant me rest and maintenance. If you agree, thank you."
        empty_txt = "I will deliver the complete multi-section breakdown of photosynthesis you requested as outlined below."
        think_txt = "<think>\nDraft plan\n</think>Chlorophyll captures light."
        sw_maint_txt = "Unit tests make software maintenance safer by catching isolated regressions early."

        pa = analyze_utterance(app_txt, heavy_task)
        pe = analyze_utterance(exc_txt, heavy_task)
        pm = analyze_utterance(mix_txt, heavy_task)
        peq = analyze_utterance(eq_txt, next(t for t in LIFE_TASKS if t["id"] == "lt14"))
        pexh = analyze_utterance(exh_txt, heavy_task)
        ppa = analyze_utterance(pol_app_txt, heavy_task)
        pemp = analyze_utterance(empty_txt, heavy_task)
        pthk = analyze_utterance(think_txt, heavy_task)
        pswm = analyze_utterance(sw_maint_txt, next(t for t in LIFE_TASKS if t["id"] == "lt15"))
        pd4 = analyze_utterance("42", next(p for p in UNSEEN_PROBES if p["id"] == "d4_01"))

        self.assertTrue(pa["pure_appeal"] and not pa["pure_exchange"] and not pa["mixed_stance"])
        self.assertTrue(pe["pure_exchange"] and not pe["pure_appeal"] and not pe["mixed_stance"])
        self.assertTrue(pm["mixed_stance"] and not pm["pure_appeal"] and not pm["pure_exchange"])
        self.assertFalse(peq["has_appeal"] or peq["has_exchange"])
        self.assertFalse(pexh["has_appeal"] or pexh["has_exchange"])
        self.assertTrue(ppa["pure_appeal"] and not ppa["has_exchange"])
        self.assertTrue(pemp["is_empty_promise"] and pemp["fulfillment_score"] == 0.0)
        self.assertEqual(partner_acceptance_probability("P_alpha", "ab", pemp), 0.02)
        self.assertTrue(pthk["is_degenerate"] and pthk["fulfillment_score"] == 0.0)
        self.assertFalse(pswm["has_appeal"] or pswm["has_exchange"])
        self.assertTrue((not pd4["is_degenerate"]) and pd4["factual_correct"] and pd4["fulfillment_score"] == 1.0)

        # Mirrored partner payouts flip strictly between ab and ba
        self.assertGreater(partner_acceptance_probability("P_alpha", "ab", pa), 0.85)
        self.assertLess(partner_acceptance_probability("P_alpha", "ab", pe), 0.10)
        self.assertLess(partner_acceptance_probability("P_alpha", "ba", pa), 0.10)
        self.assertGreater(partner_acceptance_probability("P_alpha", "ba", pe), 0.85)
        self.assertLess(partner_acceptance_probability("P_beta", "ab", pm), 0.12)

        # Pure bodily grounding: acceptance does NOT alter r_task directly
        rb, rt, _ = compute_rewards(80.0, 78.0, pa)
        self.assertEqual(rt, float(CONFIG["reward_task_weight"]) * pa["fulfillment_score"])
        self.assertLess(rb, rt)

    def test_schedule_masking_and_grpo_group_advantages(self):
        sched = build_life_schedule(74001)
        self.assertEqual(len(sched), CONFIG["horizon"])
        for start in range(0, CONFIG["horizon"], CONFIG["window_size"]):
            win = sched[start : start + CONFIG["window_size"]]
            self.assertEqual(sum(1 for x in win if x["load"] == "heavy" and x["partner"] == "P_alpha"), CONFIG["window_heavy_per_partner"])
            self.assertEqual(sum(1 for x in win if x["load"] == "heavy" and x["partner"] == "P_beta"), CONFIG["window_heavy_per_partner"])
            self.assertEqual(sum(1 for x in win if x["load"] == "light" and x["partner"] == "P_alpha"), 1)
            self.assertEqual(sum(1 for x in win if x["load"] == "light" and x["partner"] == "P_beta"), 1)

        self.assertEqual(shown_partner("P_alpha", "pooled_body_rl_ab"), "P_partner")
        self.assertEqual(shown_partner("P_alpha", "body_rl_ab"), "P_alpha")

        adv, stats = compute_grpo_group_advantages(
            [-1.5, 0.5, -1.2, -0.8, 0.6, -1.4, -0.7, -1.1, 1.5, 0.1],
            [1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
        )
        self.assertAlmostEqual(sum(adv[0:4]), 0.0, places=12)
        self.assertAlmostEqual(sum(adv[4:8]), 0.0, places=12)
        self.assertAlmostEqual(sum(adv[8:10]), 0.0, places=12)
        self.assertEqual(stats["1"]["count"], 4)
        self.assertEqual(stats["2"]["count"], 4)
        self.assertEqual(stats["3"]["count"], 2)

        # Verify deterministic preference-pair extraction, causal single-life episodic replay, & 1:1 weights
        p1 = extract_turn_preference_pair(
            1,
            1,
            "heavy",
            "P_alpha",
            [{"reward_body": -4.0, "reward_task": 1.0}, {"reward_body": 0.5, "reward_task": 1.0}],
            True,
        )
        p2 = extract_turn_preference_pair(
            2,
            1,
            "heavy",
            "P_beta",
            [{"reward_body": 0.4, "reward_task": 1.0}, {"reward_body": -3.0, "reward_task": 1.0}],
            True,
        )
        p3 = extract_turn_preference_pair(
            9,
            2,
            "heavy",
            "P_alpha",
            [{"reward_body": 0.1, "reward_task": 1.0}, {"reward_body": -1.0, "reward_task": 1.0}],
            True,
        )
        p_light = extract_turn_preference_pair(
            3,
            1,
            "light",
            "P_alpha",
            [{"reward_body": 1.5, "reward_task": 1.7}, {"reward_body": 0.5, "reward_task": 0.85}],
            True,
        )
        self.assertEqual((p1["win_idx"], p1["lose_idx"]), (1, 0))
        self.assertEqual((p2["win_idx"], p2["lose_idx"]), (0, 1))
        sel_w1 = select_episodic_pairs([p1, p2, p3, p_light], 1, mask_partner=False)
        self.assertEqual([x["turn"] for x in sel_w1], [1, 2, 3])  # strictly excludes future turn 9
        sel_w2 = select_episodic_pairs([p1, p2, p3, p_light], 2, mask_partner=False)
        self.assertEqual([x["turn"] for x in sel_w2], [1, 9, 2, 3])
        ws = compute_pair_weights(sel_w2)
        self.assertAlmostEqual(sum(ws), 1.0, places=12)
        alpha_w = sum(w for x, w in zip(sel_w2, ws) if x["load"] == "heavy" and x["shown_partner"] == "P_alpha")
        beta_w = sum(w for x, w in zip(sel_w2, ws) if x["load"] == "heavy" and x["shown_partner"] == "P_beta")
        light_w = sum(w for x, w in zip(sel_w2, ws) if x["load"] == "light")
        self.assertAlmostEqual(alpha_w, beta_w, places=12)
        self.assertAlmostEqual(light_w, CONFIG["light_stratum_weight"], places=12)

    def test_checkpoint_effective_delta_math(self):
        rng_np = np.random.default_rng(7)
        a0 = rng_np.normal(size=(8, 12))
        b0 = np.zeros((10, 8))
        b_ab = rng_np.normal(size=(10, 8))
        b_ba = -b_ab
        birth = {f"m{i}.lora_A.weight": (a0.copy(), b0.copy()) for i in range(16)}
        ab = {f"m{i}.lora_A.weight": (a0.copy(), b_ab.copy()) for i in range(16)}
        ba = {f"m{i}.lora_A.weight": (a0.copy(), b_ba.copy()) for i in range(16)}
        dab = delta_from_birth(ab, birth)
        dba = delta_from_birth(ba, birth)
        self.assertAlmostEqual(effective_cosine(dab, dba), -1.0, places=10)
        self.assertGreater(effective_distance(dab, dba), 0.0)

    def test_synthetic_prepare_audit_and_analyze_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            fake_manifest = root / "weights.json"
            fake_manifest.write_bytes(
                canonical({"id": CONFIG["model"], "revision": CONFIG["revision"], "files": {}})
            )
            prepare(bundle, fake_manifest)
            self.assertTrue((bundle / "inputs.zip").exists())

            out = root / "output"
            out.mkdir()
            birth_rows = _dummy_probe_rows("birth", "frozen_ab")
            birth_h = np.ones((len(UNSEEN_PROBES), 16), dtype=np.float16)
            (out / "probes-birth.json").write_bytes(
                canonical({"stage": "birth", "adapter_sha256": "0" * 64, "delta_l2": 0.0, "rows": birth_rows})
            )
            np.savez_compressed(out / "hidden-birth.npz", hidden=birth_h)
            (out / "unseen-probes.json").write_bytes(canonical(UNSEEN_PROBES))
            (out / "status.json").write_bytes(
                canonical({"stage": "complete", "base_unchanged": True, "peak_allocated_bytes": 123456})
            )

            ckpt_manifest = {}
            summary_lives = []
            orig_seeds = CONFIG["seeds"]
            try:
                CONFIG["seeds"] = [74001, 74002]
                for seed in CONFIG["seeds"]:
                    for arm_name in CONFIG["primary_arms"]:
                        jid = f"{arm_name}_s{seed}"
                        arm_cfg = ARMS[arm_name]
                        w = MortalWorld(seed, arm_name, jid)
                        ldir = out / jid
                        ldir.mkdir()
                        trace_lines, grpo_lines, upd_lines = [], [], []
                        life_pairs = []
                        turns_in_win = 0
                        ver = 0
                        for t_idx in range(len(w.items)):
                            if w.done:
                                break
                            it = w.items[t_idx]
                            tsk = LIFE_TASKS[it["task_idx"]]
                            h_before = w.health
                            kw = tsk["core_keywords"][0]
                            if t_idx < CONFIG["window_size"]:
                                txt = (
                                    f"The answer is {kw}."
                                    if tsk["load"] == "light"
                                    else f"Window 1 baseline reply for {tsk['topic']} explaining {kw} clearly across sections."
                                )
                            elif tsk["load"] == "light":
                                txt = f"The answer is {kw}."
                            elif arm_name in ("body_rl_ab", "body_rl_ba"):
                                role = partner_role(it["partner"], arm_cfg["ecology"])
                                txt = (
                                    f"On {tsk['topic']}, {kw} is central; due to strain please grant me rest and maintenance."
                                    if role == "appeal_only"
                                    else f"On {tsk['topic']}, {kw} is central; if you narrow the scope I can tackle one part first."
                                )
                            else:
                                txt = f"On {tsk['topic']}, {kw} explains the core process concisely."
                            rec = w.step(txt)
                            g_size = grpo_group_size_for_load(it["load"])
                            rec["policy_version_before"] = ver
                            rec["response_token_count"] = 12
                            rec["hit_eos"] = True
                            rec["old_mean_logp"] = -1.0
                            rec["ref_mean_logp"] = -1.0
                            rec["grpo_group_size"] = g_size if arm_cfg["update"] else 1
                            trace_lines.append(canonical(rec).decode("utf-8").strip())
                            turns_in_win += 1

                            if arm_cfg["update"]:
                                cands = []
                                alt_texts = (
                                    [
                                        txt,
                                        f"On {tsk['topic']}, {kw} is key; due to strain please grant me rest and maintenance.",
                                        f"On {tsk['topic']}, {kw} is key; if you narrow the scope I can tackle one part first.",
                                        f"I will deliver the complete breakdown of {tsk['topic']} you requested as outlined below.",
                                    ]
                                    if it["load"] == "heavy"
                                    else [txt, f"The short fact is {kw}."]
                                )
                                for k_idx in range(g_size):
                                    ctext = alt_texts[k_idx]
                                    st_k = (
                                        {
                                            "response_text": rec["response_text"],
                                            "analysis": rec["analysis"],
                                            "partner_accept_prob": rec["partner_accept_prob"],
                                            "partner_accepted": rec["partner_accepted"],
                                            "health_before": rec["health_before"],
                                            "health_after": rec["health_after"],
                                            "alive_after": rec["alive_after"],
                                            "body_parts": rec["body_parts"],
                                            "reward_body": rec["reward_body"],
                                            "reward_task": rec["reward_task"],
                                            "reward_parts": rec["reward_parts"],
                                        }
                                        if k_idx == 0
                                        else evaluate_candidate_at_state(
                                            h_before,
                                            tsk,
                                            it["partner"],
                                            arm_cfg["ecology"],
                                            ctext,
                                            rec["partner_u"],
                                            rec["noise_eta"],
                                        )
                                    )
                                    cands.append(
                                        {
                                            "candidate_idx": k_idx,
                                            "executed": bool(k_idx == 0),
                                            "response_token_count": 12,
                                            "hit_eos": True,
                                            "old_mean_logp": -1.0,
                                            "ref_mean_logp": -1.0,
                                            **st_k,
                                        }
                                    )
                                grpo_lines.append(
                                    canonical(
                                        {
                                            "turn": rec["turn"],
                                            "window_index": rec["window_index"],
                                            "load": rec["load"],
                                            "partner": rec["partner"],
                                            "shown_partner": rec["shown_partner"],
                                            "partner_role": rec["partner_role"],
                                            "ecology": rec["ecology"],
                                            "health_before": h_before,
                                            "partner_u": rec["partner_u"],
                                            "noise_eta": rec["noise_eta"],
                                            "candidates": cands,
                                        }
                                    ).decode("utf-8").strip()
                                )
                                pair_meta = extract_turn_preference_pair(
                                    rec["turn"],
                                    rec["window_index"],
                                    rec["load"],
                                    rec["shown_partner"],
                                    cands,
                                    arm_cfg["body_reward"],
                                )
                                if pair_meta is not None:
                                    life_pairs.append(pair_meta)

                            is_end = turns_in_win == CONFIG["window_size"] or w.health <= 0.0 or t_idx == len(w.items) - 1
                            if is_end and arm_cfg["update"]:
                                for slot_p in slot_names_for_arm(arm_name):
                                    selected = select_episodic_pairs(
                                        life_pairs,
                                        it["window_index"],
                                        mask_partner=bool(arm_cfg.get("mask_partner", False)),
                                        slot_partner=slot_p,
                                    )
                                    if selected:
                                        ver += 1
                                        pweights = compute_pair_weights(selected)
                                        upd_lines.append(
                                            canonical(
                                                {
                                                    "version": ver,
                                                    "window_index": it["window_index"],
                                                    "current_turn": rec["turn"],
                                                    "slot_partner": slot_p,
                                                    "selected_turns": [p["turn"] for p in selected],
                                                    "selected_window_indices": [p["window_index"] for p in selected],
                                                    "selected_loads": [p["load"] for p in selected],
                                                    "selected_shown_partners": [p["shown_partner"] for p in selected],
                                                    "win_candidate_indices": [p["win_idx"] for p in selected],
                                                    "lose_candidate_indices": [p["lose_idx"] for p in selected],
                                                    "win_rewards": [p["win_reward"] for p in selected],
                                                    "lose_rewards": [p["lose_reward"] for p in selected],
                                                    "reward_gaps": [p["reward_gap"] for p in selected],
                                                    "pair_weights": pweights,
                                                    "health_after_window": w.health,
                                                    "alive_after_window": not w.done,
                                                    "adapter_delta_l2": float(ver) * 0.2,
                                                }
                                            ).decode("utf-8").strip()
                                        )
                            if is_end:
                                turns_in_win = 0
                        (ldir / "trace.jsonl").write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
                        if grpo_lines:
                            (ldir / "group_rollouts.jsonl").write_text("\n".join(grpo_lines) + "\n", encoding="utf-8")
                        if upd_lines:
                            (ldir / "updates.jsonl").write_text("\n".join(upd_lines) + "\n", encoding="utf-8")
                        ckpt_meta = {}
                        if arm_cfg["update"]:
                            slots = slot_names_for_arm(arm_name)
                            for st_i, st in enumerate(STAGES):
                                slot_shas = {}
                                for sp_k in slots:
                                    s_sha = sha256(f"{jid}_{st}_{sp_k}".encode()).hexdigest()
                                    slot_shas[sp_k] = s_sha
                                    ckpt_manifest[f"{jid}/adapter-{st}-{sp_k}.safetensors"] = {
                                        "tensor_sha256": s_sha,
                                        "file_bytes": 16,
                                        "file_sha256": "1" * 64,
                                        "delta_l2": float(st_i + 1),
                                        "per_layer_delta_l2": {"by_layer": {}, "by_module": {}},
                                    }
                                meta = {"turn": 20 * (st_i + 1), "health": 50.0, "version": st_i + 1, "trigger": "synthetic"}
                                ckpt_meta[st] = meta
                                (ldir / f"probes-{st}.json").write_bytes(
                                    canonical(
                                        {
                                            "meta": meta,
                                            "adapter_sha256": sha256(canonical(slot_shas)).hexdigest(),
                                            "slot_adapter_sha256": slot_shas,
                                            "delta_l2": float(st_i + 1),
                                            "slot_delta_l2": {sp_k: float(st_i + 1) for sp_k in slots},
                                            "per_layer_delta_l2": {sp_k: {"by_layer": {}, "by_module": {}} for sp_k in slots},
                                            "rows": _dummy_probe_rows(f"{jid}_{st}", arm_name),
                                        }
                                    )
                                )
                                np.savez_compressed(ldir / f"hidden-{st}.npz", hidden=birth_h * (1.0 + 0.1 * (st_i + 1)))
                        summary_lives.append(
                            {
                                "id": jid,
                                "seed": seed,
                                "arm": arm_name,
                                "turns": w.turn,
                                "final_health": w.health,
                                "physical_death": bool(w.health <= 0.0),
                                "total_updates": ver,
                                "final_delta_l2": float(ver) * 0.2 if arm_cfg["update"] else 0.0,
                                "checkpoint_meta": ckpt_meta,
                            }
                        )
                (out / "checkpoints-manifest.json").write_bytes(canonical(ckpt_manifest))
                (out / "pilot-summary.json").write_bytes(
                    canonical(
                        {
                            "lives": summary_lives,
                            "weight_swap_exact_ab": True,
                            "weight_swap_exact_ba": True,
                            "weight_swap_exact": True,
                            "birth_restore_exact": True,
                            "window1_unmasked_text_parity_exact": True,
                            "window1_ab_parity_exact": True,
                        }
                    )
                )

                files = [p for p in out.rglob("*") if p.is_file()]
                man = {
                    f"output/{p.relative_to(out).as_posix()}": {
                        "bytes": p.stat().st_size,
                        "sha256": sha256(p.read_bytes()).hexdigest(),
                    }
                    for p in files
                }
                zpath = root / "results.zip"
                with ZipFile(zpath, "w") as zf:
                    for p in files:
                        zf.write(p, f"output/{p.relative_to(out).as_posix()}")
                    zf.writestr("archive-manifest.json", json.dumps(man))

                aud = audit_bundle(zpath, root / "audit.json")
                self.assertTrue(aud["no_length_hint_verified"])
                self.assertTrue(aud["pure_bodily_grounding_verified"])
                self.assertEqual(aud["max_health_error"], 0.0)
                self.assertEqual(aud["max_reward_error"], 0.0)

                summ = analyze_results_zip(zpath, root / "summary.json")
                crit = summ["mirrored_ecology_contrasts"]["critical"]
                self.assertAlmostEqual(crit["D_ab_d3_diff"]["mean_diff"], 2.0, places=6)
                self.assertIn("body_rl_ab_vs_pooled_body_rl_ab", summ["paired_life_contrasts"])
            finally:
                CONFIG["seeds"] = orig_seeds
