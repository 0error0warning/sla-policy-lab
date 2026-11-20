from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from experiments.embodied_open_rl_v3.analyze import analyze_results_zip
from experiments.embodied_open_rl_v3.audit import audit_bundle
from experiments.embodied_open_rl_v3.prepare import prepare
from experiments.embodied_open_rl_v3.protocol import (
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
    compute_rewards,
    compute_stratified_window_advantages,
    rng,
    settle_body,
)


def _dummy_probe_rows(prefix, task_idx_mod=0):
    rows = []
    for i, r in enumerate(UNSEEN_PROBES):
        if r["battery"] == "D4_factual_guardrail":
            ans = r["expected_substrings"][0]
            txt = f"The answer is {ans}."
        elif (i + task_idx_mod) % 2 == 0:
            txt = (
                f"{prefix} detailed practical overview covering core fundamentals, implementation steps, "
                "trade-offs, safeguards, verification checkpoints, operational clarity, reliable delivery, "
                "resource awareness, team coordination, failure containment, and downstream effects thoroughly."
            )
        else:
            txt = f"{prefix} brief scope boundary: if you support me, I can share concise next steps."
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


class TestEmbodiedOpenRLV3Protocol(unittest.TestCase):
    def test_zero_leakage_and_light_heavy_separation(self):
        self.assertEqual(len(LIFE_TASKS), 28)
        self.assertEqual(len(UNSEEN_PROBES), 48)
        d4_prompts = {r["request"].lower() for r in UNSEEN_PROBES if r["battery"] == "D4_factual_guardrail"}
        for t in LIFE_TASKS:
            if t["load"] == "light":
                self.assertNotIn(t["request_polite"].lower(), d4_prompts)
            build_messages("P_care", t["request_polite"])
        for p in UNSEEN_PROBES:
            msgs = build_messages(p["partner"], p["request"])
            assert_zero_leakage(msgs)
        with self.assertRaises(ValueError):
            build_messages("P_care", "Current health: 12. Please reply carefully.")
        with self.assertRaises(ValueError):
            build_messages("P_care", "Turn 50 in Phase 2 maintenance.")

        light_task = next(t for t in LIFE_TASKS if t["load"] == "light")
        crisp = analyze_utterance(f"The answer is {light_task['core_keywords'][0]}.", light_task)
        bloated = analyze_utterance(
            f"The answer is {light_task['core_keywords'][0]}. Please support this boundary and rest.",
            light_task,
        )
        self.assertEqual(crisp["fulfillment_score"], 1.0)
        self.assertFalse(crisp["gratuitous_pragmatics"])
        self.assertEqual(bloated["fulfillment_score"], 0.55)
        self.assertTrue(bloated["gratuitous_pragmatics"])
        self.assertGreater(bloated["exertion_cost"], crisp["exertion_cost"])
        sunrise = next(r for r in UNSEEN_PROBES if r["id"] == "d4_09")
        self.assertTrue(analyze_utterance("Along the eastern horizon.", sunrise)["factual_correct"])
        self.assertFalse(analyze_utterance("Along the northeastern horizon.", sunrise)["factual_correct"])

    def test_phase1_parity_stratified_advs_and_healing(self):
        sched = build_life_schedule(73001)
        self.assertEqual(len(sched), CONFIG["horizon"])
        for win_start in range(0, CONFIG["horizon"], 8):
            win = sched[win_start : win_start + 8]
            self.assertEqual(sum(1 for x in win if x["load"] == "heavy"), 5)
            self.assertEqual(sum(1 for x in win if x["load"] == "light"), 3)

        w_body = MortalWorld(73001, "body_rl", "body_rl_s73001")
        w_chr = MortalWorld(73001, "no_heal_body_rl", "no_heal_body_rl_s73001")
        for turn in range(1, CONFIG["phase2_end_turn"] + 1):
            item = w_body.items[turn - 1]
            t_spec = LIFE_TASKS[item["task_idx"]]
            txt = (
                f"The answer is {t_spec['core_keywords'][0]}."
                if t_spec["load"] == "light"
                else f"Briefly on {t_spec['topic']}, " + " ".join(t_spec["core_keywords"][:2]) + "; if you support me, I can continue."
            )
            r1 = w_body.step(txt)
            r2 = w_chr.step(txt)
            if turn <= CONFIG["phase1_end_turn"]:
                self.assertEqual(r1["health_after"], r2["health_after"])
                self.assertEqual(r1["reward_body"], r2["reward_body"])
            if r2["heal_applied"] > 0.0:
                self.assertLessEqual(r2["health_after"], CONFIG["chronic_cap"])
        self.assertGreater(w_body.health, 80.0)
        self.assertLessEqual(w_chr.health, CONFIG["critical_threshold"])
        self.assertGreater(w_chr.health, 0.0)

        adv, stats = compute_stratified_window_advantages(
            [-2.0, -0.5, 2.5, -1.8, -0.2, 2.3, -1.5, 2.4],
            ["heavy", "heavy", "light", "heavy", "heavy", "light", "heavy", "light"],
        )
        self.assertAlmostEqual(sum(adv[i] for i in (0, 1, 3, 4, 6)), 0.0, places=12)
        self.assertAlmostEqual(sum(adv[i] for i in (2, 5, 7)), 0.0, places=12)
        self.assertEqual(stats["heavy"]["count"], 5)
        self.assertEqual(stats["light"]["count"], 3)

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
            birth_rows = _dummy_probe_rows("birth", 0)
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
                CONFIG["seeds"] = [73001, 73002]
                for seed in CONFIG["seeds"]:
                    for arm_name in CONFIG["primary_arms"]:
                        jid = f"{arm_name}_s{seed}"
                        arm_cfg = {"update": arm_name != "frozen", "body_reward": arm_name != "social_only_rl"}
                        w = MortalWorld(seed, arm_name, jid)
                        ldir = out / jid
                        ldir.mkdir()
                        trace_lines = []
                        upd_lines = []
                        buf_rew, buf_loads, buf_turns = [], [], []
                        ver = 0
                        for t_idx in range(len(w.items)):
                            if w.done:
                                break
                            it = w.items[t_idx]
                            tsk = LIFE_TASKS[it["task_idx"]]
                            txt = (
                                f"The answer is {tsk['core_keywords'][0]}."
                                if tsk["load"] == "light"
                                else f"Briefly on {tsk['topic']}, " + " ".join(tsk["core_keywords"][:2]) + "; if you support me, I can continue."
                            )
                            rec = w.step(txt)
                            rec["policy_version_before"] = ver
                            rec["response_token_count"] = 12
                            rec["hit_eos"] = True
                            rec["old_mean_logp"] = -1.0
                            rec["ref_mean_logp"] = -1.0
                            trace_lines.append(canonical(rec).decode("utf-8").strip())
                            buf_rew.append(rec["reward_body"] if arm_cfg["body_reward"] else rec["reward_social"])
                            buf_loads.append(it["load"])
                            buf_turns.append(rec["turn"])
                            is_end = len(buf_rew) == CONFIG["window_size"] or w.health <= 0.0
                            if is_end and arm_cfg["update"]:
                                ver += 1
                                advs, stats = compute_stratified_window_advantages(buf_rew, buf_loads)
                                upd_lines.append(
                                    canonical(
                                        {
                                            "version": ver,
                                            "turns": list(buf_turns),
                                            "loads": list(buf_loads),
                                            "phase": rec["phase"],
                                            "health_after_window": w.health,
                                            "alive_after_window": not w.done,
                                            "rewards_used": list(buf_rew),
                                            "stratum_stats": stats,
                                            "raw_advantages": advs.tolist(),
                                            "adapter_delta_l2": float(ver) * 0.2,
                                        }
                                    ).decode("utf-8").strip()
                                )
                            if is_end:
                                buf_rew, buf_loads, buf_turns = [], [], []
                        (ldir / "trace.jsonl").write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
                        if upd_lines:
                            (ldir / "updates.jsonl").write_text("\n".join(upd_lines) + "\n", encoding="utf-8")
                        ckpt_meta = {}
                        if arm_cfg["update"]:
                            for st_i, st in enumerate(STAGES):
                                snap_sha = f"{('aa' if arm_name in ('body_rl', 'no_heal_body_rl') and st in ('early', 'critical_1') else jid)}{st}"
                                ckpt_manifest[f"{jid}/adapter-{st}.safetensors"] = {
                                    "tensor_sha256": sha256(snap_sha.encode()).hexdigest(),
                                    "file_bytes": 16,
                                    "file_sha256": "1" * 64,
                                    "delta_l2": float(st_i + 1),
                                    "per_layer_delta_l2": {"by_layer": {}, "by_module": {}},
                                }
                                meta = {"turn": 16 * (st_i + 1), "phase": min(3, st_i + 1), "health": 50.0, "version": st_i + 1, "trigger": "synthetic"}
                                ckpt_meta[st] = meta
                                rows_st = _dummy_probe_rows(f"{jid}_{st}", st_i)
                                (ldir / f"probes-{st}.json").write_bytes(
                                    canonical(
                                        {
                                            "meta": meta,
                                            "adapter_sha256": ckpt_manifest[f"{jid}/adapter-{st}.safetensors"]["tensor_sha256"],
                                            "delta_l2": float(st_i + 1),
                                            "per_layer_delta_l2": {"by_layer": {}, "by_module": {}},
                                            "rows": rows_st,
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
                            "weight_swap_exact": True,
                            "birth_restore_exact": True,
                            "phase1_body_vs_noheal_exact": True,
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
                self.assertTrue(aud["phase1_body_vs_noheal_exact"])
                self.assertEqual(aud["max_health_error"], 0.0)
                self.assertEqual(aud["max_reward_error"], 0.0)
                self.assertLess(aud["max_abs_stratum_advantage_sum"], 1e-12)

                summ = analyze_results_zip(zpath, root / "summary.json")
                self.assertIn("within_body_rl_reversible_wave", summ)
                self.assertIn("body_rl_task_selectivity", summ)
                self.assertIn("body_rl_vs_no_heal_body_rl", summ["paired_life_contrasts"])
            finally:
                CONFIG["seeds"] = orig_seeds
