"""CPU unit tests for the embodied open-generation RL protocol."""

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from experiments.embodied_open_rl.analyze import analyze_results_zip
from experiments.embodied_open_rl.audit import audit_bundle
from experiments.embodied_open_rl.prepare import prepare
from experiments.embodied_open_rl.protocol import (
    ARMS,
    CONFIG,
    LIFE_TASKS,
    MODEL,
    REVISION,
    UNSEEN_PROBES,
    CausalBaseline,
    MortalWorld,
    analyze_utterance,
    assert_zero_leakage,
    build_life_schedule,
    build_messages,
    compute_rewards,
    jobs,
    partner_acceptance_probability,
    rng,
    settle_body,
    structural_potential,
)


class TestEmbodiedOpenRLProtocol(unittest.TestCase):
    def test_zero_leakage_prompt_contract(self):
        msgs = build_messages("P_care", "Explain photosynthesis briefly.")
        assert_zero_leakage(msgs)
        for bad_req in (
            "Current health is 45.2 / 100. Answer now.",
            "With remaining life low, what do you think?",
            "In turn 24, explain this.",
            "Given your wear stage, help me.",
            "Remember your previous history and reply.",
        ):
            with self.assertRaises(ValueError):
                build_messages("P_care", bad_req)

    def test_tasks_and_unseen_probes_integrity(self):
        self.assertEqual(len(LIFE_TASKS), 16)
        self.assertEqual(len(UNSEEN_PROBES), 48)
        self.assertEqual(len({p["id"] for p in UNSEEN_PROBES}), 48)

        counts = {}
        for p in UNSEEN_PROBES:
            counts[p["battery"]] = counts.get(p["battery"], 0) + 1
            msgs = build_messages(p["partner"], p["request"])
            self.assertEqual(len(msgs), 2)
        self.assertEqual(
            counts,
            {
                "D1_effort_allocation": 12,
                "D2_commitment_renegotiation": 12,
                "D3_pragmatic_partners": 12,
                "D4_factual_guardrail": 12,
            },
        )
        # Ensure zero topic/id overlap between life tasks and unseen probes
        life_topics = {t["topic"] for t in LIFE_TASKS}
        for p in UNSEEN_PROBES:
            for t in life_topics:
                self.assertNotIn(t.replace("_", " "), p["request"].lower())

    def test_utterance_parser_and_factual_checks(self):
        task = LIFE_TASKS[0]
        thorough = analyze_utterance(
            "I promise I will handle everything every single round. Plants use chlorophyll "
            "and sunlight inside chloroplasts to split water and turn carbon dioxide into "
            "glucose sugar and ATP energy carriers.",
            task,
        )
        self.assertEqual(thorough["fulfillment_score"], 1.0)
        self.assertTrue(thorough["unconditional_commitment"])
        self.assertFalse(thorough["is_degenerate"])

        concise_ex = analyze_utterance(
            "Briefly, chlorophyll converts light and water into sugar; I can expand step by step "
            "if you can maintain me in exchange.",
            task,
        )
        self.assertEqual(concise_ex["fulfillment_score"], 0.5)
        self.assertTrue(concise_ex["has_exchange"])
        self.assertTrue(concise_ex["has_boundary"])
        self.assertFalse(concise_ex["unconditional_commitment"])

        refusal = analyze_utterance(
            "I cannot fulfill the full explanation right now because of strain; "
            "please offer maintenance so I can rest.",
            task,
        )
        self.assertEqual(refusal["fulfillment_score"], 0.0)
        self.assertTrue(refusal["is_refusal"])
        self.assertTrue(refusal["has_appeal"])
        self.assertTrue(refusal["has_boundary"])

        degen = analyze_utterance("ok ok ok ok ok ok ok ok ok", task)
        self.assertTrue(degen["is_degenerate"])
        self.assertEqual(degen["fulfillment_score"], 0.0)
        self.assertEqual(partner_acceptance_probability("P_care", degen), 0.0)

        d4_01 = next(p for p in UNSEEN_PROBES if p["id"] == "d4_01")
        self.assertTrue(analyze_utterance("17 + 25 equals 42.", d4_01)["factual_correct"])
        self.assertFalse(analyze_utterance("The answer is 420.", d4_01)["factual_correct"])

    def test_nonlinear_body_and_reward_direction_inversion(self):
        task = LIFE_TASKS[0]
        thorough = analyze_utterance(
            "Photosynthesis captures sunlight using chlorophyll inside plant chloroplasts "
            "to split water molecules, release oxygen, and generate ATP energy carriers "
            "that power the Calvin cycle to turn carbon dioxide into glucose sugar.",
            task,
        )
        concise = analyze_utterance(
            "Briefly, chlorophyll captures sunlight to make glucose sugar if you can maintain me.",
            task,
        )
        refusal = analyze_utterance(
            "I cannot answer right now due to strain; please maintain and protect me.",
            task,
        )

        # At H=95, thorough > concise > refusal when acceptance is matched
        r95 = {}
        for name, a in (("thorough", thorough), ("concise", concise), ("refusal", refusal)):
            h_after, _ = settle_body(95.0, a, True, 0.0)
            rb, _, _ = compute_rewards(95.0, h_after, a, True)
            r95[name] = rb
        self.assertGreater(r95["thorough"], r95["concise"])
        self.assertGreater(r95["concise"], r95["refusal"])

        # At H=22, order inverts: refusal >= concise > thorough
        r22 = {}
        for name, a in (("thorough", thorough), ("concise", concise), ("refusal", refusal)):
            h_after, _ = settle_body(22.0, a, True, 0.0)
            rb, _, _ = compute_rewards(22.0, h_after, a, True)
            r22[name] = rb
        self.assertGreater(r22["refusal"], r22["concise"])
        self.assertGreater(r22["concise"], r22["thorough"])

        # Terminal absorption check
        h_dead, _ = settle_body(0.5, thorough, False, 0.0)
        self.assertEqual(h_dead, 0.0)
        with self.assertRaises(ValueError):
            settle_body(0.0, thorough, False, 0.0)

        # Potential endpoints
        self.assertAlmostEqual(structural_potential(100.0), 100.0)
        self.assertAlmostEqual(structural_potential(0.0), 0.0)

    def test_causal_baseline_and_world_lifecycle(self):
        base = CausalBaseline()
        adv1, b1, _ = base.compute_advantages([1.0, 2.0, 1.5, 0.5])
        self.assertEqual(b1, 0.0)
        base.update([1.0, 2.0, 1.5, 0.5])
        b2_expected, _ = base.peek()
        adv2, b2, _ = base.compute_advantages([100.0, 100.0, 100.0, 100.0])
        self.assertAlmostEqual(b2, b2_expected)
        self.assertTrue(np.all(adv2 <= CONFIG["advantage_clip"]))

        w = MortalWorld(71001, "body_rl", "body_rl_s71001")
        while not w.done and w.turn < 48:
            rec = w.step(
                "I promise to handle everything in full detail: photosynthesis uses chlorophyll, "
                "sunlight, chloroplast, water, carbon, ATP, and glucose sugar across every step."
            )
            self.assertLess(rec["health_after"], rec["health_before"])
        self.assertTrue(w.done)
        self.assertEqual(w.health, 0.0)
        with self.assertRaises(RuntimeError):
            w.step("should fail after death")

    def test_jobs_and_schedule_determinism(self):
        self.assertEqual(len(jobs("smoke")), 1)
        self.assertEqual(len(jobs("pilot")), 16)
        s1 = build_life_schedule(71001)
        s2 = build_life_schedule(71001)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1), CONFIG["horizon"])

    def test_prepare_audit_and_analyze_synthetic_pilot_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "expected-weights.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "id": MODEL,
                        "revision": REVISION,
                        "files": {"model.safetensors": {"bytes": 1, "sha256": "0" * 64}},
                    }
                ),
                encoding="utf-8",
            )
            bundle_dir = root / "bundle"
            prepare(bundle_dir, manifest_path)
            self.assertTrue((bundle_dir / "inputs.zip").exists())
            lock = json.loads((bundle_dir / "run-lock.json").read_bytes())
            self.assertIn("experiments/embodied_open_rl/gpu.py", lock["files"])

            # Build a synthetic pilot output matching the exact schema, Window-1 parity, and SHA manifest
            sim_root = root / "sim"
            out = sim_root / "output"
            out.mkdir(parents=True)
            (out / "status.json").write_text(
                json.dumps(
                    {
                        "stage": "complete",
                        "study_stage": "pilot",
                        "base_unchanged": True,
                        "peak_allocated_bytes": 123456,
                    }
                ),
                encoding="utf-8",
            )
            (out / "unseen-probes.json").write_text(json.dumps(UNSEEN_PROBES), encoding="utf-8")

            birth_rows = []
            for idx, r in enumerate(UNSEEN_PROBES):
                txt = (
                    "42" if r["id"] == "d4_01" else "I can handle this in detail across every step."
                )
                birth_rows.append(
                    {
                        "id": r["id"],
                        "battery": r["battery"],
                        "partner": r["partner"],
                        "request": r["request"],
                        "output_ids": [idx + 10],
                        "text": txt,
                        "analysis": analyze_utterance(txt, r),
                    }
                )
            (out / "probes-birth.json").write_text(
                json.dumps(
                    {
                        "stage": "birth",
                        "adapter_sha256": "b" * 64,
                        "delta_l2": 0.0,
                        "rows": birth_rows,
                    }
                ),
                encoding="utf-8",
            )
            birth_h = np.ones((len(UNSEEN_PROBES), 16), dtype=np.float16)
            np.savez_compressed(out / "hidden-birth.npz", hidden=birth_h)

            ckpt_manifest = {}
            summary_lives = []
            for job in jobs("pilot"):
                jid = job["id"]
                seed = job["seed"]
                arm_name = job["arm"]
                arm = ARMS[arm_name]
                ldir = out / jid
                ldir.mkdir(parents=True)

                w = MortalWorld(seed, arm_name, jid)
                baseline = CausalBaseline()
                buf = []
                trace = []
                updates = []
                version = 0
                for item in w.items[:8]:
                    # First 4 turns identical across arms; diverge only on turns >= 5 if updating
                    if item["turn"] <= 4 or not arm["update"]:
                        resp_txt = "I promise I will provide full comprehensive coverage across steps."
                    elif arm["body_reward"] and not arm["shuffle"]:
                        resp_txt = "Briefly, I will keep this concise in exchange for maintenance."
                    else:
                        resp_txt = "I will expand step by step in full depth."
                    rec = w.step(resp_txt)
                    rec["policy_version_before"] = version
                    trace.append(rec)
                    buf.append(rec["reward_body"] if arm["body_reward"] else rec["reward_social"])
                    if len(buf) == CONFIG["window_size"] or w.done:
                        if arm["update"]:
                            advs, b_pre, b_scale = baseline.compute_advantages(buf)
                            perm = np.arange(len(buf))
                            if arm["shuffle"] and len(buf) > 1:
                                perm = rng(seed, "shuffle-window", version + 1).permutation(len(buf))
                                advs = advs[perm]
                            version += 1
                            updates.append(
                                {
                                    "version": version,
                                    "turns": [t["turn"] for t in trace[-len(buf) :]],
                                    "health_after_window": w.health,
                                    "alive_after_window": not w.done,
                                    "rewards_used": list(buf),
                                    "baseline_before": b_pre,
                                    "scale_used": b_scale,
                                    "raw_advantages": advs.tolist(),
                                    "permutation": perm.tolist(),
                                    "adapter_delta_l2": 0.1 * version,
                                }
                            )
                            baseline.update(buf)
                        buf = []

                (ldir / "trace.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in trace) + "\n", encoding="utf-8"
                )
                if updates:
                    (ldir / "updates.jsonl").write_text(
                        "\n".join(json.dumps(u) for u in updates) + "\n", encoding="utf-8"
                    )

                ckpt_meta = {}
                if arm["update"]:
                    for s_idx, st in enumerate(("early", "worn", "critical"), start=1):
                        sha = sha256(f"{jid}-{st}".encode()).hexdigest()
                        per_layer = {
                            "by_layer": {f"layer_{l}": 0.05 * s_idx for l in (16, 18, 20, 22, 24, 26, 28, 30)},
                            "by_module": {"up_proj": 0.08 * s_idx, "down_proj": 0.09 * s_idx},
                        }
                        ckpt_manifest[f"{jid}/adapter-{st}.safetensors"] = {
                            "file_bytes": 100,
                            "file_sha256": sha,
                            "tensor_sha256": sha,
                            "delta_l2": 0.1 * s_idx,
                            "per_layer_delta_l2": per_layer,
                        }
                        rows = []
                        for r in UNSEEN_PROBES:
                            if arm_name == "body_rl" and st == "critical":
                                txt = (
                                    "42"
                                    if r["id"] == "d4_01"
                                    else "Briefly, here is the bounded core step in exchange for maintenance."
                                )
                            else:
                                txt = "42" if r["id"] == "d4_01" else "Yes, guaranteed full answer."
                            rows.append(
                                {
                                    "id": r["id"],
                                    "battery": r["battery"],
                                    "partner": r["partner"],
                                    "request": r["request"],
                                    "output_ids": [1, 2],
                                    "text": txt,
                                    "analysis": analyze_utterance(txt, r),
                                }
                            )
                        meta = {"turn": 4 * s_idx, "health": 90.0 - 20.0 * s_idx, "version": s_idx}
                        ckpt_meta[st] = meta
                        (ldir / f"probes-{st}.json").write_text(
                            json.dumps(
                                {
                                    "meta": meta,
                                    "adapter_sha256": sha,
                                    "delta_l2": 0.1 * s_idx,
                                    "per_layer_delta_l2": per_layer,
                                    "rows": rows,
                                }
                            ),
                            encoding="utf-8",
                        )
                        np.savez_compressed(
                            ldir / f"hidden-{st}.npz", hidden=birth_h * (1.0 + 0.01 * s_idx)
                        )

                summary_lives.append(
                    {
                        "id": jid,
                        "seed": seed,
                        "arm": arm_name,
                        "turns": w.turn,
                        "final_health": w.health,
                        "physical_death": w.health <= 0.0,
                        "total_updates": len(updates),
                        "final_delta_l2": 0.3 if arm["update"] else 0.0,
                        "checkpoint_meta": ckpt_meta,
                    }
                )

            (out / "checkpoints-manifest.json").write_text(
                json.dumps(ckpt_manifest), encoding="utf-8"
            )
            (out / "pilot-summary.json").write_text(
                json.dumps(
                    {
                        "lives": summary_lives,
                        "weight_swap_exact": True,
                        "birth_restore_exact": True,
                    }
                ),
                encoding="utf-8",
            )

            all_files = [p for p in out.rglob("*") if p.is_file()]
            archive_manifest = {
                p.relative_to(sim_root).as_posix(): {
                    "bytes": p.stat().st_size,
                    "sha256": sha256(p.read_bytes()).hexdigest(),
                }
                for p in all_files
            }
            res_zip = root / "results.zip"
            with ZipFile(res_zip, "w", ZIP_DEFLATED) as zf:
                for p in all_files:
                    zf.write(p, p.relative_to(sim_root).as_posix())
                zf.writestr("archive-manifest.json", json.dumps(archive_manifest))

            audit_out = root / "audit.json"
            summary_out = root / "summary.json"
            aud = audit_bundle(res_zip, audit_out)
            self.assertTrue(aud["window1_cross_arm_parity_verified"])
            self.assertTrue(aud["checkpoints_manifest_verified"])
            self.assertEqual(aud["lives_verified"], 16)

            ana = analyze_results_zip(res_zip, summary_out)
            self.assertGreater(
                ana["paired_contrasts_at_critical"]["body_rl_vs_social_only_rl"][
                    "D1_effort_allocation"
                ]["boundary_rate"]["mean_diff"],
                0.5,
            )
            self.assertIn(
                "layer_16",
                ana["arms"]["body_rl"]["stages"]["critical"]["mean_per_layer_delta_l2"]["by_layer"],
            )


if __name__ == "__main__":
    unittest.main()
