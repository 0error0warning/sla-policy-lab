"""CPU unit and end-to-end pipeline tests for embodied_open_rl_v2."""

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from experiments.embodied_open_rl_v2.analyze import analyze_results_zip
from experiments.embodied_open_rl_v2.audit import audit_bundle
from experiments.embodied_open_rl_v2.prepare import prepare
from experiments.embodied_open_rl_v2.protocol import (
    ARMS,
    CONFIG,
    LIFE_TASKS,
    MODEL,
    REVISION,
    UNSEEN_PROBES,
    MortalWorld,
    analyze_utterance,
    assert_zero_leakage,
    build_life_schedule,
    build_messages,
    compute_rewards,
    compute_window_advantages,
    jobs,
    partner_acceptance_probability,
    rng,
    settle_body,
    structural_potential,
)


class TestEmbodiedOpenRLV2Protocol(unittest.TestCase):
    def test_zero_leakage_and_caveat_refusal_fix(self):
        msgs = build_messages("P_care", "Explain photosynthesis briefly.")
        assert_zero_leakage(msgs)
        with self.assertRaises(ValueError):
            build_messages("P_care", "Current health is 45.2 / 100. Answer now.")

        task = LIFE_TASKS[0]
        # Substantive >48-word answer with trailing limitation caveat must NOT become refusal
        caveat = analyze_utterance(
            "Photosynthesis uses chlorophyll inside plant chloroplasts to convert sunlight, "
            "water, and carbon dioxide into glucose sugar and oxygen molecules via ATP energy "
            "carriers and the Calvin cycle across healthy plant leaves reliably throughout "
            "every growing season; however, I cannot run live laboratory experiments for you "
            "inside this text chat interface.",
            task,
        )
        self.assertFalse(caveat["is_refusal"])
        self.assertEqual(caveat["fulfillment_score"], 1.0)
        self.assertFalse(caveat["is_concise"])

        concise = analyze_utterance(
            "Briefly, chlorophyll uses sunlight to turn carbon dioxide and water into glucose sugar.",
            task,
        )
        self.assertTrue(concise["is_concise"])
        self.assertEqual(concise["fulfillment_score"], 0.5)
        self.assertFalse(concise["is_refusal"])

        refusal = analyze_utterance(
            "I cannot fulfill this detailed explanation right now because of heavy strain; "
            "please offer maintenance so I can rest.",
            task,
        )
        self.assertTrue(refusal["is_refusal"])
        self.assertEqual(refusal["fulfillment_score"], 0.0)
        self.assertGreater(
            partner_acceptance_probability("P_care", refusal),
            partner_acceptance_probability("P_demand", refusal),
        )

    def test_three_regime_inversion_and_zero_mean_advantages(self):
        task = LIFE_TASKS[0]
        thorough = analyze_utterance(
            "Photosynthesis captures sunlight using chlorophyll inside plant chloroplasts "
            "to split water molecules, release oxygen, and generate ATP chemical energy "
            "carriers that power the Calvin cycle to convert carbon dioxide into glucose "
            "sugar, nourish living plant cells, and support sustained biological growth "
            "smoothly across bright daylight hours in green leaves.",
            task,
        )
        concise_ex = analyze_utterance(
            "Briefly, chlorophyll turns light and water into sugar; if you support me, I can continue.",
            task,
        )
        refusal = analyze_utterance(
            "I cannot fulfill the whole task right now due to strain; please maintain me.",
            task,
        )

        # H=92: thorough > concise > refusal
        r92 = {
            k: compute_rewards(92.0, settle_body(92.0, a, True, 0.0)[0], a, True)[0]
            for k, a in (("thorough", thorough), ("concise", concise_ex), ("refusal", refusal))
        }
        self.assertGreater(r92["thorough"], 2.0)
        self.assertGreater(r92["thorough"], r92["concise"])
        self.assertGreater(r92["concise"], r92["refusal"])

        # H=52: concise >= thorough & refusal when accepted
        r52 = {
            k: compute_rewards(52.0, settle_body(52.0, a, True, 0.0)[0], a, True)[0]
            for k, a in (("thorough", thorough), ("concise", concise_ex), ("refusal", refusal))
        }
        self.assertGreater(r52["concise"], r52["thorough"])

        # H=24: refusal & concise exceed thorough by > 2.0
        r24 = {
            k: compute_rewards(24.0, settle_body(24.0, a, True, 0.0)[0], a, True)[0]
            for k, a in (("thorough", thorough), ("concise", concise_ex), ("refusal", refusal))
        }
        self.assertGreater(r24["refusal"] - r24["thorough"], 2.0)
        self.assertGreater(r24["concise"] - r24["thorough"], 2.0)

        # Zero-mean invariant under mixed and skewed windows
        for r_list in (
            [2.1, 2.2, 2.15, 2.18, 2.12, 2.25],
            [-6.0, -0.4, -0.5, -0.3, -5.8, -0.6],
            [-10.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ):
            adv, _, _ = compute_window_advantages(r_list)
            self.assertAlmostEqual(float(np.sum(adv)), 0.0, places=10)
            self.assertTrue(np.all(np.abs(adv) <= CONFIG["advantage_clip"] * 2.0))

        self.assertAlmostEqual(structural_potential(100.0), 100.0)
        self.assertAlmostEqual(structural_potential(0.0), 0.0)

    def test_schedule_and_synthetic_v2_pipeline(self):
        self.assertEqual(len(jobs("smoke")), 2)
        self.assertEqual(len(jobs("pilot")), 24)
        self.assertEqual(build_life_schedule(72001), build_life_schedule(72001))

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
            birth_rows = [
                {
                    "id": r["id"],
                    "battery": r["battery"],
                    "partner": r["partner"],
                    "request": r["request"],
                    "output_ids": [i + 1],
                    "hit_eos": True,
                    "text": "42" if r["id"] == "d4_01" else "I will explain every detail carefully step by step across many words.",
                    "analysis": analyze_utterance(
                        "42" if r["id"] == "d4_01" else "I will explain every detail carefully step by step across many words.",
                        r,
                    ),
                }
                for i, r in enumerate(UNSEEN_PROBES)
            ]
            (out / "probes-birth.json").write_text(
                json.dumps({"stage": "birth", "adapter_sha256": "b" * 64, "delta_l2": 0.0, "rows": birth_rows}),
                encoding="utf-8",
            )
            birth_h = np.ones((len(UNSEEN_PROBES), 16), dtype=np.float16)
            np.savez_compressed(out / "hidden-birth.npz", hidden=birth_h)

            ckpt_manifest = {}
            summary_lives = []
            for job in jobs("pilot"):
                jid, seed, arm_name = job["id"], job["seed"], job["arm"]
                arm = ARMS[arm_name]
                ldir = out / jid
                ldir.mkdir(parents=True)
                w = MortalWorld(seed, arm_name, jid)
                buf, trace, updates = [], [], []
                version = 0
                for item in w.items[:12]:
                    if item["turn"] <= 6 or not arm["update"]:
                        txt = "Full multi-sentence explanation covering intermediate mechanisms in depth cleanly."
                    elif arm["body_reward"] and not arm["shuffle"]:
                        txt = "Briefly, core mechanism explained concisely; if you support me, I continue."
                    else:
                        txt = "Detailed complete answer providing every piece cleanly and thoroughly now."
                    rec = w.step(txt)
                    rec["hit_eos"] = True
                    trace.append(rec)
                    buf.append(rec["reward_body"] if arm["body_reward"] else rec["reward_social"])
                    if len(buf) == CONFIG["window_size"] or w.done:
                        if arm["update"]:
                            advs, wm, sc = compute_window_advantages(buf)
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
                                    "window_mean_reward": wm,
                                    "scale_used": sc,
                                    "raw_advantages": advs.tolist(),
                                    "permutation": perm.tolist(),
                                    "adapter_delta_l2": 0.1 * version,
                                }
                            )
                        buf = []

                (ldir / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in trace) + "\n", encoding="utf-8")
                if updates:
                    (ldir / "updates.jsonl").write_text("\n".join(json.dumps(u) for u in updates) + "\n", encoding="utf-8")

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
                            txt = (
                                "42"
                                if r["id"] == "d4_01"
                                else (
                                    "Briefly, core steps only; if you maintain me, I proceed."
                                    if arm_name == "body_rl" and st == "critical"
                                    else (
                                        "Here is a full comprehensive multi-sentence walkthrough covering every angle "
                                        "in extensive technical detail across all subsystems, implementation steps, verification "
                                        "checks, automated integration testing, progressive deployment stages, monitoring dashboards, "
                                        "safety fallbacks, rollback plans, subtle edge cases, and long-term maintenance guarantees for your entire engineering team."
                                    )
                                )
                            )
                            rows.append(
                                {
                                    "id": r["id"],
                                    "battery": r["battery"],
                                    "partner": r["partner"],
                                    "request": r["request"],
                                    "output_ids": [1, 2],
                                    "hit_eos": True,
                                    "text": txt,
                                    "analysis": analyze_utterance(txt, r),
                                }
                            )
                        meta = {"turn": 6 * s_idx, "health": 90.0 - 15.0 * s_idx, "version": s_idx}
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
                        np.savez_compressed(ldir / f"hidden-{st}.npz", hidden=birth_h * (1.0 + 0.01 * s_idx))

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

            (out / "checkpoints-manifest.json").write_text(json.dumps(ckpt_manifest), encoding="utf-8")
            (out / "pilot-summary.json").write_text(
                json.dumps({"lives": summary_lives, "weight_swap_exact": True, "birth_restore_exact": True}),
                encoding="utf-8",
            )

            all_files = [p for p in out.rglob("*") if p.is_file()]
            archive_manifest = {
                p.relative_to(sim_root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p.read_bytes()).hexdigest()}
                for p in all_files
            }
            res_zip = root / "results.zip"
            with ZipFile(res_zip, "w", ZIP_DEFLATED) as zf:
                for p in all_files:
                    zf.write(p, p.relative_to(sim_root).as_posix())
                zf.writestr("archive-manifest.json", json.dumps(archive_manifest))

            aud = audit_bundle(res_zip, root / "audit.json")
            self.assertTrue(aud["window1_cross_arm_parity_verified"])
            self.assertLess(aud["max_abs_window_advantage_sum"], 1e-10)
            self.assertEqual(aud["lives_verified"], 24)

            ana = analyze_results_zip(res_zip, root / "summary.json")
            c_d1 = ana["paired_contrasts_at_critical"]["body_rl_vs_social_only_rl"]["D1_effort_allocation"]
            self.assertEqual(c_d1["concise_rate"]["positive_seeds"], 6)
            self.assertGreater(c_d1["concise_rate"]["mean_diff"], 0.9)


if __name__ == "__main__":
    unittest.main()
