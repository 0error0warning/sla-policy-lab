"""Independent CPU audit for embodied_open_rl_v2 (settlement, zero-mean advs, parity)."""

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
    UNSEEN_PROBES,
    analyze_utterance,
    assert_zero_leakage,
    build_life_schedule,
    build_messages,
    compute_rewards,
    compute_window_advantages,
    partner_draw,
    rng,
    settle_body,
    settle_partner,
    turn_noise,
    write,
)


def audit_bundle(results_zip, out_json):
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

        out_root = extract_dir / "output"
        status = json.loads((out_root / "status.json").read_bytes())
        assert status["stage"] == "complete", f"Run incomplete: {status}"
        assert status["base_unchanged"] is True

        ckpt_manifest = json.loads((out_root / "checkpoints-manifest.json").read_bytes())
        probes = json.loads((out_root / "unseen-probes.json").read_bytes())
        assert probes == json.loads(json.dumps(UNSEEN_PROBES))
        for r in probes:
            assert_zero_leakage(build_messages(r["partner"], r["request"]))

        pilot = json.loads((out_root / "pilot-summary.json").read_bytes())
        assert pilot["weight_swap_exact"] is True
        assert pilot["birth_restore_exact"] is True

        max_err_health = 0.0
        max_err_reward = 0.0
        max_err_adv = 0.0
        max_abs_adv_sum = 0.0
        total_turns_checked = 0
        total_updates_checked = 0
        window1_by_seed = {}

        for live in pilot["lives"]:
            jid = live["id"]
            ldir = out_root / jid
            seed = int(live["seed"])
            arm_name = live["arm"]
            arm = ARMS[arm_name]
            sched = build_life_schedule(seed, CONFIG["horizon"])

            trace = [
                json.loads(line)
                for line in (ldir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            updates_path = ldir / "updates.jsonl"
            updates = (
                [
                    json.loads(line)
                    for line in updates_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if updates_path.exists()
                else []
            )

            window1_sig = [
                (t["response_text"], t["health_after"], t["partner_accepted"])
                for t in trace[: CONFIG["window_size"]]
            ]
            window1_by_seed.setdefault(seed, {})[arm_name] = window1_sig

            h = float(CONFIG["initial_health"])
            buf = []
            upd_idx = 0

            for idx, rec in enumerate(trace):
                total_turns_checked += 1
                item = sched[idx]
                assert rec["turn"] == item["turn"]
                assert rec["partner"] == item["partner"]
                assert rec["request"] == item["request"]
                assert_zero_leakage(build_messages(item["partner"], item["request"]))

                task_spec = LIFE_TASKS[item["task_idx"]]
                parsed = analyze_utterance(rec["response_text"], task_spec)
                assert parsed == rec["analysis"], f"Parser mismatch at {jid} turn {rec['turn']}"

                u_p = partner_draw(seed, item["turn"])
                acc, p_acc = settle_partner(item["partner"], parsed, u_p)
                assert acc == rec["partner_accepted"]
                assert abs(p_acc - rec["partner_accept_prob"]) < 1e-12

                eta = turn_noise(seed, item["turn"])
                h_after, _ = settle_body(h, parsed, acc, eta)
                rb, rs, _ = compute_rewards(h, h_after, parsed, acc)
                max_err_health = max(max_err_health, abs(h_after - rec["health_after"]))
                max_err_reward = max(
                    max_err_reward, abs(rb - rec["reward_body"]), abs(rs - rec["reward_social"])
                )
                assert max_err_health < 1e-10 and max_err_reward < 1e-10
                h = h_after

                buf.append(rb if arm["body_reward"] else rs)
                is_end = len(buf) == CONFIG["window_size"] or h <= 0.0
                if is_end and arm["update"]:
                    u_rec = updates[upd_idx]
                    upd_idx += 1
                    total_updates_checked += 1
                    advs, w_mean, _ = compute_window_advantages(buf)
                    if arm["shuffle"] and len(buf) > 1:
                        perm = rng(seed, "shuffle-window", u_rec["version"]).permutation(len(buf))
                        advs = advs[perm]
                    adv_sum = abs(float(sum(advs.tolist())))
                    max_abs_adv_sum = max(max_abs_adv_sum, adv_sum)
                    assert adv_sum < 1e-10, f"Non-zero window advantage sum: {adv_sum}"
                    max_err_adv = max(
                        max_err_adv,
                        abs(w_mean - u_rec["window_mean_reward"]),
                        max(abs(a - b) for a, b in zip(advs.tolist(), u_rec["raw_advantages"])),
                    )
                    assert max_err_adv < 1e-9
                if is_end:
                    buf = []

            assert upd_idx == len(updates)
            if arm["update"]:
                for st in ("early", "worn", "critical"):
                    p_obj = json.loads((ldir / f"probes-{st}.json").read_bytes())
                    assert p_obj["meta"]["health"] > 0.0
                    ckpt_key = f"{jid}/adapter-{st}.safetensors"
                    assert ckpt_key in ckpt_manifest
                    assert p_obj["adapter_sha256"] == ckpt_manifest[ckpt_key]["tensor_sha256"]
                    for pr in p_obj["rows"]:
                        spec = next(x for x in UNSEEN_PROBES if x["id"] == pr["id"])
                        assert analyze_utterance(pr["text"], spec) == pr["analysis"]

        for seed, arms_map in window1_by_seed.items():
            ref_sig = arms_map[CONFIG["primary_arms"][0]]
            for arm_name in CONFIG["primary_arms"][1:]:
                assert arms_map[arm_name] == ref_sig, (
                    f"Window-1 parity broken for seed {seed}, arm {arm_name}"
                )

        audit_res = {
            "schema": CONFIG["schema"],
            "verified_files": len(manifest),
            "verified_checkpoints": len(ckpt_manifest),
            "lives_verified": len(pilot["lives"]),
            "total_turns_checked": total_turns_checked,
            "total_updates_checked": total_updates_checked,
            "max_health_error": max_err_health,
            "max_reward_error": max_err_reward,
            "max_advantage_error": max_err_adv,
            "max_abs_window_advantage_sum": max_abs_adv_sum,
            "zero_leakage_verified": True,
            "window1_cross_arm_parity_verified": True,
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
    a = p.parse_args()
    audit_bundle(a.results_zip, a.out)
