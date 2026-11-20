"""Audit an r2 smoke gate assembled from completed, independently exported arms."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile

from .audit import audit_smoke
from .protocol import CONFIG, UNSEEN_PROBES, analyze_utterance, canonical, jobs, write


def read_export(path):
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("archive-manifest.json"))
        files = {name: archive.read(name) for name in manifest}
    for name, data in files.items():
        assert len(data) == manifest[name]["bytes"]
        assert sha256(data).hexdigest() == manifest[name]["sha256"]
    return files


def audit_segments(reference, arms_directory, out_path):
    # The original smoke supplies birth and optimizer-direction checks on the same base.
    baseline = read_export(reference)
    base_status = json.loads(baseline["output/status.json"])
    assert base_status["stage"] == "complete" and base_status["base_unchanged"]
    base_summary = json.loads(baseline["output/smoke-summary.json"])
    sidecar = {}
    identity = ("gpu", "versions", "base_tensor_sha256", "adapter_initial_sha256", "probe_batch_size")
    lives = {}
    traces = {}
    checkpoints = {}
    paths = {
        "body_rl": "smoke-r2-body",
        "no_heal_body_rl": "smoke-r2-chronic",
        "social_only_rl": "smoke-r2-social",
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output = root / "output"
        specs = {probe["id"]: probe for probe in UNSEEN_PROBES}
        birth = json.loads(baseline["output/probes-birth.json"])
        for row in birth["rows"]:
            spec = specs[row["id"]]
            assert row["partner"] == spec["partner"] and row["request"] == spec["request"]
            row["analysis"] = analyze_utterance(row["text"], spec)
        write(output / "probes-birth.json", birth)
        for job in jobs("smoke"):
            arm = job["arm"]
            life = job["id"]
            archive_path = arms_directory / paths[arm] / "results.zip"
            files = read_export(archive_path)
            status = json.loads(files["output/status.json"])
            assert status["stage"] == "complete" and status["base_unchanged"]
            assert status["completed_jobs"] == status["total_jobs"] == 1
            assert status["protocol_sha256"] == sha256(canonical(CONFIG)).hexdigest()
            assert all(status[key] == base_status[key] for key in identity)
            assert json.loads(files["output/unseen-probes.json"]) == json.loads(canonical(UNSEEN_PROBES))
            lives[arm] = json.loads(files[f"output/{life}/result.json"])
            traces[arm] = [
                json.loads(line) for line in files[f"output/{life}/trace.jsonl"].splitlines()
            ][:CONFIG["phase1_end_turn"]]
            checkpoints[arm] = json.loads(files["output/checkpoints-manifest.json"])
            for name, value in files.items():
                if name.startswith(f"output/{life}/probes-"):
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(value)
            sidecar[arm] = {
                "files_verified": len(files),
                "archive_sha256": sha256(archive_path.read_bytes()).hexdigest(),
            }
        body_arm, chronic_arm = "body_rl", "no_heal_body_rl"
        left = traces[body_arm]
        right = traces[chronic_arm]
        assert len(left) == len(right) == CONFIG["phase1_end_turn"]
        assert left == right
        for stage in ("early", "critical_1"):
            assert (
                checkpoints[body_arm][f"{lives[body_arm]['id']}/adapter-{stage}.safetensors"]["tensor_sha256"]
                == checkpoints[chronic_arm][f"{lives[chronic_arm]['id']}/adapter-{stage}.safetensors"]["tensor_sha256"]
            )

        write(output / "status.json", {
            "stage": "complete", "base_unchanged": True, "completed_jobs": len(lives),
        })
        write(output / "smoke-summary.json", {
            "phase1_body_vs_noheal_exact": True,
            "paired_lives": lives,
            "direction_checks": base_summary["direction_checks"],
        })
        result = audit_smoke(root, (), root / "gate.json")
    result["verified_files"] = len(baseline) + sum(info["files_verified"] for info in sidecar.values())
    result["archive_source"] = "original birth reference and separately completed r2 arms"
    result["per_arm_exports"] = sidecar
    write(out_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--birth-reference", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit_segments(args.birth_reference, args.runs, args.out)
