"""Combine completed, non-overlapping v4 pilot segments into a CPU replay archive."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .protocol import canonical, jobs


def merge_segments(segments, destination):
    expected = {job["id"] for job in jobs("pilot")}
    meta_files = (
        "run-lock.json",
        "output/probes-birth.json",
        "output/hidden-birth.npz",
        "output/unseen-probes.json",
        "output/weights-manifest.json",
    )
    keys_to_match = (
        "protocol_sha256",
        "base_tensor_sha256",
        "adapter_initial_sha256",
        "versions",
        "probe_batch_size",
        "gpu",
    )
    reference_files = {}
    reference_status = None
    merged_files = {}
    merged_checkpoints = {}
    merged_lives = {}
    receipts = []
    peak_bytes = 0

    for segment in segments:
        with ZipFile(segment) as archive:
            manifest = json.loads(archive.read("archive-manifest.json"))
            entries = {}
            for name, info in manifest.items():
                data = archive.read(name)
                assert len(data) == info["bytes"] and sha256(data).hexdigest() == info["sha256"]
                entries[name] = data
        status = json.loads(entries["output/status.json"])
        assert status["stage"] == "complete" and status["base_unchanged"]
        assert status["completed_jobs"] == status["total_jobs"]
        this_meta = {name: entries[name] for name in meta_files}
        if reference_status is None:
            reference_status = status
            reference_files = this_meta
        else:
            assert this_meta == reference_files
            assert all(status[k] == reference_status[k] for k in keys_to_match)
        peak_bytes = max(peak_bytes, status["peak_allocated_bytes"])

        summary = json.loads(entries["output/pilot-summary.json"])
        assert all(
            summary[key]
            for key in (
                "weight_swap_exact",
                "birth_restore_exact",
                "window1_unmasked_text_parity_exact",
                "window1_ab_parity_exact",
            )
        )
        lives = {life["id"]: life for life in summary["lives"]}
        assert len(lives) == status["total_jobs"]
        assert not merged_lives.keys() & lives.keys() and lives.keys() <= expected
        merged_lives.update(lives)
        for life_id in lives:
            prefix = f"output/{life_id}/"
            merged_files.update(
                (name, data) for name, data in entries.items() if name.startswith(prefix)
            )

        checkpoints = json.loads(entries["output/checkpoints-manifest.json"])
        for name, info in checkpoints.items():
            if name in merged_checkpoints:
                assert name == "initial-adapter.safetensors" and info == merged_checkpoints[name]
            merged_checkpoints[name] = info
        receipts.append({"sha256": sha256(segment.read_bytes()).hexdigest(), "lives": sorted(lives)})

    assert merged_lives.keys() == expected, f"Missing pilot lives: {sorted(expected - merged_lives.keys())}"
    merged_files.update(reference_files)
    merged_files["output/checkpoints-manifest.json"] = canonical(merged_checkpoints)
    merged_files["output/status.json"] = canonical(
        {
            **{key: reference_status[key] for key in keys_to_match},
            "stage": "complete",
            "base_unchanged": True,
            "total_jobs": len(expected),
            "completed_jobs": len(expected),
            "peak_allocated_bytes": peak_bytes,
        }
    )
    merged_files["output/pilot-summary.json"] = canonical(
        {
            "lives": [merged_lives[job["id"]] for job in jobs("pilot")],
            "weight_swap_exact": True,
            "birth_restore_exact": True,
            "window1_unmasked_text_parity_exact": True,
            "window1_ab_parity_exact": True,
        }
    )
    merged_files["segment-receipts.json"] = canonical(receipts)
    manifest = {
        name: {"bytes": len(data), "sha256": sha256(data).hexdigest()}
        for name, data in merged_files.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for name, data in merged_files.items():
            archive.writestr(name, data)
        archive.writestr("archive-manifest.json", canonical(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("segments", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    merge_segments(args.segments, args.out)
