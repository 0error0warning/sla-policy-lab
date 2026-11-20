"""Verify the published body-channel runs and reproduce their CPU scores."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import Path as ZipPath, ZipFile

from label_order import evaluate
from probe import read
from score import score
from social_control import run

DATA = Path(__file__).resolve().parents[2] / "artifacts/model-studies/2026-11-10/body-channel"


def check_members(archive, manifest_name):
    names = archive.namelist()
    manifest = json.loads(archive.read(manifest_name))
    assert len(names) == len(set(names)) and set(names) == set(manifest) | {manifest_name}
    for name, expected in manifest.items():
        payload = archive.read(name)
        assert len(payload) == expected["bytes"] and sha256(payload).hexdigest() == expected["sha256"], name
    return len(manifest)


def main():
    publication = read(DATA / "publication.json")
    archive_bytes = (DATA / "body-channel-results.zip").read_bytes()
    assert len(archive_bytes) == publication["archive_bytes"]
    assert sha256(archive_bytes).hexdigest() == publication["archive_sha256"]
    assert sha256((DATA / "REPORT.md").read_bytes()).hexdigest() == publication["report_sha256"]
    with ZipFile(BytesIO(archive_bytes)) as archive:
        files = check_members(archive, "SHA256SUMS.json")
        root = ZipPath(archive)
        initial, crossed = root / "initial", root / "label-order"
        source_hash = sha256((root / "source/probe.py").read_bytes()).hexdigest()
        with ZipFile(BytesIO((initial / "gpu-results.zip").read_bytes())) as first, \
                ZipFile(BytesIO((crossed / "gpu-results.zip").read_bytes())) as followup:
            for folder, exported in ((initial, first), (crossed, followup)):
                files += check_members(exported, "archive-manifest.json")
                expected = read(folder / "export-receipt.json")
                assert expected["worker_returncode"] == 0
                assert expected["sha256"] == sha256((folder / "gpu-results.zip").read_bytes()).hexdigest()
                assert read(folder / "frozen/run-lock.json")["probe.py"] == source_hash
            first_raw, followup_raw = ZipPath(first) / "output", ZipPath(followup) / "output"
            first_score = score(initial / "frozen", first_raw)
            crossed_score = evaluate(crossed / "frozen", followup_raw, first_raw)
            assert first_score == read(initial / "scored.json")
            assert crossed_score == read(crossed / "scored.json")
        repeated = crossed_score["diagonal_reproduction"]
        assert len(repeated) == publication["repeat_prompt_sets"]
        assert all(row["exact_scores_and_patches_reproduced"] for row in repeated)
        social = run()
        assert social == read(root / "social-control.json")
        branches = 2 * len(social["paired_rollouts"])
        assert branches == publication["cpu_wording_branches"]
    print(json.dumps({"archive_members_verified": files, "both_scores_reproduced": True,
                      "exact_repeated_prompt_sets": len(repeated), "cpu_wording_branches_replayed": branches}))


if __name__ == "__main__":
    main()
