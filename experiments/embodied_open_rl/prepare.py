"""Prepare an isolated, hash-locked GPU bundle (no GPU or credential access)."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from .protocol import CONFIG, MODEL, REVISION, write


def prepare(destination, manifest_path):
    source = Path(__file__).resolve().parent
    repo = source.parents[1]
    expected = json.loads(Path(manifest_path).read_bytes())
    assert expected["id"] == MODEL and expected["revision"] == REVISION

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)

    pkg_dir = destination / "experiments" / "embodied_open_rl"
    pkg_dir.mkdir(parents=True, exist_ok=False)
    (destination / "experiments" / "__init__.py").write_text("", encoding="utf-8")

    for name in ("__init__.py", "protocol.py", "gpu.py"):
        shutil.copyfile(source / name, pkg_dir / name)

    shutil.copyfile(
        repo / "experiments" / "social_closed_loop" / "requirements-gpu.txt",
        destination / "requirements-gpu.txt",
    )
    write(destination / "protocol.json", CONFIG)
    write(destination / "expected-weights.json", expected)

    files = {}
    for p in sorted(destination.rglob("*")):
        if p.is_file():
            files[p.relative_to(destination).as_posix()] = sha256(p.read_bytes()).hexdigest()

    write(
        destination / "run-lock.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "files": files,
        },
    )

    zip_path = destination / "inputs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(destination.rglob("*")):
            if p.is_file() and p.name != "inputs.zip":
                zf.write(p, p.relative_to(destination).as_posix())

    print(
        json.dumps(
            {
                "prepared": str(destination.resolve()),
                "inputs_zip": str(zip_path.resolve()),
                "inputs_sha256": sha256(zip_path.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("destination", type=Path)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/model-studies/2026-11-10/frozen-v1/raw/model_0/weights-manifest.json"
        ),
    )
    a = p.parse_args()
    prepare(a.destination, a.manifest)
