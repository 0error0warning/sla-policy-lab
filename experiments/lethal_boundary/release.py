"""Verify and unpack the released body-boundary experiment into a new directory."""

import argparse
import hashlib
import json
from pathlib import Path
from shutil import copyfile
from zipfile import ZipFile

from experiments.direct_policy.unpack import unpack as unpack_gpu
from experiments.internal_body.release import safe, verify_manifest


def unpack_checked(archive, root, input_package=False):
    with ZipFile(archive) as z:
        entries = z.namelist()
        if len(set(entries)) != len(entries):
            raise ValueError("Duplicate archive entry")
        meta = "run-lock.json" if input_package else "members.json"
        records = json.loads(z.read(meta))
        if input_package:
            records = records["files"]
        if set(entries) != (set(records) | {meta}):
            raise ValueError("Unexpected archive membership")
        for name, checksum in records.items():
            payload = z.read(name)
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError(name)
            target = safe(root, name)
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        if input_package:
            (root / meta).write_bytes(z.read(meta))


def unpack(art, dest):
    verify_manifest(art)
    dest.mkdir(parents=True, exist_ok=False)
    for src in sorted((art / "seeds").glob("*/*.zip")):
        dst = dest / src.parent.name / src.stem
        unpack_checked(src, dst)
    for f in (art / "common").iterdir():
        copyfile(f, dest / "core" / f.name)
    for stage in (art / "gpu").iterdir():
        rec = json.loads((stage / "receipt.json").read_bytes())
        archive = stage / "results.zip"
        if not archive.exists():
            archive = dest / (stage.name + "-transport.zip")
            with archive.open("xb") as joined:
                for part in rec["chunks"]:
                    raw = (stage / part["name"]).read_bytes()
                    if (
                        len(raw) != part["bytes"]
                        or hashlib.sha256(raw).hexdigest() != part["sha256"]
                    ):
                        raise ValueError(part["name"])
                    joined.write(raw)
        dst = dest / "gpu" / stage.name
        unpack_gpu(archive, stage / "receipt.json", dst)
        launch = json.loads((dst / "launch-receipt.json").read_bytes())
        if (
            hashlib.sha256((stage / "inputs.zip").read_bytes()).hexdigest()
            != launch["input_sha256"]
        ):
            raise ValueError("GPU input mismatch")
        unpack_checked(stage / "inputs.zip", dst / "inputs", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    a = parser.parse_args()
    if a.destination:
        unpack(a.artifact, a.destination)
    else:
        verify_manifest(a.artifact)
    print("Verified boundary release", flush=True)
