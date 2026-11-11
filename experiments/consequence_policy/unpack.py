"""Verify the published study and reconstruct the runnable GPU/CPU directory layout."""

import argparse
import json
from hashlib import sha256
from pathlib import Path, PurePosixPath
from shutil import copyfile
from zipfile import ZipFile

from experiments.direct_policy.unpack import unpack as unpack_gpu


def unpack(artifact, destination):
    manifest = json.loads((artifact / "MANIFEST.json").read_bytes())
    actual = {
        p.relative_to(artifact).as_posix() for p in artifact.rglob("*") if p.is_file()
    }
    if actual != set(manifest) | {"MANIFEST.json"}:
        raise ValueError("Manifest membership differs from delivered files")
    for name, expected in manifest.items():
        data = (artifact / name).read_bytes()
        if (
            len(data) != expected["bytes"]
            or sha256(data).hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"Invalid digest for {name}")
    destination.mkdir(parents=True, exist_ok=False)
    for mode in ("capture", "verify", "numerics", "ordering"):
        unpack_gpu(
            artifact / "gpu" / f"{mode}-results.zip",
            artifact / "gpu" / f"{mode}-receipt.json",
            destination / mode,
        )
    results = destination / "results"
    results.mkdir()
    for f in (artifact / "summaries").rglob("*"):
        if f.is_file():
            target = results / f.relative_to(artifact / "summaries")
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(f, target)
    for bundle in sorted((artifact / "seeds").glob("*.zip")):
        with ZipFile(bundle) as z:
            for info in z.infolist():
                rel = PurePosixPath(info.filename)
                if (
                    rel.is_absolute()
                    or ".." in rel.parts
                    or "\\" in info.filename
                    or ":" in info.filename
                ):
                    raise ValueError(f"Unexpected archive path {info.filename!r}")
                target = results.joinpath(*rel.parts)
                if (
                    not target.resolve().is_relative_to(results.resolve())
                    or target.exists()
                ):
                    raise ValueError(f"Invalid or duplicate target {rel}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(info))
    return len(manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    args = p.parse_args()
    print("Verified files:", unpack(args.artifact, args.destination))
