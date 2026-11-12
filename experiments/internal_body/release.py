"""Verify and unpack the published experiments, including exact individual parameters."""

import argparse
import io
import json
from hashlib import sha256
from pathlib import Path, PurePosixPath
from shutil import copyfile
from zipfile import ZipFile

from experiments.direct_policy.unpack import unpack as unpack_gpu
from experiments.internal_body.checkpoints import restore


def safe(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name:
        raise ValueError(f"Invalid member {name}")
    target = root.joinpath(*rel.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(name)
    return target


def verify_manifest(artifact):
    m = json.loads((artifact / "MANIFEST.json").read_bytes())
    files = {
        p.relative_to(artifact).as_posix() for p in artifact.rglob("*") if p.is_file()
    }
    if files != set(m) | {"MANIFEST.json"}:
        raise ValueError("Manifest membership mismatch")
    for name, record in m.items():
        path = safe(artifact, name)
        with path.open("rb") as stream:
            h = sha256()
            for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(data)
        if path.stat().st_size != record["bytes"] or h.hexdigest() != record["sha256"]:
            raise ValueError(name)
    return len(m)


def unpack(artifact, dest):
    n = verify_manifest(artifact)
    dest.mkdir(parents=True, exist_ok=False)
    for series in (artifact / "seeds").iterdir():
        root = dest / series.name
        root.mkdir()
        for bundle in sorted(series.glob("*.zip")):
            with ZipFile(bundle) as z:
                records = json.loads(z.read("members.json"))
                assert len(z.namelist()) == len(set(z.namelist())) and set(
                    z.namelist()
                ) == set(records) | {"members.json"}
                for member, info in records.items():
                    path = safe(root, info["path"])
                    if path.exists():
                        raise ValueError(f"Duplicate result {path}")
                    data = z.read(member)
                    if sha256(data).hexdigest() != info["sha256"]:
                        raise ValueError(member)
                    path.parent.mkdir(exist_ok=True, parents=True)
                    if info.get("encoding") == "lossless-checkpoint-v1":
                        restore(io.BytesIO(data), path)
                    else:
                        path.write_bytes(data)
    for path in (artifact / "summaries").rglob("*"):
        if path.is_file():
            target = safe(dest, path.relative_to(artifact / "summaries").as_posix())
            if target.exists():
                raise ValueError(target)
            target.parent.mkdir(exist_ok=True, parents=True)
            copyfile(path, target)
    for stage in (artifact / "gpu").iterdir():
        if not stage.is_dir() or not (stage / "receipt.json").exists():
            continue
        receipt = json.loads((stage / "receipt.json").read_bytes())
        archive = stage / "results.zip"
        if not archive.is_file():
            archive = dest / (stage.name + "-verified-transport.zip")
            with archive.open("xb") as f:
                for j in range(len(receipt["chunks"])):
                    f.write((stage / f"part-{j:03}.bin").read_bytes())
        unpack_gpu(
            archive,
            stage / "receipt.json",
            dest / "gpu" / stage.name,
            allow_failed=True,
        )
    return n


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", required=True, type=Path)
    p.add_argument("--destination", type=Path)
    a = p.parse_args()
    n = (
        unpack(a.artifact, a.destination)
        if a.destination
        else verify_manifest(a.artifact)
    )
    print(f"Verified {n} published files")
