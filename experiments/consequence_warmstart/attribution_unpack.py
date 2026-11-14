"""Verify the attribution result bundle and extract into a new directory."""

import argparse
import json
from hashlib import sha256
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def unpack(artifact, destination):
    manifest = json.loads((artifact / "MANIFEST.json").read_bytes())
    actual = {
        p.relative_to(artifact).as_posix() for p in artifact.rglob("*") if p.is_file()
    }
    if actual != set(manifest) | {"MANIFEST.json"}:
        raise ValueError("Public artifact membership mismatch")
    for name, info in manifest.items():
        value = (artifact / name).read_bytes()
        if len(value) != info["bytes"] or sha256(value).hexdigest() != info["sha256"]:
            raise ValueError(f"Public digest mismatch: {name}")
    members = json.loads((artifact / "result-members.json").read_bytes())
    destination.mkdir(parents=True, exist_ok=False)
    with ZipFile(artifact / "results.zip") as z:
        if set(z.namelist()) != set(members) or len(z.namelist()) != len(members):
            raise ValueError("Result archive membership mismatch")
        for name, record in members.items():
            relative = PurePosixPath(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in name
                or ":" in name
            ):
                raise ValueError("Unsafe member")
            target = destination.joinpath(*relative.parts)
            if (
                not target.resolve().is_relative_to(destination.resolve())
                or target.exists()
            ):
                raise ValueError("Unsafe or duplicate target")
            value = z.read(name)
            if (
                len(value) != record["bytes"]
                or sha256(value).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"Member digest mismatch: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
    return len(members)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print("verified members", unpack(args.artifact, args.out))
