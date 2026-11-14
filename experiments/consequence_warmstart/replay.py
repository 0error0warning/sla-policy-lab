"""Hash-verify published CPU results and reconstruct analysis input, without GPU."""

import argparse
import json
from hashlib import sha256
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from experiments.direct_policy.protocol import write


def unpack(artifact, destination):
    manifest = json.loads((artifact / "MANIFEST.json").read_bytes())
    actual = {
        p.relative_to(artifact).as_posix() for p in artifact.rglob("*") if p.is_file()
    }
    if actual != set(manifest) | {"MANIFEST.json"}:
        raise ValueError("Artifact membership differs")
    for name, info in manifest.items():
        content = (artifact / name).read_bytes()
        if (
            len(content) != info["bytes"]
            or sha256(content).hexdigest() != info["sha256"]
        ):
            raise ValueError(f"Artifact digest differs: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    for archive in sorted((artifact / "seeds").glob("*.zip")):
        with ZipFile(archive) as z:
            for member in z.infolist():
                relative = PurePosixPath(member.filename)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or ":" in member.filename
                    or "\\" in member.filename
                ):
                    raise ValueError("Unsafe archive path")
                target = destination.joinpath(*relative.parts)
                if target.exists() or not target.resolve().is_relative_to(
                    destination.resolve()
                ):
                    raise ValueError("Duplicate or outside destination")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(member))
    cases = [json.loads(p.read_bytes()) for p in sorted(destination.glob("s*.json"))]
    if len(cases) != 32:
        raise ValueError("Incomplete case list")
    write(destination / "index.json", cases)
    return len(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print("verified files", unpack(args.artifact, args.out))
