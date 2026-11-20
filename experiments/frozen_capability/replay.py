"""Verify public result archives and reproduce CPU scoring without model weights."""

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from .analysis import run
from .design import serial


def unpack(archive, destination):
    with ZipFile(archive) as z:
        manifest = json.loads(z.read("archive-manifest.json"))
        assert len(z.namelist()) == len(set(z.namelist()))
        assert set(z.namelist()) == set(manifest) | {"archive-manifest.json"}
        assert z.testzip() is None
        for name, expected in manifest.items():
            relative = PurePosixPath(name)
            assert (
                not relative.is_absolute()
                and ".." not in relative.parts
                and "\\" not in name
            )
            path = destination.joinpath(*relative.parts).resolve()
            assert path.is_relative_to(destination.resolve())
            data = z.read(name)
            assert len(data) == expected["bytes"]
            assert hashlib.sha256(data).hexdigest() == expected["sha256"]
            if path.exists():
                assert path.read_bytes() == data, str(path)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)


def replay(manifest_path, metadata, work, out):
    specification = json.loads(manifest_path.read_bytes())
    work.mkdir(parents=True, exist_ok=True)
    sources = []
    for segment in specification["segments"]:
        assert re.fullmatch(r"[a-zA-Z0-9_-]+", segment["name"])
        archive = (manifest_path.parent / segment["archive"]).resolve()
        inputs = (manifest_path.parent / segment["inputs"]).resolve()
        assert (
            hashlib.sha256(archive.read_bytes()).hexdigest()
            == segment["archive_sha256"]
        )
        assert (
            hashlib.sha256(inputs.read_bytes()).hexdigest() == segment["input_sha256"]
        )
        destination = work / segment["name"]
        unpack(archive, destination)
        sources.append(
            {
                "name": segment["name"],
                "inputs": str(inputs),
                "source": str((destination / "output").resolve()),
                "partial": segment["partial"],
            }
        )
    execution_manifest = work / "sources.json"
    execution_manifest.write_bytes(serial({"segments": sources}))
    original_inputs = Path(sources[0]["inputs"])
    result = run(None, original_inputs, metadata, out, execution_manifest)
    # Preserve portable provenance in the scored data. Actual token/input checks
    # ran against extracted byte-identical files; full paths are not published.
    import gzip

    parsed = out / "parsed-cases.jsonl.gz"
    with gzip.open(parsed, "rt", encoding="utf8") as file:
        rows = [json.loads(line) for line in file]
    for row in rows:
        origin = row["provenance"]
        if origin:
            path = Path(origin["raw_path"])
            origin["raw_path"] = (
                f"{origin['segment']}/output/{row['model']}/records/{path.name}"
            )
    with gzip.open(parsed, "wt", encoding="utf8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for flag in ("manifest", "metadata", "work", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(replay(a.manifest, a.metadata, a.work, a.out), indent=2))
