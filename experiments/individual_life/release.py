"""Verify and unpack raw private lineages, the exact Qwen launch bundle and its export."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from experiments.direct_policy.unpack import unpack as unpack_gpu
from experiments.internal_body.release import safe, verify_manifest
from experiments.lethal_boundary.release import unpack_checked


def join_parts(stage, destination, parts):
    """Reconstruct exactly the archive whose launch or export receipt is checked next."""
    merged = destination.open("xb")
    try:
        with merged:
            for part in parts:
                data = safe(stage, part["name"]).read_bytes()
                if (
                    len(data) != part["bytes"]
                    or sha256(data).hexdigest() != part["sha256"]
                ):
                    raise ValueError(part["name"])
                merged.write(data)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def unpack(artifact, dest):
    verify_manifest(artifact)
    dest.mkdir(parents=True, exist_ok=False)
    for bundle in sorted((artifact / "seeds").glob("*/*/s*.zip")):
        wave, sense, filename = bundle.relative_to(artifact / "seeds").parts
        unpack_checked(bundle, dest / wave / sense / Path(filename).stem)
    for stage in sorted((artifact / "gpu").iterdir()):
        receipt = stage / "receipt.json"
        archive = stage / "results.zip"
        if not archive.exists():
            archive = dest / (stage.name + "-transport.zip")
            join_parts(stage, archive, json.loads(receipt.read_bytes())["chunks"])
        full = dest / "gpu" / stage.name
        unpack_gpu(archive, receipt, full)
        launch = json.loads((full / "launch-receipt.json").read_bytes())
        inputs = stage / "inputs.zip"
        if not inputs.exists():
            parts = json.loads((stage / "input-transport.json").read_bytes())
            if parts["sha256"] != launch["input_sha256"]:
                raise ValueError(f"Input transport mismatch for {stage.name}")
            inputs = dest / (stage.name + "-inputs-transport.zip")
            join_parts(stage, inputs, parts["chunks"])
            if inputs.stat().st_size != parts["bytes"]:
                raise ValueError(f"Input transport size mismatch for {stage.name}")
        if sha256(inputs.read_bytes()).hexdigest() != launch["input_sha256"]:
            raise ValueError(f"Input hash mismatch for {stage.name}")
        unpack_checked(inputs, full / "inputs", True)
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--destination", type=Path)
    a = p.parse_args()
    if a.destination:
        unpack(a.artifact, a.destination)
    else:
        verify_manifest(a.artifact)
    print("Verified individual-life release", flush=True)
