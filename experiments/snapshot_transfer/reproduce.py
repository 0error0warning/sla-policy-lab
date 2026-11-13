"""Verify the GPU transport and regenerate the 48 input snapshots and CPU analyses."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.individual_life.release import unpack
from experiments.snapshot_transfer.analysis import analyze
from experiments.snapshot_transfer.inputs import PARENT, build


def reproduce(artifact, parent, dest):
    raw = unpack(artifact, dest / "raw") / "gpu/stmap"
    prepared = dest / "recovered-inputs"
    build(parent, prepared)
    checks = {}
    for f in sorted(prepared.iterdir()):
        other = raw / "inputs" / f.name
        if f.suffix == ".npz":
            with np.load(f) as a, np.load(other) as b:
                assert a.files == b.files
                for k in a.files:
                    np.testing.assert_array_equal(a[k], b[k])
        else:
            assert f.read_bytes() == other.read_bytes()
        checks["inputs/" + f.name] = sha256(other.read_bytes()).hexdigest()
    out = dest / "analysis"
    analyze(prepared, raw / "output", out)
    for f in sorted(out.iterdir()):
        other = artifact / "analysis" / f.name
        assert f.read_bytes() == other.read_bytes(), f.name
        checks["analysis/" + f.name] = sha256(f.read_bytes()).hexdigest()
    proof = {
        "recovered_snapshot_inputs": 3,
        "byte_identical_analyses": len(list(out.iterdir())),
        "sha256": checks,
    }
    write(dest / "proof.json", proof)
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--dest", type=Path, required=True)
    a = parser.parse_args()
    print(reproduce(a.artifact, a.parent, a.dest))
