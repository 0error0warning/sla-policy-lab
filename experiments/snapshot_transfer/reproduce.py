"""Verify the GPU transport and regenerate the 48 input snapshots and CPU analyses."""

import argparse
import json
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
    norm_errors = []
    for f in sorted(prepared.iterdir()):
        other = raw / "inputs" / f.name
        if f.suffix == ".npz":
            with np.load(f) as a, np.load(other) as b:
                assert a.files == b.files
                for k in a.files:
                    np.testing.assert_array_equal(a[k], b[k])
        else:
            recovered, launched = (json.loads(path.read_bytes()) for path in (f, other))
            for a, b in zip(recovered["cases"], launched["cases"], strict=True):
                # Recomputing a norm of the identical FP32 arrays can change the
                # last FP64 bits across BLAS reduction orders. No weight uses it.
                norm_errors.append(
                    abs(a.pop("parameter_step") - b.pop("parameter_step"))
                )
            assert len(norm_errors) == 48 and max(norm_errors) < 1e-12
            assert recovered == launched
        checks["inputs/" + f.name] = sha256(other.read_bytes()).hexdigest()
    out = dest / "analysis"
    analyze(raw / "inputs", raw / "output", out)
    for f in sorted(out.iterdir()):
        other = artifact / "analysis" / f.name
        assert f.read_bytes() == other.read_bytes(), f.name
        checks["analysis/" + f.name] = sha256(f.read_bytes()).hexdigest()
    proof = {
        "recovered_snapshot_inputs": 3,
        "recomputed_norm_max_error": max(norm_errors),
        "all_other_case_fields_and_weight_words_exact": True,
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
