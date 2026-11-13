"""Verify parent evidence, parameter budgets, sampled healthy trajectories and exported tables."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.individual_life.release import unpack
from experiments.order_transfer.analysis import analyze
from experiments.order_transfer.inputs import SNAPSHOTS, SOURCE, build


def reproduce(artifact, parent, source, dest):
    restored = unpack(artifact, dest / "raw") / "gpu" / "ordmix"
    inputs = dest / "recovered-inputs"
    build(parent, source, inputs)
    matched = {}
    for f in sorted(inputs.iterdir()):
        matching = restored / "inputs" / f.name
        if f.suffix == ".npz":
            with np.load(f) as a, np.load(matching) as b:
                assert a.files == b.files
                for k in a.files:
                    np.testing.assert_array_equal(a[k], b[k])
        else:
            assert f.read_bytes() == matching.read_bytes(), f.name
        matched[f.name] = sha256(matching.read_bytes()).hexdigest()
    out = dest / "analysis"
    analyze(inputs, restored / "output", out)
    checked = {}
    for f in sorted(out.iterdir()):
        assert f.read_bytes() == (artifact / "analysis" / f.name).read_bytes(), f.name
        checked[f.name] = sha256(f.read_bytes()).hexdigest()
    result = {
        "verified_transports": True,
        "parent_inputs_recovered": matched,
        "byte_identical_analyses": checked,
    }
    write(dest / "proof.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--parent", type=Path, default=SNAPSHOTS)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--dest", type=Path, required=True)
    a = p.parse_args()
    print(reproduce(a.artifact, a.parent, a.source, a.dest))
