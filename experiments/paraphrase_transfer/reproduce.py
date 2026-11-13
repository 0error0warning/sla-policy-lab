"""Recreate the signed input experience and re-audit measured outputs from the public archives."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.individual_life.release import unpack
from experiments.order_transfer.analysis import load
from experiments.paraphrase_transfer.analysis import analyze
from experiments.paraphrase_transfer.inputs import PARENT, build
from experiments.paraphrase_transfer.stability import compare


def reproduce(artifact, parent, dest):
    base = unpack(artifact, dest / "raw") / "gpu"
    raw = base / "phrmix"
    recovered = dest / "recovered-inputs"
    build(parent, recovered)
    comparisons = {}
    for f in sorted(recovered.iterdir()):
        original = raw / "inputs" / f.name
        if f.suffix == ".npz":
            with np.load(f) as a, np.load(original) as b:
                assert a.files == b.files
                for k in a.files:
                    np.testing.assert_array_equal(a[k], b[k])
        else:
            assert f.read_bytes() == original.read_bytes(), f.name
        comparisons[f.name] = sha256(original.read_bytes()).hexdigest()
    checked = {}
    runs = {}
    for name, output in (("", raw), ("repeat", base / "phrcheck")):
        out = dest / "analysis" / name
        for f in recovered.iterdir():
            assert (raw / "inputs" / f.name).read_bytes() == (
                output / "inputs" / f.name
            ).read_bytes(), f.name
        analyze(recovered, output / "output", out)
        runs[name] = load(out / "summary.json")
        for f in sorted(out.iterdir()):
            if f.is_file():
                expected = artifact / "analysis" / name / f.name
                assert f.read_bytes() == expected.read_bytes(), f.name
                checked[f.relative_to(dest / "analysis").as_posix()] = sha256(
                    f.read_bytes()
                ).hexdigest()
    st = dest / "analysis/stability.json"
    compare(
        recovered,
        raw / "output",
        base / "phrcheck/output",
        parent,
        st,
        runs[""],
        runs["repeat"],
    )
    assert st.read_bytes() == (artifact / "analysis/stability.json").read_bytes()
    checked["stability.json"] = sha256(st.read_bytes()).hexdigest()
    result = {
        "transport_verified": True,
        "recovered_parent_inputs": comparisons,
        "byte_identical_analyses": checked,
    }
    write(dest / "proof.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--parent", type=Path, default=PARENT)
    p.add_argument("--dest", type=Path, required=True)
    a = p.parse_args()
    print(reproduce(a.artifact, a.parent, a.dest))
