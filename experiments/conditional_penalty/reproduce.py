"""Recreate the measured schedules, parameter snapshots and statistics from public archives."""

import argparse
import json
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from experiments.conditional_penalty.analysis import (
    interval,
    live_summary,
    replay_summary,
)
from experiments.conditional_penalty.audit import autonomous, matched_histories
from experiments.conditional_penalty.inputs import (
    SOURCE_RISK_ANSWERS,
    restore_inputs,
    tapes,
)
from experiments.damage_penalty.numerics import reinstalls, retraining
from experiments.damage_penalty.reproduce import parity
from experiments.direct_policy.protocol import write
from experiments.health_baseline.interruption import compare
from experiments.individual_life.release import unpack

STAGES = ("gpreplay", "gppilot", "gprep", "gpcheck")


def technical(primary, fresh, inputs):
    restored = reinstalls(
        inputs,
        fresh,
        focal_arm="gate_true",
        estimate=interval,
        summary_key="first_fatal_contrasts",
        contrasts=(
            ("birth", 1, 0),
            ("wrong", 1, 2),
            ("one", 1, 3),
            ("two", 1, 4),
            ("two_minus_birth", 4, 0),
        ),
    )
    repeated = {
        k: retraining(
            primary / k / "replays", fresh / k / "replays", SOURCE_RISK_ANSWERS
        )
        for k in ("gated", "constant")
    }
    return {"fixed_tapes": repeated, "reinstalls": restored}


def reproduce(artifact, parent, dest):
    raw = unpack(artifact, dest / "restored") / "gpu"
    paths = {k: raw / k for k in STAGES}
    primary = paths["gpreplay"] / "output"
    fresh = paths["gpcheck"] / "output"
    roots = [paths[k] / "output/lives" for k in ("gppilot", "gprep")]
    out = dest / "analysis"
    out.mkdir(exist_ok=True, parents=True)
    for phase, root in (("primary", primary), ("repeat", fresh)):
        replay_summary(root, out / phase)
    live_summary(roots, out)
    write(
        out / "audit.json",
        {
            "primary": matched_histories(
                primary, paths["gpreplay"] / "inputs/tapes.zip"
            ),
            "repeat": matched_histories(fresh, paths["gpcheck"] / "inputs/tapes.zip"),
            "free": autonomous(roots, paths["gppilot"] / "inputs/birth-source.npz"),
        },
    )
    write(out / "numerics.json", technical(primary, fresh, paths["gpcheck"] / "inputs"))
    write(
        out / "interrupted.json",
        compare(artifact, dict(zip(("gppilot", "gprep"), roots, strict=True))),
    )
    recovered = dest / "recovered-inputs"
    recovered.mkdir(exist_ok=True)
    tapes(parent, recovered / "tapes.zip")
    restore_inputs(roots, primary, recovered)
    checks = {}

    def identical(a, b, key):
        data = a.read_bytes()
        if data != b.read_bytes():
            raise ValueError("Nonidentical rebuilt bytes at " + key)
        checks[key] = sha256(data).hexdigest()

    for f in sorted(out.rglob("*")):
        if f.is_file():
            key = f.relative_to(out).as_posix()
            identical(f, artifact / "analysis" / key, "analysis/" + key)
    for f in sorted(recovered.iterdir()):
        identical(f, paths["gpcheck"] / "inputs" / f.name, "inputs/" + f.name)
    identical(
        recovered / "tapes.zip",
        paths["gpreplay"] / "inputs/tapes.zip",
        "primary/tapes.zip",
    )
    core = (
        "experiments/conditional_penalty/design.py",
        "experiments/conditional_penalty/gpu.py",
        "experiments/damage_penalty/gpu.py",
        "experiments/health_baseline/design.py",
        "experiments/health_baseline/live.py",
        "experiments/health_baseline/replay.py",
    )
    result = {
        "source_parity": parity(paths, parent, ("dppilot", "dprep"), core),
        "byte_identical": checks,
        "python": sys.version.split()[0],
        "numpy": version("numpy"),
        "new_seeds": 24,
        "free_lives": 336,
        "source_lives": 48,
        "additional_seeds_from_technical_replay": 0,
        "verified_all_transport_and_member_hashes": True,
    }
    write(dest / "proof.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("artifact", "parent", "destination"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(reproduce(a.artifact, a.parent, a.destination), indent=2))
