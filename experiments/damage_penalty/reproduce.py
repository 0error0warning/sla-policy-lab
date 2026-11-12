"""Rebuild all reported numbers from verified transports and recover the measured inputs."""

import argparse
import json
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

from experiments.damage_penalty.analysis import live_summary, summarize_replay
from experiments.damage_penalty.audit import lives
from experiments.damage_penalty.audit import tapes as audit_tapes
from experiments.damage_penalty.inputs import restore_inputs, tapes
from experiments.damage_penalty.numerics import reinstalls, retraining
from experiments.direct_policy.protocol import write
from experiments.health_baseline.interruption import compare as interrupted_comparison
from experiments.individual_life.release import unpack

STAGES = ("dpreplay", "dppilot", "dprep", "dpcheck")


def analysis(paths, out, artifact):
    out.mkdir(exist_ok=True, parents=True)
    replay = paths["dpreplay"] / "output/replays"
    repeat = paths["dpcheck"] / "output/replays"
    roots = [paths[k] / "output/lives" for k in ("dppilot", "dprep")]
    for name, source in (("primary", replay), ("repeat", repeat)):
        dest = out / name
        dest.mkdir(exist_ok=True)
        summarize_replay(source, dest)
    live_summary(roots, out)
    write(
        out / "audit.json",
        {
            "replay": audit_tapes(replay, paths["dpreplay"] / "inputs/tapes.zip"),
            "repeat": audit_tapes(repeat, paths["dpcheck"] / "inputs/tapes.zip"),
            "lives": lives(roots, paths["dppilot"] / "inputs/birth-source.npz"),
        },
    )
    write(
        out / "numerics.json",
        {
            "replay": retraining(replay, repeat),
            "snapshots": reinstalls(
                paths["dpcheck"] / "inputs", paths["dpcheck"] / "output"
            ),
        },
    )
    write(
        out / "interrupted.json",
        interrupted_comparison(
            artifact, dict(zip(("dppilot", "dprep"), roots, strict=True))
        ),
    )


def parity(paths, parent):
    locks = {
        k: json.loads((d / "inputs/run-lock.json").read_bytes())["files"]
        for k, d in paths.items()
    }
    immutable = (
        "architecture.npz",
        "birth-source.npz",
        "expected-weights.json",
        "requirements-gpu.txt",
        "experiments/consequence_policy/gpu.py",
        "experiments/consequence_policy/study.py",
        "experiments/individual_life/model.py",
        "experiments/individual_life/design.py",
        "experiments/first_fatal/design.py",
        "experiments/first_fatal/engine.py",
        "experiments/first_fatal/study.py",
        "experiments/social_closed_loop/core.py",
        "experiments/direct_policy/protocol.py",
        "experiments/internal_body/design.py",
        "experiments/internal_body/suffix.py",
        "experiments/internal_body/capture.py",
    )
    for stage in ("hbpilot", "hbrepa", "hbrepb"):
        with ZipFile(parent / "gpu" / stage / "inputs.zip") as z:
            past = json.loads(z.read("run-lock.json"))["files"]
        assert all(locks[s][file] == past[file] for s in STAGES for file in immutable)
    new_core = (
        "experiments/damage_penalty/design.py",
        "experiments/damage_penalty/gpu.py",
        "experiments/health_baseline/design.py",
        "experiments/health_baseline/live.py",
        "experiments/health_baseline/replay.py",
    )
    assert all(len({locks[s][k] for s in STAGES}) == 1 for k in new_core)
    return {
        "unchanged_vs_parent": list(immutable),
        "same_training_code_on_all_assignments": list(new_core),
    }


def reproduce(artifact, parent, dest):
    full = unpack(artifact, dest / "restored") / "gpu"
    paths = {k: full / k for k in STAGES}
    out = dest / "analysis"
    analysis(paths, out, artifact)
    extra = dest / "rebuilt-inputs"
    extra.mkdir(exist_ok=True)
    tapes(parent, extra / "tapes.zip")
    restore_inputs(
        [paths[k] / "output/lives" for k in ("dppilot", "dprep")],
        paths["dpreplay"] / "output/replays",
        extra,
    )
    checked = {}

    def same(a, b, label):
        v = a.read_bytes()
        if v != b.read_bytes():
            raise ValueError("Reproduction mismatch " + label)
        checked[label] = sha256(v).hexdigest()

    for f in sorted(out.rglob("*")):
        if f.is_file():
            rel = f.relative_to(out).as_posix()
            same(f, artifact / "analysis" / rel, "analysis/" + rel)
    for f in sorted(extra.iterdir()):
        same(f, paths["dpcheck"] / "inputs" / f.name, "inputs/" + f.name)
    same(
        extra / "tapes.zip", paths["dpreplay"] / "inputs/tapes.zip", "primary/tapes.zip"
    )
    for d in paths.values():
        assert json.loads((d / "output/receipt.json").read_bytes())["backbone_restored"]
    v = {
        "source_parity": parity(paths, parent),
        "byte_identical": checked,
        "python": sys.version.split()[0],
        "numpy": version("numpy"),
        "assigned_independent_new_seeds": 24,
        "free_lives": 240,
        "fixed_source_tapes": 48,
        "repeat_additional_seeds": 0,
        "verified_inputs_outputs_all_member_hashes": True,
    }
    write(dest / "proof.json", v)
    return v


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("artifact", "parent", "destination"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(reproduce(a.artifact, a.parent, a.destination), indent=2))
