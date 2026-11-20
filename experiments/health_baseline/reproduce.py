"""Rebuild the release analyses from checked raw exports. No GPU or retraining is involved."""

import argparse
import ast
import json
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

from experiments.direct_policy.protocol import write
from experiments.health_baseline.analysis import lives_summary, replay_summary
from experiments.health_baseline.audit import audit_lives, audit_replay
from experiments.health_baseline.device_check import run as device_check
from experiments.health_baseline.interruption import compare as interruption_check
from experiments.health_baseline.numerics import repeats, restoration
from experiments.health_baseline.tapes import build as recover_tapes
from experiments.health_baseline.witness import build as recover_weights
from experiments.individual_life.release import unpack


def training_parity(parent, full):
    stages = ("hbreplay", "hbpilot", "hbrepa", "hbrepb", "hbrepeat", "hbverify")
    locks = {
        s: json.loads((full / s / "inputs/run-lock.json").read_bytes())["files"]
        for s in stages
    }
    reused = (
        "architecture.npz",
        "birth-source.npz",
        "expected-weights.json",
        "requirements-gpu.txt",
        "experiments/direct_policy/protocol.py",
        "experiments/social_closed_loop/core.py",
        "experiments/individual_life/model.py",
        "experiments/individual_life/design.py",
        "experiments/first_fatal/design.py",
        "experiments/consequence_policy/gpu.py",
        "experiments/consequence_policy/study.py",
        "experiments/internal_body/capture.py",
        "experiments/internal_body/design.py",
        "experiments/internal_body/suffix.py",
        "experiments/lethal_boundary/states.py",
    )
    functions = {
        "experiments/first_fatal/study.py": ("arrivals", "bits", "hashed", "probe"),
        "experiments/first_fatal/engine.py": ("ordinary", "tail", "paired", "prepare"),
    }

    def expressions(src, names):
        nodes = {
            n.name: ast.dump(n)
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.FunctionDef) and n.name in names
        }
        assert set(nodes) == set(names)
        return nodes

    for prior in ("ffpilot", "ffconfirm"):
        with ZipFile(parent / "gpu" / prior / "inputs.zip") as z:
            old = json.loads(z.read("run-lock.json"))["files"]
            for k in reused:
                assert all(locks[stage][k] == old[k] for stage in stages), k
            for file, names in functions.items():
                expected = expressions(z.read(file), names)
                for stage in stages:
                    assert (
                        expressions(
                            (full / stage / "inputs" / file).read_bytes(), names
                        )
                        == expected
                    ), file
    for name, waves in (
        ("gpu", stages),
        ("design", stages),
        ("replay", ("hbreplay", "hbrepeat")),
        ("live", stages[1:4]),
    ):
        k = "experiments/health_baseline/" + name + ".py"
        assert len({locks[stage][k] for stage in waves}) == 1
    return {
        "identical_parent_files": list(reused),
        "matching_parent_function_expressions": functions,
        "matched_primary_repeat_and_free_training_sources": True,
    }


def run(artifact, parent, dest):
    full = unpack(artifact, dest / "restored") / "gpu"
    roots = [full / s / "output/lives" for s in ("hbpilot", "hbrepa", "hbrepb")]
    source = full / "hbreplay/inputs"
    original = full / "hbreplay/output/replays"
    repeated = full / "hbrepeat/output/replays"
    last = full / "hbverify"
    analysis = dest / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    replay_summary(original, analysis)
    lives_summary(roots, analysis)
    write(
        analysis / "audit.json",
        {
            "live": audit_lives(roots, source / "birth-source.npz"),
            "replay": audit_replay(original, source / "tapes.zip"),
            "repeated_replay": audit_replay(
                repeated, full / "hbrepeat/inputs/tapes.zip"
            ),
        },
    )
    write(
        analysis / "numerics.json",
        {
            "replay_repeat": repeats(original, repeated),
            "restored": restoration(last / "inputs", last / "output"),
        },
    )
    write(
        analysis / "device-check.json",
        device_check(
            source / "tapes.zip", original, roots, last / "inputs", last / "output"
        ),
    )
    write(
        analysis / "interrupted-retry.json",
        interruption_check(
            artifact, dict(zip(("hbpilot", "hbrepa", "hbrepb"), roots, strict=True))
        ),
    )
    extra = dest / "reconstructed-inputs"
    recover_weights(roots, original, extra)
    recover_tapes(parent, extra / "tapes.zip")
    verified = {}

    def same(a, b, key):
        x = a.read_bytes()
        if x != b.read_bytes():
            raise ValueError("Different reconstructed bytes in " + key)
        verified[key] = sha256(x).hexdigest()

    for f in sorted(analysis.iterdir()):
        same(f, artifact / "analysis" / f.name, "analysis/" + f.name)
    for f in sorted(extra.iterdir()):
        same(
            f,
            (source if f.name == "tapes.zip" else last / "inputs") / f.name,
            "inputs/" + f.name,
        )
    for mode in ("hbreplay", "hbpilot", "hbrepa", "hbrepb", "hbrepeat", "hbverify"):
        assert json.loads((full / mode / "output/receipt.json").read_bytes())[
            "backbone_restored"
        ]
    proof = {
        "source_parity": training_parity(parent, full),
        "analysis_python": sys.version.split()[0],
        "analysis_numpy": version("numpy"),
        "raw_transports_member_hashes_and_launch_inputs_verified": True,
        "stages": 6,
        "source_lives_replayed": 48,
        "fresh_autonomous_lives": sum(
            len(list(t.glob("s*/*/*/life.json"))) for t in roots
        ),
        "repeat_adds_independent_seeds": 0,
        "identical_rebuilt_outputs": verified,
    }
    write(dest / "reproduction.json", proof)
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("artifact", "parent", "destination"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.artifact, a.parent, a.destination), indent=2))
