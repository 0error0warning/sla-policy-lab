"""Rebuild the release analyses from checked raw exports. No GPU or retraining is involved."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from experiments.direct_policy.protocol import write
from experiments.health_baseline.analysis import lives_summary, replay_summary
from experiments.health_baseline.audit import audit_lives, audit_replay
from experiments.health_baseline.device_check import run as device_check
from experiments.health_baseline.numerics import repeats, restoration
from experiments.health_baseline.tapes import build as recover_tapes
from experiments.health_baseline.witness import build as recover_weights
from experiments.individual_life.release import unpack


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
