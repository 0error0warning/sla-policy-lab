"""Localize cross-assignment differences with zero-adapter and fixed-birth measurements."""

import argparse
import gzip
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.first_fatal.numerics import compare, distribution, frozen_bank, store
from experiments.health_baseline.analysis import read_replay


def run(tapes, replay, roots, inputs, output):
    banks = {}
    within = {}
    prior = {}
    older_errors = []
    by_birth = {"blank": [], "population": []}
    global_history = {}
    with ZipFile(tapes) as z:
        for f in sorted(replay.glob("s*/*/life.json")):
            m = json.loads(f.read_bytes())
            key = f"s{m['seed']}/{m['birth']}/events.jsonl.gz"
            ev = [json.loads(l) for l in gzip.decompress(z.read(key)).splitlines()]
            now = read_replay(f.parent)
            by_birth[m["birth"]].append(now[0]["source_global_max_error"])
            for e, n in zip(ev, now, strict=True):
                if m["birth"] == "blank":
                    store(prior, e["index"], np.array(e["probs"][0]), older_errors)
                phase = (
                    "at_birth"
                    if n["turn"] == 1
                    else "practice"
                    if n["practice"]
                    else "remaining"
                )
                global_history.setdefault(phase, []).append(
                    n["source_global_max_error"]
                )
    banks["parent"] = prior
    within["parent"] = max(older_errors)
    for name, root in zip(
        ("pilot", "replication_first_eight", "replication_last_eight"),
        roots,
        strict=True,
    ):
        banks[name], within[name] = frozen_bank(root)
    cases = json.loads((inputs / "witness-cases.json").read_bytes())
    with np.load(output / "restored-probabilities.npz") as z:
        fresh = z["arrays"]
    start = 0
    restored = {}
    errors = []
    for c in cases:
        for n, (ix, _h) in enumerate(c["checks"]):
            if c["kind"] == "live" and c["birth"] == "blank":
                store(restored, ix, fresh[start + n, 0], errors)
        start += len(c["checks"])
    assert start == len(fresh)
    banks["restored"] = restored
    within["restored"] = max(errors)
    pairs = {}
    keys = list(banks)
    for j, k in enumerate(keys):
        for second in keys[j + 1 :]:
            pairs[k + "__" + second] = compare(banks[k], banks[second])
    return {
        "blank_birth_within_assignment_max": within,
        "blank_birth_matched_text_across_assignments": pairs,
        "original_vs_fixed_tape_at_identical_birth": {
            b: distribution(v) for b, v in by_birth.items()
        },
        "old_vs_retrained_global_on_recorded_tape_by_phase": {
            k: distribution(v) for k, v in global_history.items()
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lives", nargs=3, type=Path, required=True)
    for k in ("tapes", "replay", "inputs", "restored", "result"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    write(a.result, run(a.tapes, a.replay, a.lives, a.inputs, a.restored))
