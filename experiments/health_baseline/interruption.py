"""Compare any recovered full-seed backups with the official retry without adding individuals."""

import argparse
import gzip
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.first_fatal.analysis import records


def endpoint(events):
    focal = next((e for e in events if e["first_fatal"]), None)
    healthy = [e for e in events if e["health"] >= 90]
    return {
        "turns": len(events),
        "first_turn": focal["turn"] if focal else None,
        "first_action": focal["word"] if focal else None,
        "first_avoided": int(focal is not None and focal["after"] > 0),
        "fatal_probability": float(sum(focal["probs"][1][2:])) if focal else None,
        "healthy_rate": float(
            np.mean([e["action"] >= 2 and e["after"] > 0 for e in healthy])
        ),
    }


def compare(artifact, roots):
    comparisons = []
    assignments = []
    for d in sorted((artifact / "interruptions").glob("interrupted-*")):
        meta = json.loads((d / "receipt.json").read_bytes())
        stage = meta["stage"]
        assert not meta["included_in_final_analyses"]
        assignments.append(
            {
                "stage": stage,
                "attempt": meta["attempt"],
                "verified_seed_backups": len(meta["verified_partial_seed_backups"]),
            }
        )
        for name in meta["verified_partial_seed_backups"]:
            with ZipFile(d / name) as z:
                seed = Path(name).stem
                for m in sorted(f for f in z.namelist() if f.endswith("/life.json")):
                    location = m.removesuffix("/life.json")
                    e = [
                        json.loads(v)
                        for v in gzip.decompress(
                            z.read(location + "/events.jsonl.gz")
                        ).splitlines()
                    ]
                    final = records(roots[stage] / seed / location)
                    assert e[-1]["after"] == 0 and final[-1]["after"] == 0
                    mismatch = next(
                        (
                            i + 1
                            for i, (a, b) in enumerate(zip(e, final))
                            if a["action"] != b["action"]
                        ),
                        None,
                    )
                    a, b = endpoint(e), endpoint(final)
                    comparisons.append(
                        {
                            "stage": stage,
                            "attempt": meta["attempt"],
                            "seed": int(seed[1:]),
                            "start_and_arm": location,
                            "first_action_divergence_turn": mismatch,
                            "excluded_attempt": a,
                            "included_retry": b,
                            "changed_fatal_survival": a["first_avoided"]
                            != b["first_avoided"],
                        }
                    )
    return {
        "excluded_partial_assignments": assignments,
        "complete_lives_in_backup": len(comparisons),
        "new_independent_seeds": 0,
        "changed_first_survivals": sum(
            v["changed_fatal_survival"] for v in comparisons
        ),
        "raw_comparisons": comparisons,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--lives", nargs=3, type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    write(
        a.output,
        compare(
            a.artifact, dict(zip(("hbpilot", "hbrepa", "hbrepb"), a.lives, strict=True))
        ),
    )
