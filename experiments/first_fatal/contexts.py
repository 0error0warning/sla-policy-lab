"""Post-outcome decomposition of the actual versus balanced-panel context at identical risk."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.first_fatal.analysis import describe, records
from experiments.first_fatal.design import BIRTHS
from experiments.individual_life.design import rows


def run(roots, out):
    items = rows()
    cases = []
    for d in sorted(
        f.parent for root in roots for f in root.glob("s*/*/true1/life.json")
    ):
        m = json.loads((d / "life.json").read_bytes())
        e = records(d)[m["first_fatal"] - 1]
        assert e["first_fatal"]
        with np.load(d / "predecision.npz") as z:
            q = z["panel"][0, :, :5, 2:].astype(float).sum(-1)
            ix = z["indices"]
            same = z["same_request"][0, :5, 2:].astype(float).sum(-1)
        source = items[e["index"]]
        social = np.array(
            [
                items[i]["partner"] == source["partner"]
                and items[i]["frame"] == source["frame"]
                for i in ix
            ]
        )
        matched = social & np.array(
            [items[i]["layout"] == source["layout"] for i in ix]
        )
        assert matched.sum() == 1 and social.sum() == 24
        alternatives = [same, q[matched][0], q[social].mean(0), q.mean(0)]
        v = {"seed": m["seed"], "birth": m["birth"]}
        for label, index in (("birth", 0), ("entry", 4), ("near_wrong", 3)):
            effects = np.array([a[1] - a[index] for a in alternatives])
            for k, x in zip(
                ("actual", "matched_new_text", "all_orders_same_social", "all_held"),
                effects,
                strict=True,
            ):
                v[k + "_vs_" + label] = float(x)
            for k, x in zip(
                ("new_task_and_wording", "balance_orders", "balance_social"),
                np.diff(effects),
                strict=True,
            ):
                v[k + "_effect_change_vs_" + label] = float(x)
        cases.append(v)
    summary = {
        b: {
            k: describe([v for v in cases if v["birth"] == b], k)
            for k in cases[0]
            if k not in ("birth", "seed")
        }
        for b in BIRTHS
    }
    write(out, {"read_only_post_outcome": True, "per_life": cases, "groups": summary})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lives", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.lives, a.output)
