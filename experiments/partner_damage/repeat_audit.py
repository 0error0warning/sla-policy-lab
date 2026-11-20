"""Evaluate within-device invariance and cross-device differences without selecting a preferred run."""

import argparse
import json
from pathlib import Path
import numpy as np
from experiments.direct_policy.protocol import write
from experiments.lifetime_cooperation.design import ITEMS
from experiments.loreft_omission.analysis import measures
from .analysis import dump, panel_tv
from .design import CELLS, HEADLINES


def read(p):
    return json.loads(p.read_bytes())


def verify(first, second, inputs, out):
    anchors = np.load(inputs / "anchors.npz")
    events = read(inputs / "events.json")
    cases = [e for e in events if e["focus"] or e["key"] == "s77101t1"]
    receipt = read(second / "receipt.json")
    assert receipt["frozen_backbone"] and len(receipt["checks"]) == len(cases) == 3
    rows = []
    probs = []
    compare = []
    for e in cases:
        key = e["key"]
        target = HEADLINES.get((e["seed"], e["turn"]), e["factual"]["action"])
        orig = e["factual"]
        ind = 0 if orig["maintenance"] else 3
        rr = read(second / key / "receipt.json")
        test = rr["anchor"]
        with np.load(second / key / "legacy.npz") as z:
            assert np.array_equal(z["native_before"], z["before"]) and np.array_equal(
                z["before"], z["slot0_before"]
            )
            assert np.array_equal(z["source"], z["source0"]) and np.array_equal(
                z["post_weights"], z["slot0_post_weights"]
            )
            assert test["slot_post_hash_equal"] and test["original_vs_factorial"]
            assert test["archive_slot_hash_equal"] == np.array_equal(
                z["post_weights"], anchors[key + "_post"]
            )
        arrays = []
        for wave, d in (("first", first / "cases" / key), ("repeat", second / key)):
            meta = read(d / "receipt.json")
            with np.load(d / "factorial.npz") as z, np.load(d / "swaps.npz") as sw:
                p = z["panels"].astype(float)
                weights = z["cell_weights"]
                hy = sw["panels"]
                assert (
                    p.shape == (5, 4, 32, 4)
                    and np.array_equal(p[1], p[2])
                    and np.array_equal(p[3], p[4])
                )
                assert np.array_equal(weights[0], weights[1]) and np.array_equal(
                    weights[2], weights[3]
                )
                assert np.array_equal(hy[0], p[0]) and np.array_equal(hy[1], p[ind + 1])
                assert np.array_equal(hy[0], hy[2]) and np.array_equal(hy[1], hy[3])
                partner = ITEMS[orig["row"]]["partner"]
                changes = panel_tv(hy[0], hy[1], partner)
                ss = panel_tv(hy[1], hy[4], partner)
                record = dict(
                    key=key,
                    wave=wave,
                    target=target,
                    pre_anchor_max_pp=100
                    * max(abs(p[0] - anchors[key + "_prepanel"]).flat),
                    factual_post_anchor_max_pp=100
                    * max(abs(p[ind + 1] - anchors[key + "_postpanel"]).flat),
                    hash_matches=meta["source_update_hash"]
                    == meta["source_expected_hash"],
                    factual_source_tv=changes["source"],
                    factual_other_tv=changes["other"],
                    gradient_cosine=meta["gradient_cosine"],
                    swapped_other_tv=ss["other"],
                )
                rows.append(record)
                arrays.append((p, weights.copy()))
                for j, label in enumerate(("pre", *CELLS)):
                    probs.append(
                        dict(
                            key=key,
                            wave=wave,
                            snapshot=e["turn"] + 1,
                            condition=label,
                            **measures(p[j], e["seed"], target),
                        )
                    )
            if e["focus"]:
                with np.load(d / "continuation.npz") as z:
                    for j, t in enumerate(z["turns"]):
                        m = z["panels"][j]
                        assert np.array_equal(m[0], m[1]) and np.array_equal(m[2], m[3])
                        for i, label in enumerate(CELLS):
                            probs.append(
                                dict(
                                    key=key,
                                    wave=wave,
                                    snapshot=int(t),
                                    condition=label,
                                    **measures(m[i], e["seed"], target),
                                )
                            )
        p1, w1 = arrays[0]
        p2, w2 = arrays[1]
        compare.append(
            dict(
                key=key,
                slot_error=test["slot_pre_error"],
                pre_cross_run_max_pp=float(abs(p1[0] - p2[0]).max() * 100),
                max_all_cross_run_pp=float(abs(p1 - p2).max() * 100),
                factual_weight_cross_run_equal=np.array_equal(w1[ind], w2[ind]),
                repeat_matches_archive=test["archive_slot_hash_equal"],
                first_matches_archive=rows[-2]["hash_matches"],
            )
        )
    anchors.close()
    dump(out / "repeat-check.csv", rows)
    dump(out / "repeat-cells.csv", probs)
    dump(out / "cross-run.csv", compare)
    write(
        out / "recheck.json",
        dict(cases=len(cases), within_device_slots_identical=True, comparison=compare),
    )
    return compare


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("first", "second", "inputs", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.first, a.second, a.inputs, a.out), indent=2))
