"""Describe the largest observed one-update probability shifts on identical inputs."""

import argparse
import json
from pathlib import Path
import numpy as np
from .analysis import dump_table, load_trace, measures
from .design import ACTIONS, ITEMS, PILOT, branch_key, histories


def examine(root, inputs, out):
    tape = histories(inputs)
    ranked = []
    winners = []
    for seed in PILOT:
        src, ref, _, _ = tape[seed]
        folder = root / "lives" / f"s{seed}"
        events = json.loads((folder / "plan.json").read_bytes())["events"]
        with np.load(folder / "normal.npz") as z:
            panels = z["panels"]
            probs = z["actual"]
        readouts = [[measures(x, seed, a) for a in range(4)] for x in panels]
        earlier = {}
        for e in events:
            t = e["turn"]
            target = e["action"]
            old = e["previous"]
            fam = e["family"]
            first = earlier.get(fam, 1)
            earlier[fam] = t
            candidates = []
            for u in range(first, t):
                pre = readouts[u - 1]
                post = readouts[u]
                item = src[u - 1]
                r = ITEMS[item["row"]]
                deltas = {p: post[target][p] - pre[target][p] for p in ("p86", "p93")}
                old_delta = {
                    p: post[old][p] - pre[old][p] if old is not None else 0.0
                    for p in deltas
                }
                gain = {p: deltas[p] - old_delta[p] for p in deltas}
                candidates.append(
                    dict(
                        seed=seed,
                        event=e["id"],
                        anchor=t,
                        target=ACTIONS[target],
                        previous=ACTIONS[old] if old is not None else "",
                        from_update=first,
                        update=u,
                        lag=t - u,
                        partner=("P86", "P93")[r["partner"] - 2],
                        frame=r["frame"],
                        task=r["task"],
                        source_action=ACTIONS[item["action"]],
                        maintenance=item["maintenance"],
                        health_before=item["health"],
                        health_after=item["after"],
                        damage=item["health"] - item["after"],
                        raw_rul=item["reward"],
                        signal=item["signal"],
                        actual_step=ref[u - 1]["updates"]["loreft"]["actual_l2"],
                        gain_target_p86=deltas["p86"],
                        gain_target_p93=deltas["p93"],
                        gain_previous_p86=old_delta["p86"],
                        gain_previous_p93=old_delta["p93"],
                        margin_change_p86=gain["p86"],
                        margin_change_p93=gain["p93"],
                        min_margin_change=min(gain.values()),
                    )
                )
            candidates.sort(key=lambda x: (-x["min_margin_change"], x["update"]))
            ranked.extend(dict(rank=i + 1, **v) for i, v in enumerate(candidates))
            if not candidates:
                continue
            best = candidates[0]
            u = best["update"]
            # Only test a propagated single-step attribution if this same omission already ran.
            tested = (folder / "forks" / branch_key((u,)) / "trace.npz").exists()
            end_effect = {}
            if tested:
                cf, _ = load_trace(folder, panels, probs, (u,))
                factual = readouts[t - 1][target]
                alt = measures(cf[t - 1], seed, target)
                end_effect = {
                    f"single_skip_at_anchor_{p}": factual[p] - alt[p]
                    for p in ("p86", "p93")
                }
            winners.append(
                dict(
                    **best,
                    number_considered=len(candidates),
                    single_skip_measured=tested,
                    single_skip_at_anchor_p86=end_effect.get(
                        "single_skip_at_anchor_p86", ""
                    ),
                    single_skip_at_anchor_p93=end_effect.get(
                        "single_skip_at_anchor_p93", ""
                    ),
                )
            )
    dump_table(out / "step-drivers.csv", ranked)
    dump_table(out / "largest-steps.csv", winners)
    return winners


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    examine(a.root, a.inputs, a.out)
