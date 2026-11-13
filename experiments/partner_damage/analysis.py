"""Separate the online feedback path from identity encoding, policy changes and layer controls."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from experiments.direct_policy.protocol import write
from experiments.individual_life.analysis import describe
from experiments.lifetime_cooperation.design import ITEMS, LEVELS, PILOT
from experiments.loreft_omission.analysis import measures
from .design import CELLS, GRID, HEADLINES, PARTNER, SMALL, STRENGTH


def load_json(path):
    return json.loads(path.read_bytes())


def dump(path, rows):
    with path.open("w", encoding="utf8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0])
        wr.writeheader()
        wr.writerows(rows)


def tv(a, b):
    return 50.0 * np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).sum(
        -1
    )


def partition(values, partner, people):
    values = values.reshape(-1)
    people = np.broadcast_to(people, values.shape)
    return {
        side: float(np.mean(values[people == p]))
        for side, p in (("source", partner), ("other", 5 - partner))
    }


def panel_tv(a, b, partner):
    return partition(tv(a, b).reshape(-1), partner, np.tile(PARTNER, len(LEVELS)))


def small_tv(a, b, partner):
    return partition(
        tv(a, b), partner, np.array([ITEMS[GRID[j]]["partner"] for _, j in SMALL])
    )


def put_metrics(record, name, d):
    record.update({name + "_" + k: v for k, v in d.items()})


def aggregate(rows, keys):
    return (
        {
            k: describe(
                [
                    np.mean([r[k] for r in rows if r["seed"] == s])
                    for s in sorted({x["seed"] for x in rows})
                ]
            )
            for k in keys
        }
        if rows
        else {}
    )


def decode(root, dest):
    meta = load_json(root / "representation-rows.json")
    pairs = [(j, j + 1) for j in range(0, len(meta), 2)]
    assert all(
        meta[a]["partner"] == 2
        and meta[b]["partner"] == 3
        and all(
            meta[a][k] == meta[b][k] for k in ("split", "frame", "layout", "health")
        )
        for a, b in pairs
    )
    tr = [pair for pair in pairs if meta[pair[0]]["split"] == 0]
    assert len(tr) == 48
    results = []
    null = []
    with np.load(root / "representations.npz") as z:
        for stage in ("early_id", "early_decision", "terminal_decision"):
            x = z[stage].astype(float)
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            d = x[[b for _, b in tr]] - x[[a for a, _ in tr]]
            center = np.mean(x[[n for pair in tr for n in pair]], axis=0)
            gen = np.random.default_rng(78073)
            # Fixed ridge on the class-mean contrast, followed by normalization for scoring.
            # The same isotropic ridge scale cancels in the sign of this direction.
            dirs = (
                np.vstack(
                    [np.ones(len(tr)), gen.choice((-1.0, 1.0), size=(128, len(tr)))]
                )
                @ d
                / len(tr)
                / (1.0 + 0.01)
            )
            dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
            for h in LEVELS:
                te = [
                    q
                    for q in pairs
                    if meta[q[0]]["split"] == 1 and meta[q[0]]["health"] == h
                ]
                a, b = [x[[u[j] for u in te]] for j in (0, 1)]
                pa = (a - center) @ dirs.T
                pb = (b - center) @ dirs.T
                acc = (
                    (pa < 0).mean(axis=0)
                    + (pb > 0).mean(axis=0)
                    + 0.5 * ((pa == 0).mean(axis=0) + (pb == 0).mean(axis=0))
                ) / 2
                order = (pb > pa).mean(axis=0) + 0.5 * (pb == pa).mean(axis=0)
                for j in range(1, len(acc)):
                    null.append(
                        dict(
                            layer=stage,
                            health=h,
                            permutation=j,
                            accuracy=acc[j],
                            paired_order=order[j],
                        )
                    )
                results.append(
                    dict(
                        layer=stage,
                        health=h,
                        test_pairs=len(te),
                        accuracy=acc[0],
                        paired_order=order[0],
                        cosine_between_ids=float(np.mean((a * b).sum(-1))),
                        null_accuracy_mean=float(acc[1:].mean()),
                        null_accuracy_q95=float(np.quantile(acc[1:], 0.95)),
                        null_order_q95=float(np.quantile(order[1:], 0.95)),
                    )
                )
    dump(dest / "identity-readout.csv", results)
    dump(dest / "permuted-readout.csv", null)
    return results


def analyze(root, inputs, dest):
    dest.mkdir(parents=True, exist_ok=True)
    events = load_json(inputs / "events.json")
    physical = load_json(inputs / "physical-audit.json")
    cases = []
    cell_rows = []
    late = []
    site_rows = []
    flows = []
    for ev in events:
        seed, t, key = ev["seed"], ev["turn"], ev["key"]
        e = ev["factual"]
        person = ITEMS[e["row"]]["partner"]
        act = e["action"]
        target = HEADLINES.get((seed, t), act)
        f = root / "cases" / key
        info = load_json(f / "receipt.json")
        row = dict(
            seed=seed,
            turn=t,
            case=key,
            ordinary=ev["ordinary"],
            focus=ev["focus"],
            partner=person,
            was_maintained=e["maintenance"],
            action=act,
            target=target,
            self_health=e["health"],
            gradient_cosine=info["gradient_cosine"],
        )
        with np.load(f / "factorial.npz") as z:
            no, *cell = z["panels"].astype(float)
            identity = z["identity_train"].astype(float)
            fact = cell[0 if e["maintenance"] else 3]
            direct = max(abs(cell[0] - cell[1]).max(), abs(cell[2] - cell[3]).max())
            row["direct_social_probability_max"] = float(direct)
            put_metrics(row, "factual_tv", panel_tv(no, fact, person))
            put_metrics(row, "body_tv", panel_tv(cell[0], cell[2], person))
            put_metrics(
                row, "train_ID_swap", panel_tv(identity[0], identity[1], person)
            )
            put_metrics(
                row,
                "same_multiplier_ID_swap",
                panel_tv(identity[0], identity[2], person),
            )
            row["pre_source_vs_swapped_train_tv"] = float(
                tv(z["train_before"], z["train_swapped_before"])
            )
            row["body_light_minus_heavy_signal"] = (
                info["cells"][CELLS[0]]["signal"] - info["cells"][CELLS[3]]["signal"]
            )
            for n, entry in enumerate([no, *cell]):
                name = "no_update" if n == 0 else CELLS[n - 1]
                signal = None if n == 0 else info["cells"][name]["signal"]
                dmg = None if n == 0 else info["cells"][name]["damage"]
                m = measures(entry, seed, target)
                m0 = measures(no, seed, target)
                cell_rows.append(
                    dict(
                        seed=seed,
                        turn=t,
                        key=key,
                        case=name,
                        source=person,
                        action=act,
                        target=target,
                        signal=signal,
                        damage=dmg,
                        actual_step=0.0
                        if n == 0
                        else info["steps"][n - 1]["actual_l2"],
                        **m,
                        target_change_p86=m["p86"] - m0["p86"],
                        target_change_p93=m["p93"] - m0["p93"],
                    )
                )
        with np.load(f / "swaps.npz") as z:
            pre, post, rm, restore, sw, gate = z["panels"].astype(float)
            row["direct_transfer_delta_cosine"] = (
                float(np.nanmean(z["delta_cosine"]))
                if np.isfinite(z["delta_cosine"]).any()
                else 0.0
            )
            row["nonzero_delta_pairs"] = int(np.isfinite(z["delta_cosine"]).sum() // 2)
            put_metrics(row, "swap_vs_factual", panel_tv(post, sw, person))
            put_metrics(row, "gated", panel_tv(pre, gate, person))
            put_metrics(row, "remove", panel_tv(pre, rm, person))
            put_metrics(row, "restore", panel_tv(post, restore, person))
            for k, h in enumerate(LEVELS):
                for a in range(4):
                    al = post[k, :, a] - pre[k, :, a]
                    cp = sw[k, :, a] - post[k, :, a]
                    flows.append(
                        dict(
                            seed=seed,
                            turn=t,
                            case=key,
                            health=h,
                            action=a,
                            factual_source_pp=100 * al[PARTNER == person].mean(),
                            factual_other_pp=100 * al[PARTNER != person].mean(),
                            swapped_instead_of_own_source_pp=100
                            * cp[PARTNER == person].mean(),
                            swapped_instead_of_own_other_pp=100
                            * cp[PARTNER != person].mean(),
                        )
                    )
        with np.load(f / "sites.npz") as z:
            q = z["panels"].astype(float)
            c = z["calibration"].astype(float)
            ix = [0, 0, info["site"]["early_index"], info["site"]["terminal_index"]]
            for j, name in enumerate(
                ("early", "terminal", "early_matched", "terminal_matched")
            ):
                pos = j % 2
                index = ix[j]
                d = small_tv(q[:, 0], q[:, j + 1], person)
                site_rows.append(
                    dict(
                        seed=seed,
                        turn=t,
                        case=key,
                        ordinary=ev["ordinary"],
                        focus=ev["focus"],
                        site=name,
                        matched_pair=info["site"]["matched"],
                        grid_factor=STRENGTH[index],
                        source_training_tv=tv(c[pos, 0], c[pos, index + 1]),
                        source_panel_tv=d["source"],
                        other_panel_tv=d["other"],
                        specificity=d["source"] - d["other"],
                    )
                )
                put_metrics(row, "site_" + name, d)
                row["specificity_" + name] = d["source"] - d["other"]
            row["site_pair_valid"] = info["site"]["matched"]
        if ev["focus"]:
            with np.load(f / "continuation.npz") as z:
                for idx, turn in enumerate(z["turns"]):
                    for j, name in enumerate(CELLS):
                        late.append(
                            dict(
                                seed=seed,
                                source_turn=t,
                                at=int(turn),
                                case=name,
                                target=target,
                                **measures(z["panels"][idx, j], seed, target),
                            )
                        )
        row["location_specificity_delta"] = (
            row["specificity_early_matched"] - row["specificity_terminal_matched"]
        )
        cases.append(row)
    for file, rows in (
        ("cases", cases),
        ("factorial-cells", cell_rows),
        ("action-flow", flows),
        ("positions", site_rows),
        ("continuation", late),
    ):
        dump(dest / (file + ".csv"), rows)
    readouts = decode(root, dest)
    ordinary = [r for r in cases if r["ordinary"]]
    strong = [
        r for r in ordinary if min(r["factual_tv_source"], r["factual_tv_other"]) >= 1
    ]
    matched = [r for r in ordinary if r["site_pair_valid"]]
    keys = (
        "factual_tv_source",
        "factual_tv_other",
        "body_tv_source",
        "body_tv_other",
        "train_ID_swap_source",
        "train_ID_swap_other",
        "same_multiplier_ID_swap_other",
        "gradient_cosine",
        "swap_vs_factual_other",
        "gated_source",
        "gated_other",
        "direct_transfer_delta_cosine",
        "pre_source_vs_swapped_train_tv",
        "body_light_minus_heavy_signal",
    )
    sitename = (
        "site_early_source",
        "site_early_other",
        "site_terminal_source",
        "site_terminal_other",
        "specificity_early",
        "specificity_terminal",
        "site_early_matched_source",
        "site_early_matched_other",
        "site_terminal_matched_source",
        "site_terminal_matched_other",
        "specificity_early_matched",
        "specificity_terminal_matched",
        "location_specificity_delta",
    )
    signed = np.array([[r["light_signal"], r["heavy_signal"]] for r in physical], float)
    summary = dict(
        seeds=list(PILOT),
        cases=len(cases),
        ordinary=len(ordinary),
        focus=sum(r["focus"] for r in cases),
        direct_max=max(r["direct_social_probability_max"] for r in cases),
        max_remove=max(r["remove_other"] for r in cases),
        max_restore=max(r["restore_other"] for r in cases),
        all_non_silent=len(physical),
        factual_physics_accepted=sum(r["maintenance"] for r in physical),
        physics_distinct=int(np.sum(abs(signed[:, 0] - signed[:, 1]) > 1e-9)),
        physics_sign_flip=int(np.sum((signed[:, 0] > 0) & (signed[:, 1] < 0))),
        ordinary_means=aggregate(ordinary, keys),
        measurable_events=len(strong),
        measurable_seeds=len({r["seed"] for r in strong}),
        measurable_means=aggregate(strong, keys),
        location_all=aggregate(ordinary, sitename),
        location_matched=aggregate(matched, sitename),
        site_matches=len(matched),
        site_match_seeds=len({r["seed"] for r in matched}),
        readouts=readouts,
    )
    write(dest / "summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    s = analyze(a.root, a.inputs, a.out)
    print("pairs", s["cases"], "direct", s["direct_max"], "matched", s["site_matches"])
