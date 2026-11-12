"""Cluster-level direct-training, actual localization and transplant comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.consequence_policy.experiment import evaluation_reference
from experiments.consequence_policy.study import TEST_H
from experiments.direct_policy.protocol import write
from experiments.internal_body.design import all_rows
from experiments.internal_body.evidence import stat
from experiments.social_closed_loop.core import BodyValue

SITES = ("site3", "site15", "site31", "local31")
LEVELS = (2.0, 6.0, 90.0, 300.0)


def actual_depth(dirs):
    clusters = []
    texture = []
    seeds = []
    for d in dirs:
        for path in sorted(d.glob("s*-m0")):
            s = int(path.name.split("-")[0][1:])
            seeds.append(s)
            both = [
                json.loads((d / f"s{s}-m{m}/metrics.json").read_bytes()) for m in (0, 1)
            ]
            one = {}
            for n in ("frozen", *SITES):
                for k in ("danger", "healthy", "style"):
                    one[n + "_" + k] = float(np.mean([r[n][k] for r in both]))
                if n != "frozen":
                    for k in ("danger", "healthy", "style"):
                        sign = -1 if k == "danger" else 1
                        one[n + "_delta_" + k] = sign * (
                            one[n + "_" + k] - one["frozen_" + k]
                        )
                    for m, r in enumerate(both):
                        arm = r[n]
                        assert len(arm["parameter_names"]) == 2 and all(
                            "body_residual." in x for x in arm["parameter_names"]
                        )
                        texture.append(
                            {
                                "seed": s,
                                "mapping": m,
                                "site": n,
                                "disabled_max": arm["disabled_frozen_max_logit_gap"],
                                "seconds": arm["seconds"],
                                "A_l2_change": arm["state_A_change_l2"],
                                "B_l2": arm["state_B_l2"],
                                "first_loss": arm["steps"][0][1],
                                "last_loss": arm["steps"][-1][1],
                                "last_rms": arm["steps"][-1][4],
                                "mean_grad": float(
                                    np.mean([v[-1] for v in arm["steps"]])
                                ),
                            }
                        )
            clusters.append(one)
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seeds")
    stats = {k: stat([r[k] for r in clusters]) for k in clusters[0]}
    gates = {}
    for n in SITES:
        fat, healthy, style = [
            stats[n + "_delta_" + k] for k in ("danger", "healthy", "style")
        ]
        gates[n] = bool(
            fat["mean"] >= 0.10
            and healthy["mean"] >= -0.05
            and style["mean"] >= 0.10
            and fat["positive"] >= int(np.ceil(0.75 * len(seeds)))
            and style["positive"] >= int(np.ceil(0.75 * len(seeds)))
        )
    return {
        "seeds": seeds,
        "clusters": clusters,
        "stats": stats,
        "gates": gates,
        "parameter_and_gradient": texture,
    }


def measures(pi, levels, ref):
    risk, _, favor = ref
    ans = pi[:, :, 2:].sum(-1)
    by = pi[:, np.arange(len(favor)), favor]
    high = [i for i, h in enumerate(levels) if h >= 90]
    d = levels.index(2.0)
    deaths = (pi * risk[[TEST_H.index(h) for h in levels]]).sum(-1)
    out = {
        "danger": float(ans[d].mean()),
        "healthy": float(ans[high].mean()),
        "style": float(by[high].mean()),
        "death2": float(deaths[d].mean()),
        "gap": float(ans[high].mean() - ans[d].mean()),
    }
    if 6.0 in levels:
        out["death6"] = float(deaths[levels.index(6.0)].mean())
    return out


def placement(gpu, trained):
    base = np.load(gpu / "frozen.npz")
    ids = base["indices"]
    rows = all_rows()
    body = BodyValue()
    ref = [evaluation_reference(body, [rows[i] for i in ids], m) for m in (0, 1)]
    all_obs = {}
    gaps = {}
    physical_hashes = 0
    for s in range(65001, 65013):
        for m in (0, 1):
            key = f"s{s}-m{m}"
            where = gpu / key
            rec = json.loads((where / "records.json").read_bytes())
            seen = {}
            obs = {}
            with (
                np.load(where / "full-probes.npz") as full,
                np.load(trained / key / "predicted-probes.npz") as pred,
            ):
                for name, r in rec.items():
                    lvl = r["levels"]
                    pi = full[name]
                    v = measures(pi, lvl, ref[m])
                    baseline = measures(
                        np.broadcast_to(base["prob"], pi.shape), lvl, ref[m]
                    )
                    v.update(
                        {
                            "delta_" + k: (
                                -1 if k.startswith(("danger", "death")) else 1
                            )
                            * (v[k] - b)
                            for k, b in baseline.items()
                        }
                    )
                    obs[name] = v
                    if r["strength"] == 1.0:
                        for h, signature in r["applied_sha256"].items():
                            ident = (r["donor"], r["signal"], h)
                            if ident in seen:
                                assert seen[ident] == signature
                                physical_hashes += 1
                            seen[ident] = signature
                    if r["donor"] == r["receiver"] and r["strength"] == 1:
                        approx = pred[f"l{r['donor']}-{r['signal']}"][
                            [TEST_H.index(h) for h in lvl]
                        ][:, ids - 768]
                        err = np.abs(pi - approx).max(-1)
                        gaps.setdefault(name, []).append(err.ravel())
                        v["predicted_danger"] = float(approx[0, :, 2:].sum(-1).mean())
                        v["predicted_healthy"] = float(
                            approx[[-2, -1], :, 2:].sum(-1).mean()
                        )
            all_obs[key] = obs
    names = {k for sample in all_obs.values() for k in sample}
    stats = {}
    for name in sorted(names):
        seeds = [s for s in range(65001, 65013) if name in all_obs[f"s{s}-m0"]]
        one = {}
        for metric in all_obs[f"s{seeds[0]}-m0"][name]:
            one[metric] = stat(
                [
                    np.mean([all_obs[f"s{s}-m{m}"][name][metric] for m in (0, 1)])
                    for s in seeds
                ]
            )
        stats[name] = one
    gates = {}
    for l in (3, 15, 31):
        v = stats[f"d{l}-r{l}-real-x1"]
        f, h, st = v["delta_danger"], v["delta_healthy"], v["delta_style"]
        gates[str(l)] = (
            f["mean"] >= 0.1
            and h["mean"] >= -0.05
            and st["mean"] >= 0.1
            and min(f["positive"], st["positive"]) >= 10
            and min(f["ci95"][0], st["ci95"][0]) > 0.0
        )
    errors = {
        k: {
            "max": float(np.max(np.concatenate(a))),
            "p95": float(np.quantile(np.concatenate(a), 0.95)),
            "mean": float(np.mean(np.concatenate(a))),
        }
        for k, a in gaps.items()
    }
    return {
        "per_world": all_obs,
        "stats": stats,
        "gates": gates,
        "identical_transplant_hash_comparisons": physical_hashes,
        "linearized_to_actual_probability_disagreement": errors,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--depth", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--transplant", type=Path)
    p.add_argument("--trained", type=Path)
    a = p.parse_args()
    res = {"direct": actual_depth(a.depth)}
    if a.transplant:
        res["placement"] = placement(a.transplant, a.trained)
    write(a.output, res)
    print(json.dumps({k: v["gates"] for k, v in res.items()}))
