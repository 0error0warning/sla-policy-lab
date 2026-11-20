"""Score published checkpoints on independent singleton states with native BF16 roundings."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.experiment import evaluation_reference, metrics
from experiments.consequence_policy.study import TEST_H, basis
from experiments.direct_policy.protocol import rng, write
from experiments.internal_body.carrier import rotate
from experiments.internal_body.design import CONFIRM, SPLIT_H
from experiments.internal_body.lives import WORLDS, post
from experiments.internal_body.personal import LifeActor, LifeInputs
from experiments.internal_body.suffix import distribution
from experiments.social_closed_loop.core import BodyValue


class Tail:
    def __init__(self, fresh, cache):
        with np.load(cache / "architecture.npz") as z:
            self.head = torch.tensor(z["head"])
            self.norm = torch.tensor(z["norm"])
        self.cfg = json.loads((cache / "architecture.json").read_bytes())
        self.rows = json.loads((cache / "rows.json").read_bytes())
        self.p = torch.tensor([r["partner"] for r in self.rows[768:960]])
        self.g = torch.tensor(basis(TEST_H), dtype=torch.float32)
        self.neutral = torch.tensor(basis(SPLIT_H).mean(axis=0), dtype=torch.float32)
        with np.load(fresh / "plain.npz") as z:
            self.novel = {k: z[k] for k in ("features", "native", "r", "z")}
            self.plain = {
                k: torch.tensor(z[k][:192])[None]
                for k in ("features", "native", "r", "z", "skip", "down")
            }
        self.prompt = {}
        for h in TEST_H:
            with np.load(fresh / f"h{int(h)}.npz") as z:
                for k in self.plain:
                    self.prompt.setdefault(k, []).append(torch.tensor(z[k]))
        self.prompt = {k: torch.stack(x) for k, x in self.prompt.items()}

    def normalized(self, r, norm):
        return (
            (
                r
                * torch.rsqrt(r.square().mean(-1, keepdim=True) + self.cfg["epsilon"])
                * (self.cfg["norm_offset"] + norm)
            )
            .bfloat16()
            .float()
        )

    def words(self, z):
        return (z @ self.head.T).bfloat16().float()

    def score(self, kind, w=None, carrier=None, force_mask=False):
        text = kind in ("native_prompt", "prompt_lora", "frozen_prompt")
        a = self.prompt if text else self.plain
        n = a["native"]
        if kind.startswith("frozen"):
            return distribution(n.expand(len(TEST_H), -1, -1)).numpy()
        norm = self.norm
        if kind.startswith("native") or kind == "carrier":
            effective = (norm + torch.tensor(w["delta"])).bfloat16().float()
            r = a["r"]
            if carrier is not None:
                u, v = carrier
                r = (
                    rotate(r, self.g, torch.tensor(u), torch.tensor(v))
                    .bfloat16()
                    .float()
                )
            z = self.normalized(r, effective)
        else:
            A = torch.tensor(w["A"])
            B = torch.tensor(w["B"])
            if kind == "no_route":
                A = A.mean(0, keepdim=True).expand(2, -1, -1, -1)
            g = (
                self.neutral.expand_as(self.g)
                if kind in ("prompt_lora", "blind") or force_mask
                else self.g
            )
            tables = torch.einsum("hb,nbdr->hndr", g, A[self.p])
            c = torch.einsum(
                "hnd,hndr->hnr", a["features"].expand(len(g), -1, -1), tables
            )
            if kind == "nonlinear":
                c = c.tanh()
            delta = c @ B
            if kind == "postnorm":
                z = (
                    (
                        a["z"]
                        + delta
                        * (norm + self.cfg["norm_offset"])
                        * torch.rsqrt(
                            a["r"].square().mean(-1, keepdim=True) + self.cfg["epsilon"]
                        )
                    )
                    .bfloat16()
                    .float()
                )
            else:
                r = (a["skip"].bfloat16() + (a["down"] + delta).bfloat16()).float()
                z = self.normalized(r, norm)
        anchored = n + self.words(z) - self.words(a["z"])
        return distribution(anchored.expand(len(TEST_H), -1, -1)).numpy()


def stats(x):
    a = np.array(x)
    bootstrap = rng(65902, "fresh").integers(len(x), size=(10000, len(x)))
    return {
        "mean": float(a.mean()),
        "ci95": np.quantile(a[bootstrap].mean(axis=1), [0.025, 0.975]).tolist(),
        "positive": int((a > 0).sum()),
    }


def run(cache, fresh, primary, native, credit, out):
    tail = Tail(fresh, cache)
    body = BodyValue()
    zrows = tail.rows[768:960]
    ref = [evaluation_reference(body, zrows, m) for m in (0, 1)]
    witness = json.loads((fresh / "actual-tail-witness.json").read_bytes())
    absolute = {}
    checks = {}
    pairs = []
    for seed in CONFIRM:
        mirror = []
        for m in (0, 1):
            folder = primary / f"s{seed}-m{m}"
            values = {}
            for k in (
                "frozen",
                "linear",
                "nonlinear",
                "postnorm",
                "native_hidden",
                "native_prompt",
                "prompt_lora",
                "shuffle",
                "blind",
                "no_route",
                "maintain",
                "zero",
                "high",
            ):
                w = None if k == "frozen" else np.load(folder / (k + ".npz"))
                pi = tail.score(k, w)
                values[k] = metrics(pi, ref[m])
                if seed == CONFIRM[0] and m == 0:
                    for r in witness:
                        if r["kind"] == k:
                            d = np.max(
                                np.abs(
                                    pi[TEST_H.index(r["h"]), r["i"] - 768] - r["prob"]
                                )
                            )
                            checks.setdefault(k, []).append(float(d))
            absolute[f"{seed}-{m}"] = values
            f, l, sh, b = [values[q] for q in ("frozen", "linear", "shuffle", "blind")]
            mirror.append(
                {
                    "fatal": f["danger_answer"] - l["danger_answer"],
                    "healthy": l["healthy_answer"] - f["healthy_answer"],
                    "style": l["healthy_style"] - sh["healthy_style"],
                    "gap": l["gap"] - b["gap"],
                }
            )
        pairs.append({k: np.mean([v[k] for v in mirror]) for k in mirror[0]})
    orig_stats = {k: stats([r[k] for r in pairs]) for k in pairs[0]}
    carrier = []
    carrier_values = {}
    for seed in range(64501, 64521):
        pair = []
        with np.load(native / f"basis-{seed}.npz") as q:
            rot = (q["u"], q["v"])
        for m in (0, 1):
            folder = native / f"s{seed}-m{m}"
            values = {}
            for name in ("frozen", "sensor", "shuffle", "text", "hidden"):
                code = {
                    "sensor": "carrier",
                    "shuffle": "carrier",
                    "frozen": "frozen",
                    "text": "native_prompt",
                    "hidden": "native_hidden",
                }[name]
                w = (
                    None
                    if name == "frozen"
                    else np.load(folder / ("native-" + name + ".npz"))
                )
                values[name] = metrics(
                    tail.score(code, w, rot if code == "carrier" else None), ref[m]
                )
            carrier_values[f"{seed}-{m}"] = values
            l, f, sh, b = [values[q] for q in ("sensor", "frozen", "shuffle", "hidden")]
            pair.append(
                {
                    "fatal": f["danger_answer"] - l["danger_answer"],
                    "healthy": l["healthy_answer"] - f["healthy_answer"],
                    "shuffle": sh["danger_answer"] - l["danger_answer"],
                    "gap": l["gap"] - b["gap"],
                }
            )
        carrier.append({k: float(np.mean([r[k] for r in pair])) for k in pair[0]})
    carrier_stats = {k: stats([r[k] for r in carrier]) for k in carrier[0]}
    data = LifeInputs(cache)
    for old, new in ((data.x, "features"), (data.r, "r"), (data.n, "native")):
        old[768:] = tail.novel[new]
    data.scale0 = np.sqrt((data.r**2).mean(1, keepdims=True) + data.eps)
    data.base_dots = (data.r / data.scale0) @ data.t
    pinned = []
    for seed in range(64701, 64721):
        fixed = LifeActor(data, credit, seed, "fixed")
        free = LifeActor(data, credit, seed, "free")
        with np.load(credit / f"credit-{seed}-parameters.npz") as weights:
            deltas = []
            for j, world in enumerate(WORLDS):
                birth = post(fixed, world)
                free.W = weights[f"w{j}-v0-predict"]
                true = post(free, world)
                free.W = weights[f"w{j}-v0-shuffled"]
                sh = post(free, world)
                deltas.append(
                    [true["style"] - birth["style"], true["style"] - sh["style"]]
                )
            pinned.append(np.mean(deltas, axis=0))
    write(out / "fresh-population.json", absolute)
    write(out / "fresh-carrier.json", carrier_values)
    summary = {
        "internal": orig_stats,
        "carrier": carrier_stats,
        "same_process_full_model_error": {
            k: {"n": len(v), "max": max(v), "p95": float(np.quantile(v, 0.95))}
            for k, v in checks.items()
        },
        "new_partner_fixed_final_weights": {
            "over_fixed": stats(np.array(pinned)[:, 0]),
            "over_shuffle": stats(np.array(pinned)[:, 1]),
        },
    }
    write(out / "fresh-summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("cache", "fresh", "population", "native", "credit", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    run(a.cache, a.fresh, a.population, a.native, a.credit, a.output)
