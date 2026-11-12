"""Separate frozen-prefix and tail arithmetic differences on the independent transition panel."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.study import basis, contexts
from experiments.direct_policy.protocol import write
from experiments.internal_body.suffix import InternalPolicy, distribution
from experiments.lethal_boundary.design import heldout_panel
from experiments.lethal_boundary.states import Bank, at, expand_bf16


def compare(x, y):
    return np.abs(x - y).max(-1).ravel()


def run(gpu, cache, arch, study, out):
    receipt = json.loads((gpu / "receipt.json").read_bytes())
    h = np.asarray(receipt["result"]["health"])
    panel = heldout_panel()
    rows = contexts()[768:]
    ps = np.array([rows[i]["partner"] for i in panel])
    pp = torch.tensor(ps)
    bank = Bank(cache, arch)
    gate = torch.as_tensor(basis(h), dtype=torch.float32)
    factor = bank.norm + bank.cfg["norm_offset"]

    def normalize(x):
        return (
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + bank.cfg["epsilon"])
            * factor
        )

    def words(z):
        return (z @ bank.head.T).bfloat16().float()

    errors = {
        k: []
        for k in (
            "frozen_old_vs_fresh",
            "matched_smooth_old_vs_fresh",
            "fresh_smooth_vs_bf16",
            "matched_bf16_vs_actual",
            "old_smooth_vs_actual",
        )
    }
    with (
        np.load(gpu / "fresh-components.npz") as fc,
        np.load(gpu / "full-probes.npz") as actual,
        torch.no_grad(),
    ):
        for j, k in ((1, 7), (2, 15)):

            def get(name, idx=j):
                return torch.tensor(expand_bf16(fc[f"c{idx}-{name}"]))

            x = torch.tensor(fc[f"c{j}-x"])
            n = torch.tensor(fc[f"c{j}-n"])
            r, z, skip, down = (get(name) for name in ("r", "z", "skip", "down"))
            assert torch.equal((skip.bfloat16() + down.bfloat16()).float(), r)
            with np.load(cache / f"k{k:02d}.npz") as f:
                take = np.searchsorted(f["idx"], np.asarray(panel) + 768)
                old = bank.prepare(
                    (expand_bf16(f["r"][take]), f["x"][take], f["n"][take]), ps
                )
            native = distribution(n).numpy()
            errors["frozen_old_vs_fresh"].append(
                compare(distribution(old["n"]).numpy()[0, :, None, :], native)
            )
            for s in (67301, 67302, 67303, 67304):
                for m in (0, 1):
                    for name in ("fixed", "border"):
                        key = f"s{s}-m{m}-record-{name}-c{j}"
                        model = InternalPolicy.load(
                            bank, study / f"s{s}/m{m}" / (name + ".npz")
                        )
                        first = (
                            distribution(model.scores(at(old, h)))
                            .numpy()
                            .transpose(1, 0, 2)
                        )
                        table = torch.einsum("hb,pbdr->phdr", gate, model.A)
                        delta = torch.einsum("nhd,nhdr->nhr", x, table[pp]) @ model.B
                        smooth = distribution(
                            n + ((normalize(r + delta) - normalize(r)) @ bank.head.T)
                        ).numpy()
                        quant = (skip.bfloat16() + (down + delta).bfloat16()).float()
                        scored = distribution(
                            n + (words(normalize(quant).bfloat16().float()) - words(z))
                        ).numpy()
                        full = actual[key]
                        for label, a, b in (
                            ("matched_smooth_old_vs_fresh", first, smooth),
                            ("fresh_smooth_vs_bf16", smooth, scored),
                            ("matched_bf16_vs_actual", scored, full),
                            ("old_smooth_vs_actual", first, full),
                        ):
                            errors[label].append(compare(a, b))
    report = {}
    for k, parts in errors.items():
        ar = np.concatenate(parts)
        report[k] = {
            "mean_max_action_error": float(ar.mean()),
            "p95": float(np.quantile(ar, 0.95)),
            "max": float(ar.max()),
            "count": int(ar.size),
            "over_one_point": int((ar > 0.01).sum()),
        }
    out.mkdir(exist_ok=True, parents=True)
    write(out / "transition-numerics.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("gpu", "cache", "arch", "study", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    run(a.gpu, a.cache, a.arch, a.study, a.output)
