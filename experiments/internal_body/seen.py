"""Rescore the exact 256 seen state entries in each small-budget direct training run."""

import numpy as np
import torch

from experiments.consequence_policy.study import KL
from experiments.direct_policy.protocol import write
from experiments.internal_body.capture import encode
from experiments.internal_body.depth import SITES, Site, forward
from experiments.internal_body.design import SPLIT_H, all_rows
from experiments.internal_body.suffix import distribution


def run(actor, root, out):
    rows = all_rows()[:960]
    ids = [encode(actor, r) for r in rows]
    summary = []
    for seed in range(64211, 64215):
        data = np.load(root / f"seen-{seed}.npz")
        ix, hs = data["contexts"], data["health_slots"]
        for m in (0, 1):
            results = {}
            pis = {}
            for name in ("frozen", *SITES):
                site = None
                if name != "frozen":
                    with np.load(root / f"projection-{seed}.npz") as z:
                        projection = z[f"l{SITES[name][0]}"]
                    site = Site(actor, name, projection)
                    with (
                        np.load(root / f"s{seed}-m{m}" / (name + ".npz")) as z,
                        torch.no_grad(),
                    ):
                        site.A.copy_(torch.tensor(z["A"], device="cuda"))
                        site.B.copy_(torch.tensor(z["B"], device="cuda"))
                try:
                    all_pi = []
                    with torch.no_grad():
                        for ids8, h in zip(ix, hs):
                            chunks = []
                            for j in (0, 4):
                                ii = ids8[j : j + 4]
                                hh = h[j : j + 4]
                                if site is not None:
                                    site.condition(
                                        [rows[i]["partner"] for i in ii],
                                        [SPLIT_H[k] for k in hh],
                                    )
                                chunks.append(
                                    distribution(forward(actor, ids, ii)).cpu().numpy()
                                )
                            all_pi.append(np.concatenate(chunks))
                    pi = np.stack(all_pi)
                    pis[name] = pi
                    base = pis["frozen"]
                    diff = (pi - base) * data[f"scores_m{m}"]
                    kl = np.mean(np.sum(pi * np.log(pi / base), axis=-1))
                    gain = np.mean(diff.sum(axis=-1))
                    answer = pi[:, :, 2:].sum(-1)
                    low = np.isin(hs, [0, 1])
                    high = np.isin(hs, [3, 4, 6, 7])
                    results[name] = {
                        "chosen_tape_gain": float(gain),
                        "kl_to_frozen": float(kl),
                        "net_objective_gain": float(gain - KL * kl),
                        "low_answer": float(answer[low].mean()),
                        "healthy_answer": float(answer[high].mean()),
                        "max_single_probability_change": float(np.abs(pi - base).max()),
                    }
                finally:
                    if site is not None:
                        site.close(actor)
            key = f"s{seed}-m{m}"
            np.savez_compressed(out / (key + ".npz"), **pis)
            write(out / (key + ".json"), results)
            summary.append(key)
            write(out / "status.json", {"stage": "seen_state_rescore", "world": key})
    return {
        "evaluated_worlds": summary,
        "seen_states_per_fit": 256,
        "site_conditions": 4,
        "matched_full_qwen_forward": True,
    }
