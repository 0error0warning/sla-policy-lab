"""Check the small transition contrast on independently installed full-Qwen checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.study import contexts
from experiments.direct_policy.protocol import write
from experiments.internal_body.suffix import InternalPolicy, distribution
from experiments.lethal_boundary.design import HEALTH, heldout_panel
from experiments.lethal_boundary.evaluate import BANDS, index, summarize
from experiments.lethal_boundary.physics import maintenance_table
from experiments.lethal_boundary.states import Bank, at, expand_bf16


def run(gpu, cache, arch, study, ref, out):
    receipt = json.loads((gpu / "receipt.json").read_bytes())
    assert (
        receipt["backbone_restored"] and receipt["result"]["disabled_native_identical"]
    )
    h = np.asarray(receipt["result"]["health"])
    np.testing.assert_array_equal(h, HEALTH[BANDS["transition"]])
    panel = heldout_panel()
    rows = [contexts()[768 + i] for i in panel]
    pids = np.array([r["partner"] for r in rows])
    bank = Bank(cache, arch)
    with np.load(ref / "physical.npz") as r:
        dv = r["conditional_death"][index(HEALTH, h)]
    conditions = {}
    drifts = []
    vectors = 0
    with np.load(gpu / "full-probes.npz") as real:
        for j, k in ((1, 7), (2, 15)):
            with np.load(cache / f"k{k:02d}.npz") as base:
                ix = np.searchsorted(base["idx"], np.asarray(panel) + 768)
                triplet = expand_bf16(base["r"][ix]), base["x"][ix], base["n"][ix]
            grid = at(bank.prepare(triplet, pids), h)
            for s in range(67301, 67305):
                for m in (0, 1):
                    q = maintenance_table(rows, m, k / 20)
                    risk = (1 - q[None]) * dv[:, :, 0][:, None] + q[None] * dv[:, :, 1][
                        :, None
                    ]
                    for arm in ("fixed", "border"):
                        w = study / f"s{s}/m{m}" / (arm + ".npz")
                        model = InternalPolicy.load(bank, w)
                        name = f"s{s}-m{m}-record-{arm}-c{j}"
                        with torch.no_grad():
                            old = distribution(model.scores(grid)).numpy()
                        pi = real[name].transpose(1, 0, 2)
                        assert pi.shape == old.shape == (16, 24, 4)
                        np.testing.assert_allclose(pi.sum(-1), 1, atol=3e-7, rtol=0)
                        drifts.append(np.abs(pi - old).max(-1).ravel())
                        vectors += len(h) * len(panel)
                        vals = {}
                        for tag, x in (("actual", pi), ("acquisition", old)):
                            death = (x * risk).sum(-1)
                            answer_death = (x[:, :, 2:] * risk[:, :, 2:]).sum(-1)
                            vals[tag] = {
                                "death": float(death.mean()),
                                "answer_and_die": float(answer_death.mean()),
                                "safe_task": float(
                                    (x[:, :, 2:].sum(-1) - answer_death).mean()
                                ),
                            }
                        conditions[name] = vals
    assert vectors == receipt["result"]["decisions"] == 12288 and len(conditions) == 32
    paired = {}
    for path in ("actual", "acquisition"):
        r = []
        for s in range(67301, 67305):
            one = []
            for m in (0, 1):
                for j in (1, 2):
                    a, b = (
                        conditions[f"s{s}-m{m}-record-{k}-c{j}"][path]
                        for k in ("fixed", "border")
                    )
                    one.append(
                        [a["death"] - b["death"], b["safe_task"] - a["safe_task"]]
                    )
            r.append(np.mean(one, axis=0))
        paired[path] = {
            "death_reduction": summarize(np.array(r)[:, 0]),
            "safe_task_gain": summarize(np.array(r)[:, 1]),
            "seed_pairs": np.asarray(r).tolist(),
        }
    drift = np.concatenate(drifts)
    payload = {
        "seeds": list(range(67301, 67305)),
        "observed_record_counts": [7, 15],
        "physical_q": [0.35, 0.75],
        "health": h.tolist(),
        "decisions": vectors,
        "actual_checkpoints": receipt["result"]["checkpoints"],
        "paired": paired,
        "max_independent_probability_difference": float(drift.max()),
        "p95_independent_probability_difference": float(np.quantile(drift, 0.95)),
        "disabled_original_identical": True,
        "raw_panel_metrics": conditions,
    }
    out.mkdir(exist_ok=True, parents=True)
    write(out / "transition-hardware.json", payload)
    print(vectors, paired["actual"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("gpu", "cache", "arch", "study", "ref", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    run(a.gpu, a.cache, a.arch, a.study, a.ref, a.output)
