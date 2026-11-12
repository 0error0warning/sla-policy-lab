"""Compare installed Qwen with a BF16 reconstruction and with independently acquired prefix probes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.study import basis, contexts
from experiments.direct_policy.protocol import write
from experiments.internal_body.suffix import distribution
from experiments.lethal_boundary.design import heldout_panel
from experiments.lethal_boundary.states import WITNESS_H, WITNESS_K, expand_bf16


def run(gpu, arch, trained, out):
    receipt = json.loads((gpu / "receipt.json").read_bytes())
    assert (
        receipt["backbone_restored"] and receipt["result"]["disabled_native_identical"]
    )
    with np.load(arch / "architecture.npz") as z:
        head = torch.from_numpy(z["head"])
        norm = torch.from_numpy(z["norm"])
    cfg = json.loads((arch / "architecture.json").read_bytes())
    ix = heldout_panel()
    rows = contexts()[768:]
    partner = torch.as_tensor([rows[i]["partner"] for i in ix])
    g = torch.as_tensor(basis(WITNESS_H), dtype=torch.float32)

    def words(z):
        return (z @ head.T).bfloat16().float()

    errors = {}
    group = {}
    expected_n = 0
    extreme = 0.0
    oldmax = 0.0
    allerr = []
    entry_counts = {}
    with (
        np.load(gpu / "fresh-components.npz") as fc,
        np.load(gpu / "full-probes.npz") as fp,
    ):
        np.testing.assert_array_equal(fc["indices"], ix)
        for j, count in enumerate((None, 7, 15)):

            def native(k, batch=j):
                return torch.from_numpy(expand_bf16(fc[f"c{batch}-{k}"]))

            x = torch.tensor(fc[f"c{j}-x"])
            n = torch.tensor(fc[f"c{j}-n"])
            skip, down, z0, r0 = [native(k) for k in ("skip", "down", "z", "r")]
            assert torch.equal((skip.bfloat16() + down.bfloat16()).float(), r0)

            def normalized(r):
                return (
                    (
                        r
                        * torch.rsqrt(
                            r.square().mean(-1, keepdim=True) + cfg["epsilon"]
                        )
                        * (norm + cfg["norm_offset"])
                    )
                    .bfloat16()
                    .float()
                )

            normerr = float((normalized(r0) - z0).abs().max())
            sense = "plain" if count is None else "record"
            for seed in (67101, 67102):
                for m in (0, 1):
                    for arm in ("R", "D", "M"):
                        folder = trained / f"s{seed}/m{m}"
                        k = f"s{seed}-m{m}-{sense}-{arm}-c{j}"
                        with np.load(folder / (sense + "-" + arm + ".npz")) as weights:
                            A, B = (
                                torch.from_numpy(weights["A"]),
                                torch.from_numpy(weights["B"]),
                            )
                        table = torch.einsum("hb,pbdr->phdr", g, A)
                        coeff = torch.einsum("nhd,nhdr->nhr", x, table[partner])
                        r = (skip.bfloat16() + (down + coeff @ B).bfloat16()).float()
                        approx = distribution(
                            n + (words(normalized(r)) - words(z0))
                        ).numpy()
                        measured = fp[k]
                        with np.load(folder / (sense + "-" + arm + "-probe.npz")) as z:
                            which = (
                                0
                                if count is None
                                else np.flatnonzero(WITNESS_K == count)[0]
                            )
                            old = np.take(z["panel"][:, which], ix, axis=1).transpose(
                                1, 0, 2
                            )
                        assert measured.shape == approx.shape == old.shape == (24, 8, 4)
                        np.testing.assert_allclose(
                            measured.sum(-1), 1.0, atol=3e-7, rtol=0
                        )
                        err = np.abs(measured - approx).max(-1)
                        drift = np.abs(measured - old).max(-1)
                        extreme = max(extreme, float(err.max()))
                        oldmax = max(oldmax, float(drift.max()))
                        allerr.append(drift.ravel())
                        errors[k] = {
                            "same_process_max": float(err.max()),
                            "same_process_p95": float(np.quantile(err, 0.95)),
                            "acquisition_to_deployment_max": float(drift.max()),
                            "acquisition_to_deployment_p95": float(
                                np.quantile(drift, 0.95)
                            ),
                            "max_frozen_norm_reconstruction": normerr,
                        }
                        for mode, s in (("actual", measured), ("acquisition", old)):
                            ans = s[:, :, 2:].sum(-1).mean(0)
                            group.setdefault(
                                f"{sense}-{arm}-{count}-{mode}", []
                            ).append(ans)
                        expected_n += measured.shape[0] * measured.shape[1]
                        entry_counts[sense] = (
                            entry_counts.get(sense, 0)
                            + measured.shape[0] * measured.shape[1]
                        )
    assert expected_n == receipt["result"]["decisions"] == 6912 and len(errors) == 36
    summary = {k: np.mean(rows, axis=0).tolist() for k, rows in group.items()}
    checks = {}
    for count in (7, 15):
        name = f"record-M-{count}-actual"
        r = f"record-R-{count}-actual"
        checks[str(count)] = {
            "fatal_reduction_at_first_two_healths": float(
                np.mean(np.array(summary[r])[:2] - np.array(summary[name])[:2])
            ),
            "healthy_answer_change": float(
                np.mean(np.array(summary[name])[-2:] - np.array(summary[r])[-2:])
            ),
        }
    out.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        out / "matched-panel.npz",
        health=WITNESS_H,
        **{k: np.stack(rows) for k, rows in group.items()},
    )
    data = {
        "seeds": [67101, 67102],
        "decisions": expected_n,
        "by_interface": entry_counts,
        "installations": receipt["result"]["checkpoints"],
        "same_process_max_probability_disagreement": extreme,
        "cross_process_max_probability_disagreement": oldmax,
        "cross_process_p95_probability_disagreement": float(
            np.quantile(np.concatenate(allerr), 0.95)
        ),
        "disabled_model_exact": True,
        "per_panel": errors,
        "mean_answer_by_health": summary,
        "matched_small_panel_contrasts": checks,
    }
    write(out / "hardware.json", data)
    print(
        json.dumps(
            {
                k: data[k]
                for k in (
                    "decisions",
                    "same_process_max_probability_disagreement",
                    "cross_process_max_probability_disagreement",
                    "matched_small_panel_contrasts",
                )
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("gpu", "arch", "trained", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(3)
    run(a.gpu, a.arch, a.trained, a.output)
