"""Inventory learned arrays and native BF16 effective changes for the reported cohorts."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.internal_body.design import CONFIRM


def bf16(a):
    x = np.ascontiguousarray(a, dtype="<f4").view("<u4")
    return ((x + np.uint32(0x7FFF) + ((x >> 16) & 1)) >> 16).astype("<u2")


def main():
    p = argparse.ArgumentParser()
    for name in ("cache", "population", "native", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    initial = (
        np.random.default_rng(64800).normal(0, 0.06, (2, 12, 33, 4)).astype(np.float32)
    )
    with np.load(a.cache / "architecture.npz") as z:
        w = z["norm"]
    records = []
    for seed in CONFIRM:
        for m in (0, 1):
            folder = a.population / f"s{seed}-m{m}"
            for mode in (
                "linear",
                "postnorm",
                "nonlinear",
                "no_route",
                "blind",
                "shuffle",
                "zero",
                "maintain",
                "high",
                "prompt_lora",
            ):
                with np.load(folder / (mode + ".npz")) as z:
                    A, B = z["A"], z["B"]
                records.append(
                    {
                        "seed": seed,
                        "mapping": m,
                        "mode": mode,
                        "A_shape": list(A.shape),
                        "B_shape": list(B.shape),
                        "learned_elements": A.size + B.size,
                        "A_l2_delta": float(np.linalg.norm(A - initial)),
                        "B_l2_delta": float(np.linalg.norm(B)),
                        "A_changed": int(np.count_nonzero(A != initial)),
                        "B_changed": int(np.count_nonzero(B)),
                        "B_sha256": sha256(B.tobytes()).hexdigest(),
                    }
                )
    original = []
    for seed in range(64501, 64521):
        for m in (0, 1):
            for name in ("sensor", "shuffle", "hidden", "text"):
                with np.load(a.native / f"s{seed}-m{m}" / f"native-{name}.npz") as z:
                    delta = z["delta"]
                effective = bf16(w + delta)
                original.append(
                    {
                        "seed": seed,
                        "mapping": m,
                        "mode": name,
                        "path": "model.language_model.norm.weight",
                        "shape": list(w.shape),
                        "fp32_master_delta_l2": float(np.linalg.norm(delta)),
                        "changed_effective_bf16": int(
                            np.count_nonzero(effective != bf16(w))
                        ),
                        "original_bf16_sha256": sha256(bf16(w).tobytes()).hexdigest(),
                        "effective_bf16_sha256": sha256(
                            effective.tobytes()
                        ).hexdigest(),
                    }
                )
    write(
        a.output,
        {
            "internal": records,
            "original_scale": original,
            "parameters": {
                "output_reference5": 3168,
                "internal_conditional": 13408,
                "original_norm": 2560,
                "personal_unrestricted": 76800,
                "personal_tied": 38400,
                "personal_supplied_direction": 2,
                "matched_nuisance_route": 76800,
            },
        },
    )
    print(len(records), "internal, ", len(original), "native snapshots")


if __name__ == "__main__":
    main()
