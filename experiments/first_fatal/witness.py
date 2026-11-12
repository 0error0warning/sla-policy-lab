"""Restore saved, pre-feedback adapters on a fresh complete model and re-score the first request."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.first_fatal.analysis import records


def prepare(roots, out):
    out.mkdir(parents=True, exist_ok=True)
    weights = {}
    cases = []
    expect = []
    paths = sorted(f for root in roots for f in root.glob("s*/*/*/life.json"))
    for j, f in enumerate(paths):
        meta = json.loads(f.read_bytes())
        d = f.parent
        if meta["first_fatal"] is None:
            continue
        ev = records(d)
        first = ev[meta["first_fatal"] - 1]
        assert first["first_fatal"]
        roles = [
            "birth",
            "first",
            "first_wrong" if meta["rule"] == "true1" else "birth",
            "first_near_wrong" if meta["rule"] == "true1" else "birth",
            "entry",
            "first",
        ]
        keys = []
        with np.load(d / "parameters.npz") as z:
            for name in roles:
                data = z[name]
                key = sha256(data.tobytes()).hexdigest()
                assert key == meta["snapshot_hashes"][name]
                weights[key] = data
                keys.append(key)
        checks = []
        with np.load(d / "predecision.npz") as z:
            idx = z["indices"]
            for n, h in enumerate(z["health"]):
                probe = (j + 32 * n) % len(idx)
                checks.extend([[first["index"], float(h)], [int(idx[probe]), float(h)]])
                expect.extend([z["same_request"][n], z["panel"][n, probe]])
        cases.append(
            {
                "seed": meta["seed"],
                "birth": meta["birth"],
                "rule": meta["rule"],
                "turn": first["turn"],
                "u": first["u"][0],
                "weights": keys,
                "checks": checks,
            }
        )
    assert cases
    write(out / "witness-cases.json", cases)
    np.savez_compressed(out / "witness-weights.npz", **weights)
    np.savez_compressed(out / "witness-expected.npz", arrays=np.stack(expect))


def run(backend, root, out):
    import torch

    cases = json.loads((root / "witness-cases.json").read_bytes())
    models = {}
    values = []
    with np.load(root / "witness-weights.npz") as z, torch.no_grad():
        for key in z.files:
            data = z[key]
            assert sha256(data.tobytes()).hexdigest() == key
            model = backend.new("blank")
            n = model.A.numel()
            for param, v in zip(model.live(), [data[:n], data[n:]], strict=True):
                param.copy_(
                    torch.as_tensor(v.view("<f4"), device=param.device).view_as(param)
                )
            models[key] = model
    backend.bind(next(iter(models.values())), None, None)
    try:
        with torch.no_grad():
            for j, case in enumerate(cases, 1):
                weights = [models[key].live() for key in case["weights"]]
                for i, h in case["checks"]:
                    values.append(backend.ordinary(i, h, weights).cpu().numpy())
                if j % 12 == 0:
                    write(
                        out / "status.json",
                        {"stage": "restored_first_decisions", "cases": j},
                    )
    finally:
        backend.close()
    got = np.stack(values)
    with np.load(root / "witness-expected.npz") as z:
        expected = z["arrays"]
    diff = np.abs(got - expected).max(axis=(1, 2))
    np.savez_compressed(out / "restored-probabilities.npz", arrays=got)
    return {
        "restored_lives": len(cases),
        "complete_panels": len(values),
        "exact_panels": int((diff == 0).sum()),
        "max_error": float(diff.max()),
        "p95": float(np.quantile(diff, 0.95)),
        "training_steps": 0,
    }


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--lives", nargs="+", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    v = a.parse_args()
    prepare(v.lives, v.output)
