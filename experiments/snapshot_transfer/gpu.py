"""Read-only complete-model scoring of archived, unmodified individual adapters."""

import argparse
import json
import time
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.direct_policy.protocol import write
from experiments.first_fatal.engine import Qwen
from experiments.individual_life.model import correction
from experiments.internal_body.suffix import distribution
from experiments.snapshot_transfer.inputs import CONTEXTS


class PrecisionProbe:
    """Observe the native last-token computation without replacing any forward result."""

    def __init__(self, backend):
        self.backend = backend
        self.active = False
        self.data = {}
        self.handles = [
            backend.full.layer.register_forward_hook(self.down),
            backend.lm.norm.register_forward_pre_hook(self.residual),
            backend.actor.model.lm_head.register_forward_hook(self.head),
        ]

    def compare(self, name, tensor):
        v = tensor[3].float() - tensor[2].float()
        self.data[name + "_changed"] = int(torch.count_nonzero(v).item())
        self.data[name + "_norm"] = float(torch.linalg.vector_norm(v).item())

    def down(self, module, args, output):
        if self.active:
            full = self.backend.full
            x = args[0][:, -1].float() @ full.proj
            self.compare(
                "fp32_correction", correction(x, full.g, full.partner, full.weights)
            )
            self.compare("native_down", output[:, -1])

    def residual(self, module, args):
        if self.active:
            self.compare("native_residual", args[0][:, -1])

    def head(self, module, args, output):
        if self.active:
            hidden = args[0][:, -1]
            self.compare("native_hidden", hidden)
            ids = self.backend.actor.ids
            # Re-score only the same four token IDs, keeping the actual BF16 hidden.
            assert module.bias is None
            alt = hidden.float() @ module.weight[ids].float().T
            self.data["fp32_head"] = distribution(alt)[2:4].cpu().tolist()
            self.data["native_logits"] = output[:, -1, ids][2:4].float().cpu().tolist()

    def close(self):
        for h in self.handles:
            h.remove()


def load_adapters(backend, root):
    models = {}
    with (
        np.load(root / "transfer-weights.npz", allow_pickle=False) as z,
        torch.no_grad(),
    ):
        for key in z.files:
            data = z[key]
            assert sha256(data.tobytes()).hexdigest() == key
            m = backend.new("blank")
            n = m.A.numel()
            for param, values in zip(m.live(), (data[:n], data[n:]), strict=True):
                param.copy_(
                    torch.as_tensor(values.view("<f4"), device=param.device).view_as(
                        param
                    )
                )
            np.testing.assert_array_equal(m.bits(), data)
            models[key] = m
    return models


def run(root):
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    case_file = json.loads((root / "transfer-cases.json").read_bytes())
    cases = case_file["cases"]
    assert len(cases) == 48 and tuple(case_file["contexts"]) == CONTEXTS
    actor = FrozenQwen(root / "expected-weights.json", out)
    backend = Qwen(actor, root)
    params = sum(p.numel() for p in actor.model.parameters())
    models = load_adapters(backend, root)
    backend.bind(next(iter(models.values())), None, None)
    probe = PrecisionProbe(backend)
    results = []
    repeats = []
    diagnostics = []
    try:
        with torch.no_grad():
            for n, c in enumerate(cases, 1):
                copies = [models[key].live() for key in c["weights"]]
                values = []
                diag = []
                for step, (i, h) in enumerate(c["checks"]):
                    probe.active = step < 3
                    probe.data = {}
                    result = backend.ordinary(i, h, copies).cpu().numpy()
                    np.testing.assert_array_equal(result[1], result[5])
                    if probe.active:
                        diag.append(probe.data)
                    values.append(result)
                probe.active = False
                repeat = []
                for i, h in reversed(c["checks"][:3]):
                    q = backend.ordinary(i, h, copies[::-1]).cpu().numpy()[::-1]
                    repeat.append(q.copy())
                results.append(values)
                repeats.append(repeat[::-1])
                diagnostics.append(diag)
                if n % 8 == 0:
                    write(
                        out / "status.json",
                        {"stage": "scoring_snapshots", "completed": n, "total": 48},
                    )
    finally:
        probe.close()
        backend.close()
    actor.verify_unchanged()
    assert params == sum(v.numel() for v in actor.model.parameters())
    for key, m in models.items():
        assert sha256(m.bits().tobytes()).hexdigest() == key
    result = np.array(results)
    repeat = np.array(repeats)
    assert result.shape == (48, 6, 6, 4) and repeat.shape == (48, 3, 6, 4)
    np.savez_compressed(out / "probabilities.npz", primary=result, repeat=repeat)
    write(out / "precision.json", diagnostics)
    with np.load(root / "transfer-anchors.npz") as z:
        original = np.abs(result[:, 0, 2] - z["original_before"]).max(axis=1)
        successor = np.abs(result[:, 2] - z["successor"]).max(axis=(1, 2))
    repeat_error = np.abs(result[:, :3] - repeat).max(axis=(2, 3))
    write(
        out / "receipt.json",
        {
            "mode": "stmap",
            "cases": len(cases),
            "training_steps": 0,
            "ordinary_forwards": backend.stats["full_forwards"],
            "gpu": torch.cuda.get_device_name(),
            "elapsed_seconds": time.monotonic() - start,
            "peak_bytes": torch.cuda.max_memory_allocated(),
            "backbone_unchanged": True,
            "snapshot_hashes_after": len(models),
            "params": params,
            "original_exact": int((original == 0).sum()),
            "original_max": float(original.max()),
            "successor_exact": int((successor == 0).sum()),
            "successor_max": float(successor.max()),
            "repeat_exact": int((repeat_error == 0).sum()),
            "repeat_max": float(repeat_error.max()),
            "versions": {
                k: version(k) for k in ("torch", "transformers", "numpy", "scipy")
            },
        },
    )
    write(out / "status.json", {"stage": "complete", "completed": len(cases)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stmap"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
