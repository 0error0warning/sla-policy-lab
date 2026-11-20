"""Technical repeat anchors: historical six-slot arrangement, canonical text IDs and client versions."""

import platform
from hashlib import sha256
from importlib.metadata import version

import numpy as np
import torch

from experiments.order_transfer.gpu import from_bits


def original_interface(backend, cases, root):
    with np.load(root / "phrase-weights.npz") as z:
        zeros_and_pairs = []
        for c in cases:
            zeros_and_pairs.append(
                (
                    from_bits(backend, z[c["pre_hash"]]),
                    from_bits(backend, z[c["pr15_hash"]]),
                )
            )

    def tokens():
        return [
            (
                i,
                sha256(
                    np.asarray(backend.inputs.ids[i], dtype="<i4").tobytes()
                ).hexdigest(),
                int(backend.inputs.slot[i]),
            )
            for i in sorted({i for c in cases for i, h in c["checks"][:3]})
        ]

    initial_tokens = tokens()

    def score():
        assert initial_tokens == tokens()
        all_scores = []
        with torch.no_grad():
            for c, (pre, baseline) in zip(cases, zeros_and_pairs, strict=True):
                backend.bind(pre, None, None)
                try:
                    weights = [
                        m.live()
                        for m in (pre, baseline, baseline, baseline, pre, baseline)
                    ]
                    scores = [
                        backend.ordinary(i, h, weights).cpu().numpy()[[0, 1]]
                        for i, h in c["checks"][:3]
                    ]
                finally:
                    backend.close()
                all_scores.append(scores)
        return np.array(all_scores)

    return score, initial_tokens


def previous_slot_check(backend, c, models):
    pre, original, aug, archive = models
    pairs = []
    with torch.no_grad():
        backend.bind(pre, None, None)
        try:
            for target in (original, archive):
                # PR #15 puts the original four-order model in row 2.
                weights = [m.live() for m in (pre, aug, target, original, pre, target)]
                pairs.append(
                    [
                        backend.ordinary(i, h, weights).cpu().numpy()[2]
                        for i, h in c["checks"][:3]
                    ]
                )
        finally:
            backend.close()
    return np.array(pairs)


def environment():
    device = torch.cuda.get_device_properties(0)
    return {
        "device": device.name,
        "compute_capability": [device.major, device.minor],
        "memory_bytes": device.total_memory,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "triton": version("triton"),
        "transformers": version("transformers"),
    }
