"""Fresh-process execution-order audit on one L4, using separate compilation caches."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.consequence_policy.study import contexts
from experiments.direct_policy.protocol import write


def child(root, name):
    out = root / "output" / name
    out.mkdir(parents=True, exist_ok=True)
    model = FrozenQwen(root / "expected-weights.json", out)
    rows = contexts()
    prefixes = [model.encode(row) for row in rows]
    tested = [i for i, r in enumerate(rows) if r["split"] == "test"]
    groups = defaultdict(list)
    for i, ids in enumerate(prefixes):
        groups[len(ids)].append(i)
    batches = [
        group[start : start + 8]
        for length in sorted(groups)
        for group in [groups[length]]
        for start in range(0, len(group), 8)
    ]
    if name == "one_batch_first":
        model.batch([prefixes[i] for i in batches[0]])
    if name == "all_batches_first":
        for batch in batches:
            model.batch([prefixes[i] for i in batch])
    a, b = [], []
    for i in tested:
        hidden, logits = model.batch([prefixes[i]])
        a.append(hidden[0].astype(np.float16))
        b.append(logits[0])
    np.savez_compressed(
        out / "singleton.npz",
        hidden=np.asarray(a),
        logits=np.asarray(b),
        context=np.array(tested),
    )
    repeats = []
    for j in range(0, len(tested), 8):
        hidden, logits = model.batch([prefixes[tested[j]]])
        repeats.append(
            np.array_equal(logits[0], b[j]) and np.array_equal(hidden[0], a[j])
        )
    model.verify_unchanged()
    write(
        out / "check.json",
        {
            "repeated_exact": sum(bool(v) for v in repeats),
            "repeated": len(repeats),
            "first_batch_size": len(batches[0]),
            "first_batch_tokens": len(prefixes[batches[0][0]]),
            "first_test_tokens": len(prefixes[tested[0]]),
            "hardware": model.torch.cuda.get_device_name(),
        },
    )
    write(out / "status.json", {"stage": "complete"})


def run(root):
    out = root / "output"
    out.mkdir(exist_ok=True)
    names = (
        "singleton_first_a",
        "one_batch_first",
        "all_batches_first",
        "singleton_first_b",
    )
    start = time.monotonic()
    for name in names:
        write(out / "status.json", {"stage": "new_process", "condition": name})
        env = dict(
            os.environ,
            TRITON_CACHE_DIR=f"/tmp/cq-triton-{name}",
            TORCHINDUCTOR_CACHE_DIR=f"/tmp/cq-inductor-{name}",
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.consequence_policy.ordering",
                "--root",
                str(root),
                "--child",
                name,
            ],
            env=env,
            check=True,
            timeout=900,
        )
    data = {name: np.load(out / name / "singleton.npz") for name in names}
    comparisons = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            comparisons.append(
                {
                    "a": a,
                    "b": b,
                    "hidden_equal": bool(
                        np.array_equal(data[a]["hidden"], data[b]["hidden"])
                    ),
                    "logits_equal": bool(
                        np.array_equal(data[a]["logits"], data[b]["logits"])
                    ),
                    "hidden_rms": float(
                        np.sqrt(
                            np.mean(
                                (data[a]["hidden"].astype(float) - data[b]["hidden"])
                                ** 2
                            )
                        )
                    ),
                }
            )
    write(
        out / "receipt.json",
        {
            "mode": "ordering",
            "elapsed_seconds": time.monotonic() - start,
            "conditions": names,
            "comparisons": comparisons,
            "within_process": {
                n: json.loads((out / n / "check.json").read_bytes()) for n in names
            },
        },
    )
    write(out / "status.json", {"stage": "complete"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--child", required=True)
    args = p.parse_args()
    child(args.root, args.child)
