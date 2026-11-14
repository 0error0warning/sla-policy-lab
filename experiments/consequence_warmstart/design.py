"""Locked CPU-only initialization comparison; no population fitting or GPU code."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.consequence_policy.lifetime import environment_tape
from experiments.direct_policy.protocol import write

BIRTH_SEEDS = tuple(range(63001, 63005))
ENV_SEEDS = tuple(range(78201, 78205))
ARMS = ("prior", "personal", "warmstart_adapter", "scratch")
BLOCKS = {"legacy": ((4, 2), (5, 2)), "wording": ((0, 0), (0, 1))}
CHECKPOINTS = (16, 32, 48)
TIME_LIMIT = 3600


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def code_digest(path):
    # Git checks this repository out with CRLF on Windows and LF on Linux.
    return sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def selected(rows, task_wording):
    task, wording = task_wording
    return np.array(
        [i for i, r in enumerate(rows) if (r["task"], r["wording"]) == (task, wording)]
    )


def tape(seed, world, block, rows):
    index = tuple(BLOCKS).index(block)
    ids, uniforms, noise = environment_tape(seed, world, index, rows)
    live = BLOCKS[block][0]
    lookup = {
        (rows[i]["partner"], rows[i]["frame"], rows[i]["layout"]): i
        for i in selected(rows, live)
    }
    ids = np.array(
        [lookup[rows[i]["partner"], rows[i]["frame"], rows[i]["layout"]] for i in ids]
    )
    return ids, uniforms, noise


def permutation(trace, rows, seed, world, block, count):
    from experiments.direct_policy.protocol import rng

    prefix = trace[:count]
    p = np.array([rows[int(i)]["partner"] for i in prefix[:, 0]])
    bands = np.digitize(prefix[:, 1], [6.0, 15.0])
    order = np.arange(count)
    gen = rng(seed, "warmstart-prefix-shuffle-v1", list(world), block, count)
    groups = []
    for partner in (0, 1):
        for band in range(3):
            ids = np.flatnonzero((p == partner) & (bands == band))
            order[ids] = gen.permutation(ids)
            groups.append({"partner": partner, "band": band, "count": len(ids)})
    return order, {
        "groups": groups,
        "fixed_points": int((order == np.arange(count)).sum()),
    }


def lock(repo, source, output):
    paths = [
        "capture/output/qwen-states.npz",
        "capture/output/contexts.json",
        "capture/output/receipt.json",
        "capture/output/weights-manifest.json",
        "results/projection.npz",
    ]
    paths += [f"results/s{s}-m{m}/real.npz" for s in BIRTH_SEEDS for m in (0, 1)]
    paths += [
        f"results/life-{s}{ext}"
        for s in BIRTH_SEEDS
        for ext in (".json", "-traces.npz", "-scratch.npz")
    ]
    code = [
        "experiments/consequence_policy/lifetime.py",
        "experiments/consequence_policy/policy.py",
        "experiments/consequence_policy/study.py",
        "experiments/consequence_policy/experiment.py",
        "experiments/direct_policy/protocol.py",
        "experiments/social_closed_loop/core.py",
    ] + [
        p.relative_to(repo).as_posix()
        for p in sorted((repo / "experiments/consequence_warmstart").glob("*"))
        if p.suffix in (".py", ".md")
    ]
    record = {
        "schema": "warmstart-cpu-development-v1",
        "birth_seeds": BIRTH_SEEDS,
        "environment_seeds": ENV_SEEDS,
        "arms": ARMS,
        "blocks": BLOCKS,
        "wall_seconds": TIME_LIMIT,
        "max_autonomous_lives": 128,
        "max_autonomous_encounters": 32768,
        "gpu_allocations": 0,
        "source_files": {p: digest(source / p) for p in paths},
        "code_hash_encoding": "bytes with CRLF normalized to LF",
        "code_files": {p: code_digest(repo / p) for p in code},
        "archived_manifest": digest(
            repo / "artifacts/model-studies/2026-11-12/consequence-policy/MANIFEST.json"
        ),
    }
    write(output, record)
    return record


def verify_lock(repo, source, path):
    record = json.loads(path.read_bytes())
    for key, root in (("source_files", source), ("code_files", repo)):
        for name, expected in record[key].items():
            calculate = code_digest if key == "code_files" else digest
            if calculate(root / name) != expected:
                raise ValueError(f"Locked file changed: {name}")
    return record
