"""Exploratory post-run description of action collapse and direction transfer. No model fitting."""

import argparse
import json
from pathlib import Path
import numpy as np
from .analysis import csv_out
from .design import PILOT, ROLES


def examine(root, dest):
    rows = []
    direction = []
    with np.load(root / "steering" / "donors.npz") as z:
        donors = np.stack([z[f"s{s}"] for s in PILOT])
    for idx, seed in enumerate(PILOT):
        path = root / "histories" / f"s{seed}" / "stable"
        p = (
            np.load(path / "final-panel.npy")[:, 1, 2:]
            .astype(float)
            .reshape(len(ROLES), -1, 4)
        )
        logs = [
            json.loads(x) for x in (path / "updates.jsonl").read_bytes().splitlines()
        ]
        tape = np.asarray([e["probabilities"] for e in logs])
        others = np.delete(donors, idx, axis=0)
        avg = others.mean(0)
        norm = np.linalg.norm(avg)
        denom = norm * np.linalg.norm(donors[idx])
        direction.append(
            dict(
                seed=seed,
                direction_norm=norm,
                alpha=norm / 2,
                direction_coherence=norm / np.linalg.norm(others, axis=1).mean(),
                cosine_with_held_seed=(
                    float(avg @ donors[idx] / denom) if denom else ""
                ),
            )
        )
        for j, name in enumerate(ROLES):
            dist = p[j]
            top = dist.argmax(axis=1)
            counts = np.bincount(top, minlength=4)
            tv = np.abs(tape[:, 0] - tape[:, j]).sum(1) / 2
            high = (tape[:, j].max(1) >= 0.9) & (tv >= 0.3)
            first = next(
                (k + 1 for k in range(len(logs) - 2) if high[k : k + 3].all()), ""
            )
            rows.append(
                dict(
                    seed=seed,
                    method=name,
                    healthy_matched_cells=len(dist),
                    max_probability_mean=float(dist.max(1).mean()),
                    modal_action=int(counts.argmax()),
                    cells_with_modal_top=int(counts.max()),
                    cells_with_top_over_90=int((dist.max(1) > 0.9).sum()),
                    first_three_high_probability_divergent_requests=first,
                )
            )
    dest.mkdir(exist_ok=True, parents=True)
    csv_out(dest / "action-concentration.csv", rows)
    csv_out(dest / "direction-alignment.csv", direction)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    examine(a.root, a.out)
