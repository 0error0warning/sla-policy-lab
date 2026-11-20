"""Paired real Qwen training on actual personal histories, followed by leave-one-seed-out steering."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.consequence_policy.gpu import FrozenQwen
from experiments.direct_policy.protocol import write
from experiments.first_fatal.engine import NumericalMismatch
from .adapters import signature
from .analysis import analyze_replay
from .design import (
    ITEMS,
    LEVELS,
    PANEL,
    PILOT,
    ROLES,
    WORLDS,
    MODES,
    INTERVENTIONS,
    orientation,
    leave_one_out,
    tape_inputs,
)
from .engine import Engine, People


def save_events(path, events):
    with path.open("w", encoding="utf8", newline="\n") as f:
        for e in events:
            f.write(
                json.dumps(e, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )


@torch.no_grad()
def panel(engine):
    result = np.empty((len(ROLES), 2, 4, 32, 4), dtype=np.float32)
    for w, indices in enumerate(PANEL):
        for hi, h in enumerate(LEVELS):
            for j, i in enumerate(indices):
                p = engine.probe(int(i), h, check=j == (2 * hi + w))
                result[:, w, hi, j] = p.cpu().numpy()
    return result


def replay_one(engine, folder, events):
    folder.mkdir(parents=True, exist_ok=False)
    person = People(engine.q)
    fixed = person.state()
    np.savez_compressed(folder / "birth.npz", **fixed)
    engine.bind(person)
    try:
        birth = panel(engine)
        assert np.array_equal(birth, np.broadcast_to(birth[0], birth.shape))
        drift = 0.0
        actual_birth_diff = 0.0
        orth = 0.0
        checked = 0
        logs = []
        max_step = {r: 0.0 for r in ROLES[1:]}
        for k, e in enumerate(events):
            # Actual replays use a fresh complete Qwen forward with all six actors.
            p = engine.decision(
                e["row"], e["health"], check=k in (0, 23, len(events) - 1)
            )
            pp = p.detach().cpu().numpy()
            if k == 0:
                assert np.array_equal(pp, np.broadcast_to(pp[0], pp.shape))
            actual_birth_diff = max(
                actual_birth_diff, float(abs(pp[0] - np.array(e["prob"][1])).max())
            )
            drift = max(drift, float(abs(pp[1] - np.array(e["prob"][0])).max()))
            log = engine.update(p, e)
            for name, update in log.items():
                checked += 1
                orth = max(orth, update["orthogonality_error"])
                max_step[name] = max(max_step[name], update["actual_l2"])
            logs.append(
                dict(
                    turn=e["turn"],
                    row=e["row"],
                    health=e["health"],
                    after=e["after"],
                    action=e["action"],
                    wrong_action=e["wrong_action"],
                    signal=e["signal"],
                    reward=e["reward"],
                    maintenance=e["maintenance"],
                    origin_behavior=e["prob"][0],
                    probabilities=pp.tolist(),
                    updates=log,
                )
            )
            if k == 23:
                np.savez_compressed(folder / "at24.npz", **person.state())
        np.savez_compressed(folder / "last-living.npz", **person.state())
        np.save(folder / "birth-panel.npy", birth[0], allow_pickle=False)
        result = panel(engine)
        assert np.array_equal(result[0], birth[0])
        np.save(folder / "final-panel.npy", result, allow_pickle=False)
        save_events(folder / "updates.jsonl", logs)
        write(
            folder / "receipt.json",
            dict(
                turns=len(events),
                updates=checked,
                birth_max_wrt_archive=actual_birth_diff,
                native_max_wrt_archive=drift,
                orthogonality_max=orth,
                actual_step_max=max_step,
                final_hashes={k: signature(v) for k, v in person.state().items()},
            ),
        )
    finally:
        engine.close()
    return person


@torch.no_grad()
def donor_vectors(engine, people):
    means = {}
    for seed in PILOT:
        stable, changed = [people[seed, w] for w in WORLDS]
        engine.bind(stable)
        rows = []
        try:
            for i in PANEL[0]:
                row = ITEMS[i]
                if row["partner"] == 2 and row["task"] == 4:
                    _, (a, b) = engine.learned_pair(
                        int(i), 90.0, (stable.loreft, changed.loreft)
                    )
                    rows.append(
                        orientation(seed, row) * (a - b).cpu().numpy().astype(float)
                    )
            assert len(rows) == 8
            means[seed] = np.mean(rows, axis=0)
        finally:
            engine.close()
    return means


@torch.no_grad()
def steer_one(engine, folder, seed, people, means):
    folder.mkdir(parents=True, exist_ok=False)
    st, ch = [people[seed, w] for w in WORLDS]
    unit, alpha, controls = leave_one_out(means, seed)
    dirs = torch.tensor(np.vstack([unit, controls]), device="cuda", dtype=torch.float32)
    np.savez_compressed(
        folder / "directions.npz",
        unit=unit,
        controls=controls,
        alpha=np.array(alpha),
        seeds=[s for s in PILOT if s != seed],
    )
    probs = np.empty((2, len(INTERVENTIONS), 2, 4, 32, 4), dtype=np.float32)
    engine.bind(st)
    try:
        for w, indices in enumerate(PANEL):
            for k, h in enumerate(LEVELS):
                for j, i in enumerate(indices):
                    for a in range(2):
                        probs[a, :, w, k, j] = (
                            engine.steering(
                                int(i),
                                h,
                                (st.loreft, ch.loreft),
                                a,
                                dirs,
                                alpha,
                                check=j == (w + k),
                            )
                            .cpu()
                            .numpy()
                        )
    finally:
        engine.close()
    np.save(folder / "probabilities.npy", probs, allow_pickle=False)
    return alpha


def run(root, mode):
    start = time.monotonic()
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    for f, h in {
        "architecture.npz": "0a555b686b89724a85600e2e7d5cbb73d51222700beae8c069355897ed482232",
        "birth-source.npz": "95a68bf40d4b56beb127f92b696040906b141aab63c84e030ae2c8734d27fd40",
    }.items():
        assert sha256((root / f).read_bytes()).hexdigest() == h
    actor = FrozenQwen(root / "expected-weights.json", out)
    count = sum(p.numel() for p in actor.model.parameters())
    engine = Engine(actor, root)
    if mode == "samealive":
        from .online import run_online

        gate = json.loads((root / "replay-gate.json").read_bytes())
        assert gate["behavior"]["passed"] and gate["behavior"]["seeds"] == list(PILOT)
        tapes = {}
        ids = sorted({int(i) for i in PANEL.ravel()})
    else:
        tapes = tape_inputs(root / "source-results.zip")
        ids = sorted(
            set(PANEL.ravel().tolist())
            | {e["row"] for seq in tapes.values() for e in seq}
        )
    write(out / "status.json", dict(stage="calibrating", contexts=len(ids)))
    try:
        engine.q.prepare(ids)
    except NumericalMismatch as err:
        engine.q.fast = False
        engine.fallback.append(dict(kind="initial", message=str(err)))
    if mode == "samealive":
        run_online(engine, out)
    else:
        people = {}
        for n, ((seed, world), events) in enumerate(tapes.items()):
            write(
                out / "status.json",
                dict(stage="paired_history", completed=n, seed=seed, world=world),
            )
            people[seed, world] = replay_one(
                engine, out / "histories" / f"s{seed}" / world, events
            )
        means = donor_vectors(engine, people)
        target = out / "steering"
        target.mkdir()
        np.savez_compressed(
            target / "donors.npz", **{f"s{s}": v for s, v in means.items()}
        )
        for n, seed in enumerate(PILOT):
            write(
                out / "status.json",
                dict(stage="held_seed_steering", completed=n, seed=seed),
            )
            steer_one(engine, target / f"s{seed}", seed, people, means)
        analyze_replay(out, out / "analysis")
    actor.verify_unchanged()
    assert count == sum(p.numel() for p in actor.model.parameters())
    write(
        out / "receipt.json",
        dict(
            mode=mode,
            elapsed=time.monotonic() - start,
            device=torch.cuda.get_device_name(),
            frozen_backbone=True,
            model_parameters=count,
            peak_bytes=torch.cuda.max_memory_allocated(),
            roles=ROLES,
            conditions=INTERVENTIONS,
            paired_semantic=engine.checked,
            paired_gradients=engine.grad_checks,
            custom_full=engine.custom_full,
            max_probability_error=engine.max_probability_difference,
            max_gradient_error=engine.max_gradient_difference,
            fallbacks=engine.fallback,
            counts=engine.q.stats,
            contexts=len(ids),
            versions={
                k: version(k) for k in ("torch", "numpy", "transformers", "scipy")
            },
        ),
    )
    write(out / "status.json", dict(stage="complete", mode=mode))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    args = ap.parse_args()
    run(args.root, args.mode)
