"""Read-only complete-Qwen probes on exact snapshots reconstructed from living individuals."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.direct_policy.protocol import write
from experiments.individual_life.design import PROBE_H


def load_events(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def prepare(roots, out):
    out.mkdir(parents=True, exist_ok=True)
    weights = {}
    targets = []
    for root in roots:
        for f in sorted(root.glob("*/s*/*/true1/life.json")):
            d = f.parent
            meta = json.loads(f.read_bytes())
            ev = load_events(d / "life.jsonl")
            with np.load(d / "parameters.npz") as z:
                initial = z["birth"]
                changes = z["main_xor"]
            living = meta["living_probes"]
            for marker in ("birth", "t32", "first_low"):
                if marker not in living:
                    continue
                level = living[marker]
                k = level["version"]
                j = level["after_turn"]
                assert j < len(ev) and k == j
                now = ev[j]
                assert (
                    now["before_learning_version"] == k
                    and now["health_before"] == level["actual_alive_health"] > 0
                )
                key = {
                    "name": d.relative_to(root).as_posix(),
                    "marker": marker,
                    "world": meta["world"],
                    "visible": meta["visible"],
                    "turn": j,
                    "health": now["health_before"],
                    "context": now["context"],
                    "previous": now["previous_decision_version"],
                    "recorded_current": now["prob_current"],
                    "recorded_previous": now["prob_previous"],
                }
                for version, label in ((k, "current"), (key["previous"], "pre")):
                    bits = initial.copy()
                    if version:
                        bits ^= np.bitwise_xor.reduce(changes[:version], axis=0)
                    value = sha256(bits.tobytes()).hexdigest()
                    assert value == (
                        meta["birth_sha256"]
                        if version == 0
                        else ev[version - 1]["after_hash"]
                    )
                    weights[value] = bits
                    key[label] = value
                targets.append(key)
    assert targets and len({(a["name"], a["marker"]) for a in targets}) == len(targets)
    write(out / "probe-snapshots.json", targets)
    np.savez_compressed(out / "probe-parameters.npz", **weights)


def hardware(actor, root, out):
    import torch

    from experiments.individual_life.model import Complete, Native

    native = Native(root, root, "cuda")
    backend = Complete(actor, native, root)
    refs = json.loads((root / "probe-snapshots.json").read_bytes())
    n = len(native.probe_ix)
    models = {}
    with np.load(root / "probe-parameters.npz") as z:
        for key in z.files:
            bits = z[key]
            assert sha256(bits.tobytes()).hexdigest() == key
            model = native.new("blank")
            cut = model.A.numel()
            with torch.no_grad():
                for param, arr in zip(
                    model.live(), (bits[:cut], bits[cut:]), strict=True
                ):
                    param.copy_(
                        torch.tensor(arr.view("<f4"), device="cuda").view_as(param)
                    )
            models[key] = model
    backend.bind(next(iter(models.values())), None)
    probs = {}
    anchors = []
    calls = 0
    try:
        for v in (True, False):
            snapshots = sorted({r["current"] for r in refs if r["visible"] == v})
            for k in range(0, len(snapshots), 6):
                take = snapshots[k : k + 6]
                weights = [models[key].live() for key in take]
                weights += [weights[0]] * (6 - len(take))
                scores = np.empty((len(take), len(PROBE_H), n, 4), np.float32)
                for hi, h in enumerate(PROBE_H):
                    for j, index in enumerate(native.probe_ix):
                        panel = backend.panel(int(index), h, v, weights)
                        scores[:, hi, j] = panel[: len(take)].detach().cpu().numpy()
                        del panel
                        calls += 1
                if not v:
                    np.testing.assert_array_equal(
                        scores, np.broadcast_to(scores[:, :1], scores.shape)
                    )
                for key, value in zip(take, scores, strict=True):
                    probs[str(int(v)) + "-" + key] = value
                write(
                    out / "status.json",
                    {
                        "stage": "read_only_full_probes",
                        "completed_unique_snapshots": len(probs),
                        "forward_calls": calls,
                    },
                )
        for r in refs:
            previous = models[r["pre"]].live()
            current = models[r["current"]].live()
            panel = (
                backend.panel(
                    r["context"],
                    r["health"],
                    r["visible"],
                    [previous, current, current, previous, current, previous],
                )
                .detach()
                .cpu()
                .numpy()
            )
            assert (
                np.max(np.abs(panel[0] - panel[3])) < 3e-5
                and np.max(np.abs(panel[1] - panel[4])) < 3e-5
            )
            anchors.append(panel[:2].copy())
        np.savez_compressed(
            out / "full-checkpoints.npz",
            **probs,
            anchors=np.stack(anchors),
            contexts=native.probe_ix,
            health=PROBE_H,
        )
        write(out / "probe-snapshots.json", refs)
    finally:
        backend.close()
    return {
        "installed_living_snapshots": len(refs),
        "unique_sensor_and_weights": len(probs),
        "heldout_contexts": n,
        "ordinary_probe_forwards": calls,
        "same_state_anchor_forwards": len(anchors),
        "complete_action_distributions": sum(
            a.shape[0] * a.shape[1] for a in probs.values()
        ),
        "new_lives": 0,
        "optimizer_steps": 0,
    }


def analyze(roots, hardware, out):
    from experiments.individual_life.analysis import Evaluator, describe
    from experiments.individual_life.design import rows

    refs = json.loads((hardware / "probe-snapshots.json").read_bytes())
    ev = Evaluator(PROBE_H, rows())
    levels = {}
    groups = {}
    raw = []
    changes = []
    anchor = []
    anchor_d = []
    with np.load(hardware / "full-checkpoints.npz") as z:
        assert np.array_equal(
            z["contexts"],
            [i for i, r in enumerate(ev.rows) if (r["task"], r["wording"]) == (5, 4)],
        )
        assert np.array_equal(z["health"], PROBE_H)
        for idx, r in enumerate(refs):
            tag = r["name"]
            v = str(int(r["visible"]))
            current = z[v + "-" + r["current"]]
            root = next(root for root in roots if (root / tag / "life.json").is_file())
            with np.load(root / tag / "living-probes.npz") as cache:
                approx = cache[r["marker"] + "-main"]
                prefix = cache["birth-main"]
            birth = next(x for x in refs if x["name"] == tag and x["marker"] == "birth")
            base = z[v + "-" + birth["current"]]
            raw.extend(np.max(np.abs(current - approx), axis=-1).ravel().tolist())
            changes.extend(
                np.max(np.abs((current - base) - (approx - prefix)), axis=-1)
                .ravel()
                .tolist()
            )
            orig = np.array([r["recorded_previous"], r["recorded_current"]])
            fresh = z["anchors"][idx]
            anchor.append(float(np.max(np.abs(orig - fresh))))
            anchor_d.append(
                float(np.max(np.abs((orig[1] - orig[0]) - (fresh[1] - fresh[0]))))
            )
            scored = ev.probe(current, r["world"])
            born = ev.probe(base, r["world"])
            change = {k: scored[k] - born[k] for k in scored}
            sense, seed, birth, _rule = tag.split("/")
            wave = "confirm" if int(seed[1:]) >= 68101 else "pilot"
            levels.setdefault(tag, {})[r["marker"]] = {
                "turn": r["turn"],
                "real_health": r["health"],
                "probability_metrics": scored,
                "change": change,
            }
            groups.setdefault("/".join((wave, sense, birth, r["marker"])), []).append(
                change
            )
    summary = {
        k: {a: describe([q[a] for q in group]) for a in group[0]}
        for k, group in groups.items()
    }

    def error(arr):
        return {
            "count": len(arr),
            "max": float(np.max(arr)),
            "p95": float(np.quantile(arr, 0.95)),
            "mean": float(np.mean(arr)),
        }

    out.mkdir(exist_ok=True, parents=True)
    write(
        out / "full-body-probes.json",
        {
            "by_life": levels,
            "groups": summary,
            "suffix_to_complete_absolute": error(raw),
            "suffix_to_complete_effect": error(changes),
            "original_to_replayed_full_probability": error(anchor),
            "original_to_replayed_full_single_update_effect": error(anchor_d),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "analyze"))
    parser.add_argument("--lives", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", type=Path)
    a = parser.parse_args()
    if a.mode == "prepare":
        prepare(a.lives, a.output)
    elif a.hardware:
        analyze(a.lives, a.hardware, a.output)
    else:
        parser.error("analyze requires --hardware")
