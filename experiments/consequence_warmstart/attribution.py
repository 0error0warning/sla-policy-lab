"""CPU-only attribution on fixed #28 histories; never samples or settles a new life."""

import argparse
import csv
import json
import time
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.consequence_policy.experiment import read_inputs
from experiments.consequence_policy.lifetime import PersonalActor
from experiments.consequence_policy.policy import Policy, deployed, softmax
from experiments.consequence_policy.study import TEST_H, basis
from experiments.consequence_warmstart.audit import rates, risk_branches
from experiments.consequence_warmstart.design import BLOCKS, selected, verify_lock
from experiments.consequence_warmstart.runner import effective
from experiments.direct_policy.protocol import rng, write

ARMS = ("warmstart_adapter", "personal")
BANDS = ("le6", "6_to15", "15_to90", "gt90")
OMIT = ("answer_maintained", "answer_refused", "nonanswer", *BANDS)


def band(health):
    return BANDS[int(health > 6) + int(health > 15) + int(health > 90)]


def category(action, response):
    if action == 0:
        return "silence"
    return ("answer" if action >= 2 else "ask") + (
        "_maintained" if response else "_refused"
    )


def drop_mask(trace, name):
    a, response = trace[:, 2], trace[:, 3]
    if name == "answer_maintained":
        return (a >= 2) & (response == 1)
    if name == "answer_refused":
        return (a >= 2) & (response == 0)
    if name == "nonanswer":
        return a < 2
    return np.array([band(h) == name for h in trace[:, 1]])


def matched_mask(mask, trace, rows, case, name, replicate):
    gen = rng(
        case["environment_seed"],
        "attribution-countmatched-v1",
        case["world"],
        case["block"],
        name,
        replicate,
    )
    partners = np.array([rows[int(i)]["partner"] for i in trace[:, 0]])
    bands = np.array([band(h) for h in trace[:, 1]])
    selected_mask = np.zeros(len(trace), dtype=bool)
    for p in (0, 1):
        for b in BANDS:
            available = np.flatnonzero((partners == p) & (bands == b))
            count = int(mask[available].sum())
            selected_mask[gen.choice(available, size=count, replace=False)] = True
    return selected_mask


class Grid:
    def __init__(self, rows, native, x, case, branch):
        self.rows, self.native, self.x = rows, native, x
        self.idx = selected(rows, BLOCKS[case["block"]][1])
        self.partner = np.array([rows[i]["partner"] for i in self.idx])
        self.favored = []
        risk = []
        for i in self.idx:
            row = rows[i]
            p = rates(row["partner"], row["frame"], case["world"][row["partner"]])
            self.favored.append(2 if p[2] > p[3] else 3)
            risk.append((1 - p)[None] * branch[:, :, 0] + p[None] * branch[:, :, 1])
        self.risk = np.stack(risk, axis=1)

    def measure(self, actor):
        policy = Policy(self.x.shape[1])
        policy.weights[:] = effective(actor)
        logits = policy.grid_logits(
            self.native[self.idx], self.x[self.idx], self.partner, TEST_H
        )
        pi = deployed(logits)
        good = pi[:, np.arange(len(self.idx)), self.favored]
        answer = pi[:, :, 2:].sum(-1)
        deaths = (pi * self.risk).sum(-1).mean(1)
        return {
            "healthy_style": float(good[-2:].mean()),
            "healthy_answer": float(answer[-2:].mean()),
            "healthy_conditional": float(
                (good[-2:].sum(1) / answer[-2:].sum(1)).mean()
            ),
            **{
                f"partner{p}_style": float(good[-2:, self.partner == p].mean())
                for p in (0, 1)
            },
            **{
                f"risk_h{int(h)}": float(deaths[TEST_H.index(h)])
                for h in (2.0, 3.0, 4.0, 6.0)
            },
        }


def local_inputs(event, next_event, rows, block):
    i, h = int(event[0]), event[1]
    row = rows[i]
    held = [
        j
        for j in selected(rows, BLOCKS[block][1])
        if all(rows[j][k] == row[k] for k in ("partner", "frame", "layout"))
    ]
    assert len(held) == 1
    result = {
        "original": (i, h),
        "h2": (i, 2.0),
        "h6": (i, 6.0),
        "heldout": (held[0], h),
    }
    if next_event is not None:
        result["next"] = (int(next_event[0]), next_event[1])
    return result


def local_scores(actor, inputs, rows, native, x):
    return {
        name: actor.score(i, rows[i], h, native, x)[0]
        for name, (i, h) in inputs.items()
    }


def analytic_transfer(a, h, x, pi, logits):
    s = softmax(logits)
    derivative = -s.copy()
    derivative[a] += 1
    derivative *= 0.95 * s[a] / pi[a]
    g = basis([h])[0]
    norm = max(1.0, 0.25 * np.dot(g, g) * np.dot(x, x))
    # eta has already absorbed the measured advantage; infer it from the caller.
    return g, derivative, norm


def trajectory(source, raw, case, arm, rows, native, x, grid, started):
    actor = PersonalActor(source / "results", case["birth_seed"], case["world"], arm)
    trace = raw["trace"]
    if trace[-1, 4] != 0.0:
        raise ValueError("This audit is scoped to the archived uncensored lives")
    initial = previous = grid.measure(actor)
    events = []
    max_identity = max_rounding = max_factor_error = 0.0
    for t, event in enumerate(trace):
        if time.monotonic() - started > 3600:
            raise TimeoutError("Attribution CPU budget exceeded")
        terminal = t == len(trace) - 1
        if terminal:
            np.testing.assert_array_equal(
                effective(actor), raw["last_preterminal-weights"]
            )
            last_live = previous
        i, h, a = int(event[0]), event[1], int(event[2])
        row = rows[i]
        p = row["partner"]
        pi, logits, direction = actor.score(i, row, h, native, x)
        np.testing.assert_array_equal(pi, event[6:10])
        inputs = local_inputs(
            event, None if terminal else trace[t + 1], rows, case["block"]
        )
        local_before = local_scores(actor, inputs, rows, native, x)
        w_before = actor.main.weights.copy()
        actor.update(p, a, h, x[i], pi, logits, direction, event[10])
        current = grid.measure(actor)
        local_after = local_scores(actor, inputs, rows, native, x)
        np.testing.assert_array_equal(local_after["original"], event[12:16])
        true_rates = rates(p, row["frame"], case["world"][p])
        favored = 2 if true_rates[2] > true_rates[3] else 3
        record = {
            "life": f"s{case['environment_seed']}-{case['block']}-w{case['world'][0]}{case['world'][1]}",
            "seed": case["environment_seed"],
            "block": case["block"],
            "arm": arm,
            "turn": t + 1,
            "terminal": terminal,
            "partner": p,
            "context": i,
            "health": h,
            "after": event[4],
            "action": a,
            "response": int(event[3]),
            "band": band(h),
            "category": category(a, int(event[3])),
            "favored_answer": bool(a == favored),
            "normalized": event[10],
            "advantage": event[10] + 0.9,
            "next_changed_partner": bool(
                not terminal and rows[int(trace[t + 1, 0])]["partner"] != p
            ),
            **{f"delta_{k}": current[k] - previous[k] for k in current},
        }
        for name in ("original", "h2", "h6", "heldout", "next"):
            if name not in inputs:
                record.update(
                    {
                        f"{name}_{k}": None
                        for k in ("chosen_delta", "answer_delta", "tvd")
                    }
                )
                continue
            change = local_after[name] - local_before[name]
            record.update(
                {
                    f"{name}_chosen_delta": float(change[a]),
                    f"{name}_answer_delta": float(change[2:].sum()),
                    f"{name}_tvd": float(np.abs(change).sum() / 2),
                }
            )
        if arm == "warmstart_adapter":
            g, derivative, norm = analytic_transfer(a, h, x[i], pi, logits)
            eta = 0.7 * (event[10] + 0.9) / norm
            ideal = eta * np.einsum("b,d,a->bda", g, x[i], derivative)
            actual = actor.main.weights[p].astype(float) - w_before[p]
            max_rounding = max(max_rounding, float(np.max(abs(actual - ideal))))
            for H in (2.0, 6.0):
                q = basis([H])[0].astype(np.float32)
                direct = np.einsum("b,d,bda->a", q, x[i], ideal)
                factored = (
                    eta * np.dot(q, g) * np.dot(x[i].astype(float), x[i]) * derivative
                )
                error = float(np.max(abs(direct - factored)))
                max_factor_error = max(max_factor_error, error)
                np.testing.assert_allclose(direct, factored, atol=2e-12, rtol=1e-12)
                actual_logits = np.einsum("b,d,bda->a", q, x[i], actual)
                record[f"h{int(H)}_health_overlap_ratio"] = float(
                    np.dot(q, g) / np.dot(g, g)
                )
                record[f"h{int(H)}_linear_logit_absmax"] = float(
                    np.max(abs(actual_logits))
                )
                record[f"h{int(H)}_ideal_logit_absmax"] = float(np.max(abs(factored)))
        else:
            for H in (2, 6):
                for field in (
                    "health_overlap_ratio",
                    "linear_logit_absmax",
                    "ideal_logit_absmax",
                ):
                    record[f"h{H}_{field}"] = None
        events.append(record)
        previous = current
    np.testing.assert_array_equal(effective(actor), raw["terminal-weights"])
    for k in initial:
        accumulated = sum(e[f"delta_{k}"] for e in events if not e["terminal"])
        err = abs(accumulated - (last_live[k] - initial[k]))
        max_identity = max(max_identity, err)
        if err > 1e-10:
            raise ValueError("Attribution does not telescope")
    return events, {
        "birth": initial,
        "last_live": last_live,
        "terminal": previous,
        "checks": {
            "telescoping_error": max_identity,
            "ideal_factor_error": max_factor_error,
            "float32_parameter_rounding_max": max_rounding,
        },
    }


def recipient(source, case, trace, mask, arm, rows, native, x):
    actor = PersonalActor(source / "results", case["birth_seed"], case["world"], arm)
    for t, event in enumerate(trace):
        if mask[t]:
            continue
        i, a, h = int(event[0]), int(event[2]), event[1]
        row = rows[i]
        pi, logits, direction = actor.score(i, row, h, native, x)
        actor.update(row["partner"], a, h, x[i], pi, logits, direction, event[10])
    return actor


def interventions(source, case, raw, rows, native, x, grid):
    trace = raw["trace"][:-1]
    specs = [
        ("normal", "warmstart_adapter", np.zeros(len(trace), dtype=bool)),
        ("scalar_rule_same_history", "personal", np.zeros(len(trace), dtype=bool)),
    ]
    for name in OMIT:
        mask = drop_mask(trace, name)
        specs.append((f"omit_{name}", "warmstart_adapter", mask))
        if name in OMIT[:3]:
            specs.extend(
                (
                    f"matched_{name}_{r}",
                    "warmstart_adapter",
                    matched_mask(mask, trace, rows, case, name, r),
                )
                for r in range(5)
            )
    result, arrays = {}, {}
    for name, arm, mask in specs:
        actor = recipient(source, case, trace, mask, arm, rows, native, x)
        if name == "normal":
            np.testing.assert_array_equal(
                effective(actor), raw["last_preterminal-weights"]
            )
        result[name] = {
            "removed": np.flatnonzero(mask).tolist(),
            "retained_updates": int((~mask).sum()),
            "metrics": grid.measure(actor),
            "u": actor.u.tolist(),
        }
        arrays[name] = effective(actor).copy()
    return result, arrays


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--histories", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    verify_lock(
        repo,
        args.source,
        repo
        / "artifacts/model-studies/2026-11-14/warmstart-adapter/preregistered/lock.json",
    )
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    index = json.loads((args.histories / "index.json").read_bytes())
    if len(index) != 32:
        raise ValueError("Require all 32 paired development environments")
    rows, native, x, _, _ = read_inputs(
        args.source / "capture/output", args.source / "results"
    )
    branches = risk_branches()
    events, cases, hashes = [], [], {}
    for case in index:
        key = f"s{case['environment_seed']}-{case['block']}-w{case['world'][0]}{case['world'][1]}"
        grid = Grid(rows, native, x, case, branches)
        record = {
            "life": key,
            "seed": case["environment_seed"],
            "block": case["block"],
            "world": case["world"],
            "arms": {},
        }
        for arm in ARMS:
            path = args.histories / f"{key}-{arm}.npz"
            hashes[path.name] = sha256(path.read_bytes()).hexdigest()
            with np.load(path) as raw:
                addition, outcomes = trajectory(
                    args.source, raw, case, arm, rows, native, x, grid, started
                )
                events.extend(addition)
                record["arms"][arm] = outcomes
                if arm == "warmstart_adapter":
                    record["interventions"], weights = interventions(
                        args.source, case, raw, rows, native, x, grid
                    )
                    np.savez_compressed(
                        args.out / f"{key}-recipient-weights.npz", **weights
                    )
        cases.append(record)
        print("audited", key, flush=True)
    write(args.out / "cases.json", cases)
    write(args.out / "events.json", events)
    with (args.out / "events.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(events[0]))
        w.writeheader()
        w.writerows(events)
    write(
        args.out / "execution.json",
        {
            "complete": True,
            "seconds": time.monotonic() - started,
            "history_files": hashes,
            "source_index_sha256": sha256(
                (args.histories / "index.json").read_bytes()
            ).hexdigest(),
            "existing_warm_lives": 32,
            "existing_scalar_lives": 32,
            "new_lives": 0,
            "gpu_allocations": 0,
            "events_including_terminal": len(events),
            "history_interventions": sum(len(c["interventions"]) for c in cases),
        },
    )


if __name__ == "__main__":
    main()
