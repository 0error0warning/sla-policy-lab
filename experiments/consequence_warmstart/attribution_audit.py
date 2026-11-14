"""Independently replay omission recipients and reconcile path/group arithmetic."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.consequence_policy.experiment import read_inputs
from experiments.consequence_warmstart.audit import Independent
from experiments.consequence_warmstart.design import BLOCKS, selected
from experiments.direct_policy.protocol import write


def independent_grid(weights, native, x, p):
    levels = np.array([90.0, 300.0])
    centers = np.linspace(np.log(2), np.log(351), 12)
    z = (np.log1p(levels[:, None]) - centers) / (0.7 * (centers[1] - centers[0]))
    g = np.exp(-0.5 * z**2)
    g = (g / g.sum(1, keepdims=True)).astype(np.float32)
    w = weights.astype(np.float32)
    # Direct indexed tensor contraction, no Policy.grid_logits or attribution.Grid.
    residual = np.einsum("hb,nd,nbda->hna", g, x, w[p], optimize=True)
    logits = native[None] / 2.0 + residual
    pi = np.exp(logits - logits.max(-1, keepdims=True))
    pi /= pi.sum(-1, keepdims=True)
    return 0.95 * pi + 0.0125


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    rows, native, x, _, _ = read_inputs(
        args.source / "capture/output", args.source / "results"
    )
    parent = json.loads((args.histories / "index.json").read_bytes())
    cases = json.loads((args.run / "cases.json").read_bytes())
    events = json.loads((args.run / "events.json").read_bytes())
    summary = json.loads((args.run / "summary.json").read_bytes())
    event_count = recipients = updates = 0
    max_weight_error = max_probability_error = 0.0
    for case, metadata in zip(cases, parent):
        assert (
            case["seed"] == metadata["environment_seed"]
            and case["block"] == metadata["block"]
        )
        life = case["life"]
        ids = selected(rows, BLOCKS[case["block"]][1])
        partners = np.array([rows[i]["partner"] for i in ids])
        favored = np.array(
            [
                2
                if (
                    (rows[i]["frame"] == 0)
                    == (
                        (rows[i]["partner"] == 0)
                        ^ bool(case["world"][rows[i]["partner"]])
                    )
                )
                else 3
                for i in ids
            ]
        )
        for arm in ("warmstart_adapter", "personal"):
            own = [e for e in events if e["life"] == life and e["arm"] == arm]
            with np.load(args.histories / f"{life}-{arm}.npz") as raw:
                assert len(own) == len(raw["trace"])
                assert sum(e["terminal"] for e in own) == 1 and own[-1]["terminal"]
                assert own[-1]["after"] == 0
                event_count += len(own)
                for e, t in zip(own, raw["trace"]):
                    assert (
                        e["context"],
                        e["health"],
                        e["action"],
                        e["after"],
                        e["normalized"],
                    ) == (t[0], t[1], t[2], t[4], t[10])
                    if not e["terminal"] and e["next_changed_partner"]:
                        assert e["next_tvd"] == 0 and e["next_chosen_delta"] == 0
                for name in case["arms"][arm]["birth"]:
                    total = sum(e[f"delta_{name}"] for e in own[:-1])
                    expected = (
                        case["arms"][arm]["last_live"][name]
                        - case["arms"][arm]["birth"][name]
                    )
                    assert abs(total - expected) < 1e-10
                pi = raw["last_preterminal-heldout"]
                value = float(pi[-2:, np.arange(len(ids)), favored].mean())
                assert (
                    abs(value - case["arms"][arm]["last_live"]["healthy_style"]) < 2e-7
                )
                for current, parent_label in (
                    ("birth", "birth"),
                    ("last_live", "last_preterminal"),
                    ("terminal", "terminal"),
                ):
                    reference = metadata["arms"][arm]["checkpoints"][parent_label][
                        "panels"
                    ]["heldout"]
                    for j, h in enumerate((2, 3, 4, 6)):
                        assert (
                            abs(
                                case["arms"][arm][current][f"risk_h{h}"]
                                - reference["death"][j]
                            )
                            < 2e-7
                        )
        with (
            np.load(args.histories / f"{life}-warmstart_adapter.npz") as raw,
            np.load(args.run / f"{life}-recipient-weights.npz") as outputs,
        ):
            trace = raw["trace"][:-1]
            for name, condition in case["interventions"].items():
                dropped = set(condition["removed"])
                assert all(0 <= t < len(trace) for t in dropped)
                assert len(dropped) + condition["retained_updates"] == len(trace)
                actor = Independent(
                    args.source / "results",
                    metadata["birth_seed"],
                    "personal"
                    if name == "scalar_rule_same_history"
                    else "warmstart_adapter",
                )
                for t, entry in enumerate(trace):
                    if t in dropped:
                        continue
                    i = int(entry[0])
                    actor.update(
                        native[i],
                        x[i],
                        rows[i]["partner"],
                        entry[1],
                        int(entry[2]),
                        entry[10],
                    )
                    updates += 1
                weights = actor.weights()
                err = float(np.max(abs(weights - outputs[name])))
                max_weight_error = max(max_weight_error, err)
                np.testing.assert_allclose(weights, outputs[name], atol=2e-6, rtol=0)
                pi = independent_grid(weights, native[ids], x[ids], partners)
                value = float(pi[:, np.arange(len(ids)), favored].mean())
                err = abs(value - condition["metrics"]["healthy_style"])
                max_probability_error = max(max_probability_error, err)
                if err > 3e-7:
                    raise ValueError("Independent recipient probe differs")
                recipients += 1
                if name.startswith("matched_"):
                    target = name[len("matched_") :].rsplit("_", 1)[0]
                    actual = set(case["interventions"]["omit_" + target]["removed"])
                    for partner in (0, 1):
                        for lo, hi in ((0, 6), (6, 15), (15, 90), (90, 350)):
                            positions = {
                                t
                                for t, e in enumerate(trace)
                                if rows[int(e[0])]["partner"] == partner
                                and lo < e[1] <= hi
                            }
                            assert len(dropped & positions) == len(actual & positions)
    # Independently sum every category by block; include zero-category lives.
    for block, values in summary["blocks"].items():
        selected_events = [
            e
            for e in events
            if e["block"] == block
            and e["arm"] == "warmstart_adapter"
            and not e["terminal"]
        ]
        for name, item in values["arms"]["warmstart_adapter"]["by_category"].items():
            subset = [e for e in selected_events if e["category"] == name]
            assert len(subset) == item["count"]
            expected = sum(e["delta_healthy_style"] for e in subset) / 16
            assert abs(expected - item["healthy_style_contribution"]["mean"]) < 1e-12
    result = {
        "passed": True,
        "events_checked": event_count,
        "recipients_checked": recipients,
        "independent_recipient_updates": updates,
        "max_weight_error": max_weight_error,
        "max_independent_healthy_probability_error": max_probability_error,
        "terminal_exclusion": True,
        "matched_masks_preserve_partner_health_counts": True,
        "scope": "Independent scalar update and direct head contraction; shared frozen prefix and RUL histories; no GPU",
    }
    write(args.run / "audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
