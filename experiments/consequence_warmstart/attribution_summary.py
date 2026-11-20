"""Descriptive, seed-clustered summaries of the fixed existing-history attribution."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.consequence_warmstart.attribution import ARMS, BANDS, OMIT
from experiments.consequence_warmstart.design import BLOCKS, ENV_SEEDS
from experiments.direct_policy.protocol import write

METRICS = (
    "healthy_style",
    "healthy_answer",
    "healthy_conditional",
    "risk_h2",
    "risk_h3",
    "risk_h4",
    "risk_h6",
)


def mean(values):
    return float(np.mean(values)) if len(values) else None


def paired(cases, values):
    seeds = {
        str(s): mean([v for c, v in zip(cases, values) if c["seed"] == s])
        for s in ENV_SEEDS
    }
    numbers = list(seeds.values())
    return {
        "mean": mean(numbers),
        "seeds": seeds,
        "range": [min(numbers), max(numbers)],
    }


def family(events, cases, field, levels):
    result = {}
    for level in levels:
        chosen = [e for e in events if e[field] == level]
        values = [
            sum(e["delta_healthy_style"] for e in chosen if e["life"] == c["life"])
            for c in cases
        ]
        result[str(level)] = {
            "count": len(chosen),
            "positive_advantage": sum(e["advantage"] > 0 for e in chosen),
            "negative_advantage": sum(e["advantage"] < 0 for e in chosen),
            "mean_health": mean([e["health"] for e in chosen]),
            "mean_advantage": mean([e["advantage"] for e in chosen]),
            "healthy_style_contribution": paired(cases, values),
            "mean_update_healthy_style": mean(
                [e["delta_healthy_style"] for e in chosen]
            ),
            "mean_life_risk_h6_change": sum(e["delta_risk_h6"] for e in chosen)
            / len(cases),
            "mean_life_risk_h2_change": sum(e["delta_risk_h2"] for e in chosen)
            / len(cases),
        }
    return result


def summarize(events, all_cases):
    result = {
        "scope": "existing histories only; exploratory; four clusters",
        "blocks": {},
    }
    for block in BLOCKS:
        cases = [c for c in all_cases if c["block"] == block]
        arms = {}
        for arm in ARMS:
            ev = [
                e
                for e in events
                if e["block"] == block and e["arm"] == arm and not e["terminal"]
            ]
            terminal = [
                e
                for e in events
                if e["block"] == block and e["arm"] == arm and e["terminal"]
            ]
            for e in ev:
                e["answer_evidence"] = (
                    (
                        ("favored" if e["favored_answer"] else "unfavored")
                        + ("_maintained" if e["response"] == 1 else "_refused")
                    )
                    if e["action"] >= 2
                    else "nonanswer"
                )
            arms[arm] = {
                "nonterminal_updates": len(ev),
                "gains": {
                    k: paired(
                        cases,
                        [
                            c["arms"][arm]["last_live"][k] - c["arms"][arm]["birth"][k]
                            for c in cases
                        ],
                    )
                    for k in METRICS
                },
                "by_category": family(
                    ev,
                    cases,
                    "category",
                    (
                        "answer_maintained",
                        "answer_refused",
                        "ask_maintained",
                        "ask_refused",
                        "silence",
                    ),
                ),
                "by_health": family(ev, cases, "band", BANDS),
                "by_answer_evidence": family(
                    ev,
                    cases,
                    "answer_evidence",
                    (
                        "favored_maintained",
                        "unfavored_maintained",
                        "favored_refused",
                        "unfavored_refused",
                        "nonanswer",
                    ),
                ),
                "terminal": {
                    "count": len(terminal),
                    "bands": {b: sum(e["band"] == b for e in terminal) for b in BANDS},
                    "actions": {
                        str(a): sum(e["action"] == a for e in terminal)
                        for a in range(4)
                    },
                    "effects": {
                        k: mean([e[f"delta_{k}"] for e in terminal]) for k in METRICS
                    },
                },
                "next_request": {
                    str(changed): {
                        "count": sum(e["next_changed_partner"] == changed for e in ev),
                        "mean_tvd": mean(
                            [
                                e["next_tvd"]
                                for e in ev
                                if e["next_changed_partner"] == changed
                            ]
                        ),
                        "mean_abs_chosen_delta": mean(
                            [
                                abs(e["next_chosen_delta"])
                                for e in ev
                                if e["next_changed_partner"] == changed
                            ]
                        ),
                    }
                    for changed in (False, True)
                },
            }
            if arm == "warmstart_adapter":
                transfer = {}
                for b in BANDS:
                    subset = [e for e in ev if e["band"] == b]
                    transfer[b] = {"count": len(subset)}
                    for name in ("original", "h2", "h6", "heldout"):
                        transfer[b][name] = {
                            "mean_abs_chosen_delta": mean(
                                [abs(e[f"{name}_chosen_delta"]) for e in subset]
                            ),
                            "mean_tvd": mean([e[f"{name}_tvd"] for e in subset]),
                        }
                    for h in (2, 6):
                        overlaps = [e[f"h{h}_health_overlap_ratio"] for e in subset]
                        linear = [e[f"h{h}_linear_logit_absmax"] for e in subset]
                        transfer[b][f"h{h}_overlap"] = {
                            "mean": mean(overlaps),
                            "median": float(np.median(overlaps)) if overlaps else None,
                            "max": max(overlaps) if overlaps else None,
                        }
                        transfer[b][f"h{h}_linear_absmax"] = {
                            "mean": mean(linear),
                            "max": max(linear) if linear else None,
                        }
                arms[arm]["local_transfer"] = transfer
        omissions = {}
        for name in OMIT:
            conditions = [c["interventions"][f"omit_{name}"] for c in cases]
            entry = {
                "removed_total": sum(len(x["removed"]) for x in conditions),
                "lost_healthy_gain": paired(
                    cases,
                    [
                        c["interventions"]["normal"]["metrics"]["healthy_style"]
                        - v["metrics"]["healthy_style"]
                        for c, v in zip(cases, conditions)
                    ],
                ),
                "endpoint": {
                    k: mean([v["metrics"][k] for v in conditions]) for k in METRICS
                },
            }
            if name in OMIT[:3]:
                matches = [
                    [c["interventions"][f"matched_{name}_{r}"] for r in range(5)]
                    for c in cases
                ]
                entry["matched_endpoint"] = {
                    k: mean(
                        [mean([v["metrics"][k] for v in group]) for group in matches]
                    )
                    for k in METRICS
                }
                entry["excess_loss_vs_matched"] = paired(
                    cases,
                    [
                        mean([v["metrics"]["healthy_style"] for v in group])
                        - condition["metrics"]["healthy_style"]
                        for group, condition in zip(matches, conditions)
                    ],
                )
                entry["matched_removed_counts_verified"] = all(
                    all(len(v["removed"]) == len(condition["removed"]) for v in group)
                    for group, condition in zip(matches, conditions)
                )
            omissions[name] = entry
        structural = {
            k: paired(
                cases,
                [
                    c["interventions"]["scalar_rule_same_history"]["metrics"][k]
                    - c["arms"]["warmstart_adapter"]["birth"][k]
                    for c in cases
                ],
            )
            for k in METRICS
        }
        result["blocks"][block] = {
            "arms": arms,
            "omissions": omissions,
            "scalar_on_warm_history_gain": structural,
        }
    result["checks"] = {
        k: max(c["arms"][a]["checks"][k] for c in all_cases for a in ARMS)
        for k in (
            "telescoping_error",
            "ideal_factor_error",
            "float32_parameter_rounding_max",
        )
    }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    summary = summarize(
        json.loads((args.run / "events.json").read_bytes()),
        json.loads((args.run / "cases.json").read_bytes()),
    )
    write(args.run / "summary.json", summary)
    print(
        json.dumps(
            {
                "checks": summary["checks"],
                "blocks": {
                    b: {
                        "gains": v["arms"]["warmstart_adapter"]["gains"],
                        "categories": v["arms"]["warmstart_adapter"]["by_category"],
                        "omissions": v["omissions"],
                    }
                    for b, v in summary["blocks"].items()
                },
            }
        )
    )
