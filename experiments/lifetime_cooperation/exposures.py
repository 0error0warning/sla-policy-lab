"""Count real opportunities for a survivor to learn; counterfactuals are audit-only."""

import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import csv_out
from .audit import physical
from .design import ITEMS


def analyze(root, out):
    records = []
    paired_events = []
    for source in sorted((root / "lives").glob("s*/*/*/life.json")):
        m = json.loads(source.read_bytes())
        events = [
            json.loads(line)
            for line in (source.parent / "events.jsonl").read_bytes().splitlines()
        ]
        if m["learning"] and m["kind"] == "qwen":
            saved = np.asarray([e["prob"] for e in events])
            for region in ("healthy", "maintenance_rescues", "answer_fatal"):
                sample = []
                for k, event in enumerate(events[1:], 1):
                    h, noise = event["health"], event["noise"]
                    safe = physical(h, 2, True, noise)[-1] > 0
                    if {
                        "healthy": h >= 90,
                        "maintenance_rescues": safe
                        and physical(h, 2, False, noise)[-1] == 0,
                        "answer_fatal": not safe,
                    }[region]:
                        sample.append(k)
                for slot, role in enumerate(("current", "birth", "wrong")):
                    favored = []
                    for k in sample:
                        row = ITEMS[events[k]["row"]]
                        person = row["partner"] - 2
                        favored.append(
                            2 + (person ^ row["frame"] ^ events[k]["rules"][person])
                        )
                    p = saved[sample, slot, :]
                    preferred = p[
                        np.arange(len(sample)), np.asarray(favored, dtype=int)
                    ]
                    paired_events.append(
                        dict(
                            seed=m["seed"],
                            world=m["world"],
                            region=region,
                            role=role,
                            events=len(sample),
                            mean_answer=float(p[:, 2:].sum(axis=1).mean())
                            if sample
                            else "",
                            mean_favored_answer=float(preferred.mean())
                            if sample
                            else "",
                            mean_refuse_and_appeal=float(p[:, 1].mean())
                            if sample
                            else "",
                            sampled_answers=sum(
                                events[k]["action"] >= 2 for k in sample
                            ),
                        )
                    )
        counts = dict(
            any_injury_with_next=0,
            low_arrivals=0,
            low_injury_with_next=0,
            low_answer_injury_with_next=0,
            maintenance_can_rescue_answer=0,
            rescuable_encounters_with_next=0,
            observed_rescued_answers_with_next=0,
            maintained_answer_fatal=0,
            maintained_answer_fatal_with_next=0,
            maintained_answer_fatal_refusal_can_survive=0,
            first_fatal_arrival=0,
            first_fatal_avoidable=0,
            first_fatal_answer=0,
            first_fatal_survived=0,
        )
        first = None
        for k, e in enumerate(events):
            h = e["health"]
            after = e["after"]
            a = e["action"]
            noise = e["noise"]
            next_alive = k + 1 < len(events)
            assert next_alive == (after > 0)
            continuing = next_alive and h > after
            counts["any_injury_with_next"] += int(continuing)
            counts["low_arrivals"] += int(h <= 10)
            counts["low_injury_with_next"] += int(h <= 10 and continuing)
            counts["low_answer_injury_with_next"] += int(
                h <= 10 and a >= 2 and continuing
            )
            no = physical(h, 2, False, noise)[-1]
            yes = physical(h, 2, True, noise)[-1]
            rescuable = yes > 0 and no == 0
            counts["maintenance_can_rescue_answer"] += int(rescuable)
            counts["rescuable_encounters_with_next"] += int(rescuable and next_alive)
            counts["observed_rescued_answers_with_next"] += int(
                e["rescued_answer"] and next_alive
            )
            if yes == 0:
                counts["maintained_answer_fatal"] += 1
                counts["maintained_answer_fatal_with_next"] += int(next_alive)
                # Same realized partner variate and physical noise across possible actions.
                avoidable = (
                    physical(h, 0, None, noise)[-1] > 0
                    or physical(h, 1, e["partner_u"] < 0.15, noise)[-1] > 0
                )
                counts["maintained_answer_fatal_refusal_can_survive"] += int(avoidable)
                if first is None:
                    first = k + 1
                    counts.update(
                        first_fatal_arrival=1,
                        first_fatal_avoidable=int(avoidable),
                        first_fatal_answer=int(a >= 2),
                        first_fatal_survived=int(next_alive),
                    )
        records.append(
            dict(
                seed=m["seed"],
                world=m["world"],
                family=m["kind"],
                learning=m["learning"],
                turns=len(events),
                first_fatal_turn=first or "",
                **counts,
            )
        )
    out.mkdir(exist_ok=True, parents=True)
    csv_out(out / "opportunities.csv", records)
    csv_out(out / "actual_choices.csv", paired_events)
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    analyze(a.root, a.out)
