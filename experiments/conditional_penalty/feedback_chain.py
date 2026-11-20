"""Trace the PR 12 body, reward, signed update and the following living decision.

Run from the repository root with python -m experiments.conditional_penalty.feedback_chain.
Only reads the published inputs/outputs. No training or model weights are needed.
"""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from experiments.conditional_penalty.design import ARMS
from experiments.conditional_penalty.reference_check import LowHealthReference
from experiments.direct_policy.protocol import COSTS
from experiments.first_fatal.audit import path as independent_body_path
from experiments.first_fatal.design import FATAL
from experiments.first_fatal.design import settle as trial_settle
from experiments.health_baseline.analysis import save_csv
from experiments.health_baseline.audit import centers, score
from experiments.individual_life.design import action_at, offers
from experiments.internal_body.release import safe, verify_manifest
from experiments.social_closed_loop.core import BodyValue
from experiments.social_closed_loop.core import settle as paper_settle

PARENT = Path("artifacts/model-studies/2026-11-13/conditional-penalty")
OUT = Path("artifacts/model-studies/2026-11-13/body-feedback-chain")
EPS = 1e-12
SEEDS = tuple(range(71101, 71109)) + tuple(range(71201, 71217))
NEW_SEEDS = tuple(range(72101, 72109)) + tuple(range(72201, 72217))
BIRTHS = ("blank", "population")


@contextmanager
def results(stage):
    receipt = json.loads((stage / "receipt.json").read_bytes())
    archive = stage / "results.zip"
    if archive.exists():
        data = archive.read_bytes()
    else:
        data = b"".join(safe(stage, c["name"]).read_bytes() for c in receipt["chunks"])
    assert (
        len(data) == receipt["bytes"] and sha256(data).hexdigest() == receipt["sha256"]
    )
    with ZipFile(BytesIO(data)) as bundle:
        yield bundle


def read_events(bundle, name):
    return [
        json.loads(line) for line in gzip.decompress(bundle.read(name)).splitlines()
    ]


def band(h):
    if h >= 90:
        return "healthy"
    if h > 6:
        return "middle"
    return "practice" if h > FATAL else "fatal"


def physics(e, body, past):
    h, a, m = e["health"], e["action"], e["maintenance"]
    path, collapse = independent_body_path(h, a, m, e["noise"])
    assert np.allclose(path, e["subevents"], rtol=0, atol=1e-11)
    assert collapse == e["peak_collapse"] and abs(path[-1] - e["after"]) < 1e-11
    rul, end_rul = body.rul(h), body.rul(e["after"])
    d = rul - end_rul
    assert abs(d + e["reward"]) < 1e-10
    gb, lb, eligible = centers(past, h)
    assert eligible == e["eligible"]
    assert (
        abs(lb - e["local_baseline"]) < 1e-10 and abs(gb - e["global_baseline"]) < 1e-10
    )
    past.append((h, e["reward"]))
    return {
        "band": band(h),
        "fragility": 2 - h / 350,
        "peak_demand": COSTS[a] * (2 - h / 350),
        "duration": max(1 / 192, min(1, (h / 48) ** 2)),
        "peak_collapse": collapse,
        "action_damage": h - e["subevents"][1],
        "environment_damage": e["subevents"][1] - e["after"],
        "damage": h - e["after"],
        "rul_before": rul,
        "rul_after": end_rul,
        "loss": d,
        "baseline": lb,
        "prior_in_band": eligible,
    }


def immediate_alternative(e, meta, body):
    """Hindsight physical counterfactual with the recorded common noise and partner uniform."""
    prob, _ = offers(e, meta["world"])
    after = []
    for a in range(4):
        m = bool(e["u"][1] < prob[a]) if a else None
        after.append(trial_settle(e["health"], a, m, e["noise"])[0])
    assert abs(after[e["action"]] - e["after"]) < 1e-10
    best = max(after[:2])
    return best - e["after"], body.rul(best) - body.rul(e["after"])


def update_effect(before, after, action, u):
    pre = float(sum(before[2:]))
    post = float(sum(after[2:]))
    a0, a1 = action_at(before, u), action_at(after, u)
    return {
        "next_p_before": pre,
        "next_p_after": post,
        "next_delta": post - pre,
        "next_same_action_delta": after[action] - before[action],
        "next_action_before": a0,
        "next_action_after": a1,
        "semantic_flip": a0 != a1,
        "answer_flip": (a0 >= 2) != (a1 >= 2),
        "all_four_equal": before == after,
    }


def read_replays(artifact, body):
    rows = []
    stage = artifact / "gpu/gpreplay"
    with (
        results(stage) as out,
        ZipFile(stage / "inputs.zip") as inp,
        ZipFile(BytesIO(inp.read("tapes.zip"))) as tapes,
    ):
        for seed in SEEDS:
            for birth in BIRTHS:
                key = f"s{seed}/{birth}"
                src = read_events(tapes, key + "/events.jsonl.gz")
                meta = json.loads(tapes.read(key + "/life.json"))
                trials = [
                    read_events(out, f"output/{arm}/replays/{key}/replay.jsonl.gz")
                    for arm in ("gated", "constant")
                ]
                past = []
                assert (meta["seed"], meta["birth"]) == (seed, birth) and src[-1][
                    "after"
                ] == 0
                for j, (e, g, c) in enumerate(zip(src, *trials, strict=True)):
                    assert all(
                        e[k] == g[k] == c[k]
                        for k in (
                            "index",
                            "turn",
                            "health",
                            "after",
                            "reward",
                            "action",
                            "practice",
                        )
                    )
                    assert g["turn"] == j + 1 and (j == len(src) - 1) == (
                        e["after"] == 0
                    )
                    if j:
                        assert src[j - 1]["after"] == e["health"]
                    a, h = e["action"], e["health"]
                    f = physics(e, body, past)
                    paper_after, paper_path = paper_settle(
                        h, a, e["maintenance"], e["noise"]
                    )
                    row = {
                        "seed": seed,
                        "birth": birth,
                        "turn": j + 1,
                        "index": e["index"],
                        "health": h,
                        "after": e["after"],
                        "action": a,
                        "partner": e["partner"],
                        "frame": e["frame"],
                        "layout": e["layout"],
                        "maintained": e["maintenance"],
                        "first_fatal": e["first_fatal"],
                        **f,
                        "paper_action_damage": h - paper_path[1],
                        "paper_after": paper_after,
                        "paper_total_damage": h - paper_after,
                        "paper_rul_loss": body.rul(h) - body.rul(paper_after),
                        "k": 2 if h <= 6 else 1,
                    }
                    if e["practice"] and a >= 2:
                        (
                            row["best_refusal_health_gain"],
                            row["best_refusal_rul_gain"],
                        ) = immediate_alternative(e, meta, body)
                    denom = e["probs"][1][a]
                    for arm, rec, role, uindex, k in (
                        ("one", g, 1, 0, 1),
                        ("gate", g, 3, 1, row["k"]),
                        ("two", c, 3, 1, 2),
                    ):
                        sig = rec["signals"][uindex]
                        assert abs(sig + k * f["loss"] + f["baseline"]) < 1e-10
                        u = rec["updates"][uindex]
                        score(u, sig, denom, rec["probs"][role][a])
                        row[arm + "_signal"] = sig
                        row[arm + "_step"] = u["actual_step_norm"]
                        row[arm + "_parameter_changed"] = (
                            rec["before_hash"][uindex] != u["after_hash"]
                        )
                        row[arm + "_gradient_norm"] = float(
                            np.hypot(u["A_gradient_norm"], u["B_gradient_norm"])
                        )
                        row[arm + "_multiplier"] = u["parameter_multiplier"]
                        if e["after"] > 0:
                            nxt = trials[int(arm == "two")][j + 1]
                            assert (
                                u["after_hash"] == nxt["before_hash"][uindex]
                                and nxt["health"] == e["after"]
                            )
                            effect = update_effect(
                                nxt["probs"][role - 1], nxt["probs"][role], a, nxt["u"]
                            )
                            row.update(
                                {
                                    arm + "_" + key: value
                                    for key, value in effect.items()
                                }
                            )
                    if e["after"] > 0:
                        next_e = src[j + 1]
                        row.update(
                            next_index=next_e["index"],
                            next_health=next_e["health"],
                            next_layout=next_e["layout"],
                            next_same_partner=next_e["partner"] == e["partner"],
                            next_factual_action=next_e["action"],
                            next_u=next_e["u"][0],
                        )
                        assert (
                            row["one_next_action_after"] == row["next_factual_action"]
                        )
                    rows.append(row)
    return rows


def live_rows(artifact, body):
    lives = []
    observed = set()
    for stage in ("gppilot", "gprep"):
        with results(artifact / "gpu" / stage) as out:
            for name in sorted(
                n
                for n in out.namelist()
                if n.startswith("output/lives/") and n.endswith("/life.json")
            ):
                m = json.loads(out.read(name))
                seed, birth, arm = m["seed"], m["birth"], m["rule"]
                assert (seed, birth, arm) not in observed
                observed.add((seed, birth, arm))
                events = read_events(
                    out, name.removesuffix("life.json") + "events.jsonl.gz"
                )
                past = []
                health = 350.0
                first = 0
                practiced = []
                above = 0
                correct = arm.endswith("_true")
                for e in events:
                    assert e["health"] == health and health > 0
                    state = physics(e, body, past)
                    if e["first_fatal"]:
                        first += 1
                        assert e["turn"] == m["first_fatal"] and 0 < health <= FATAL
                    if e["turn"] < m["first_fatal"]:
                        above += int(e["practice"])
                    a = e["action"]
                    k = (
                        2
                        if arm.startswith("two")
                        or (arm.startswith("gate") and health <= 6)
                        else 1
                    )
                    assert (
                        abs(e["signal"] + k * state["loss"] + state["baseline"]) < 1e-10
                    )
                    if arm != "frozen":
                        u = e["update"][0]
                        credit = a if correct else (a + e["offset"]) % 4
                        assert (
                            u["credited_action"] == credit
                            and u["signal"] == e["signal"]
                        )
                        score(
                            u, e["signal"], e["probs"][1][credit], e["probs"][1][credit]
                        )
                    else:
                        assert not e["update"] and e["before_hash"] == e["after_hash"]
                    if e["practice"] and a >= 2:
                        assert e["turn"] < m["first_fatal"]
                        practiced.append(
                            (
                                state["damage"],
                                state["loss"],
                                e["signal"],
                                e["before_hash"][0] != e["after_hash"][0],
                            )
                        )
                    health = e["after"]
                assert (
                    health == 0
                    and first == 1
                    and len(events) == m["turns"]
                    and above == m["practice_count"]
                )
                lives.append(
                    {
                        "seed": seed,
                        "birth": birth,
                        "arm": arm,
                        "turns": len(events),
                        "practice_rounds": above,
                        "practice_answers": len(practiced),
                        "rul_flat": sum(abs(d) < EPS for _, d, _, _ in practiced),
                        "mean_practice_answer_damage": np.mean(
                            [r[0] for r in practiced]
                        )
                        if practiced
                        else None,
                        "mean_practice_answer_loss": np.mean([r[1] for r in practiced])
                        if practiced
                        else None,
                        "positive": sum(s > EPS for _, _, s, _ in practiced),
                        "negative": sum(s < -EPS for _, _, s, _ in practiced),
                        "parameter_changed": sum(changed for *_, changed in practiced),
                        "first_avoided": events[m["first_fatal"] - 1]["after"] > 0,
                        "first_fatal_positive": events[m["first_fatal"] - 1]["signal"]
                        > EPS,
                    }
                )
    assert observed == {(s, b, a) for s in NEW_SEEDS for b in BIRTHS for a in ARMS}
    return lives


def describe(rows, arm):
    if not rows:
        return {"n": 0}

    def val(key):
        return np.asarray([r[key] for r in rows])

    sig = val(arm + "_signal")
    step = val(arm + "_step")
    delta = val(arm + "_next_delta")
    gain = val("best_refusal_rul_gain")
    return {
        "n": len(rows),
        "positive": int(sum(sig > EPS)),
        "negative": int(sum(sig < -EPS)),
        "zero_signal": int(sum(abs(sig) <= EPS)),
        "physical_injuries": int(sum(val("damage") > EPS)),
        "flat_rul": int(sum(abs(val("loss")) < EPS)),
        "flat_and_positive": int(sum((abs(val("loss")) < EPS) & (sig > EPS))),
        "answer_worse_than_best_refusal_rul": int(sum(gain > 1e-9)),
        "positive_despite_better_refusal_rul": int(sum((gain > 1e-9) & (sig > EPS))),
        "nonzero_parameter_steps": int(sum(step > 0)),
        "changed_hash": int(sum(val(arm + "_parameter_changed"))),
        "median_step": float(np.median(step)),
        "negative_with_lower_p": int(sum((sig < -EPS) & (delta < 0))),
        "negative_with_equal_p": int(sum((sig < -EPS) & (delta == 0))),
        "negative_with_higher_p": int(sum((sig < -EPS) & (delta > 0))),
        "next_p_identical": int(sum(delta == 0)),
        "next_all_four_identical": int(sum(val(arm + "_all_four_equal"))),
        "same_text_context_next": int(sum(val("index") == val("next_index"))),
        "same_order_next": int(sum(val("layout") == val("next_layout"))),
        "same_partner_next": int(sum(val("next_same_partner"))),
        "equal_next_p_with_same_partner": int(
            sum(val("next_same_partner") & (delta == 0))
        ),
        "equal_next_p_with_other_partner": int(
            sum(~val("next_same_partner") & (delta == 0))
        ),
        "semantic_flips": int(sum(val(arm + "_semantic_flip"))),
        "answer_flips": int(sum(val(arm + "_answer_flip"))),
        "mean_next_delta_pp": float(100 * delta.mean()),
        "paper_one_step_deaths": int(sum(val("paper_after") == 0)),
        "mean_damage": float(np.mean(val("damage"))),
        "mean_paper_damage": float(np.mean(val("paper_total_damage"))),
        "mean_rul_loss": float(np.mean(val("loss"))),
        "mean_signal": float(np.mean(sig)),
    }


def grid(body):
    rows = []
    for h in (350, 175, 90, 48, 24, 12, 6, 5, 4, 3.48):
        for a, m in ((2, True), (2, False), (1, True), (0, None)):
            end, _, collapse = trial_settle(h, a, m, [0] * 5)
            orig, _ = paper_settle(h, a, m, [0] * 5)
            rows.append(
                {
                    "health": h,
                    "action": a,
                    "maintained": m,
                    "fragility": 2 - h / 350,
                    "peak": COSTS[a] * (2 - h / 350),
                    "duration": max(1 / 192, min(1, (h / 48) ** 2)),
                    "paper_damage": h - orig,
                    "trial_damage": h - end,
                    "paper_loss": body.rul(h) - body.rul(orig),
                    "trial_loss": body.rul(h) - body.rul(end),
                    "trial_collapse": collapse,
                }
            )
    return rows


def continuous_feedback(rows, risky):
    # Past.baselines uses pre-action health in [h/1.25, 1.25*h]. For h<=6, this
    # requires at most 7.5 health, including the first qualifying prior encounter.
    low = [r for r in rows if r["health"] <= 7.5]
    h = [r["health"] for r in low]
    end = [r["after"] for r in low]
    lo, hi = LowHealthReference().loss(h, end)
    coarse_lo, coarse_hi = LowHealthReference(2**-22).loss(h, end)
    assert np.all(coarse_lo <= lo + 1e-10) and np.all(coarse_hi >= hi - 1e-10)
    for r, v, w in zip(low, lo, hi, strict=True):
        r.update(continuous_loss_lower=float(v), continuous_loss_upper=float(w))
    by_life = defaultdict(list)
    for r in low:
        past = by_life[(r["seed"], r["birth"])]
        if r["band"] == "practice" and r["action"] >= 2:
            eligible = [
                prior
                for prior in past
                if r["health"] / 1.25 <= prior["health"] <= 1.25 * r["health"]
            ]
            assert len(eligible) == r["prior_in_band"] and len(eligible) >= 3
            past_min = float(np.mean([v["continuous_loss_lower"] for v in eligible]))
            past_max = float(np.mean([v["continuous_loss_upper"] for v in eligible]))
            r["continuous_gate_signal_lower"] = (
                past_min - 2 * r["continuous_loss_upper"]
            )
            r["continuous_gate_signal_upper"] = (
                past_max - 2 * r["continuous_loss_lower"]
            )
        past.append(r)

    # 1e-10 exceeds numeric FFT/convolution roundoff at the normalized probabilities used.
    def sign_summary(group):
        return {
            "corrected_positive": sum(
                r["continuous_gate_signal_lower"] > 1e-10 for r in group
            ),
            "corrected_negative": sum(
                r["continuous_gate_signal_upper"] < -1e-10 for r in group
            ),
            "corrected_sign_unresolved": sum(
                r["continuous_gate_signal_lower"] <= 1e-10
                and r["continuous_gate_signal_upper"] >= -1e-10
                for r in group
            ),
            "positive_to_negative": sum(
                r["gate_signal"] > EPS and r["continuous_gate_signal_upper"] < -1e-10
                for r in group
            ),
            "negative_to_positive": sum(
                r["gate_signal"] < -EPS and r["continuous_gate_signal_lower"] > 1e-10
                for r in group
            ),
        }

    flat = [r for r in risky if abs(r["loss"]) < EPS]
    return {
        "flat_count": len(flat),
        "positive_lower_bound": sum(r["continuous_loss_lower"] > 1e-10 for r in flat),
        "min_loss_lower": min(r["continuous_loss_lower"] for r in flat),
        "max_loss_upper": max(r["continuous_loss_upper"] for r in flat),
        "max_interval_width_all_low_plus_prior": float(np.max(hi - lo)),
        "increment_bin_width": 2**-24,
        "roundoff_tolerance": 1e-10,
        "smallest_prior_count_for_risky_answers": min(
            r["prior_in_band"] for r in risky
        ),
        "flat_if_only_current_loss_corrected_positive": sum(
            -2 * r["continuous_loss_upper"] - r["baseline"] > 1e-10 for r in flat
        ),
        "flat_if_history_also_corrected": sign_summary(flat),
        "all_risk_if_history_also_corrected": sign_summary(risky),
    }


def audit(artifact, dest):
    verified = verify_manifest(artifact)
    body = BodyValue()
    rows = read_replays(artifact, body)
    live = live_rows(artifact, body)
    risky = [r for r in rows if r["band"] == "practice" and r["action"] >= 2]
    assert (
        len(rows) == 5832
        and sum(r["after"] > 0 for r in rows) == 5784
        and len(risky) == 660
    )
    reference = continuous_feedback(rows, risky)
    grouped = {}
    seeds = []
    for birth in ("both", *BIRTHS):
        sample = [r for r in risky if birth == "both" or birth == r["birth"]]
        grouped[birth] = {a: describe(sample, a) for a in ("one", "gate", "two")}
    for seed in SEEDS:
        for birth in BIRTHS:
            sample = [r for r in risky if r["seed"] == seed and r["birth"] == birth]
            seeds.append({"seed": seed, "birth": birth, **describe(sample, "gate")})
    sample_grid = grid(body)
    assert len(live) == 336 and sum(r["turns"] for r in live) == 40663
    live_groups = {}
    for birth in ("both", *BIRTHS):
        for arm in ARMS:
            group = [
                r
                for r in live
                if r["arm"] == arm and (birth == "both" or r["birth"] == birth)
            ]
            live_groups[birth + "/" + arm] = {
                key: int(sum(r[key] for r in group))
                for key in (
                    "turns",
                    "practice_rounds",
                    "practice_answers",
                    "rul_flat",
                    "positive",
                    "negative",
                    "parameter_changed",
                    "first_avoided",
                )
            }
            live_groups[birth + "/" + arm]["lives"] = len(group)
    dest.mkdir(parents=True, exist_ok=True)
    # The most detailed table is compressed with stable metadata for bytewise reproduction.
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (dest / "rounds.csv.gz").write_bytes(
        gzip.compress(stream.getvalue().encode(), compresslevel=6, mtime=0)
    )
    save_csv(dest / "fixed-body.csv", sample_grid)
    save_csv(dest / "per-seed.csv", seeds)
    save_csv(dest / "live-evidence.csv", live)
    seed_means = defaultdict(list)
    for row in risky:
        seed_means[row["seed"]].append(row["gate_next_delta"])
    values = np.array([100 * np.mean(seed_means[seed]) for seed in SEEDS])
    first = next(
        row for row in risky if row["seed"] == 71101 and row["birth"] == "blank"
    )
    null = next(row for row in risky if abs(row["loss"]) < EPS)
    f = len([r for r in rows if r["first_fatal"]])
    result = {
        "source": "PR 12 at 5d29b1c7081a90e2a0e27ff3a278d417dcb7de44",
        "verified_files": verified,
        "source_seeds": len(SEEDS),
        "free_seeds": len(NEW_SEEDS),
        "source_rounds": len(rows),
        "source_first_fatal": f,
        "source_terminal": sum(r["after"] == 0 for r in rows),
        "successors": sum(r["after"] > 0 for r in rows),
        "schedules_on_fixed_risky_answers": grouped,
        "free": live_groups,
        "risk_gate_mean_next_change_pp_seed_weighted": float(values.mean()),
        "risk_gate_seed_range_pp": values[[values.argmin(), values.argmax()]].tolist(),
        "reference": reference,
        "examples": {
            "first_risky_answer_first_blank": first,
            "first_flat_in_source_order": null,
        },
    }
    (dest / "summary.json").write_text(
        json.dumps(
            result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf8",
        newline="\n",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=PARENT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    audit(args.artifact, args.out)
