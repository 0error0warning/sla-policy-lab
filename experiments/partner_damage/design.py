"""Factual traces, health-only learning signal, and independently crossed social/physical outcomes."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np

from experiments.direct_policy.protocol import settle, write
from experiments.lifetime_cooperation.design import ITEMS, LEVELS, LOOKUP, PANEL, PILOT
from experiments.social_closed_loop.core import BodyValue

SHA17 = "742ce68fafdeab6d50f59b4029c48f17bab63ab67cbde830d83a85dba6139abc"
SHA19 = "bf40fadb5ed4c10e9a3f4a75761239b9719de68bedbb0e7ae6ce1d188b9942e8"
GRID = PANEL[1]
PARTNER = np.asarray([ITEMS[i]["partner"] for i in GRID])
HEADLINES = {(77101, 33): 2, (77101, 37): 1}
CELLS = ("accept_light", "decline_light", "accept_heavy", "decline_heavy")
DOSES = ((True, True), (False, True), (True, False), (False, False))
HYBRIDS = ("pre", "post", "removed", "restored", "swap_increment", "source_routed")
SITES = ("early", "terminal", "early_matched", "terminal_matched")
STRENGTH = (1.0, 0.5, 0.25, 0.125, 0.0625)
SMALL = tuple(
    (k, j) for k in (1, 2) for j, i in enumerate(GRID) if ITEMS[i]["layout"] == 0
)
TOKEN_PANELS = tuple(
    (w, h, p, f, l, LOOKUP[p, f, 4 + w, 3 + w, l])
    for w in (0, 1)
    for h in ((90.0,) if w == 0 else LEVELS)
    for f in (0, 1)
    for l in range(24)
    for p in (2, 3)
)


def swapped(i):
    r = ITEMS[i]
    return LOOKUP[5 - r["partner"], r["frame"], r["task"], r["wording"], r["layout"]]


PAIRS = np.asarray([list(GRID).index(swapped(int(i))) for i in GRID])


def feedback(body, e, social, after):
    """A full outcome record is retained. The existing advantage reads only post-health and past baseline."""
    raw = body.rul(after) - body.rul(e["health"])
    return dict(
        maintenance=social,
        after=float(after),
        damage=float(e["health"] - after),
        reward=raw,
        signal=raw - e["baseline"],
    )


def factorial(body, e):
    if not e["action"]:
        raise ValueError("silent refusal has no observed partner response")
    outcomes = {
        rewarded: settle(e["health"], e["action"], rewarded, e["noise"])[0]
        for rewarded in (True, False)
    }
    return {
        name: feedback(body, e, social, outcomes[physics])
        for name, (social, physics) in zip(CELLS, DOSES, strict=True)
    }


def verified_inputs(src17, src19, dest):
    for path, sha in zip((src17, src19), (SHA17, SHA19), strict=True):
        assert sha256(path.read_bytes()).hexdigest() == sha
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    catalog = []
    traces = {}
    params = {}
    body = BodyValue()
    with ZipFile(src17) as a, ZipFile(src19) as b:
        for seed in PILOT:
            events = [
                json.loads(v)
                for v in a.read(
                    f"output/lives/s{seed}/stable/qwen_learn/events.jsonl"
                ).splitlines()
            ][:-1]
            traces[str(seed)] = events
            with np.load(BytesIO(b.read(f"output/lives/s{seed}/normal.npz"))) as z:
                chosen = {}
                total = 0.0
                for k, e in enumerate(events):
                    assert abs(e["baseline"] - (total - 8) / (8 + k)) < 1e-9
                    assert (
                        abs(
                            e["signal"]
                            - (
                                body.rul(e["after"])
                                - body.rul(e["health"])
                                - e["baseline"]
                            )
                        )
                        < 1e-9
                    )
                    total += e["reward"]
                    if not e["action"]:
                        continue
                    person = ITEMS[e["row"]]["partner"]
                    m = e["maintenance"]
                    cf = factorial(body, e)
                    idx = 0 if m else 3
                    assert abs(cf[CELLS[idx]]["after"] - e["after"]) < 1e-10
                    rows.append(
                        dict(
                            seed=seed,
                            turn=k + 1,
                            partner=person,
                            action=e["action"],
                            maintenance=m,
                            health=e["health"],
                            light_damage=cf[CELLS[0]]["damage"],
                            heavy_damage=cf[CELLS[3]]["damage"],
                            light_signal=cf[CELLS[0]]["signal"],
                            heavy_signal=cf[CELLS[3]]["signal"],
                        )
                    )
                    chosen.setdefault((person, m), k + 1)
                for t in sorted(
                    set(chosen.values()) | {u for s, u in HEADLINES if s == seed}
                ):
                    e = events[t - 1]
                    key = f"s{seed}t{t}"
                    role = dict(
                        seed=seed,
                        turn=t,
                        key=key,
                        ordinary=t in chosen.values(),
                        focus=(seed, t) in HEADLINES,
                        factual=e,
                        cell_records=factorial(body, e),
                    )
                    catalog.append(role)
                    for n, value in (
                        ("pre", z["weights"][t - 1]),
                        ("post", z["weights"][t]),
                        ("prepanel", z["panels"][t - 1]),
                        ("postpanel", z["panels"][t]),
                    ):
                        params[key + "_" + n] = value
                    if role["focus"]:
                        for k in (
                            (35, len(events) + 1) if t == 33 else (46, len(events) + 1)
                        ):
                            params[key + f"_d{k}"] = z["panels"][k - 1]
    write(dest / "events.json", catalog)
    write(dest / "tapes.json", traces)
    write(dest / "physical-audit.json", rows)
    np.savez_compressed(dest / "anchors.npz", **params)
    return dict(
        selected=len(catalog),
        ordinary=sum(e["ordinary"] for e in catalog),
        focus=sum(e["focus"] for e in catalog),
        nonsilent=len(rows),
        nonterminal=sum(map(len, traces.values())),
        counts=dict(Counter(str(r["maintenance"]) for r in rows)),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source17", type=Path, required=True)
    p.add_argument("--source19", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    a = p.parse_args()
    print(verified_inputs(a.source17, a.source19, a.dest))
