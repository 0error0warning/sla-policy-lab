"""Check archives, actual actions, event alignment, no-op snapshots and scalar effect estimates on CPU."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from experiments.direct_policy.protocol import write
from experiments.lifetime_cooperation.audit import physical
from .analysis import analyze
from .design import GRID, ITEMS, PARTNER, branch_key, branches, histories


def sig(x):
    return sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest()


def manifest(path):
    with ZipFile(path) as z:
        m = json.loads(z.read("archive-manifest.json"))
        assert set(z.namelist()) == set(m) | {"archive-manifest.json"}
        for k, v in m.items():
            s = z.read(k)
            assert len(s) == v["bytes"] and sha256(s).hexdigest() == v["sha256"]


def classifications(grid, actual, b0):
    healthy = grid[:, 2:]
    best = healthy.argmax(-1)
    strong = healthy.max(-1) >= 0.9
    labels = []
    for k in range(len(grid)):
        all_four = []
        for a in range(4):
            vals = [
                np.count_nonzero(
                    (best[k, h, PARTNER == p] == a) & strong[k, h, PARTNER == p]
                )
                for h in (0, 1)
                for p in (2, 3)
            ]
            if min(vals) >= 12:
                all_four.append(a)
        labels.append(all_four[0] if all_four else -1)
    actual_label = [
        int(v.argmax())
        if max(v) >= 0.9 and sum(abs(v.astype(float) - b.astype(float))) / 2 >= 0.30
        else -1
        for v, b in zip(actual[:-1], b0[:-1], strict=True)
    ]

    def scan(trace):
        answer = []
        previous = None
        for k in range(len(trace) - 2):
            a = trace[k]
            if a >= 0 and trace[k : k + 3] == [a, a, a] and a != previous:
                answer.append((k + 1, a))
                previous = a
        return answer

    return scan(actual_label), scan(labels)


def verify(root, inputs, out, archive):
    if archive:
        manifest(archive)
    report = json.loads((root / "receipt.json").read_bytes())
    result = analyze(root, inputs, out)
    srcs = histories(inputs)
    with (out / "event-effects.csv").open(encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    evidence = {}
    omcount = 0
    preserved = 0
    counter_steps = 0
    matched = 0
    changes = 0
    ref_hash = 0
    ref_n = 0
    refmax = 0.0
    panelerr = 0.0
    actualerr = 0.0
    max_scalar = 0.0
    totalcells = 0
    for seed, (src, ref, oldbirth, oldfinal) in srcs.items():
        folder = root / "lives" / f"s{seed}"
        plan = json.loads((folder / "plan.json").read_bytes())
        with np.load(folder / "normal.npz") as z:
            panel = z["panels"]
            probs = z["actual"]
            weights = z["weights"]
            pb = z["birth_actual"]
        logs = list(
            map(json.loads, (folder / "normal-updates.jsonl").read_bytes().splitlines())
        )
        n = len(src)
        ref_n += len(ref)
        assert (
            panel.shape == (n, 4, len(GRID), 4)
            and weights.shape == (n, 10242)
            and probs.shape == (n, 4)
        )
        assert (
            max(abs(panel.sum(-1) - 1).ravel()) < 3e-6
            and panel.min() > 0
            and np.isfinite(panel).all()
        )
        assert np.array_equal(probs[0], pb[0])
        refmax = max(
            refmax,
            abs(
                probs[:-1].astype(float)
                - np.array([e["probabilities"][3] for e in ref])
            ).max(),
        )
        panelerr = max(
            panelerr,
            float(abs(panel[0] - oldbirth).max()),
            float(abs(panel[-1] - oldfinal).max()),
        )
        for k, e in enumerate(src):
            assert (
                abs(
                    physical(e["health"], e["action"], e["maintenance"], e["noise"])[-1]
                    - e["after"]
                )
                < 1e-9
            )
            assert e["turn"] == k + 1 and (e["after"] > 0) == (k < n - 1)
            if k < n - 1:
                rec = logs[k]
                q = ref[k]["updates"]["loreft"]
                assert (
                    rec["turn"] == k + 1
                    and rec["before"] == sig(weights[k])
                    and rec["after"] == sig(weights[k + 1])
                )
                ref_hash += int(
                    rec["before"] == q["before"] and rec["after"] == q["after"]
                )
                eq = rec["actual_l2"] - float(
                    np.linalg.norm(weights[k + 1].astype(float) - weights[k])
                )
                assert abs(eq) < 1e-9
        act, fix = classifications(panel, probs, pb)
        for typ, ans in (("actual", act), ("panel", fix)):
            assert sorted(
                (e["turn"], e["action"]) for e in plan["events"] if e["family"] == typ
            ) == sorted(ans)
        target = {v["id"]: v for v in plan["events"]}
        windows = branches(plan["events"])
        assert len(windows) == len(plan["forks"])
        for e in plan["events"]:
            s = e["control"]
            if s:
                assert abs(s - e["turn"]) >= 6 and s >= 4 and s + 2 <= n
                assert not any(
                    s - 3 <= q["turn"] <= s + 2
                    for q in plan["events"]
                    if q["family"] != "no_onset"
                )
                matched += 1
        per = {}
        for omission in windows:
            f = folder / "forks" / branch_key(omission)
            start = min(omission)
            with np.load(f / "trace.npz") as z:
                g = z["panels"]
                v = z["actual"]
                end = z["final_weights"]
                assert omission == tuple(z["omitted"]) and start == int(z["first"])
            assert g.shape == (n - start + 1, 4, len(GRID), 4) and v.shape == (
                n - start + 1,
                4,
            )
            assert np.array_equal(g[0], panel[start - 1]) and np.array_equal(
                v[0], probs[start - 1]
            )
            assert (
                np.isfinite(g).all() and g.min() > 0 and abs(g.sum(-1) - 1).max() < 3e-6
            )
            logs = list(
                map(json.loads, (f / "updates.jsonl").read_bytes().splitlines())
            )
            assert len(logs) == n - start
            chain = sig(weights[start - 1])
            omcount += 1
            for j, rec in enumerate(logs):
                t = start + j
                e = src[t - 1]
                assert (
                    rec["turn"] == t
                    and rec["before"] == chain
                    and rec["skipped"] == (t in omission)
                )
                chain = rec["after"]
                if rec["skipped"]:
                    assert (
                        chain == rec["before"]
                        and rec["actual_l2"] == 0
                        and np.array_equal(g[j], g[j + 1])
                    )
                    preserved += 1
                else:
                    b = e["prob"][0][e["action"]]
                    p = float(v[j, e["action"]])
                    assert (
                        abs(b - rec["behavior"]) < 1e-9
                        and abs(p - rec["current"]) < 1e-9
                    )
                    assert abs(rec["importance"] - np.clip(p / b, 0.5, 2.0)) < 1e-9
                    counter_steps += 1
            assert chain == sig(end)
            per[omission] = (start, g, v)
            totalcells += g.shape[0] * g.shape[1] * g.shape[2]
        erows = [r for r in rows if int(r["seed"]) == seed]
        for row in erows:
            e = target[row["event"]]
            a = e["action"]
            t = int(row["decision"])
            om = tuple(map(int, row["omitted"].split(",")))
            s, g, v = per[om]
            normal = panel[t - 1]
            different = g[t - s]
            for name, p in (("p86", 2), ("p93", 3)):
                j = [i for i, idx in enumerate(GRID) if ITEMS[idx]["partner"] == p]
                difference = (
                    100
                    * sum(
                        float(normal[h, i, a]) - float(different[h, i, a])
                        for h in (2, 3)
                        for i in j
                    )
                    / (2 * len(j))
                )
                error = abs(difference - float(row["delta_" + name]))
                assert error < 1e-6
                max_scalar = max(max_scalar, error)
            actualerr = max(
                actualerr,
                abs(
                    100 * (float(probs[t - 1, a]) - float(v[t - s, a]))
                    - float(row["actual_difference"])
                ),
            )
            changes += 1
        totalcells += n * 4 * len(GRID)
        evidence[seed] = dict(
            observed_turns=n,
            normal_updates=n - 1,
            actual_events=len(act),
            panel_events=len(fix),
            forks=len(windows),
        )
    assert (
        actualerr < 2e-5 and omcount == result["unique_forks"] == report["total_forks"]
    )
    data = dict(
        lives=evidence,
        normal_updates=ref_n,
        normal_hash_pairs_matching_PR18=ref_hash,
        max_PR18_actual_error=float(refmax),
        max_PR18_birth_final_panel_error=panelerr,
        forks=omcount,
        skipped_parameter_writes=preserved,
        recomputed_counterfactual_updates=counter_steps,
        directly_stored_probe_contexts=totalcells,
        ordinary_matches=matched,
        event_effect_rows_rechecked=changes,
        maximum_scalar_audit_error=max_scalar,
        probability_restore_error=actualerr,
        device_panel_check_error=report["max_panel_probability_error"],
    )
    write(out / "audit.json", data)
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.root, a.inputs, a.out, a.archive), indent=2))
