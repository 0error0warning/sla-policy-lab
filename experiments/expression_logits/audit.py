"""CPU audit from archived snapshots, raw native token logits and actual probe uniforms."""

import argparse
import csv
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.consequence_policy.study import EXPLORATION, TEMP, WORDS
from experiments.direct_policy.protocol import write
from experiments.forecast_stages.design import VIEWS
from experiments.lifetime_cooperation.audit import physical
from experiments.lifetime_cooperation.design import PILOT
from experiments.loreft_omission.audit import manifest

from .analysis import analyze
from .design import CASE_COL, CASE_ID, CASE_SEED, TWIN_ID, prepare


def verify_inputs(parent, inputs):
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        prepare(parent, tmp)
        for n in ("architecture.npz", "birth-source.npz"):
            assert (inputs / n).read_bytes() == (tmp / n).read_bytes()
        with (
            np.load(inputs / "frozen-snapshots.npz") as src,
            np.load(tmp / "frozen-snapshots.npz") as rebuilt,
        ):
            assert set(src.files) == set(rebuilt.files)
            for k in src.files:
                assert np.array_equal(src[k], rebuilt[k]), k


def sampler(z):
    # Reconstruct the policy independently of the GPU code.
    s = z.astype(float) / TEMP
    e = np.exp(s - s.max(-1, keepdims=True))
    return (1.0 - EXPLORATION) * e / e.sum(-1, keepdims=True) + EXPLORATION / 4


def choices_table(path):
    with gzip.open(path, "rt", encoding="utf8", newline="") as f:
        return {
            (int(r["seed"]), int(r["completed"]), int(r["prompt"]), r["view"]): r
            for r in csv.DictReader(f)
        }


def audit(src, inputs, parent, archive, out):
    manifest(archive)
    verify_inputs(parent, inputs)
    res = analyze(src, inputs, out)
    rec = json.loads((src / "receipt.json").read_bytes())
    assert (
        rec["frozen_backbone"]
        and rec["training_updates"] == 0
        and rec["native_logit_dtype"] == "torch.bfloat16"
    )
    assert tuple(rec["words"]) == WORDS and tuple(rec["views"]) == VIEWS
    raw = choices_table(out / "native_views.csv.gz")
    assert len(raw) == res["n_rows"] and rec["mirror_error"] == 0
    assert (
        rec["score_policy_error"] == 0 and rec["model_forward_calls"] == len(raw) // 3
    )
    max_policy_error = 0.0
    matched = 0
    corpus = {}
    with np.load(inputs / "frozen-snapshots.npz") as s:
        keys, qs, us, noise, old = (
            s[k] for k in ("keys", "q", "uniforms", "noise", "previous_prob")
        )
        for seed in PILOT:
            subset = np.flatnonzero(keys[:, 0] == seed)
            with np.load(src / "lives" / f"s{seed}" / "scores.npz") as v:
                assert np.array_equal(v["rows"], subset) and np.array_equal(
                    v["keys"], keys[subset]
                )
                z, p = v["scores"], v["prob"]
                assert (
                    z.shape == (len(subset), 3, 4)
                    and p.shape == z.shape
                    and np.isfinite(z).all()
                )
                cpu = sampler(z)
                err = float(abs(cpu - p).max())
                max_policy_error = max(max_policy_error, err)
                np.testing.assert_allclose(cpu, p, atol=2e-7, rtol=0)
                assert np.all((p >= EXPLORATION / 4 - 2e-7) & (p < 1))
                for k, n in enumerate(subset):
                    _, t, _w, j, i = map(int, keys[n])
                    u, vv = us[n]
                    real = qs[n, 1]
                    if t == 0:
                        assert np.array_equal(
                            z[k], np.broadcast_to(z[k, :1], z[k].shape)
                        )
                    for m, name in enumerate(VIEWS):
                        row = raw[(seed, t, i, name)]
                        logits = z[k, m].astype(float)
                        pp = p[k, m].astype(float)
                        assert np.array_equal(
                            logits, [float(row["z_" + word]) for word in WORDS]
                        )
                        assert np.array_equal(
                            pp, [float(row["p_" + word]) for word in WORDS]
                        )
                        assert np.array_equal(
                            qs[n, m],
                            [float(row["q_" + w]) for w in ("ask", "help", "trade")],
                        )
                        assert logits[3] - logits[2] == float(row["gap_trade_help"])
                        a = int(
                            np.searchsorted(
                                np.cumsum(pp, dtype=float)[:3], u, side="left"
                            )
                        )
                        actual = int(
                            np.searchsorted(np.cumsum(cpu[k, m])[:3], u, side="left")
                        )
                        assert a == actual and row["action"] == WORDS[a]
                        previous = int(
                            np.searchsorted(
                                np.cumsum(old[n, m], dtype=float)[:3], u, side="left"
                            )
                        )
                        assert row["archive_action"] == WORDS[previous]
                        response = None if not a else bool(vv < real[a - 1])
                        h = physical(6.0, a, response, noise[n])[-1]
                        assert float(row["u"]) == u and row["maintained"] == (
                            "" if response is None else str(response)
                        )
                        assert abs(float(row["after"]) - h) < 1e-10 and int(
                            row["completed_task"]
                        ) == int(a >= 2 and h > 0)
                        matched += int(a == previous)
                    if seed == CASE_SEED and t == 31 and i in (CASE_ID, TWIN_ID):
                        assert j == CASE_COL
                        corpus[i] = [raw[(seed, t, i, k)] for k in VIEWS]
    for k in (
        "u",
        "partner_u",
        "policy_sha",
        "health",
        "partner",
        "frame",
        "task",
        "layout",
    ):
        assert corpus[CASE_ID][0][k] == corpus[TWIN_ID][0][k]
    with (out / "archived_case_boundaries.csv").open(
        encoding="utf8", newline=""
    ) as stream:
        parent_actions = {
            (int(r["prompt"]), r["view"]): r["action"] for r in csv.DictReader(stream)
        }
    assert [parent_actions[CASE_ID, key] for key in VIEWS] == ["trade", "trade", "help"]
    assert all(parent_actions[TWIN_ID, key] == "help" for key in VIEWS)
    info = {
        "complete_native_forwards": int(rec["model_forward_calls"]),
        "checked_decisions": len(raw),
        "archive_matching_actions": matched,
        "native_probability_cpu_max_error": max_policy_error,
        "cases": res["cases"],
        "score_dtype": rec["native_logit_dtype"],
        "mirror_error": rec["mirror_error"],
        "native_reconstruction_error": rec["score_policy_error"],
        "training_updates": 0,
    }
    assert matched == res["same_archived_choices"]
    write(out / "audit.json", info)
    return info


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for field in ("src", "inputs", "parent", "archive", "out"):
        p.add_argument("--" + field, type=Path, required=True)
    print(json.dumps(audit(**vars(p.parse_args()))))
