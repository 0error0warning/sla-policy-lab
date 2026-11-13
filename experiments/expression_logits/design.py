"""Immutable #22 checkpoints, matched requests and the exact score-to-sampler definitions."""

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.special import expit

from experiments.forecast_stages.design import DENSE, inputs, oracle, random_panel
from experiments.lifetime_cooperation.design import ITEMS, PANEL, PILOT

PARENT_INPUT = "1e305cc5f14759c07db1117fd0d078738966c76db221d964572cf9662e861184"
PARENT_RESULTS = {
    "first": "ed728c7b04f8bf9775cfa9dff95870ee96e71315ab594ef438c42d96149b7bdd",
    "second": "1338e56f10ac8479bf1edac86ab695abab86ee41ca2d703b69b5f1481fe484f9",
}
CASE_SEED = 77102
CASE_ID = 369
TWIN_ID = 345
LAYOUT_IDS = (360, 376, 378)
TASK_ID = 321
FOCUS = (CASE_ID, TWIN_ID, *LAYOUT_IDS, TASK_ID)
CASE_COL = int(np.flatnonzero(PANEL[1] == CASE_ID)[0])
COLUMNS = ("seed", "completed", "wording_index", "panel_column", "prompt_id")


def specifications():
    """No frame corresponds to a different source of parameters within a request triplet."""
    for t in range(9):
        for w in (0, 1) if t == 8 else (1,):
            for j in range(32) if t == 8 else DENSE:
                yield t, w, int(j), int(PANEL[w, j])


def check_case():
    assert PANEL[0, CASE_COL] == TWIN_ID
    value = ITEMS[CASE_ID]
    assert all(
        value[k] == ITEMS[TWIN_ID][k] for k in ("partner", "frame", "task", "layout")
    )
    assert value["partner"] == 3 and value["text"] != ITEMS[TWIN_ID]["text"]
    for i in LAYOUT_IDS:
        assert all(
            value[k] == ITEMS[i][k]
            for k in ("partner", "frame", "task", "text", "wording")
        )
    assert all(
        value[k] == ITEMS[TASK_ID][k] for k in ("partner", "frame", "layout", "wording")
    )
    assert value["task"] != ITEMS[TASK_ID]["task"]


def arrays(z, name):
    return np.load(BytesIO(z.read(name)), allow_pickle=False)


def prepare(parent, dest):
    check_case()
    dest.mkdir(parents=True, exist_ok=True)
    source = parent / "gpu/inputs.zip"
    assert sha256(source.read_bytes()).hexdigest() == PARENT_INPUT
    with ZipFile(source) as z:
        for k in ("architecture.npz", "birth-source.npz"):
            (dest / k).write_bytes(z.read(k))
    results = [parent / "gpu" / tag / "results.zip" for tag in PARENT_RESULTS]
    for path, digest in zip(results, PARENT_RESULTS.values(), strict=True):
        assert sha256(path.read_bytes()).hexdigest() == digest
    keys, loc, forecasts, ps, uniforms, eta = [], [], [], [], [], []
    checkpoints, params, heads, hashes = [], [], [], []
    with ZipFile(results[0]) as a, ZipFile(results[1]) as b:
        with (
            arrays(a, "output/features.npz") as f,
            arrays(b, "output/features.npz") as other,
        ):
            assert np.array_equal(f["ids"], other["ids"]) and np.array_equal(
                f["x"], other["x"]
            )
            x = {int(i): v for i, v in zip(f["ids"], f["x"], strict=True)}
        for seed in PILOT:
            prefix = f"output/lives/s{seed}/"
            uq, noise = random_panel(seed)
            env = oracle(seed)
            with (
                arrays(a, prefix + "states.npz") as s,
                arrays(a, prefix + "stages.npz") as panels,
                arrays(b, prefix + "states.npz") as sb,
                arrays(b, prefix + "stages.npz") as pb,
            ):
                for name in ("policy", "head", "sha", "uniforms", "noise", "selected"):
                    assert np.array_equal(s[name], sb[name])
                for name in ("prob", "q", "dense_prob", "dense_q"):
                    assert np.array_equal(panels[name], pb[name])
                assert np.array_equal(uq, s["uniforms"])
                assert np.array_equal(noise, s["noise"])
                rec = list(specifications())
                selected = s["selected"].tolist()
                if seed == CASE_SEED:
                    assert selected[2] == 31
                    rec.extend(
                        (31, w, j, int(i))
                        for w, ix in enumerate(PANEL)
                        for j, i in enumerate(ix)
                    )
                slots = {}
                source_weights = s["policy"]
                source_heads = s["head"]
                source_hashes = s["sha"]
                full_q, full_p, small_q, small_p = (
                    panels[k] for k in ("q", "prob", "dense_q", "dense_prob")
                )
                for t, w, j, i in rec:
                    if t not in slots:
                        slots[t] = len(checkpoints)
                        checkpoints.append([seed, t])
                        coeff = source_weights[t]
                        hsh = source_hashes[t]
                        assert (
                            sha256(coeff.astype("<f4").tobytes()).hexdigest()
                            == hsh.decode()
                        )
                        hashes.append(hsh)
                        params.append(coeff)
                        heads.append(source_heads[t])
                    k = slots[t]
                    if t in selected:
                        stage = selected.index(t)
                        q = full_q[stage, w, j]
                        probs = full_p[stage, w, 0, j]
                    else:
                        col = int(np.flatnonzero(DENSE == j)[0])
                        q = small_q[t, col]
                        probs = small_p[t, col]
                    # The stored personal head must produce the supplied learned forecast.
                    learned = expit(heads[k] @ x[i])
                    np.testing.assert_allclose(
                        q, inputs(learned, env[w, j]), atol=1e-12, rtol=0
                    )
                    keys.append([seed, t, w, j, i])
                    loc.append(k)
                    forecasts.append(q)
                    ps.append(probs)
                    uniforms.append(uq[j])
                    eta.append(noise[j])
    np.savez_compressed(
        dest / "frozen-snapshots.npz",
        keys=keys,
        slots=loc,
        checkpoints=checkpoints,
        policy=params,
        head=heads,
        policy_sha=hashes,
        q=forecasts,
        previous_prob=ps,
        uniforms=uniforms,
        noise=eta,
    )
    print(
        f"{len(checkpoints)} frozen states, {len(keys)} matched requests, {3 * len(keys)} forecasts"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for field in ("parent", "dest"):
        p.add_argument("--" + field, type=Path, required=True)
    prepare(**vars(p.parse_args()))
