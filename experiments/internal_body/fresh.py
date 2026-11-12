"""Independent tail states for quantized replay; current body is never stored in the plain cache."""

import numpy as np
import torch

from experiments.consequence_policy.study import TEST_H
from experiments.direct_policy.protocol import write
from experiments.internal_body.capture import DecisionState, encode
from experiments.internal_body.design import all_rows
from experiments.internal_body.live import Installed, full
from experiments.internal_body.suffix import distribution


def capture(actor, root, out):
    out.mkdir(exist_ok=True)
    rows = all_rows()
    lm = actor.model.model.language_model
    with np.load(root / "tail-architecture.npz") as z:
        proj = z["projection"]
    reader = DecisionState(actor)
    parts = {}

    def skip(_, args):
        parts["skip"] = args[0][:, -1].detach().float().cpu().numpy()[0]

    def down(_, args, val):
        parts["down"] = val[:, -1].detach().float().cpu().numpy()[0]

    a = lm.layers[-1].post_attention_layernorm.register_forward_pre_hook(skip)
    b = lm.layers[-1].mlp.down_proj.register_forward_hook(down)
    frozen_error = []
    rms_error = []
    try:
        for stage, h in enumerate((None, *TEST_H)):
            ids = list(range(768, len(rows) if h is None else 960))
            data = {}
            for i in ids:
                x = reader.batch([encode(actor, rows[i], h)])
                v = dict(
                    r=x["r"][0],
                    features=(x["v"] @ proj)[0],
                    native=x["native"][0],
                    z=x["z"][0],
                    **parts,
                )
                for k, q in v.items():
                    data.setdefault(k, []).append(q.copy())
            batch = {k: np.stack(value) for k, value in data.items()}
            r = (
                (
                    torch.from_numpy(batch["skip"]).bfloat16()
                    + torch.from_numpy(batch["down"]).bfloat16()
                )
                .float()
                .numpy()
            )
            frozen_error.append(float(np.max(np.abs(r - batch["r"]))))
            with torch.no_grad():
                zn = (
                    lm.norm(
                        torch.tensor(batch["r"], device="cuda", dtype=torch.bfloat16)
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
            rms_error.append(float(np.max(np.abs(zn - batch["z"]))))
            np.savez_compressed(
                out / ("plain.npz" if h is None else f"h{int(h)}.npz"),
                index=ids,
                **batch,
            )
            write(
                out / "status.json",
                {"stage": "independent_singletons", "finished": stage + 1, "total": 9},
            )
    finally:
        a.remove()
        b.remove()
        reader.close()
    checked = []
    cfg = __import__("json").loads((root / "tail-architecture.json").read_bytes())
    selected = [
        next(
            i
            for i, r in enumerate(rows)
            if r["split"] == "test"
            and r["layout"] == j
            and r["partner"] == j % 2
            and r["frame"] == j // 2 % 2
            and r["task"] == 4 + j // 4 % 2
        )
        for j in range(24)
    ]
    for kind in ("linear", "nonlinear", "postnorm", "native_prompt", "prompt_lora"):
        with np.load(root / "tail-witness" / (kind + ".npz")) as z:
            w = {k: z[k] for k in z if k != "kind"}
        layer = Installed(actor, kind, w, proj, cfg["norm_offset"], cfg["epsilon"])
        try:
            with torch.no_grad():
                for i in selected:
                    for h in (2.0, 6.0, 90.0, 300.0):
                        layer.condition(rows[i]["partner"], h)
                        ids = encode(
                            actor,
                            rows[i],
                            h if kind in ("native_prompt", "prompt_lora") else None,
                        )
                        pi = distribution(full(actor, ids)).cpu().tolist()
                        checked.append({"kind": kind, "i": i, "h": h, "prob": pi})
        finally:
            layer.close()
    write(out / "actual-tail-witness.json", checked)
    write(
        out / "component-check.json",
        {
            "max_skip_plus_down_error": frozen_error,
            "max_native_normalization_replay_error": rms_error,
        },
    )
    return {
        "plain": len(rows) - 768,
        "prompt": len(TEST_H) * 192,
        "full_tail_comparisons": len(checked),
        "max_component_disagreement": max(frozen_error),
        "max_normalization_disagreement": max(rms_error),
    }
