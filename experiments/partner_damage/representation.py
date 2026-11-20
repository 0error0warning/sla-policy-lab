"""Cross-word, cross-task decoding of paired partner names at decision and ID tokens."""

import torch
import numpy as np
from experiments.direct_policy.protocol import write
from .design import TOKEN_PANELS, swapped


def capture(batch, out):
    actor = batch.engine.actor
    early = batch.engine.lm.layers[3]
    state = {}
    position = {}
    diff_tokens = {}
    for *_, i in TOKEN_PANELS:
        a = batch.q.inputs.ids[i]
        b = batch.q.inputs.ids[swapped(i)]
        if len(a) != len(b):
            raise ValueError("partner token alignment changed")
        differences = [j for j, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
        if not differences:
            raise ValueError("partner tokens identical")
        position[i] = differences[-1]
        diff_tokens[i] = dict(
            indices=differences,
            tokens=[actor.tok.convert_ids_to_tokens(int(a[j])) for j in differences],
            prompt_tokens=len(a),
        )

    def read_early(_, args, output):
        h = output[0] if isinstance(output, tuple) else output
        state["early_decision"] = h[0, -1].detach().float().cpu().numpy().copy()
        state["early_id"] = (
            h[0, position[state["i"]]].detach().float().cpu().numpy().copy()
        )

    def read_final(raw):
        state["terminal_decision"] = raw[0].detach().float().cpu().numpy().copy()
        return raw

    handle = early.register_forward_hook(read_early)
    data = {k: [] for k in ("early_decision", "early_id", "terminal_decision")}
    archive_rows = []
    try:
        with torch.no_grad():
            batch.engine.custom = read_final
            for w, h, p, f, l, i in TOKEN_PANELS:
                state["i"] = i
                batch.q.ordinary(i, h, batch.weights)
                for k in data:
                    data[k].append(state[k])
                archive_rows.append(
                    dict(
                        split=w,
                        health=h,
                        partner=p,
                        frame=f,
                        layout=l,
                        row=i,
                        **diff_tokens[i],
                    )
                )
    finally:
        batch.engine.custom = None
        handle.remove()
    np.savez_compressed(
        out / "representations.npz", **{k: np.asarray(v) for k, v in data.items()}
    )
    write(out / "representation-rows.json", archive_rows)
