"""Reinstall final-block LoRA on full Qwen and compare decisions on a matched health panel."""

import json

import numpy as np
import torch
from torch import nn

from experiments.consequence_policy.study import basis, contexts
from experiments.direct_policy.protocol import write
from experiments.internal_body.capture import DecisionState
from experiments.internal_body.suffix import distribution
from experiments.lethal_boundary.design import heldout_panel
from experiments.lethal_boundary.gpu import encode, pack_bf16
from experiments.lethal_boundary.states import WITNESS_H


def run(actor, root, out):
    rows = contexts()[768:]
    ix = heldout_panel()
    counts = (None, 7, 15)
    plan_file = root / "installation-plan.json"
    plan = (
        json.loads(plan_file.read_bytes())
        if plan_file.exists()
        else {
            "seeds": [67101, 67102],
            "arms": ["D", "R", "M"],
            "senses": ["plain", "record"],
            "health": WITNESS_H.tolist(),
        }
    )
    gs = torch.tensor(basis(plan["health"]), device="cuda", dtype=torch.float32)
    projection = torch.tensor(
        np.load(root / "architecture.npz")["projection"], device="cuda"
    )
    ids = {(k, i): encode(actor, rows[i], k) for k in counts for i in ix}
    lm = actor.model.model.language_model
    router = lm.layers[-1].mlp.down_proj
    state = DecisionState(actor)
    pieces = {}

    def observe_skip(_, args):
        pieces["skip"] = args[0][:, -1].detach()

    def observe_down(_, args, y):
        pieces["down"] = y[:, -1].detach()

    def observe_feature(_, args):
        pieces["feature"] = args[0][:, -1].float() @ projection

    handles = [
        lm.layers[-1].post_attention_layernorm.register_forward_pre_hook(observe_skip),
        router.register_forward_hook(observe_down),
        router.register_forward_pre_hook(observe_feature),
    ]
    base = {}
    try:
        with torch.inference_mode():
            for j, k in enumerate(counts):
                measurements = {}
                for i in ix:
                    tt = state.batch([ids[k, i]] * len(gs), include_down=False)
                    for key in ("r", "z", "n", "x", "skip", "down"):
                        if key == "n":
                            arr = tt["native"]
                        elif key == "x":
                            arr = pieces["feature"].cpu().numpy()
                        elif key in ("r", "z"):
                            arr = pack_bf16(tt[key])
                        else:
                            arr = pack_bf16(pieces[key].float().cpu().numpy())
                        measurements.setdefault(key, []).append(arr)
                base.update(
                    {f"c{j}-{key}": np.stack(v) for key, v in measurements.items()}
                )
    finally:
        for handle in handles:
            handle.remove()
        state.close()
    np.savez_compressed(out / "fresh-components.npz", **base, indices=ix)
    done = []
    results = {}
    for seed in plan["seeds"]:
        for m in (0, 1):
            for sense in plan["senses"]:
                for arm in plan["arms"]:
                    name = f"s{seed}-m{m}-{sense}-{arm}"
                    with np.load(root / "weights" / (name + ".npz")) as z:
                        block = nn.Module()
                        for key in ("A", "B"):
                            block.register_parameter(
                                key,
                                nn.Parameter(
                                    torch.tensor(z[key], device="cuda"),
                                    requires_grad=False,
                                ),
                            )
                    router.add_module("boundary_lora", block)
                    context = {}

                    def read(_, args, context=context):
                        context["x"] = args[0][:, -1].float() @ projection

                    def correct(_, args, y, context=context, block=block):
                        matrix = torch.einsum(
                            "nb,bdr->ndr", gs, block.A[context["partner"]]
                        )
                        delta = (
                            torch.einsum("nd,ndr->nr", context["x"], matrix) @ block.B
                        )
                        return torch.cat(
                            (
                                y[:, :-1],
                                (y[:, -1].float() + delta).to(y.dtype)[:, None],
                            ),
                            dim=1,
                        )

                    h1 = router.register_forward_pre_hook(read)
                    h2 = router.register_forward_hook(correct)
                    try:
                        with torch.inference_mode():
                            for j, k in enumerate(counts):
                                if (k is None) != (sense == "plain"):
                                    continue
                                vals = []
                                for i in ix:
                                    context["partner"] = rows[i]["partner"]
                                    it = torch.tensor(
                                        [ids[k, i]] * len(gs), device="cuda"
                                    )
                                    outs = actor.model(
                                        input_ids=it,
                                        attention_mask=torch.ones_like(it),
                                        use_cache=False,
                                        logits_to_keep=1,
                                    )
                                    vals.append(
                                        distribution(
                                            outs.logits[:, -1, actor.ids].float()
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                results[f"{name}-c{j}"] = np.stack(vals)
                    finally:
                        h1.remove()
                        h2.remove()
                        delattr(router, "boundary_lora")
                    done.append(name)
                    write(
                        out / "status.json",
                        {
                            "stage": "native_installations",
                            "completed": len(done),
                            "last": name,
                        },
                    )
    np.savez_compressed(out / "full-probes.npz", **results)
    frozen_after = DecisionState(actor)
    try:
        with torch.inference_mode():
            native = frozen_after.batch(
                [ids[None, ix[0]]] * len(gs), include_down=False
            )["native"]
        np.testing.assert_array_equal(native, base["c0-n"][0])
    finally:
        frozen_after.close()
    return {
        "checkpoints": len(done),
        "contexts": len(ix),
        "health": plan["health"],
        "record_counts": counts,
        "decisions": sum(x.shape[0] * x.shape[1] for x in results.values()),
        "disabled_native_identical": True,
    }
