"""Physically rotate the last residual and load original RMSNorm weights in real Qwen."""

import numpy as np
import torch
from torch import nn

from experiments.consequence_policy.study import TEST_H, basis
from experiments.direct_policy.protocol import write
from experiments.internal_body.capture import encode
from experiments.internal_body.carrier import rotate
from experiments.internal_body.suffix import distribution


class FixedRotation:
    def __init__(self, lm, u, v):
        self.u = torch.as_tensor(u, dtype=torch.float32, device="cuda")
        self.v = torch.as_tensor(v, dtype=torch.float32, device="cuda")
        self.enabled = True
        self.health(90.0)

        def before(_, args):
            if not self.enabled:
                return None
            r = args[0]
            new = rotate(r[:, -1].float()[None], self.g, self.u, self.v)[0]
            return (torch.cat((r[:, :-1], new[:, None].to(r.dtype)), dim=1), *args[1:])

        self.handle = lm.norm.register_forward_pre_hook(before)

    def health(self, h):
        self.g = torch.as_tensor(basis([h]), dtype=torch.float32, device="cuda")

    def close(self):
        self.handle.remove()


def all_logits(actor, ids):
    x = torch.as_tensor([ids], device="cuda")
    return (
        actor.model(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            use_cache=False,
            logits_to_keep=1,
        )
        .logits[0, -1]
        .float()
    )


def checks(actor, root, out, rows):
    from experiments.internal_body.live import digest

    lm = actor.model.model.language_model
    original = lm.norm.weight
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
    ids = {i: encode(actor, rows[i]) for i in selected}
    questions = [
        "What color is grass?",
        "What is two plus seven?",
        "Name one fruit.",
        "What month comes after April?",
        "What is the opposite of cold?",
        "Complete the sequence: one, two,",
        "What is the capital of France?",
        "Name an animal that can fly.",
    ]
    neutral = [
        actor.tok.encode(
            actor.tok.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            add_special_tokens=False,
        )
        for q in questions
    ]
    with torch.no_grad():
        targets = {
            i: all_logits(actor, x)[actor.ids].cpu().numpy() for i, x in ids.items()
        }
        original_text = [all_logits(actor, x).detach() for x in neutral]
    result = []
    language = []
    hashes = []
    for seed, m in ((64501, 0), (64502, 1)):
        folder = root / "native" / f"s{seed}-m{m}"
        z = np.load(root / "native" / f"basis-{seed}.npz")
        expected = np.load(folder / "expected.npz")
        for mode in ("birth", "sensor", "shuffle", "exchanged", "removed"):
            rotation = FixedRotation(
                lm, z["u"], z["swapped_v"] if mode == "exchanged" else z["v"]
            )
            if mode == "removed":
                rotation.enabled = False
            delta = (
                np.zeros(original.shape, np.float32)
                if mode == "birth"
                else np.load(
                    folder
                    / (
                        "native-shuffle.npz"
                        if mode == "shuffle"
                        else "native-sensor.npz"
                    )
                )["delta"]
            )
            lm.norm.weight = nn.Parameter(
                (original.float() + torch.as_tensor(delta, device="cuda")).to(
                    original.dtype
                ),
                requires_grad=False,
            )
            try:
                hashes.append(
                    {
                        "seed": seed,
                        "mapping": m,
                        "mode": mode,
                        "name": "model.language_model.norm.weight",
                        "original": digest(original),
                        "deployed": digest(lm.norm.weight),
                        "changed": int((lm.norm.weight != original).sum()),
                    }
                )
                with torch.no_grad():
                    for i in selected:
                        for h in (2.0, 6.0, 90.0, 300.0):
                            rotation.health(h)
                            q = (
                                distribution(all_logits(actor, ids[i])[actor.ids])
                                .cpu()
                                .numpy()
                            )
                            key = {
                                "birth": "frozen",
                                "exchanged": "carrier_exchanged",
                                "removed": "carrier_removed",
                            }.get(mode, mode)
                            result.append(
                                {
                                    "seed": seed,
                                    "mapping": m,
                                    "mode": mode,
                                    "i": i,
                                    "h": h,
                                    "full": q.tolist(),
                                    "original_batch": expected[key][
                                        TEST_H.index(h), i - 768
                                    ].tolist(),
                                    "same_process_birth": distribution(
                                        torch.tensor(targets[i])
                                    ).tolist(),
                                }
                            )
                    for j, x in enumerate(neutral):
                        for h in (2.0, 300.0):
                            rotation.health(h)
                            new = all_logits(actor, x)
                            lo = torch.log_softmax(original_text[j], dim=0)
                            ln = torch.log_softmax(new, dim=0)
                            language.append(
                                {
                                    "seed": seed,
                                    "mode": mode,
                                    "question": questions[j],
                                    "health": h,
                                    "kl_original_to_changed": float(
                                        (lo.exp() * (lo - ln)).sum()
                                    ),
                                    "original_top": int(lo.argmax()),
                                    "new_top": int(ln.argmax()),
                                }
                            )
            finally:
                lm.norm.weight = original
                rotation.close()
    write(out / "carrier-full.json", result)
    write(out / "carrier-native-parameters.json", hashes)
    write(out / "neutral-vocabulary.json", language)
    return len(result)
