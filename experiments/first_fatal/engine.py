"""Use the ordinary Qwen computation or reuse its immutable upstream tensors on the same device."""

import numpy as np
import torch

from experiments.consequence_policy.study import basis
from experiments.individual_life.design import rows
from experiments.individual_life.model import (
    Complete,
    Individual,
    correction,
)
from experiments.internal_body.capture import encode
from experiments.internal_body.suffix import distribution


class Inputs:
    def __init__(self, actor, root):
        self.rows = rows()
        self.ids = [encode(actor, row) for row in self.rows]
        self.slot = np.array([r["partner"] - 2 for r in self.rows])
        self.probe_ix = [
            i for i, r in enumerate(self.rows) if (r["task"], r["wording"]) == (5, 4)
        ]
        self.source = root / "birth-source.npz"

    def gate(self, h, visible):
        assert visible
        return torch.as_tensor(basis([h])[0], dtype=torch.float32, device="cuda")

    def new(self, birth):
        return Individual(self.source, birth, "cuda")


class NumericalMismatch(RuntimeError):
    pass


class Qwen:
    def __init__(self, actor, root):
        self.inputs = Inputs(actor, root)
        self.rows = self.inputs.rows
        self.probe_ix = self.inputs.probe_ix
        self.actor = actor
        self.full = Complete(actor, self.inputs, root)
        self.lm = actor.model.model.language_model
        self.cache = {}
        self.fast = True
        self.stats = {
            "full_forwards": 0,
            "tail_forwards": 0,
            "paired_panels": 0,
            "paired_gradients": 0,
            "max_prob_error": 0.0,
            "max_gradient_error": 0.0,
        }

    def new(self, birth):
        return self.inputs.new(birth)

    def bind(self, main, control, near):
        self.full.bind(main, control)
        if near is not None:
            self.full.layer.add_module("near_miscredit_control", near)

    def close(self):
        if hasattr(self.full.layer, "near_miscredit_control"):
            delattr(self.full.layer, "near_miscredit_control")
        self.full.close()

    def ordinary(self, i, h, weights):
        self.stats["full_forwards"] += 1
        return self.full.panel(i, h, True, weights)

    def tail(self, i, h, weights):
        self.stats["tail_forwards"] += 1
        x, skip, down = self.cache[i]
        g = self.inputs.gate(h, True)
        delta = correction(x, g, int(self.inputs.slot[i]), weights)
        residual = skip + (down.float() + delta).to(down.dtype)
        # Preserve the ordinary final norm's leading dimensions and the strided
        # single-position input to the actual full-vocabulary Qwen head.
        n = len(self.inputs.ids[i])
        prefix = residual.new_zeros(()).expand(6, n - 1, residual.shape[-1])
        z = self.lm.norm(torch.cat((prefix, residual[:, None]), dim=1))
        logits = self.actor.model.lm_head(z[:, -1:, :])
        return distribution(logits[:, -1, self.actor.ids].float())

    def paired(self, i, h, weights, requests):
        full = self.ordinary(i, h, weights)
        if not self.fast:
            return full
        tail = self.tail(i, h, weights)
        err = float((full.detach() - tail.detach()).abs().max())
        self.stats["paired_panels"] += 1
        self.stats["max_prob_error"] = max(self.stats["max_prob_error"], err)
        if err:
            raise NumericalMismatch(f"semantic probability error {err} in text {i}")
        for row, action, model in requests:
            a = torch.autograd.grad(
                full[row, action].log(), model.live(), retain_graph=True
            )
            b = torch.autograd.grad(
                tail[row, action].log(), model.live(), retain_graph=True
            )
            err = max(float((u - v).abs().max()) for u, v in zip(a, b, strict=True))
            self.stats["paired_gradients"] += 1
            self.stats["max_gradient_error"] = max(
                self.stats["max_gradient_error"], err
            )
            if err:
                raise NumericalMismatch(f"native gradient error {err} in text {i}")
        return full

    def panel(self, i, h, weights, *, check=False, requests=()):
        if check:
            return self.paired(i, h, weights, requests)
        return self.tail(i, h, weights) if self.fast else self.ordinary(i, h, weights)

    def prepare(self, ids=None):
        if ids is None:
            ids = [
                i
                for i, r in enumerate(self.rows)
                if (r["task"], r["wording"]) in ((4, 3), (5, 4))
            ]
        main = self.new("population")
        blank = self.new("blank")
        self.bind(main, None, None)
        weights = [
            blank.live(),
            main.live(),
            main.snapshot(),
            blank.snapshot(),
            blank.snapshot(),
            main.snapshot(),
        ]
        inputs = {}

        def skip(_, args):
            inputs["skip"] = args[0][:, -1].detach().clone()

        def down(_, args, res):
            inputs["x"] = (args[0][:, -1].detach().float() @ self.full.proj).clone()
            inputs["down"] = res[:, -1].detach().clone()

        handles = [
            self.lm.layers[-1].post_attention_layernorm.register_forward_pre_hook(skip),
            self.full.layer.register_forward_hook(down, prepend=True),
        ]
        try:
            for j, i in enumerate(ids):
                h = (3.36, 4.5, 90.0, 300.0)[j % 4]
                original = self.ordinary(i, h, weights)
                self.cache[i] = tuple(inputs[k] for k in ("x", "skip", "down"))
                alt = self.tail(i, h, weights)
                err = float((original.detach() - alt.detach()).abs().max())
                self.stats["max_prob_error"] = max(self.stats["max_prob_error"], err)
                self.stats["paired_panels"] += 1
                if err:
                    raise NumericalMismatch(
                        f"initial semantic probabilities differ by {err}, text {i}"
                    )
                if j % 8 == 0:
                    for k, a, subject in ((0, j % 4, blank), (1, (j // 8) % 4, main)):
                        x = torch.autograd.grad(
                            original[k, a].log(), subject.live(), retain_graph=True
                        )
                        y = torch.autograd.grad(
                            alt[k, a].log(), subject.live(), retain_graph=True
                        )
                        err = max(
                            float((u - v).abs().max())
                            for u, v in zip(x, y, strict=True)
                        )
                        self.stats["paired_gradients"] += 1
                        self.stats["max_gradient_error"] = max(
                            self.stats["max_gradient_error"], err
                        )
                        if err:
                            raise NumericalMismatch(
                                f"initial score gradients differ by {err}, text {i}"
                            )
        finally:
            for handle in handles:
                handle.remove()
            self.close()
