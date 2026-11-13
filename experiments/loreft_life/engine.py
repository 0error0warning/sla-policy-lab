"""A shared frozen health path, native Qwen physical forwards, and a verified exact tail."""

import torch

from experiments.individual_life.model import correction, derivatives
from experiments.first_fatal.engine import Qwen
from experiments.internal_body.suffix import distribution
from .adapters import LoReFT, ResidualLoRA, advance, frame, vector
from .design import INTERVENTIONS, ROLES


class People:
    def __init__(self, backend):
        self.birth = backend.new("population").snapshot()
        self.native = backend.new("population")
        self.native_wrong = backend.new("population")
        origin = frame(self.birth[0].device)
        self.loreft = LoReFT(origin)
        self.loreft_wrong = LoReFT(origin)
        self.add = ResidualLoRA(origin)
        self.subjects = (
            None,
            self.native,
            self.add,
            self.loreft,
            self.native_wrong,
            self.loreft_wrong,
        )

    def weights(self):
        return [
            self.birth,
            self.native.live(),
            self.birth,
            self.birth,
            self.native_wrong.live(),
            self.birth,
        ]

    def state(self):
        return {
            ROLES[k]: vector(m) for k, m in enumerate(self.subjects) if m is not None
        }

    def restore(self, arrays):
        with torch.no_grad():
            for name, model in zip(ROLES[1:], self.subjects[1:], strict=True):
                flat = arrays[name]
                start = 0
                for t in model.live():
                    end = start + t.numel()
                    t.copy_(
                        torch.tensor(
                            flat[start:end].reshape(tuple(t.shape)), device=t.device
                        )
                    )
                    start = end
                assert start == len(flat)


class Engine:
    def __init__(self, actor, root):
        self.q = Qwen(actor, root)
        self.lm = self.q.lm
        self.actor = actor
        self.hook = None
        self.person = None
        self.checked = 0
        self.grad_checks = 0
        self.max_probability_difference = 0.0
        self.max_gradient_difference = 0.0
        self.custom_full = 0
        self.fallback = []

    def bind(self, person):
        self.person = person
        self.q.bind(person.native, person.native_wrong, None)
        self.hook = self.lm.norm.register_forward_pre_hook(self._hook)

    def close(self):
        self.hook.remove()
        self.hook = None
        self.q.close()
        self.person = None

    def _hook(self, _module, args):
        raw = args[0]
        last = raw[:, -1]
        changed = (
            self._live_residual(last) if self.custom is None else self.custom(last)
        )
        return (torch.cat((raw[:, :-1], changed[:, None]), dim=1), *args[1:])

    def _live_residual(self, r):
        p = self.person
        h = r.float()
        deltas = [
            torch.zeros_like(h[0]),
            torch.zeros_like(h[1]),
            p.add.delta(h[2]),
            p.loreft.delta(h[3]),
            torch.zeros_like(h[4]),
            p.loreft_wrong.delta(h[5]),
        ]
        return (h + torch.stack(deltas)).to(r.dtype)

    def _cached_residual(self, i, h, weights):
        x, skip, down = self.q.cache[i]
        delta = correction(
            x, self.q.inputs.gate(h, True), int(self.q.inputs.slot[i]), weights
        )
        return skip + (down.float() + delta).to(down.dtype)

    def _read(self, i, r):
        n = len(self.q.inputs.ids[i])
        prefix = r.new_zeros(()).expand(6, n - 1, r.shape[-1])
        z = self.lm.norm.forward(torch.cat((prefix, r[:, None]), dim=1))
        logits = self.actor.model.lm_head(z[:, -1:, :])
        return distribution(logits[:, -1, self.actor.ids].float())

    def ordinary(self, i, h):
        self.custom = None
        return self.q.ordinary(i, h, self.person.weights())

    def cached(self, i, h):
        self.q.stats["tail_forwards"] += 1
        return self._read(
            i, self._live_residual(self._cached_residual(i, h, self.person.weights()))
        )

    def compare(self, a, b):
        self.checked += 1
        err = float((a.detach() - b.detach()).abs().max())
        self.max_probability_difference = max(self.max_probability_difference, err)
        if err:
            self.fallback.append(dict(kind="probability", difference=err))
            self.q.fast = False
        return err

    def decision(self, i, h, check=False):
        p = self.ordinary(i, h)
        if self.q.fast and check:
            cached = self.cached(i, h)
            if not self.compare(p, cached):
                for k, act in ((2, 2), (3, 0), (5, 3)):
                    model = self.person.subjects[k]
                    g1 = torch.autograd.grad(
                        p[k, act].log(), model.live(), retain_graph=True
                    )
                    g2 = torch.autograd.grad(
                        cached[k, act].log(), model.live(), retain_graph=True
                    )
                    err = max(
                        float((l - r).abs().max()) for l, r in zip(g1, g2, strict=True)
                    )
                    self.grad_checks += 1
                    self.max_gradient_difference = max(
                        self.max_gradient_difference, err
                    )
                    if err:
                        self.q.fast = False
                        self.fallback.append(dict(kind="gradient", difference=err))
                        break
        return p

    @torch.no_grad()
    def probe(self, i, h, check=False):
        if not self.q.fast:
            return self.ordinary(i, h)
        p = self.cached(i, h)
        if check:
            native = self.ordinary(i, h)
            self.compare(native, p)
            return native
        return p

    def update(self, p, e):
        labels = [
            e["wrong_action"] if k in (4, 5) else e["action"] for k in range(1, 6)
        ]
        requests = [
            (k, label, self.person.subjects[k]) for k, label in enumerate(labels, 1)
        ]
        grads = derivatives(p, requests)
        numbers = p.detach().cpu().numpy()
        return {
            ROLES[k]: advance(
                m, g, e["signal"], float(e["prob"][0][label]), float(numbers[k, label])
            )
            for (k, label, m), g in zip(requests, grads, strict=True)
        }

    def common_residual(self, i, h):
        if i in self.q.cache and self.q.fast:
            return self._cached_residual(i, h, [self.person.birth] * 6)[0]
        # Ordinary Qwen obtains precisely the shared activation, including visible body.
        captured = []

        def capture(last):
            captured.append(last[0].detach().clone())
            return last

        self.custom = capture
        try:
            self.q.ordinary(i, h, [self.person.birth] * 6)
        finally:
            self.custom = None
        return captured[0]

    @torch.no_grad()
    def learned_pair(self, i, h, pair):
        common = self.common_residual(i, h)
        return self._apply_pair(common, pair)

    @staticmethod
    def _apply_pair(common, pair):
        h = common.float()
        return h, [(h + m.delta(h)).to(common.dtype).float() for m in pair]

    def _variants(self, common, pair, source, dirs, alpha):
        birth, both = self._apply_pair(common, pair)
        own, opposite = both[source], both[1 - source]
        v, r1, r2 = dirs
        coef = torch.dot(own - birth, v)
        removed = own - coef * v
        projection = torch.dot(opposite - own, v)
        values = [
            own,
            own + alpha * v,
            own - alpha * v,
            removed,
            own + projection * v,
            removed + coef * v,
        ]
        for u in (r1, r2):
            sign = torch.sign(torch.dot(own - birth, u))
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            values.extend(
                [own + alpha * u, own - alpha * u, own - sign * coef.abs() * u]
            )
        assert len(values) == len(INTERVENTIONS)
        return torch.stack(values).to(common.dtype)

    @torch.no_grad()
    def steering(self, i, h, pair, source, dirs, alpha, check):
        def full_half(offset):
            self.custom = lambda raw: self._variants(raw[0], pair, source, dirs, alpha)[
                offset : offset + 6
            ]
            try:
                self.custom_full += 1
                return self.q.ordinary(i, h, [self.person.birth] * 6)
            finally:
                self.custom = None

        if not self.q.fast:
            return torch.cat([full_half(0), full_half(6)])
        common = self.common_residual(i, h)
        variants = self._variants(common, pair, source, dirs, alpha)
        cached = torch.cat([self._read(i, variants[:6]), self._read(i, variants[6:])])
        if check:
            native = torch.cat([full_half(0), full_half(6)])
            self.compare(native, cached)
            return native
        return cached
