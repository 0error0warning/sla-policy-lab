"""Read Qwen's name and request representations and feed individual forecasts into the policy."""

import numpy as np
import torch
from torch import nn

from experiments.internal_body.design import chat
from experiments.lifetime_cooperation.design import ITEMS, LEVELS, PANEL
from experiments.loreft_life.adapters import LoReFT, dot_rows, frame
from experiments.loreft_omission.gpu import Batch
from experiments.partner_damage.design import swapped

from .design import FEATURE_IDS, MODES, features


@torch.no_grad()
def encode_features(base, out):
    tok = base.engine.actor.tok
    names, texts, windows = [], [], []
    pos = {}

    def capture(_, args, output):
        h = output[0] if isinstance(output, tuple) else output
        names.append(h[0, pos["name"]].float().cpu().numpy().copy())
        texts.append(
            h[0, pos["start"] : pos["end"]].float().mean(0).cpu().numpy().copy()
        )

    handle = base.engine.lm.layers[3].register_forward_hook(capture)
    try:
        base.engine.custom = lambda h: h
        for i in FEATURE_IDS:
            row = ITEMS[i]
            original = base.q.inputs.ids[i]
            other = base.q.inputs.ids[swapped(i)]
            name = [
                k
                for k, (a, b) in enumerate(zip(original, other, strict=True))
                if a != b
            ][-1]
            prompt = tok.apply_chat_template(
                chat(row),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            start = prompt.rindex(row["text"])
            tokens = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
            assert list(tokens["input_ids"]) == list(original)
            selected = [
                k
                for k, (l, r) in enumerate(tokens["offset_mapping"])
                if r > start and l < start + len(row["text"]) and r > l
            ]
            assert (
                selected
                and selected == list(range(selected[0], selected[-1] + 1))
                and name < selected[0]
            )
            pos = {"name": name, "start": selected[0], "end": selected[-1] + 1}
            windows.append([i, *pos.values()])
            base.q.ordinary(i, 90.0, base.weights)
    finally:
        handle.remove()
        base.engine.custom = None
    x, scales = features(names, texts, FEATURE_IDS)
    np.savez_compressed(
        out / "features.npz",
        ids=FEATURE_IDS,
        name=names,
        text=texts,
        windows=windows,
        scales=scales,
        x=x[FEATURE_IDS],
    )
    return x


class ReadPolicy(LoReFT):
    def __init__(self, initial):
        super().__init__(initial)
        self.C = nn.Parameter(initial.new_zeros((initial.shape[0], 3)))

    def read_delta(self, h, q):
        coef = dot_rows(self.W - self.R, h) + self.bias
        calibrated = (q - 0.5) * (torch.linalg.vector_norm(h) / np.sqrt(3.0))
        coef = coef + self.C @ calibrated
        return (self.R * coef[:, None]).sum(0)

    def live(self):
        return (*super().live(), self.C)


class Actor:
    def __init__(self, frozen, root, out):
        self.b = Batch(frozen, root)
        self.x = encode_features(self.b, out)
        self.origin = frame("cuda")
        self.common = {}
        self.checks = 0
        self.error = 0.0
        self.fallback = []
        with torch.no_grad():
            for i in PANEL.reshape(-1):
                for h in LEVELS:
                    self.common[int(i), h] = self.b.engine.common_residual(
                        int(i), h
                    ).clone()

    def new(self, kind):
        return (LoReFT if kind == "body" else ReadPolicy)(self.origin)

    @staticmethod
    def inject(raw, roles):
        h = raw.float()
        values = []
        for j, (model, q) in enumerate(roles):
            if model is None:
                values.append(raw[j])
                continue
            delta = (
                model.read_delta(h[j], q)
                if isinstance(model, ReadPolicy)
                else model.delta(h[j])
            )
            values.append((h[j] + delta).to(raw.dtype))
        return torch.stack(values)

    def lanes(self, pairs):
        assert len(pairs) == 6
        return [
            (
                m,
                torch.as_tensor(q, dtype=torch.float32, device=self.origin.device)
                if isinstance(m, ReadPolicy)
                else None,
            )
            for m, q in pairs
        ]

    def ordinary(self, i, h, roles):
        self.b.engine.custom = lambda raw: self.inject(raw, roles)
        try:
            return self.b.q.ordinary(i, h, self.b.weights)
        finally:
            self.b.engine.custom = None

    def panel_roles(self, policies, heads, i):
        q = {k: m.read(self.x, i) for k, m in heads.items()}
        true = policies["true"]
        all_roles = [(policies[k], q[k]) for k in ("body", "true", "wrong")]
        all_roles.extend(
            (
                (true, np.full(3, 0.5)),
                (true, q["true"][[0, 2, 1]]),
                (true, heads["true"].read(self.x, swapped(i))),
                (true, q["drop93"]),
                (true, q["drop86"]),
                (None, None),
            )
        )
        assert len(all_roles) == len(MODES)
        return all_roles

    @torch.no_grad()
    def panels(self, policies, heads, check_index):
        result = np.empty((len(MODES), 2, 4, 32, 4), dtype=np.float32)
        for w, indices in enumerate(PANEL):
            for j, idx in enumerate(indices):
                i = int(idx)
                views = self.panel_roles(policies, heads, i)
                for k, h in enumerate(LEVELS):
                    for offset in (0, 6):
                        part = views[offset : offset + 6]
                        roles = self.lanes(part + [(None, None)] * (6 - len(part)))
                        if self.b.q.fast:
                            cached = self.b.engine._read(
                                i, self.inject(self.common[i, h].expand(6, -1), roles)
                            )
                            if j == (check_index + 7 * k + 3 * w) % 32:
                                native = self.ordinary(i, h, roles)
                                err = float((cached - native).abs().max())
                                self.checks += 1
                                self.error = max(self.error, err)
                                if err:
                                    self.fallback.append([w, j, k, offset, err])
                                    self.b.q.fast = False
                                    return self.panels(policies, heads, check_index)
                            p = cached
                        else:
                            p = self.ordinary(i, h, roles)
                        result[offset : offset + len(part), w, k, j] = (
                            p[: len(part)].cpu().numpy()
                        )
        return result
