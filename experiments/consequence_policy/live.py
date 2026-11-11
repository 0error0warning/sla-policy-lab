"""Install the adapter in the genuine full-vocabulary Qwen projection on a fresh GPU."""

import json

import numpy as np
import torch
from torch import nn

from experiments.consequence_policy.experiment import read_inputs
from experiments.consequence_policy.lifetime import PersonalActor, environment_tape, run_life
from experiments.consequence_policy.policy import features
from experiments.consequence_policy.study import EXPLORATION, TEMP, basis
from experiments.direct_policy.protocol import write
from experiments.social_closed_loop.core import BodyValue


class NativeBodyHead(nn.Module):
    def __init__(self, original, token_ids, center, matrix, agent):
        super().__init__()
        self.original = original
        self.register_buffer("token_ids", torch.tensor(token_ids, device="cuda"))
        self.register_buffer("center", torch.as_tensor(center, device="cuda"))
        self.register_buffer("matrix", torch.as_tensor(matrix, device="cuda"))
        self.weight_bank = nn.Parameter(torch.as_tensor(agent.main.weights.copy(), device="cuda"), requires_grad=False)
        self.direction = nn.Parameter(torch.as_tensor(agent.direction.weights.copy(), device="cuda"), requires_grad=False)
        self.personal = nn.Parameter(torch.zeros(2, device="cuda", dtype=torch.float64), requires_grad=agent.arm == "personal")
        self.changed_outside_actions = 0.

    def set_body(self, p, health):
        self.partner = p
        self.gate = torch.tensor(basis([health])[0], dtype=torch.float32, device="cuda")

    def forward(self, hidden):
        original = self.original(hidden).float()
        z = (hidden.float() - self.center) @ self.matrix
        x = torch.cat((z, torch.ones_like(z[..., :1])), dim=-1)
        g = self.gate[:, None, None]
        base = x @ (g * self.weight_bank[self.partner]).sum(0)
        self.slope = x @ (g * self.direction[self.partner]).sum(0)
        residual = base + self.personal[self.partner].float() * self.slope
        new = original.index_select(-1, self.token_ids) + TEMP * residual
        result = original.index_copy(-1, self.token_ids, new)
        if not hasattr(self, "untouched_verified"):
            delta = (result.detach() - original.detach()).abs()
            delta.index_fill_(-1, self.token_ids, 0)
            self.changed_outside_actions = float(delta.max().item())
            assert self.changed_outside_actions == 0.
            self.untouched_verified = True
        return result


class LiveInterface:
    def __init__(self, model, original, center, matrix, agent, ids, prefixes):
        self.model = model
        self.head = NativeBodyHead(original, ids, center, matrix, agent)
        self.model.lm_head = self.head
        self.prefixes = prefixes
        self.max_grad_diff = 0.
        self.max_u_diff = 0.
        self.max_fisher_diff = 0.

    def __call__(self, i, row, health, u):
        np.testing.assert_allclose(self.head.personal.detach().cpu().numpy(), u, rtol=0, atol=1e-10)
        self.head.set_body(row["partner"], health)
        ids = self.prefixes[i]
        out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                         use_cache=False, logits_to_keep=1)
        logits = out.logits[0, -1].index_select(0, self.head.token_ids) / TEMP
        self.s = torch.softmax(logits, dim=0)
        self.pi = (1. - EXPLORATION) * self.s + EXPLORATION / 4
        self.d = self.head.slope[0, -1].detach()
        return self.pi.detach().cpu().numpy(), logits.detach().cpu().numpy(), self.d.cpu().numpy()

    def commit(self, p, action, normalized, expected_u, grad, fisher):
        automatic, information = grad, fisher
        if self.head.personal.requires_grad:
            automatic = torch.autograd.grad(self.pi[action].log(), self.head.personal)[0][p].item()
            with torch.no_grad():
                slopes = (1 - EXPLORATION) * self.s / self.pi * (self.d - torch.dot(self.s, self.d))
                information = torch.dot(self.pi, slopes.square()).item()
                self.head.personal[p].add_(.5 * (normalized + .9) * automatic / (1. + information)).clamp_(-1., 1.)
            self.max_grad_diff = max(self.max_grad_diff, abs(automatic - grad))
            self.max_fisher_diff = max(self.max_fisher_diff, abs(information - fisher))
        actual = self.head.personal.detach().cpu().numpy().copy()
        self.max_u_diff = max(self.max_u_diff, float(np.max(np.abs(actual - expected_u))))
        return actual, automatic, information


def verify(backbone, root, out):
    cache, checkpoints = root / "cache", root / "checkpoints"
    rows, native, x, _, train = read_inputs(cache, checkpoints)
    with np.load(checkpoints / "projection.npz") as projection:
        center, matrix = projection["center"], projection["matrix"]
    observed = json.loads((cache / "contexts.json").read_bytes())
    test = np.flatnonzero(~train)
    prefixes = {int(i): torch.tensor([backbone.encode(rows[i])], device="cuda") for i in test}
    assert all(prefixes[i][0].tolist() == observed[i]["input_ids"] for i in test)
    fresh_z, fresh_native = [], []
    for i in test:
        z, logits = backbone.batch([observed[i]["input_ids"]])
        fresh_z.append(z[0])
        fresh_native.append(logits[0])
    fresh_z, fresh_native = np.asarray(fresh_z), np.asarray(fresh_native)
    np.savez_compressed(out / "independent-test-prefixes.npz", context=test, hidden=fresh_z.astype(np.float16), logits=fresh_native)
    fresh_x = features(fresh_z, center, matrix)
    fresh_index = {i: j for j, i in enumerate(test)}
    selected = [next(i for i in test if rows[i]["layout"] == o and rows[i]["partner"] == o % 2
                     and rows[i]["frame"] == o // 2 % 2 and rows[i]["task"] == 4 + o // 4 % 2)
                for o in range(24)]
    original = backbone.model.lm_head
    body = BodyValue()
    audits, lifelogs, receipts = [], {}, []
    try:
        for seed, world in ((63101, (0, 1)), (63102, (1, 0))):
            for arm in (("personal", "prior", "population", "frozen") if seed == 63101 else ("personal", "prior")):
                agent = PersonalActor(checkpoints, seed, world, arm)
                live = LiveInterface(backbone.model, original, center, matrix, agent, backbone.ids, prefixes)
                if (arm in ("population", "frozen")) or (seed == 63102 and arm == "prior"):
                    for i in selected:
                        row = rows[i]
                        j = fresh_index[i]
                        for health in (2., 6., 300.):
                            direct, _, _ = live(i, row, health, agent.u)
                            cached = agent.score(i, row, health, native, x)[0]
                            independent = agent.score(j, row, health, fresh_native, fresh_x)[0]
                            audits.append(dict(seed=seed, arm=arm, context=int(i), health=health,
                                               full=direct.tolist(), cached=cached.tolist(), singleton=independent.tolist()))
                tape = environment_tape(seed, world, 0, rows)
                trace, report = run_life(agent, seed, world, 0, tape, rows, native, x, body, override=live)
                key = f"s{seed}-w{world[0]}{world[1]}-{arm}"
                lifelogs[key] = trace
                np.savez_compressed(out / f"{key}-personal.npz", personal=live.head.personal.detach().cpu().numpy())
                receipt = dict(run=key, world=list(world), replica=0, seed=seed, arm=arm, metrics=report,
                               max_grad_disagreement=live.max_grad_diff, max_fisher_disagreement=live.max_fisher_diff,
                               max_update_disagreement=live.max_u_diff, other_vocabulary_change=live.head.changed_outside_actions)
                receipts.append(receipt)
                write(out / "live-metrics.json", receipts)
                write(out / "status.json", dict(stage="live", completed=len(receipts), run=key, turns=len(trace)))
    finally:
        backbone.model.lm_head = original
    np.savez_compressed(out / "end-to-end-lives.npz", **lifelogs)
    write(out / "static-full-forward.json", audits)
    diff = [np.max(np.abs(np.array(r["full"]) - r["cached"])) for r in audits]
    same = [np.max(np.abs(np.array(r["full"]) - r["singleton"])) for r in audits]
    return dict(lives=len(receipts), turns=sum(len(v) for v in lifelogs.values()), static_comparisons=len(diff),
                max_cache_probability_disagreement=float(max(diff)), p95_cache_disagreement=float(np.quantile(diff,.95)),
                max_singleton_disagreement=float(max(same)), max_grad_disagreement=max(r["max_grad_disagreement"] for r in receipts),
                max_update_disagreement=max(r["max_update_disagreement"] for r in receipts))
