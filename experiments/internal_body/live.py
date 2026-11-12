"""Actual native projection and decoder hooks, original-weight updates and live decisions."""

import json
from hashlib import sha256

import numpy as np
import torch
from torch import nn

from experiments.consequence_policy.policy import deployed
from experiments.consequence_policy.study import KL, TEMP, basis
from experiments.direct_policy.protocol import write
from experiments.internal_body.capture import DecisionState, encode
from experiments.internal_body.depth import schedule
from experiments.internal_body.design import LR_NATIVE, SPLIT_H, all_rows
from experiments.internal_body.lives import life_tape, run_one
from experiments.internal_body.personal import LifeActor, LifeInputs
from experiments.internal_body.suffix import distribution
from experiments.social_closed_loop.core import BodyValue


def digest(t):
    return sha256(
        t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def full(actor, ids):
    tokens = torch.as_tensor([ids], device="cuda")
    out = actor.model(
        input_ids=tokens,
        attention_mask=torch.ones_like(tokens),
        use_cache=False,
        logits_to_keep=1,
    )
    return out.logits[0, -1, actor.ids].float()


class HiddenModule(nn.Module):
    def __init__(self, A, B, projection, personal_shape=None):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(A, device="cuda"), requires_grad=False)
        self.B = nn.Parameter(torch.tensor(B, device="cuda"), requires_grad=False)
        self.register_buffer("projection", torch.tensor(projection, device="cuda"))
        if personal_shape is not None:
            self.W = nn.Parameter(torch.zeros(personal_shape, device="cuda"))


class Installed:
    def __init__(self, actor, kind, state, projection, offset, eps, life=None):
        self.kind, self.actor, self.eps, self.offset = kind, actor, eps, offset
        self.lm = actor.model.model.language_model
        self.handles = []
        self.old_norm = None
        self.block = None
        self.untouched = True
        if kind.startswith("native"):
            self.old_norm = self.lm.norm.weight
            updated = self.old_norm.float() + torch.as_tensor(
                state["delta"], device="cuda"
            )
            self.lm.norm.weight = nn.Parameter(
                updated.to(self.old_norm.dtype), requires_grad=False
            )
            return
        A, B = state["A"], state["B"]
        if life is not None:
            A = np.broadcast_to(A.mean(axis=0), A.shape).copy()
        self.block = HiddenModule(
            A,
            B,
            projection,
            life.W.shape if life is not None and life.W is not None else None,
        )
        down = self.lm.layers[-1].mlp.down_proj
        down.add_module("body_conditioner", self.block)

        def read(_, args):
            self.feat = args[0][:, -1].detach().float() @ self.block.projection

        def correct(_, args, value):
            t = torch.einsum("b,bdr->dr", self.g, self.block.A[self.partner])
            c = self.feat @ t
            if self.kind == "nonlinear":
                c = c.tanh()
            self.delta = c @ self.block.B
            if getattr(self.block, "W", None) is not None and getattr(
                self, "personal_on", True
            ):
                self.delta = (
                    self.delta
                    + torch.einsum(
                        "b,k,bkd->d", self.pg, self.key, self.block.W[self.partner]
                    )[None]
                )
            if self.kind == "postnorm":
                return value
            result = torch.cat(
                (
                    value[:, :-1],
                    (value[:, -1].float() + self.delta).to(value.dtype)[:, None],
                ),
                dim=1,
            )
            if self.untouched:
                assert torch.equal(result[:, :-1], value[:, :-1])
                self.untouched = False
            return result

        self.handles.extend(
            [down.register_forward_pre_hook(read), down.register_forward_hook(correct)]
        )
        if kind == "postnorm":

            def post(_, args, value):
                r = args[0][:, -1].float()
                scale = (r.square().mean(-1, keepdim=True) + self.eps).sqrt()
                delta = self.delta * (self.offset + self.lm.norm.weight.float()) / scale
                return torch.cat(
                    (
                        value[:, :-1],
                        (value[:, -1].float() + delta).to(value.dtype)[:, None],
                    ),
                    dim=1,
                )

            self.handles.append(self.lm.norm.register_forward_hook(post))

    def condition(self, partner, h, life=None, index=None):
        self.partner = int(partner) % 2
        if self.block is None:
            return
        g = basis([h])[0]
        if self.kind in ("blind", "prompt_lora"):
            g = basis(SPLIT_H).mean(axis=0)
        if life is not None:
            g, pg = life.gates(h)
            self.pg = torch.as_tensor(pg, dtype=torch.float32, device="cuda")
            self.key = torch.as_tensor(life.keys[index], device="cuda")
        self.g = torch.as_tensor(g, dtype=torch.float32, device="cuda")

    def close(self):
        for h in self.handles:
            h.remove()
        if self.block is not None:
            delattr(self.lm.layers[-1].mlp.down_proj, "body_conditioner")
        if self.old_norm is not None:
            self.lm.norm.weight = self.old_norm


def surrogate(s, kind, w, p, h, proj, norm, head, offset, eps):
    r = s["r"][0]
    score = s["native"][0]
    den = np.sqrt(np.mean(r * r) + eps)
    if kind.startswith("native"):
        new = (
            torch.from_numpy(norm + w["delta"]).to(torch.bfloat16).float().numpy()
            - norm
        )
        return score + (r / den * new) @ head.T
    g = basis([h])[0].astype(np.float32)
    if kind in ("blind", "prompt_lora"):
        g = basis(SPLIT_H).mean(axis=0).astype(np.float32)
    coefficients = s["v"][0] @ proj @ np.einsum("b,bdr->dr", g, w["A"][p])
    if kind == "nonlinear":
        coefficients = np.tanh(coefficients)
    d = coefficients @ w["B"]
    weighted = (norm + offset)[:, None] * head.T
    if kind == "postnorm":
        return score + (d @ weighted) / den
    r1 = r + d
    return score + (r1 / np.sqrt(np.mean(r1 * r1) + eps) - r / den) @ weighted


def static(actor, root, out, job, rows, proj, norm, head, cfg):
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
    levels = (2.0, 6.0, 90.0, 300.0)
    observed = {}
    prefixes = {}
    reader = DecisionState(actor)
    try:
        for i in selected:
            for h in (None, *levels):
                ids = encode(actor, rows[i], h)
                prefixes[i, h] = ids
                observed[i, h] = reader.batch([ids])
    finally:
        reader.close()
    results = []
    params = []
    for seed, m in zip(job["seeds"], (0, 1)):
        folder = root / "checkpoints" / f"s{seed}-m{m}"
        with np.load(folder / "published-probes.npz") as expected:
            for kind in (
                "linear",
                "nonlinear",
                "postnorm",
                "prompt_lora",
                "native_prompt",
                "native_hidden",
            ):
                with np.load(folder / (kind + ".npz")) as state:
                    w = {k: state[k] for k in state if k != "kind"}
                install = Installed(
                    actor, kind, w, proj, cfg["norm_offset"], cfg["epsilon"]
                )
                try:
                    if install.old_norm is not None:
                        old, new = install.old_norm, install.lm.norm.weight
                        params.append(
                            {
                                "seed": seed,
                                "kind": kind,
                                "path": "model.language_model.norm.weight",
                                "shape": list(old.shape),
                                "before": digest(old),
                                "after": digest(new),
                                "changed_elements": int((old != new).sum()),
                            }
                        )
                    else:
                        params.append(
                            {
                                "seed": seed,
                                "kind": kind,
                                "installed": [
                                    n
                                    for n, _ in actor.model.named_parameters()
                                    if ".body_conditioner." in n
                                ],
                                "rank": int(w["B"].shape[0]),
                                "trainable_elements": int(w["A"].size + w["B"].size),
                            }
                        )
                    for i in selected:
                        for h in levels:
                            text = (
                                h if kind in ("prompt_lora", "native_prompt") else None
                            )
                            install.condition(rows[i]["partner"], h)
                            with torch.no_grad():
                                actual = (
                                    distribution(full(actor, prefixes[i, text]))
                                    .cpu()
                                    .numpy()
                                )
                            predicted = deployed(
                                surrogate(
                                    observed[i, text],
                                    kind,
                                    w,
                                    rows[i]["partner"],
                                    h,
                                    proj,
                                    norm,
                                    head,
                                    cfg["norm_offset"],
                                    cfg["epsilon"],
                                )
                                / TEMP
                            )
                            hi = (2.0, 3.0, 4.0, 6.0, 10.0, 18.0, 90.0, 300.0).index(h)
                            results.append(
                                {
                                    "seed": seed,
                                    "mapping": m,
                                    "kind": kind,
                                    "context": i,
                                    "health": h,
                                    "full": actual.tolist(),
                                    "same_process": predicted.tolist(),
                                    "original_batch": expected[kind][
                                        hi, i - 768
                                    ].tolist(),
                                }
                            )
                finally:
                    install.close()
        write(out / "status.json", {"stage": "static", "seed": seed})
    write(out / "static.json", results)
    write(out / "installed.json", params)
    return len(results)


def actual_native(actor, root, out, job, rows):
    from experiments.consequence_policy.policy import feedback_scores
    from experiments.internal_body.carrier_gpu import FixedRotation

    lm = actor.model.model.language_model
    old = lm.norm.weight
    original = digest(old)
    runs = []
    body = BodyValue()
    ids = [encode(actor, r) for r in rows[:768]]
    seed = job["seeds"][0]
    with np.load(root / "native-actual-tape.npz") as z:
        scores = feedback_scores(
            z["action"],
            z["reward"],
            z["mu"],
            np.minimum(1.0, [body.rul(h) for h in SPLIT_H]),
        )
    states = schedule(seed, ids)[:16]
    for label in ("text", "hidden", "carrier"):
        visible = label == "text"
        rotation = None
        if label == "carrier":
            with np.load(root / "native" / "basis-64501.npz") as sensor:
                rotation = FixedRotation(lm, sensor["u"], sensor["v"])
        lm.norm.weight = nn.Parameter(old.detach().float().clone(), requires_grad=True)
        optimizer = torch.optim.Adam([lm.norm.weight], lr=LR_NATIVE)
        names = [n for n, p in actor.model.named_parameters() if p.requires_grad]
        assert names == ["model.language_model.norm.weight"]
        history = []
        base = {}
        try:
            for indices, levels in states:
                for i, j in zip(indices, levels):
                    key = (int(i), int(j))
                    if key not in base:
                        if rotation is not None:
                            rotation.health(SPLIT_H[j])
                        prefix = encode(actor, rows[i], SPLIT_H[j] if visible else None)
                        with torch.no_grad():
                            base[key] = (
                                prefix,
                                distribution(full(actor, prefix)).detach(),
                            )
            for step, (indices, levels) in enumerate(states):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for i, j in zip(indices, levels):
                    prefix, prior = base[int(i), int(j)]
                    if rotation is not None:
                        rotation.health(SPLIT_H[j])
                    pi = distribution(full(actor, prefix))
                    val = torch.as_tensor(scores[j, i], device="cuda")
                    loss = -(pi * val).sum() + KL * (pi * (pi / prior).log()).sum()
                    (loss / len(indices)).backward()
                    losses.append(float(loss.detach()))
                gradient = float(lm.norm.weight.grad.norm())
                assert np.isfinite(gradient)
                optimizer.step()
                history.append(
                    {
                        "step": step + 1,
                        "indices": indices.tolist(),
                        "health_slots": levels.tolist(),
                        "losses": losses,
                        "gradient": gradient,
                    }
                )
            master = lm.norm.weight.detach().clone()
            lm.norm.weight = nn.Parameter(master.to(old.dtype), requires_grad=False)
            pi = []
            with torch.no_grad():
                for i in [v for v in range(768, 960) if rows[v]["layout"] % 4 == 0]:
                    for h in (2.0, 300.0):
                        if rotation is not None:
                            rotation.health(h)
                        pp = encode(actor, rows[i], h if visible else None)
                        pi.append([i, h, *distribution(full(actor, pp)).cpu().tolist()])
            np.savez_compressed(
                out / f"native-actual-{label}.npz",
                original=old.detach().float().cpu().numpy(),
                master=master.cpu().numpy(),
                effective=lm.norm.weight.detach().float().cpu().numpy(),
                probes=pi,
            )
            runs.append(
                {
                    "channel": label,
                    "visible": visible,
                    "original": original,
                    "effective": digest(lm.norm.weight),
                    "trainable": names,
                    "changed": int((old != lm.norm.weight).sum()),
                    "steps": history,
                }
            )
        finally:
            lm.norm.weight = old
            if rotation is not None:
                rotation.close()
        assert digest(lm.norm.weight) == original
    write(out / "native-actual.json", runs)
    return len(runs)


class DirectLife:
    def __init__(self, actor, life, seed, root, proj, cfg, rule="scalar"):
        self.actor = actor
        self.rule = rule
        with np.load(root / "checkpoints" / f"generic-{seed}.npz") as z:
            self.install = Installed(
                actor,
                life.arm,
                {k: z[k] for k in ("A", "B")},
                proj,
                cfg["norm_offset"],
                cfg["epsilon"],
                life,
            )
        self.ids = {}

    def policy(self, i, h, life):
        if i not in self.ids:
            self.ids[i] = encode(self.actor, life.data.rows[i])
        self.install.condition(life.data.slot[i], h, life, i)
        if self.rule == "predict":
            self.install.personal_on = False
            with torch.no_grad():
                self.prior_log = torch.log_softmax(
                    full(self.actor, self.ids[i]) / TEMP, dim=0
                )
            self.install.personal_on = True
        scores = full(self.actor, self.ids[i])
        self.log_scores = torch.log_softmax(scores / TEMP, dim=0)
        self.pi = distribution(scores)
        return (self.pi.detach().cpu().numpy()[None],)

    def update(self, index, action, scaled, life, details):
        if life.arm == "fixed":
            return 0.0, 0.0
        w = self.install.block.W
        if self.rule == "predict":
            centered = self.log_scores[action] - self.log_scores.mean()
            movement = self.log_scores.detach() - self.prior_log
            estimate = float(movement[action] - movement.mean())
            grad = torch.autograd.grad(centered, w)[0]
            energy = float(grad.square().sum())
            with torch.no_grad():
                w.add_(
                    grad, alpha=0.55 * (3 * (scaled + 0.9) - estimate) / (0.05 + energy)
                )
                life.W[:] = w.detach().cpu().numpy()
            return float(grad.norm()), energy
        gradient = [
            torch.autograd.grad(self.pi[j].log(), w, retain_graph=j != 3)[0]
            for j in range(4)
        ]
        with torch.no_grad():
            fisher = sum(
                float(self.pi[j]) * float(g.square().sum())
                for j, g in enumerate(gradient)
            )
            gain = 3.0 if self.rule == "scalar3" else 1.0
            w.add_(
                gradient[action], alpha=gain * 0.55 * (scaled + 0.9) / (0.05 + fisher)
            )
            life.W[:] = w.detach().cpu().numpy()
        return float(gradient[action].norm()), fisher


def run(actor, root, out):
    job = json.loads((root / "live-job.json").read_bytes())
    cache = root / "live-cache"
    rows = all_rows()
    cfg = json.loads((cache / "architecture.json").read_bytes())
    with np.load(cache / "architecture.npz") as z:
        proj, norm, head = [z[s] for s in ("projection", "norm", "head")]
    count = static(actor, root, out, job, rows, proj, norm, head, cfg)
    from experiments.internal_body.carrier_gpu import checks

    carrier_count = checks(actor, root, out, rows)
    direct_native = actual_native(actor, root, out, job, rows)
    data = LifeInputs(cache)
    body = BodyValue()
    all_logs = {}
    reports = []
    snapshots = {}
    both = [
        (s, w, v, [("fixed", "fixed"), ("free", "scalar")])
        for s, w, v in zip(job["seeds"], ((0, 1), (1, 0)), (0, 1))
    ]
    both += [
        (s, w, v, [("fixed", "fixed"), ("free", "scalar3"), ("free", "predict")])
        for s, w, v in zip(job.get("credit_seeds", []), ((0, 0), (1, 0)), (0, 1))
    ]
    for seed, world, rev, conditions in both:
        tape = life_tape(seed, world, data.rows)
        for arm, rule in conditions:
            agent = LifeActor(data, root / "checkpoints", seed, arm)
            direct = DirectLife(actor, agent, seed, root, proj, cfg, rule)
            try:
                trace, rep = run_one(agent, seed, world, rev, tape, body, direct=direct)
                key = f"s{seed}-{rule}"
                all_logs[key] = trace
                snapshots[key] = agent.parameters()
                reports.append(
                    {
                        "key": key,
                        "seed": seed,
                        "rule": rule,
                        "world": world,
                        "reversal": rev,
                        "metrics": rep,
                        "trainable": [
                            n
                            for n, p in actor.model.named_parameters()
                            if p.requires_grad
                        ],
                    }
                )
                write(
                    out / "status.json",
                    {"stage": "actual_life", "completed": len(reports)},
                )
            finally:
                direct.install.close()
    write(out / "lives.json", reports)
    np.savez_compressed(out / "lifetimes.npz", **all_logs)
    np.savez_compressed(out / "personal-checkpoints.npz", **snapshots)
    return {
        "static": count,
        "carrier_static": carrier_count,
        "genuine_native_fits": direct_native,
        "lives": len(reports),
        "turns": sum(len(t) for t in all_logs.values()),
    }
