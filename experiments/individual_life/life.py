"""One causal lineage: decide, settle its own body, update private parameters, then continue."""

import json
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import torch

from experiments.consequence_policy.study import WORDS
from experiments.direct_policy.protocol import render, settle, write
from experiments.individual_life.design import (
    CHECKS,
    PROBE_H,
    action_at,
    offers,
    permutation,
    world,
)
from experiments.individual_life.model import derivatives, update


def save_lines(path, rows):
    with path.open("w", encoding="utf8", newline="\n") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def signature(bits):
    return sha256(bits.tobytes()).hexdigest()


class History:
    """Every microstep is invertible from the birth plus exact FP32-bit XORs."""

    def __init__(self, actor, shadow):
        self.birth = actor.bits()
        self.roles = ("main", "shadow") if shadow is not None else ("main",)
        self.last = {key: self.birth.copy() for key in self.roles}
        self.digests = {key: signature(self.birth) for key in self.roles}
        self.deltas = {key: [] for key in self.roles}
        self.records = []
        self.split = actor.A.numel()

    def length(self, role):
        return len(self.deltas[role])

    def append(self, role, actor, record):
        before = self.last[role]
        after = actor.bits()
        diff = after.view("<f4").astype(float) - before.view("<f4")
        checksum = signature(after)
        self.deltas[role].append(before ^ after)
        self.records.append(
            {
                **record,
                "role": role,
                "step": self.length(role),
                "before": self.digests[role],
                "after": checksum,
                "A_change_l2": float(np.linalg.norm(diff[: self.split])),
                "B_change_l2": float(np.linalg.norm(diff[self.split :])),
                "changed_coordinates": int(np.count_nonzero(before != after)),
            }
        )
        self.last[role] = after
        self.digests[role] = checksum

    def save(self, d):
        params = {
            "birth": self.birth,
            "A_shape": np.array([2, 12, 33, 4]),
            "B_shape": np.array([4, 2560]),
        }
        for role, log in self.deltas.items():
            params[role + "_xor"] = (
                np.stack(log) if log else np.empty((0, len(self.birth)), dtype="<u4")
            )
            params[role + "_final"] = self.last[role]
        np.savez_compressed(d / "parameters.npz", **params)
        save_lines(d / "updates.jsonl", self.records)


@dataclass
class Encounter:
    t: int
    index: int
    health: float
    action: int
    offset: int
    behavior: np.ndarray
    advantage: float


class Lineage:
    def __init__(self, backend, birth, visible, rule, seed):
        self.backend = backend
        self.visible = visible
        self.rule = rule
        self.seed = seed
        self.main = backend.new(birth)
        self.shadow = backend.new(birth) if rule in ("true1", "true8") else None
        self.birth = self.main.snapshot()
        self.previous = self.main.snapshot()
        self.deferred = self.main.snapshot()
        self.pre_index = self.defer_index = 0
        self.log = History(self.main, self.shadow)
        self.buffer = []

    def read(self, i, h):
        current = self.main.live()
        control = self.shadow.live() if self.shadow is not None else self.birth
        return self.backend.panel(
            i,
            h,
            self.visible,
            [
                self.previous,
                current,
                self.birth,
                control,
                self.deferred,
                (current[0].detach(), current[1].detach()),
            ],
        )

    def step(self, panel, e, main_signal, shadow_signal, turn, cause):
        main_a = (e.action + e.offset) % 4 if self.rule == "wrong1" else e.action
        tasks = [(1, main_a, self.main)]
        roles = [("main", main_signal)]
        if self.shadow is not None:
            a = (e.action + e.offset) % 4 if self.rule == "true1" else e.action
            tasks.append((3, a, self.shadow))
            roles.append(("shadow", shadow_signal))
        grads = derivatives(panel, tasks)
        numeric = panel.detach().cpu().numpy()
        for (row, a, actor), (role, feedback), g in zip(
            tasks, roles, grads, strict=True
        ):
            details = update(
                actor,
                g,
                feedback.advantage,
                float(e.behavior[a]),
                float(numeric[row, a]),
            )
            self.log.append(
                role,
                actor,
                {
                    "release_at": turn,
                    "state_action_turn": e.t,
                    "feedback_turn": feedback.t,
                    "action_executed": e.action,
                    "action_credited": a,
                    "applied_signal": feedback.advantage,
                    "applied_at_terminal": cause,
                    **details,
                },
            )

    def settle_update(self, panel, e, terminal):
        start = self.log.length("main")
        self.previous = self.main.snapshot()
        self.pre_index = start
        if self.rule in ("true1", "wrong1"):
            self.step(panel, e, e, e, e.t, terminal)
        elif self.rule in ("true8", "shuffle8"):
            self.buffer.append(e)
            if len(self.buffer) == 8 or terminal:
                order = permutation(self.seed, self.buffer[0].t, len(self.buffer))
                for j, old in enumerate(self.buffer):
                    other = self.buffer[int(order[j])]
                    prob = self.read(old.index, old.health)
                    self.step(
                        prob,
                        old,
                        old if self.rule == "true8" else other,
                        other,
                        e.t,
                        terminal,
                    )
                self.buffer.clear()
        elif self.rule != "frozen":
            raise ValueError(self.rule)
        if self.rule == "true1" and e.t % 8 == 0:
            self.deferred = self.main.snapshot()
            self.defer_index = self.log.length("main")
        return self.log.length("main") - start


def run_one(d, backend, body, tape, seed, birth, visible, rule):
    d.mkdir(parents=True, exist_ok=True)
    lineage = Lineage(backend, birth, visible, rule, seed)
    backend.bind(lineage.main, lineage.shadow)
    indices, u, noise, offset = tape
    bits = world(seed)
    h = 350.0
    prev_rewards = 0.0
    events = []
    probes = {}
    probe_t = {}
    danger_seen = False
    max_dup = 0.0

    def inspect(key, turn):
        with torch.no_grad():
            for role, actor in (("main", lineage.main), ("shadow", lineage.shadow)):
                if actor is not None:
                    probes[key + "-" + role] = np.stack(
                        [backend.probe(actor, visible, health) for health in PROBE_H]
                    )
                    if not visible:
                        v = probes[key + "-" + role]
                        np.testing.assert_array_equal(v, np.broadcast_to(v[0], v.shape))
        probe_t[key] = {
            "after_turn": turn,
            "actual_alive_health": float(h),
            "version": lineage.log.length("main"),
        }

    try:
        inspect("birth", 0)
        for k, i in enumerate(indices):
            t = k + 1
            i = int(i)
            row = backend.rows[i]
            with torch.set_grad_enabled(rule in ("true1", "wrong1")):
                panel = lineage.read(i, h)
            pi = panel.detach().cpu().numpy().copy()
            err = float(np.abs(pi[5] - pi[1]).max())
            max_dup = max(max_dup, err)
            assert err < 3e-5
            action = action_at(pi[1], u[k, 0])
            p, _ = offers(row, bits)
            maintained = bool(u[k, 1] < p[action]) if action else None
            after, body_steps = settle(h, action, maintained, noise[k])
            reward = body.rul(after) - body.rul(h)
            baseline = (-8.0 + prev_rewards) / (8 + k)
            adv = float(reward - baseline)
            e = Encounter(t, i, float(h), action, int(offset[k]), pi[1], adv)
            before_hash = lineage.log.digests["main"]
            v = lineage.log.length("main")
            old_v = lineage.pre_index
            defer_v = lineage.defer_index
            if old_v == v:
                assert float(np.abs(pi[0] - pi[1]).max()) < 3e-5
            shadow_v = (
                lineage.log.length("shadow") if lineage.shadow is not None else None
            )
            additional = {}
            if hasattr(backend, "native"):
                with torch.no_grad():
                    native = (
                        backend.native.panel(i, h, visible, [lineage.main.live()] * 6)[
                            1
                        ]
                        .cpu()
                        .numpy()
                    )
                additional["same_state_native_suffix"] = native.tolist()
            advanced = lineage.settle_update(panel, e, after <= 0)
            events.append(
                {
                    "turn": t,
                    "context": i,
                    "partner": ("P86", "P93")[row["partner"] - 2],
                    "frame": row["frame"],
                    "request": row["text"],
                    "layout": row["layout"],
                    "health_before": float(h),
                    "health_after": after,
                    "five_events": body_steps,
                    "action": action,
                    "word": WORDS[action],
                    "utterance": render(action, row["task"], 0),
                    "maintenance": maintained,
                    "action_uniform": float(u[k, 0]),
                    "partner_uniform": float(u[k, 1]),
                    "body_noise": noise[k].tolist(),
                    "reference_change": float(reward),
                    "personal_baseline_before": float(baseline),
                    "centered_signal": adv,
                    "wrong_action_offset": int(offset[k]),
                    "prob_previous": pi[0].tolist(),
                    "prob_current": pi[1].tolist(),
                    "prob_birth": pi[2].tolist(),
                    "prob_wrong_shadow": pi[3].tolist()
                    if lineage.shadow is not None
                    else None,
                    "prob_deferred_same_tape": pi[4].tolist()
                    if rule == "true1"
                    else None,
                    "counterfactual_previous_choice": action_at(pi[0], u[k, 0]),
                    "counterfactual_birth_choice": action_at(pi[2], u[k, 0]),
                    "before_learning_version": v,
                    "previous_decision_version": old_v,
                    "deferred_version": defer_v if rule == "true1" else None,
                    "shadow_version": shadow_v,
                    "before_hash": before_hash,
                    "after_hash": lineage.log.digests["main"],
                    "after_learning_version": lineage.log.length("main"),
                    "updates_after_settlement": advanced,
                    "terminal": bool(after <= 0),
                    **additional,
                }
            )
            del panel
            prev_rewards += reward
            h = after
            if h <= 0:
                break
            if t in CHECKS:
                inspect("t" + str(t), t)
            if h <= 10 and not danger_seen:
                inspect("first_low", t)
                danger_seen = True
        assert h == 0 and not lineage.buffer
        lineage.log.save(d)
        save_lines(d / "life.jsonl", events)
        np.savez_compressed(d / "living-probes.npz", **probes)
        summary = {
            "seed": seed,
            "birth": birth,
            "visible": visible,
            "rule": rule,
            "world": bits,
            "turns": len(events),
            "next_living_decisions": len(events) - 1,
            "main_steps": lineage.log.length("main"),
            "physical_death": True,
            "birth_sha256": signature(lineage.log.birth),
            "final_sha256": lineage.log.digests["main"],
            "max_duplicate_probability_error": max_dup,
            "living_probes": probe_t,
        }
        write(d / "life.json", summary)
        return summary
    finally:
        backend.close()
