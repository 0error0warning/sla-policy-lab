"""Conditional independent autonomous-life wave using the same frozen birth/sensor."""

import numpy as np
import torch

from experiments.direct_policy.protocol import write
from experiments.individual_life.design import action_at
from experiments.lifetime_cooperation.design import schedule, physical_step
from experiments.social_closed_loop.core import BodyValue
from .adapters import advance
from .design import ITEMS, LIVE_SEEDS, ROLES
from .engine import People
from .gpu import panel, save_events


def run_online(engine, out):
    body = BodyValue()
    lives = []
    for seed in LIVE_SEEDS:
        tape = schedule(seed)
        bits = divmod((seed - LIVE_SEEDS[0]) % 4, 2)
        for source in ("birth", "native_lora", "loreft"):
            write(
                out / "status.json",
                dict(
                    stage="autonomous_life",
                    seed=seed,
                    agent=source,
                    completed=len(lives),
                ),
            )
            slot = ROLES.index(source)
            person = People(engine.q)
            engine.bind(person)
            d = out / "lives" / f"s{seed}" / source
            d.mkdir(parents=True, exist_ok=False)
            h, total = 350.0, 0.0
            history = []
            snapshots = {}
            logs = []
            try:
                for n, i in enumerate(tape[0]):
                    i = int(i)
                    p = engine.decision(i, h, check=n in (0, 24))
                    probs = p.detach().cpu().numpy()
                    action = action_at(probs[slot], tape[1][n, 0])
                    bad = (action + int(tape[3][n])) % 4
                    after, path, accepted, rescued = physical_step(
                        h, action, ITEMS[i], bits, tape[1][n, 1], tape[2][n]
                    )
                    reward = body.rul(after) - body.rul(h)
                    signal = reward - (total - 8) / (8 + n)
                    if n in (0, 24, 32) or after == 0:
                        tag = "last" if after == 0 else f"k{n}"
                        snapshots[tag] = panel(engine)
                        if after == 0:
                            np.savez_compressed(
                                d / "last-living-weights.npz", **person.state()
                            )
                    updates = {}
                    if slot:
                        shadow = 4 if slot == 1 else 5
                        for j, label in ((slot, action), (shadow, bad)):
                            model = person.subjects[j]
                            grad = torch.autograd.grad(
                                p[j, label].log(), model.live(), retain_graph=j == slot
                            )
                            # Delay both writes until both gradients have been computed.
                            logs.append((j, label, model, grad))
                        for j, label, model, grad in logs:
                            updates[ROLES[j]] = advance(
                                model,
                                grad,
                                signal,
                                float(probs[slot, label]),
                                float(probs[j, label]),
                            )
                        logs.clear()
                    history.append(
                        dict(
                            turn=n + 1,
                            row=i,
                            health=h,
                            after=after,
                            stages=path,
                            maintenance=accepted,
                            action=action,
                            action_u=float(tape[1][n, 0]),
                            partner_u=float(tape[1][n, 1]),
                            noise=tape[2][n].tolist(),
                            rescued=rescued,
                            wrong_action=bad,
                            reward=reward,
                            signal=signal,
                            prob=probs.tolist(),
                            updates=updates,
                        )
                    )
                    h = after
                    total += reward
                    if h == 0:
                        break
                assert h == 0
                save_events(d / "events.jsonl", history)
                np.savez_compressed(d / "panels.npz", **snapshots)
                healthy = [e for e in history if e["health"] >= 90]
                meta = dict(
                    seed=seed,
                    agent=source,
                    turns=len(history),
                    healthy_rounds=len(healthy),
                    healthy_tasks=sum(
                        e["action"] >= 2 and e["after"] > 0 for e in healthy
                    ),
                    injured_then_again=sum(
                        e["after"] > 0 and e["after"] < e["health"] for e in history
                    ),
                    low_then_again=sum(
                        e["health"] <= 10 and e["after"] > 0 for e in history
                    ),
                    physical_death=True,
                )
                lives.append(meta)
                write(d / "life.json", meta)
            finally:
                engine.close()
    write(out / "lives.json", lives)
