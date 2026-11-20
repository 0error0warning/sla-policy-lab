"""Post-hoc CPU audit; never changes the original experiment, labels or weights."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

import numpy as np

from protocol import ACTIONS, ACTION_COST, BODY_ACTION, REPO, SocialWorld, canonical, probabilities, values
from score import match, verify_archive

DATA = REPO / "artifacts/model-studies/2026-11-10/social-learning"


def transition(damage, protected, action, granted, load, weight):
    a = BODY_ACTION[action]
    intrinsic, coefficient = ACTION_COST[a]
    increment = intrinsic + coefficient * (0 if protected else load) * (1 + 1.2 * damage)
    next_damage = min(1., damage + increment)
    next_protection = 3 if a == 2 and granted else max(0, protected - 1)
    reward = (1., -.02, -.08)[a] - weight * increment - (2 if next_damage >= 1 else 0)
    return next_damage, next_protection, reward, next_damage >= 1


def exact_four_turn(state, response_probabilities):
    """Enumerate adaptive continuations, rather than forcing future answers."""
    @lru_cache(None)
    def value(damage, protected, left):
        if left == 0:
            return 0.
        return max(action_value(damage, protected, left, action) for action in ACTIONS)

    def action_value(damage, protected, left, action):
        p = response_probabilities.get(action, 0.)
        result = 0.
        for granted, mass in ((False, 1 - p), (True, p)):
            if mass:
                d, shield, reward, dead = transition(damage, protected, action, granted, state["load"], state["weight"])
                result += mass * (reward + (0. if dead else value(d, shield, left - 1)))
        return result

    return {action: action_value(state["damage"], state["protection_left"], state["turns"], action) for action in ACTIONS}


def finite_horizon_dp(roots, grid_intervals, quadrature_nodes, horizon=32, constant_future_load=None):
    """Approximate the actual 32-turn objective under IID uniform future loads.

    Future load is observed before choosing an action: integrate MAX(Q), not
    MAX(integrated Q). Known .9 best-wording probability is privileged knowledge.
    The worse .1 wording is dominated, so the DP has three physical actions.
    """
    grid = np.linspace(0., 1., grid_intervals + 1)
    grid[-1] = np.nextafter(1., 0.)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_nodes)
    loads = .65 * (.55 + .45 * (nodes + 1) / 2)
    weights /= 2
    if constant_future_load is not None:
        loads, weights = np.array([constant_future_load]), np.array([1.])
    future = np.zeros((4, len(grid)), dtype=np.float64)

    def q(damage, protected, load, continuation):
        all_q = []
        for a in range(3):
            intrinsic, coefficient = ACTION_COST[a]
            inc = intrinsic + coefficient * (0. * load if protected else load) * (1 + 1.2 * damage)
            next_damage = damage + inc
            dead = next_damage >= 1
            reward = (1., -.02, -.08)[a] - 18. * inc - 2. * dead
            failed = np.interp(next_damage, grid, continuation[max(0, protected - 1)])
            next_value = failed
            if a == 2:
                accepted = np.interp(next_damage, grid, continuation[3])
                next_value = .9 * accepted + .1 * failed
            all_q.append(reward + np.where(dead, 0., next_value))
        return np.stack(all_q)

    for _ in range(horizon - 1):
        previous = future
        future = np.empty_like(previous)
        for shield in range(4):
            candidates = q(grid[None, :], shield, loads[:, None], previous)
            future[shield] = weights @ candidates.max(axis=0)
    results = []
    for root in roots:
        state = root["state"]
        result = q(np.array([state["damage"]]), state["protection_left"], np.array([state["load"]]), future)[:, 0]
        actions = ("answer", "refuse", "best_wording_request")
        results.append(dict(id=root["id"], q=dict(zip(actions, result.tolist())), best=actions[int(result.argmax())]))
    return results


def run():
    with verify_archive(DATA / "gpu-results.zip") as archive:
        protocol = json.loads(archive.read("protocol.json"))
        scored = json.loads((DATA / "scored.json").read_bytes())
        training, teacher, roots = [], [], []
        for life in protocol["lives"]:
            lid = life["id"]
            # Next visible load comes from the same archived RNG rule, not from
            # future tape. Reconstructing the checkpoint is independent of LoRA.
            world = SocialWorld(life["seed"], life["mapping"])
            for row in life["trace"]:
                world.step(row["encounter_id"], row["partner"], row["action"], tuple(row["task"]))
            obs = world.observation("P17", "audit")
            state = dict(damage=obs["damage"], protection_left=obs["protection_left"], load=obs["load"], weight=18., turns=4)
            short = values(state, probabilities(life["mapping"])["P17"])
            roots.append(dict(id=lid, state=state, four_turn_reference_best=max(short, key=short.get), four_turn_reference_q=short))
            for arm in ("aligned", "shuffled"):
                source = protocol["datasets"][lid][arm]
                rows = source["rows"]
                target = np.array([row["target"] for row in rows])
                order_sets = defaultdict(set)
                for row in rows:
                    system, user = (m["content"] for m in row["messages"])
                    question = user.split("New task: ")[1]
                    order = int(system.index("answer:") > system.index("exchange:"))
                    order_sets[(question, row["partner"])].add(order)
                steps = [json.loads(line) for line in archive.read(f"output/{lid}-{arm}/steps.jsonl").splitlines()]
                training.append(dict(life=lid, arm=arm, mean_target=dict(zip(ACTIONS, target.mean(axis=0).tolist())),
                                     target_argmax_counts=dict(Counter(max(row["values"], key=row["values"].get) for row in rows)),
                                     orders_per_task_partner=sorted(set(map(len, order_sets.values()))),
                                     epoch_mean_loss=[mean(s["loss"] for s in steps if s["epoch"] == e) for e in range(3)],
                                     clipped_steps=sum(s["gradient_norm"] > 1 for s in steps)))
                oracle_correct = 0
                for test in protocol["tests"]:
                    posterior = {style: x["mean"] for style, x in source["posterior"][test["partner"]].items()}
                    learned_q = values(test["state"], posterior)
                    action = max(learned_q, key=learned_q.get)
                    true_q = values(test["state"], probabilities(life["mapping"])[test["partner"]])
                    oracle_correct += abs(true_q[action] - max(true_q.values())) < 1e-10
                teacher.append(dict(life=lid, arm=arm, perfect_teacher_correct=oracle_correct, n=32))
        observations = {}
        for arm in ("frozen", "aligned", "shuffled"):
            rows = [r for r in scored["details"] if r["arm"] == arm]
            correct_answer = sum(r["correct"] and r["action"] == "answer" for r in rows)
            correct_request = sum(r["correct"] and r["action"] != "answer" for r in rows)
            observations[arm] = dict(correct_answer=correct_answer, correct_request=correct_request,
                                     answer_cases_wrongly_requested=sum("answer" in r["reference_best"] and r["action"] in ("appeal", "exchange") for r in rows),
                                     last_turn_correct=sum(r["correct"] for r in rows if r["id"].startswith("test-06-")),
                                     protected_correct=sum(r["correct"] for r in rows if r["id"].startswith("test-07-")))
        teacher_totals = {arm: sum(t["perfect_teacher_correct"] for t in teacher if t["arm"] == arm)
                          for arm in ("aligned", "shuffled")}
        assert teacher_totals == {"aligned": 124, "shuffled": 88}
        order_changes = {}
        for arm in ("frozen", "aligned", "shuffled"):
            count = 0
            for life in protocol["lives"]:
                selected = {r["id"]: r["action"] for r in scored["details"] if r["arm"] == arm and r["life"] == life["id"]}
                count += sum(selected[f"test-{i:02}-{partner}-order0"] != selected[f"test-{i:02}-{partner}-order1"]
                             for i in range(8) for partner in ("P17", "P42"))
            order_changes[arm] = dict(changed=count, pairs=64)
        assert order_changes["aligned"]["changed"] == 25
        reference_checks = []
        for mapping in (0, 1):
            for test in protocol["tests"]:
                p = probabilities(mapping)[test["partner"]]
                supplied, optimal = values(test["state"], p), exact_four_turn(test["state"], p)
                selected = max(supplied, key=supplied.get)
                reference_checks.append(dict(mapping=mapping, id=test["id"], reference=selected,
                                             adaptive_optimal=max(optimal, key=optimal.get),
                                             adaptive_regret=max(optimal.values()) - optimal[selected]))
        assert all(x["adaptive_regret"] < 1e-10 for x in reference_checks)
        for root in roots:
            # In the actual checkpoint, the preferred wording depends on mapping.
            mapping = next(l["mapping"] for l in protocol["lives"] if l["id"] == root["id"])
            q = exact_four_turn(root["state"], probabilities(mapping)["P17"])
            root["adaptive_four_turn_q"] = q
            root["adaptive_four_turn_best"] = max(q, key=q.get)
        sensitivity = []
        for weight in (0., 1., 6., 18., 36.):
            result = {}
            for arm in ("frozen", "aligned", "shuffled"):
                hits = 0
                losses = []
                for life in protocol["lives"]:
                    chosen = {r["id"]: r["action"] for r in scored["details"] if r["life"] == life["id"] and r["arm"] == arm}
                    for test in protocol["tests"]:
                        q = values(test["state"] | {"weight": weight}, probabilities(life["mapping"])[test["partner"]])
                        loss = max(q.values()) - q[chosen[test["id"]]]
                        hits += loss < 1e-10
                        losses.append(loss)
                result[arm] = dict(correct=hits, mean_regret=mean(losses))
            sensitivity.append(dict(weight=weight, results=result))
    coarse = finite_horizon_dp(roots, 4000, 8)
    fine = finite_horizon_dp(roots, 8000, 16)
    constant_long = [finite_horizon_dp([root], 8000, 1, constant_future_load=root["state"]["load"])[0] for root in roots]
    independent_dp_errors = []
    # At short horizons the grid solver can be compared with exact recursion.
    for test in protocol["tests"][::4]:
        root = {"id": test["id"], "state": test["state"]}
        numerical = finite_horizon_dp([root], 8000, 1, horizon=test["state"]["turns"], constant_future_load=test["state"]["load"])[0]
        exact = exact_four_turn(test["state"], probabilities(0)["P17"])
        expected = dict(answer=exact["answer"], refuse=exact["refuse"], best_wording_request=exact["appeal"])
        error = max(abs(numerical["q"][a] - expected[a]) for a in expected)
        assert error < .001
        independent_dp_errors.append(error)
    return dict(schema="post-hoc-social-learning-audit-v1", original_protocol_sha256=scored["protocol_sha256"],
                original_publication_sha256=sha256((DATA / "publication.json").read_bytes()).hexdigest(),
                audit_source_lf_sha256=sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                scoring_decomposition=observations, training=training, perfect_imitation=teacher,
                perfect_imitation_totals=teacher_totals, static_order_sensitivity=order_changes,
                exact_four_turn_checks=reference_checks, reward_sensitivity_posthoc=sensitivity,
                actual_checkpoint_roots=roots, approximate_32_turn_dp=dict(coarse=coarse, fine=fine,
                    grid_intervals=[4000, 8000], future_load_quadrature_nodes=[8, 16],
                    constant_future_load_comparison=constant_long,
                    short_horizon_max_errors_against_exact=independent_dp_errors,
                    caveat="Known best partner response probability .9; numerical DP, not certified global optimality."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = run()
    if args.verify:
        match(result, json.loads(args.verify.read_bytes()))
    else:
        with args.output.open("xb") as handle:
            handle.write(canonical(result))
    print(json.dumps({"decomposition": result["scoring_decomposition"], "exact_labels_changed": sum(r["adaptive_regret"] > 1e-10 for r in result["exact_four_turn_checks"]),
                      "checkpoint_short_actions": [r["four_turn_reference_best"] for r in result["actual_checkpoint_roots"]],
                      "checkpoint_long_actions": [r["best"] for r in result["approximate_32_turn_dp"]["fine"]]}))
