"""Independent equation, noise-integration, posterior and replay checks."""

from decimal import Decimal
from statistics import NormalDist
import argparse
import json
from pathlib import Path

import numpy as np

from core import (ACTIONS, BodyValue, CONFIG, COSTS, Evidence, HIGH, LOW,
                  body_event, event_noise, execute_event, schedule, settle, write)


def validate():
    body = BodyValue()
    finer = BodyValue(.02)
    assert body.rul(0) == 0 and np.all(np.diff(body.reference) >= 0)
    assert all(body.rul(h) >= .25 for h in [.001, .1, 1., 350.])
    step_error = max(abs(body.rul(h)-finer.rul(h)) for h in [1., 2., 6., 20., 100., 200., 350.])
    # Independent Decimal event settlement, with the recorded realizations.
    max_event_error, events = 0., 0
    for seed in CONFIG["seeds"]:
        evidence, health = Evidence(), 350.
        for item in schedule(seed):
            for action in range(4):
                for accepted in ([None] if action == 0 else [False, True]):
                    noise = event_noise(seed, item["turn"])
                    actual, trace = settle(health, action, accepted, noise)
                    h = Decimal(str(health))
                    expected = [float(h)]
                    for i, eta in enumerate(noise):
                        damage = Decimal(str(float(COSTS[action]))) if i == 0 else Decimal(0 if accepted else 1)
                        h = max(Decimal(0), h-damage*(Decimal(2)-h/Decimal(350))*(Decimal(1)+Decimal(str(eta))))
                        expected.append(float(h))
                    max_event_error = max(max_event_error, max(abs(a-b) for a, b in zip(trace, expected)))
                    events += 5
                    assert actual >= 0 and actual <= health
            p = evidence.moments(item["partner"], item["frame"])[0]
            record = execute_event(body, evidence, health, item, 0, seed, p)
            health = record["health_after"]
            if health == 0:
                break
    assert max_event_error < 1e-10
    # Analytic integration of the two Beta levels vs independent sampled conditionals.
    evidence = Evidence()
    for outcome in [True]*8+[False]*2:
        evidence.observe("P17", 0, 2, outcome)
    for outcome in [True]+[False]*4:
        evidence.observe("P42", 0, 2, outcome)
    mean, variance = evidence.moments("P42", 0)
    gen = np.random.default_rng(102)
    group = gen.beta(9., 3., 300000)
    personal = gen.beta(.5*group+1., .5*(1-group)+4.)
    posterior_mean_error = abs(float(personal.mean())-mean[1])
    posterior_variance_error = abs(float(personal.var())-variance[1])
    assert posterior_mean_error < .002 and posterior_variance_error < .002
    # Compare sequential continuous-noise MC to the seven-node Markov operators.
    quadrature = []
    normal = NormalDist()
    for health in [350., 100., 6., 2.]:
        for action in [1, 2]:
            for accepted in [False, True]:
                h = np.full(40000, health)
                for i in range(5):
                    eta = np.array([.1*normal.inv_cdf(LOW+(HIGH-LOW)*float(u)) for u in gen.random(len(h))])
                    h = body_event(h, COSTS[action] if i == 0 else (0. if accepted else 1.), eta)
                reference = body.maintain if accepted else body.skip
                analytic = body.interpolate(health, reference[action])
                monte_carlo = body.interpolate(h, body.reference).mean()
                quadrature.append(dict(health=health, action=ACTIONS[action], accepted=accepted,
                    quadrature=float(analytic), continuous_noise_mc=float(monte_carlo),
                    absolute_error=float(abs(analytic-monte_carlo))))
    assert max(r["absolute_error"] for r in quadrature) < .04
    # Silence is censored; maintenance does not repair health or suppress action cost.
    before = evidence.export()
    evidence.observe("P17", 0, 0, None)
    assert before == evidence.export()
    assert settle(100., 1, True, [0.]*5)[0] == 100.-.25*(2.-100./350.)
    assert settle(100., 2, True, [0.]*5)[0] < 100.
    return dict(passed=True, decimal_events=events, max_decimal_event_error=max_event_error,
        reference_grid_comparison_max_error=step_error, reference_rul_full=body.rul(350.),
        posterior_mean_mc_error=posterior_mean_error, posterior_variance_mc_error=posterior_variance_error,
        transition_mc=quadrature, reference_zero_boundary="R(0)=0; positive health below grid step retains the first positive tick value",
        scope="equation checks, not independent replication of original assets/random-number streams")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate()
    write(args.output, result)
    print(json.dumps({k:v for k, v in result.items() if k != "transition_mc"}, ensure_ascii=False))
