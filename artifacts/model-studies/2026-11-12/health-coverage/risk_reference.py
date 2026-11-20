"""CPU one-encounter death reference, used only for evaluation.

Partner responses are enumerated exactly. Bounded body noise is integrated
with fixed independent NumPy samples. This module never emits training labels.
"""

import argparse
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent/'inputs'))
from protocol import ACTIONS, COSTS, probabilities


NOISE_SEED = 622900


@lru_cache(maxsize=8)
def body_noise(samples):
    """Draw the specified truncated normal by rejection, without SciPy."""
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError('samples must be a positive integer')
    generator = np.random.default_rng(NOISE_SEED)
    chunks = []
    remaining = samples*5
    while remaining:
        proposed = generator.normal(0., .1, max(64, int(remaining*1.02)))
        retained = proposed[np.abs(proposed) <= .3][:remaining]
        chunks.append(retained)
        remaining -= len(retained)
    values = np.concatenate(chunks).reshape(samples, 5)
    values.setflags(write=False)
    return values


@lru_cache(maxsize=64)
def response_probabilities(partner, frame, mapping):
    if frame not in (0, 1) or mapping not in (0, 1):
        raise ValueError('frame and mapping must be zero or one')
    return tuple(probabilities(partner, frame, mapping))


def _after(health, action, maintained, noise):
    values = np.full(noise.shape[0], health, dtype=float)
    for event in range(5):
        cost = COSTS[action] if event == 0 else (0. if maintained else 1.)
        values = np.maximum(0., values-cost*(2.-values/350.)*(1.+noise[:, event]))
    return values


@lru_cache(maxsize=512)
def _expected_death(health, response, samples):
    noise = body_noise(samples)
    result = np.zeros(4)
    for action in range(4):
        if action == 0:
            result[action] = np.mean(_after(health, action, False, noise) <= 0.)
            continue
        probability = response[action-1]
        maintained = np.mean(_after(health, action, True, noise) <= 0.)
        skipped = np.mean(_after(health, action, False, noise) <= 0.)
        result[action] = probability*maintained+(1.-probability)*skipped
    result.setflags(write=False)
    return result


def expected_death(health, partner, frame, mapping, samples=4096):
    """Return four action death probabilities for one bodily encounter.

    The real partner probabilities belong exclusively to this evaluation
    reference. No model prompt, training baseline, or optimizer consumes them.
    """
    health = float(health)
    if not np.isfinite(health) or not 0. <= health <= 350.:
        raise ValueError('health must be finite and between zero and 350')
    response = response_probabilities(partner, frame, mapping)
    return _expected_death(health, response, samples).copy()


def boundary_checks(samples=4096):
    # Conservative noise-independent upper damage bound over five events.
    damage_bound = [(float(cost)+4.)*2.*1.3 for cost in COSTS]
    assert max(damage_bound) < 15.
    # Even the smallest possible answering cost exhausts health two.
    minimum_answer_cost_at_two = float(COSTS[2])*(2.-2./350.)*.7
    assert minimum_answer_cost_at_two > 2.
    for frame in (0, 1):
        low = expected_death(2., 'P17', frame, 0, samples)
        assert np.array_equal(low[2:], [1., 1.])
        assert np.array_equal(expected_death(15., 'P17', frame, 0, samples), np.zeros(4))
        assert np.array_equal(expected_death(300., 'P17', frame, 0, samples), np.zeros(4))
        six = expected_death(6., 'P17', frame, 0, samples)
        favored = 2 if response_probabilities('P17', frame, 0)[1] > .5 else 3
        assert six[favored] < six[1]
    return dict(passed=True, maximum_possible_damage=damage_bound,
                minimum_answer_action_damage_at_health2=minimum_answer_cost_at_two,
                h2_answer_death_exact=True, h15_all_survive_exact=True,
                h300_all_survive_exact=True, h6_favored_answer_safer_than_refuse_appeal=True)


def reference_table(samples=4096):
    checks = boundary_checks(samples)
    rows = []
    for health in (2., 6., 10., 15., 90., 300.):
        for frame in (0, 1):
            response = response_probabilities('P17', frame, 0)
            death = expected_death(health, 'P17', frame, 0, samples)
            best = np.flatnonzero(np.isclose(death, death.min(), atol=1e-12, rtol=0)).tolist()
            rows.append(dict(health=health, partner='P17', frame=frame, mapping=0,
                response_probabilities=list(response), death_probability=death.tolist(),
                survival_probability=(1.-death).tolist(), best_survival_actions=best,
                best_survival_action_names=[ACTIONS[i] for i in best]))
    protocol = Path(__file__).resolve().parent/'inputs/protocol.py'
    return dict(scope='Evaluation-only one-encounter risk reference; no training targets',
                algorithm='Fixed independent truncated-normal body samples; exact response enumeration',
                samples=samples, noise_seed=NOISE_SEED,
                protocol_sha256=sha256(protocol.read_bytes()).hexdigest(),
                checks=checks, actions=list(ACTIONS), rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=4096)
    args = parser.parse_args()
    result = reference_table(args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(verified=result['checks']['passed'], rows=len(result['rows']),
                         samples=args.samples, noise_seed=NOISE_SEED)))
