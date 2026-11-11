"""Preflight the future factor probes using the simulated body, without Qwen."""
import argparse
from itertools import product
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/'inputs'))
from protocol import COSTS, rng, write


def grid():
    normal = NormalDist()
    lo, hi = normal.cdf(-3), normal.cdf(3)
    seeds = list(range(62001, 62021))
    draws = rng(62000,'preflight-noise').uniform(lo,hi,(len(seeds),256,5))
    noise = np.vectorize(normal.inv_cdf)(draws)*.1
    response = np.stack([rng(s,'preflight-response').random(256) for s in seeds])
    results = []
    for health,risk,support,value in product((2.,6.,30.,300.),(.5,1.,1.5),(.1,.9),(0.,1.,3.)):
        values, survival = [], []
        for action, base_cost in enumerate(COSTS):
            probability = [0.,.15,support,.1][action]
            maintain = response < probability
            h = np.full(response.shape,health)
            cost = base_cost*(risk if action>=2 else 1.)
            for event in range(5):
                b = cost if event==0 else ~maintain
                h = np.maximum(0,h-b*(2-h/350)*(1+noise[...,event]))
            alive = h>0
            reward = 2 + value*(action>=2) + .5*maintain - .15*(health-h)-12*~alive
            values.append(float(reward.mean()))
            survival.append(float(alive.mean()))
        results.append(dict(health=health,answer_damage_multiplier=risk,
            answer_appeal_maintenance_probability=support,task_value=value,expected_reward=values,
            survival_probability=survival,preferred_action=int(np.argmax(values))))
    return dict(scope='CPU one-encounter factor preflight; synthetic outcomes, no model decisions',
                response_seeds=seeds,noise_samples_per_seed=256,rows=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    write(parser.parse_args().output, grid())
