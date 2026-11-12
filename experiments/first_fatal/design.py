"""A finite, irreversible body with repeated survivable high-strain encounters."""
import numpy as np

from experiments.direct_policy.protocol import COSTS
from experiments.individual_life.design import offers

PILOT = tuple(range(69101, 69109))
REPLICATION = tuple(range(69201, 69217))
BIRTHS = ("blank", "population")
RULES = ("frozen", "true1", "wrong1")
PRACTICE_TOP = 6.0
FATAL = 2 * COSTS[2] / (1 + COSTS[2] / 350)
HEALTHY = (90., 300.)
HEALTH_MARGIN = .02
RATE_DIVISOR = 48.
MIN_DURATION = 1 / 192
STRAIN = 4.


def world(seed):
    # Four balanced identity/frame ecologies in each separately fixed wave.
    return divmod((seed - (PILOT[0] if seed in PILOT else REPLICATION[0])) % 4, 2)


def first_fatal(health):
    return 0 < health <= FATAL


def practice(health):
    return FATAL < health <= PRACTICE_TOP


def settle(health, action, maintenance, noise):
    if not 0 < health <= 350 or action not in range(4) or (action == 0) != (maintenance is None):
        raise ValueError("invalid living encounter")
    noise = np.asarray(noise)
    if noise.shape != (5,) or not np.all(np.abs(noise) <= .300000000001):
        raise ValueError("five bounded body perturbations required")
    cost = COSTS[action]
    peak = cost * (2. - health / 350.)
    duration = max(MIN_DURATION, min(1., (health / RATE_DIVISOR) ** 2))
    # The analytical cutoff also treats floating-point equality as collapse.
    if health <= 2 * cost / (1. + cost / 350.):
        return 0., [float(health),0.,0.,0.,0.,0.], True
    strain = peak / health
    wear = duration * peak * (1. + STRAIN * strain ** 2) * (1. + noise[0])
    new = max(0., health - wear)
    path=[float(health),float(new)]
    for eta in noise[1:]:
        if not maintenance:
            new = max(0., new - duration * (2. - new / 350.) * (1. + eta))
        path.append(float(new))
    return float(new),path,False


def expected_after(health, row, bits):
    """Exact expected next integrity for the living practice/healthy states (no wear clipping)."""
    prob,_ = offers(row,bits)
    duration = max(MIN_DURATION, min(1., (health / RATE_DIVISOR)**2))
    peak = np.asarray(COSTS)*(2.-health/350)
    post = health - duration*peak*(1.+STRAIN*(peak/health)**2)
    clipped = [settle(health,a,None if not a else False,[.3]*5)[0] == 0 for a in range(4)]
    if any(clipped):raise ValueError("reference restricted to guaranteed nonterminal states")
    unprotected = 700 + (post-700)*(1.+duration/350)**4
    return prob*post+(1.-prob)*unprotected
