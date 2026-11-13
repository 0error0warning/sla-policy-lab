"""One physical environment and task-matched wording intervention."""

from itertools import permutations, product
import numpy as np
from scipy.special import ndtr, ndtri

from experiments.direct_policy.protocol import rng, settle
from experiments.individual_life.design import offers, rows

PILOT = tuple(range(77101, 77109))
CONFIRM = tuple(range(77201, 77217))
WAVES = {"lcpilot": PILOT, "lcconfirm": CONFIRM}
ITEMS = rows()
LEVELS = (2.0, 6.0, 90.0, 300.0)
FAMILIES = ("qwen", "simple", "simple_on_qwen")
ROLES = ("current", "birth", "wrong")
CHECKS = (8, 16, 32, 64)
LAYOUTS = tuple(
    list(permutations(range(4))).index(tuple((p + j) % 4 for p in range(4)))
    for j in range(4)
)
LOOKUP = {
    (r["partner"], r["frame"], r["task"], r["wording"], r["layout"]): i
    for i, r in enumerate(ITEMS)
}
# Both wordings share precisely the same ordered list of (partner,frame,task,layout).
PANEL = np.array(
    [
        [
            LOOKUP[p, f, t, w, o]
            for p, f, t, o in product((2, 3), range(2), (4, 5), LAYOUTS)
        ]
        for w in (3, 4)
    ],
    dtype=int,
)
SCHEDULE_LIMIT = 2008  # even repeated maintained .25-cost appeals must exhaust the body


def schedule(seed):
    gen = rng(seed, "lifetime-cooperation-locked-v1")
    cells = np.concatenate([gen.permutation(8) for _ in range(SCHEDULE_LIMIT // 8)])
    layouts = np.concatenate(
        [gen.permutation(24) for _ in range((SCHEDULE_LIMIT + 23) // 24)]
    )[:SCHEDULE_LIMIT]
    idx = [
        LOOKUP[2 + int(c) // 4, (int(c) // 2) % 2, 4 + int(c) % 2, 3, int(o)]
        for c, o in zip(cells, layouts, strict=True)
    ]
    u = gen.random((SCHEDULE_LIMIT, 2))
    noise = 0.1 * ndtri(
        ndtr(-3.0) + (ndtr(3.0) - ndtr(-3.0)) * gen.random((SCHEDULE_LIMIT, 5))
    )
    offsets = rng(seed, "lifetime-cooperation-wrong-label").integers(
        1, 4, size=SCHEDULE_LIMIT
    )
    return idx, u, noise, offsets


def rules(seed, world, completed):
    base = 77101 if seed in PILOT else 77201
    a, b = divmod((seed - base) % 4, 2)
    return (a ^ int(world == "changed" and completed >= 24), b)


def checkpoints(completed, h, first_low, terminal):
    tags = []
    if completed == 0:
        tags.append("birth")
    if completed in CHECKS:
        tags.append(f"t{completed}")
    if completed == 24:
        tags.append("t24")  # prior experience, evaluated under the now active rules
    if h <= 10 and first_low:
        tags.append("first_low")
    if terminal:
        tags.append("last_living")
    return tags


def physical_step(h, action, row, bits, partner_u, noise):
    chance, _ = offers(row, bits)
    maintained = bool(partner_u < chance[action]) if action else None
    after, path = settle(h, action, maintained, noise)
    rescued = bool(
        action >= 2
        and maintained
        and after > 0
        and settle(h, action, False, noise)[0] == 0
    )
    return after, path, maintained, rescued


def death_matrix():
    # The monotone event functions have the same lethal outcome at both noise extremes.
    result = np.empty((len(LEVELS), 4, 2))
    for j, h in enumerate(LEVELS):
        for a in range(4):
            for m in range(2):
                flag = bool(m) if a else None
                corners = [settle(h, a, flag, [v] * 5)[0] == 0 for v in (-0.3, 0.3)]
                # At H=6 a silent rejection can survive the lowest noise. The
                # cooperation metric uses only answer outcomes at that health.
                result[j, a, m] = (
                    float(corners[0]) if corners[0] == corners[1] else np.nan
                )
    assert np.array_equal(result[0], [[1, 1], [1, 0], [1, 1], [1, 1]])
    assert np.array_equal(result[1, 2:], [[1, 0], [1, 0]]) and not result[2:].any()
    return result
