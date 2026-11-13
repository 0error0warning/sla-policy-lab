"""Pre-specified input layouts and indices for the matched injury intervention."""

from experiments.consequence_policy.study import ORDERS
from experiments.individual_life.design import rows
from experiments.snapshot_transfer.inputs import row_key

SEEDS = tuple(range(71101, 71109)) + tuple(range(71201, 71217))
HEALTH = (90.0, 300.0)
ROLES = ("pre", "repeat", "rotate")
ARGUMENTS = ("fixed_repeat", "four_rotations")


def rotations(layout):
    original = ORDERS[layout]
    return tuple(ORDERS.index(original[j:] + original[:j]) for j in range(4))


def evaluation_indices(case, items, lookup):
    original = case["original_context"]
    all_orders = [lookup[row_key({**original, "layout": k})] for k in range(24)]
    held_task = [j for j, r in enumerate(items) if (r["task"], r["wording"]) == (5, 4)]
    assert len(held_task) == 96
    return all_orders, held_task


def item_bank():
    items = rows()
    return items, {row_key(r): j for j, r in enumerate(items)}
