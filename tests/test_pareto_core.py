import numpy as np

from pareto_core import (
    crowding_distances,
    dominates_with_violation,
    extract_objectives,
    prune_front_with_extremes,
)


def test_dominance_prefers_lower_violation_and_better_objectives():
    obj_a = (1.0, 1.0)
    obj_b = (1.0, 1.0)
    assert dominates_with_violation(obj_a, 0.1, obj_b, 0.2)
    assert not dominates_with_violation(obj_b, 0.2, obj_a, 0.1)

    obj_c = (0.5, 0.9)
    assert dominates_with_violation(obj_c, 0.0, obj_a, 0.0)


def test_extract_objectives_respects_scoring_order():
    entry = {"metrics": {"objectives": {"w_1": 2.0, "w_0": 1.0, "w_2": 3.0}}}
    obj, keys = extract_objectives(entry, scoring_keys=["m0", "m1", "m2"])
    assert obj == (1.0, 2.0, 3.0)
    assert keys == ["w_0", "w_1", "w_2"]


def test_prune_preserves_extremes_and_uses_crowding():
    front = ["a", "b", "c", "d", "e"]
    objectives = {
        "a": (0.1, 0.5),
        "b": (0.2, 0.4),
        "c": (0.3, 0.3),
        "d": (0.4, 0.2),
        "e": (0.5, 0.1),
    }
    violations = {k: 0.0 for k in front}
    to_remove = prune_front_with_extremes(front, objectives, violations, max_size=3)
    remaining = set(front) - set(to_remove)
    # extremes along each axis should be kept (a,e)
    assert {"a", "e"}.issubset(remaining)
    assert len(remaining) == 3

    # crowding distances still computable for completeness
    arr = np.array([objectives[k] for k in front])
    cds = crowding_distances(arr)
    assert cds.shape[0] == len(front)
