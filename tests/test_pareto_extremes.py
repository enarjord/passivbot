import math
import random

import pytest

from pareto_store import ParetoStore
from pure_funcs import calc_hash
from opt_utils import round_floats


def _make_candidate(w_values, violation=0.0):
    """Build a minimal candidate dict compatible with ParetoStore expectations."""
    return {
        "metrics": {
            "objectives": dict(w_values),
            "constraint_violation": violation,
        },
        "optimize": {"scoring": sorted(w_values.keys())},
    }


@pytest.mark.parametrize("sig_digits", [6])
def test_pareto_front_retains_per_metric_extremes(sig_digits):
    random.seed(42)

    # Build a synthetic candidate set with 3 objectives and some violating entries.
    candidates = []
    for i in range(300):
        obj = {
            "w_0": round(random.uniform(-0.002, 0.0), 8),
            "w_1": round(random.uniform(-0.001, 0.0), 8),
            "w_2": round(random.uniform(0.0, 1.0), 8),
        }
        violation = 0.0 if i % 5 else 1.0  # every 5th candidate violates constraints
        candidates.append(_make_candidate(obj, violation=violation))

    # Compute best (minimum) per-objective among non-violating candidates after rounding.
    non_violating = [
        round_floats(c, sig_digits) for c in candidates if c["metrics"]["constraint_violation"] <= 0
    ]
    assert non_violating, "Expected some non-violating candidates"

    best_per_metric = {}
    for cand in non_violating:
        for k, v in cand["metrics"]["objectives"].items():
            if k not in best_per_metric or v < best_per_metric[k]:
                best_per_metric[k] = v

    # Build Pareto front with pruning (max_size << total candidates).
    store = ParetoStore(
        directory="/tmp",
        sig_digits=sig_digits,
        flush_interval=10_000,
        max_size=50,  # force pruning to check extreme preservation
    )
    for cand in candidates:
        store.add_entry(cand)

    front = store.get_front()
    front_hashes = {calc_hash(round_floats(cfg, sig_digits)) for cfg in front}

    # Assert each metric's minimum value is represented in the pruned front.
    for metric, min_val in best_per_metric.items():
        found = False
        for cfg in front:
            cfg_val = cfg["metrics"]["objectives"].get(metric)
            if cfg_val is None:
                continue
            if math.isclose(cfg_val, min_val, rel_tol=0.0, abs_tol=1e-12):
                found = True
                break
        assert found, f"Front missing extreme for {metric}: expected {min_val}"
