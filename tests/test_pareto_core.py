import numpy as np

from config.scoring import ObjectiveSpec
from pareto_core import (
    compute_ideal,
    crowding_distances,
    detect_latest_pareto_dir,
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
    assert keys == ["m0", "m1", "m2"]


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


def test_compute_ideal_weighted_respects_mixed_objective_goals():
    values = np.array(
        [
            [0.001, 400.0],
            [0.003, 800.0],
        ]
    )
    specs = [
        ObjectiveSpec(metric="adg_strategy_pnl_rebased", goal="max"),
        ObjectiveSpec(metric="peak_recovery_hours", goal="min"),
    ]

    ideal = compute_ideal(values, mode="weighted", weights=np.array([0.25, 0.25]), objective_specs=specs)

    assert ideal[0] == np.float64(0.0025)
    assert ideal[1] == np.float64(500.0)


def test_detect_latest_pareto_dir_uses_run_name_and_requires_json(tmp_path):
    older = tmp_path / "optimize_results" / "2026-04-28T09_00_00_old" / "pareto"
    newer = tmp_path / "optimize_results" / "2026-04-28T10_00_00_new" / "pareto"
    empty_latest = tmp_path / "optimize_results" / "2026-04-28T11_00_00_empty" / "pareto"
    junk_latest = tmp_path / "optimize_results" / "2026-04-28T12_00_00_junk" / "pareto"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    empty_latest.mkdir(parents=True)
    junk_latest.mkdir(parents=True)
    (older / "older.json").write_text("{}", encoding="utf-8")
    (newer / "newer.json").write_text("{}", encoding="utf-8")
    (junk_latest / ".DS_Store").write_text("junk", encoding="utf-8")
    older_dir = older.resolve()
    newer_dir = newer.resolve()
    older_dir.touch()
    newer_dir.touch()

    assert detect_latest_pareto_dir(tmp_path / "optimize_results") == newer_dir
