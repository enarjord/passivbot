from types import SimpleNamespace
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pytest

import optimize
from optimization.native_checkpoint import load_checkpoint
from optimization.problem import PymooEvaluatorAdapter
from test_native_backend import FakeService, execute, guard_cpu, inputs, managed_arrays
from test_native_screening import suite_inputs


def rejecting_policy():
    return {
        "checks": [
            {
                "metric": "entry_retracement",
                "enabled": True,
                "min": 100,
                "sides": ["long"],
            }
        ]
    }


def vector(base):
    return optimize.config_to_individual(
        base.config, base.bounds, optimization_shape=base.optimization_shape
    )


@pytest.mark.parametrize("suite_mode", [False, True])
def test_cpu_prefilter_skips_payload_and_replay_and_activates_constraints(
    monkeypatch, suite_mode
):
    with managed_arrays() as manager:
        base, suite = suite_inputs(manager) if suite_mode else (inputs(manager), None)
        base.config["optimize"]["limits"] = []
        base.config["optimize"]["prefilter"] = rejecting_policy()
        base.config["backtest"]["dynamic_wel_by_tradability"] = False
        base.build_limit_checks()
        base.use_duplicate_guard = False
        evaluator = suite or base

        def forbidden(*_args, **_kwargs):
            pytest.fail(
                "rejected config must not build a candle payload or execute a replay"
            )

        monkeypatch.setattr(optimize, "build_backtest_payload", forbidden)
        monkeypatch.setattr(optimize, "execute_backtest", forbidden)
        payload = evaluator.evaluate(vector(base), [])
        assert payload["constraint_violation"] > 0
        assert all(
            value == payload["constraint_violation"] for value in payload["fitness"]
        )
        assert payload["metrics"]["backtest_performed"] is False
        assert PymooEvaluatorAdapter(evaluator).has_constraints


def test_passing_cpu_prefilter_continues_into_existing_payload_path(monkeypatch):
    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["backtest"]["dynamic_wel_by_tradability"] = False
        base.config["optimize"]["prefilter"] = {
            "checks": [
                {"metric": "entry_retracement", "enabled": True, "max": 100},
            ]
        }
        base.use_duplicate_guard = False

        class ReachedReplay(Exception):
            pass

        def reached(*_args, **_kwargs):
            raise ReachedReplay

        monkeypatch.setattr(optimize, "build_backtest_payload", reached)
        with pytest.raises(ReachedReplay):
            base.evaluate(vector(base), [])


def test_duplicate_rejection_keeps_its_non_replay_provenance(monkeypatch):
    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["backtest"]["dynamic_wel_by_tradability"] = False
        base.config["optimize"]["prefilter"] = rejecting_policy()
        base.duplicate_counter = defaultdict(int)
        for name in (
            "perturb_x_pct",
            "perturb_step_digits",
            "perturb_gaussian",
            "perturb_random_subset",
            "perturb_sample_some",
            "perturb_large_uniform",
        ):
            monkeypatch.setattr(base, name, lambda values: list(values))
        first = base.evaluate(vector(base), [])
        reused = base.evaluate(vector(base), [])
        assert first["fitness"] == reused["fitness"]
        assert reused["metrics"]["backtest_performed"] is False


@pytest.mark.parametrize("screening", [False, True])
def test_all_rejected_gpu_cohorts_reach_evolution_and_resume_without_replay(
    monkeypatch, tmp_path, screening
):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base, suite = (
            suite_inputs(manager, constrained=False)
            if screening
            else (inputs(manager), None)
        )
        base.config["optimize"]["limits"] = []
        base.config["optimize"]["prefilter"] = rejecting_policy()
        base.config["backtest"]["dynamic_wel_by_tradability"] = False
        base.build_limit_checks()
        records, path = [], tmp_path / "checkpoint.pkl"
        execute(
            base, SimpleNamespace(record=records.append), path, evaluator=suite or base
        )
        assert FakeService.rows == []
        assert len(records) == base.config["optimize"]["iters"]
        assert all(row["metrics"]["backtest_performed"] is False for row in records)
        state = load_checkpoint(path, base.config)
        assert state["phase"] == "idle"
        assert np.all(state["algorithm"].pop.get("G") > 0)
        before = len(records)
        execute(
            base,
            SimpleNamespace(record=records.append),
            path,
            evaluator=suite or base,
            resume=True,
        )
        assert len(records) == before and FakeService.rows == []
        base.config["optimize"]["prefilter"]["checks"][0]["min"] = 99
        with pytest.raises(ValueError, match="evaluation contract changed"):
            load_checkpoint(path, base.config)


def test_mixed_screened_cohort_keeps_config_rejections_in_tell(monkeypatch, tmp_path):
    guard_cpu(monkeypatch)
    import optimization.native_planning as planning

    def controlled(configs, params):
        # Exercise admission/selection independently of strategy measurement:
        # the scalar worsens continuously as this proposed gene moves down.
        qty = next(iter(configs.values()))[0]["bot"]["long"]["strategy"][
            "trailing_martingale"
        ]["entry"]["initial_qty_fraction"]
        if qty < 0.03:
            return (0.03 - qty) * 1e6, [{"metric": "fixture_constraint", "value": qty}]
        return 0.0, []

    monkeypatch.setattr(planning, "prefilter_candidate", controlled)
    with managed_arrays() as manager:
        base, suite = suite_inputs(manager, constrained=False)
        base.config["optimize"]["prefilter"] = rejecting_policy()
        base.config["backtest"]["dynamic_wel_by_tradability"] = False
        records, path = [], tmp_path / "checkpoint.pkl"
        # Explicitly place both outcomes in the initial cohort. A random cohort
        # need not contain both classes, even when its configured seed is fixed.
        import optimization.backends.gpu_backend as backend
        population = []
        for qty in (0.01, 0.02, 0.04, 0.05):
            config = deepcopy(base.config)
            config["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_fraction"] = qty
            population.append(optimize.config_to_individual(config, base.bounds,
                              optimization_shape=base.optimization_shape))
        monkeypatch.setattr(backend, "_build_random_sampling", lambda *_args: np.asarray(population))
        execute(base, SimpleNamespace(record=records.append), path, evaluator=suite)
        rejected = [
            row for row in records if row["metrics"].get("backtest_performed") is False
        ]
        replayed = [
            row
            for row in records
            if row["metrics"].get("backtest_performed") is not False
        ]
        assert rejected and replayed
        assert all(
            row["long_entry_initial_qty_pct"] >= 0.03 for row in FakeService.rows
        )
        assert load_checkpoint(path, base.config)["phase"] == "idle"
