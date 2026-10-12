from copy import deepcopy
import pickle
from types import SimpleNamespace

import pytest

import optimize
from optimization.native_checkpoint import load_checkpoint
from test_native_backend import execute, guard_cpu, inputs, managed_arrays
from test_native_datasets import contexts


def suite_inputs(manager, *, n_obj=2, constrained=True, all_scenarios=False):
    base = inputs(manager, n_obj=n_obj)
    base.config["optimize"]["iters"] = 12
    base.config["backtest"].update(suite_enabled=True, scenarios=[{"label":"base"}, {"label":"stress"}])
    if not constrained:
        base.config["optimize"]["limits"] = []
        base.build_limit_checks()
    base.config["optimize"]["gpu"]["screening"] = dict(
        scenarios=["base", "stress"] if all_scenarios else ["base"], survival_fraction=0.5, min_survivors=1,
    )
    return base, optimize.SuiteEvaluator(base, contexts(base, base.timestamps["binance"], lazy=True),
                                         {"default":"mean"})


@pytest.mark.parametrize("n_obj,constrained,all_scenarios", [(2, True, False), (2, False, False),
                                                           (4, True, False), (2, False, True)])
def test_screened_offspring_use_only_full_survivor_fitness_and_records(monkeypatch, tmp_path, n_obj,
                                                                     constrained, all_scenarios):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base, suite = suite_inputs(manager, n_obj=n_obj, constrained=constrained, all_scenarios=all_scenarios)
        records, path = [], tmp_path / "checkpoint.pkl"
        execute(base, SimpleNamespace(record=records.append), path, evaluator=suite)
        state = load_checkpoint(path, base.config)
        assert len(records) == state["completed"] == (12 if all_scenarios else 8)
        assert state["screened"] == (0 if all_scenarios else 8)
        assert state["phase"] == "idle"
        assert len(state["algorithm"].pop) == 4
        for individual in state["algorithm"].pop:
            assert {"F", "G", "H"} <= individual.evaluated
            assert not hasattr(individual, "screening_payload")
        for record in records:
            assert record["suite_metrics"]
            assert all(set(value["scenarios"]) == {"base", "stress"}
                       for value in record["suite_metrics"].values() if "scenarios" in value)
        completed = len(records)
        execute(base, SimpleNamespace(record=records.append), path, evaluator=suite, resume=True)
        assert len(records) == completed  # Completed checkpoints do not start another cohort.


@pytest.mark.parametrize("stop", ["screening", "promotion", "full"])
def test_screening_stage_interrupt_and_resume_preserve_full_results(monkeypatch, tmp_path, stop):
    guard_cpu(monkeypatch)
    from optimization.backends.gpu_backend import _Search
    original = _Search.checkpoint
    progress = {}
    def checkpoint(self, **kwargs):
        original(self, **kwargs)
        progress.update(phase=self.state["phase"], screened=self.state["screened"], completed=self.state["completed"])
    monkeypatch.setattr(_Search, "checkpoint", checkpoint)
    def interrupt():
        if ((stop == "screening" and progress.get("phase") == "screening" and progress.get("screened", 0) > 0)
                or (stop == "promotion" and progress.get("phase") == "generation" and progress.get("screened", 0) >= 4)
                or (stop == "full" and progress.get("completed", 0) > 4)):
            raise KeyboardInterrupt
    with managed_arrays() as manager:
        base, suite = suite_inputs(manager)
        records, path = [], tmp_path / "checkpoint.pkl"
        with pytest.raises(KeyboardInterrupt):
            execute(base, SimpleNamespace(record=records.append), path, evaluator=suite, interrupt_check=interrupt)
        state = load_checkpoint(path, base.config)
        assert state["completed"] == len(records) >= 4
        assert state["phase"] == ("screening" if stop == "screening" else "generation")
        if stop == "screening":
            assert any(hasattr(item, "screening_payload") for item in state["population"])
            assert all(not {"F", "G", "H"} <= item.evaluated for item in state["population"])
        else:
            assert len(state["population"]) == 2
            assert all(not hasattr(item, "screening_payload") for item in state["population"])
        execute(base, SimpleNamespace(record=records.append), path, evaluator=suite, resume=True)
        done = load_checkpoint(path, base.config)
        assert done["phase"] == "idle" and done["screened"] == 8
        assert done["completed"] == len(records) == 8


def test_screening_keeps_seed_bootstrap_fully_evaluated(monkeypatch, tmp_path):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base, suite = suite_inputs(manager)
        vector = optimize.config_to_individual(base.config, base.bounds, optimization_shape=base.optimization_shape)
        starts = [list(vector) for _ in range(6)]
        index = [key for key, _path in base.key_paths].index("long_entry_initial_qty_pct")
        for count, row in enumerate(starts):
            row[index] = 0.01 + count * 0.004
        records, path = [], tmp_path / "checkpoint.pkl"
        execute(base, SimpleNamespace(record=records.append), path, evaluator=suite, starts=starts)
        assert load_checkpoint(path, base.config)["screened"] == 8
        assert len(records) == 10  # Six complete seeds supply parents; four full offspring follow.


@pytest.mark.parametrize("invalid", ["unknown", "objective", "limit"])
def test_screening_rejects_bad_or_missing_required_scenarios_before_device_start(monkeypatch, tmp_path, invalid):
    guard_cpu(monkeypatch)
    import optimization.gpu.native as native
    def forbidden(**_kwargs):
        pytest.fail("invalid screening policy must fail before constructing the GPU service")
    monkeypatch.setattr(native, "CudaBacktestService", forbidden)
    with managed_arrays() as manager:
        base, suite = suite_inputs(manager)
        if invalid == "unknown":
            base.config["optimize"]["gpu"]["screening"]["scenarios"] = ["missing"]
        elif invalid == "objective":
            base.config["optimize"]["objective_scenario"] = "stress"
            suite = optimize.SuiteEvaluator(base, suite.contexts, suite.reducer_cfg)
        else:
            base.config["optimize"]["limits"][0]["scenario"] = "stress"
            suite = optimize.SuiteEvaluator(base, suite.contexts, suite.reducer_cfg)
        with pytest.raises(ValueError, match="unknown labels|retain explicitly selected"):
            execute(base, SimpleNamespace(record=lambda _row: None), tmp_path / "checkpoint.pkl", evaluator=suite)


def test_standalone_screening_requires_a_suite(monkeypatch, tmp_path):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["optimize"]["gpu"]["screening"] = {"scenarios":["base"]}
        with pytest.raises(ValueError, match="requires a prepared scenario suite"):
            execute(base, SimpleNamespace(record=lambda _row: None), tmp_path / "checkpoint.pkl")


@pytest.mark.parametrize("policy", [{"min_survivors":4}, {"survival_fraction":1.0}])
def test_noop_screening_policy_uses_one_full_stage(monkeypatch, tmp_path, policy):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base, suite = suite_inputs(manager)
        base.config["optimize"]["gpu"]["screening"].update(policy)
        records, path = [], tmp_path / "checkpoint.pkl"
        execute(base, SimpleNamespace(record=records.append), path, evaluator=suite)
        state = load_checkpoint(path, base.config)
        assert state["completed"] == len(records) == 12
        assert state["screened"] == 0


@pytest.mark.parametrize("corrupt", ["shape", "penalty", "evaluated", "version"])
def test_resume_rejects_malformed_partial_screening_evidence(monkeypatch, tmp_path, corrupt):
    guard_cpu(monkeypatch)
    from optimization.backends.gpu_backend import _Search
    def stop(_self):
        raise KeyboardInterrupt
    monkeypatch.setattr(_Search, "promote_screening", stop)
    with managed_arrays() as manager:
        base, suite = suite_inputs(manager)
        path = tmp_path / "checkpoint.pkl"
        with pytest.raises(KeyboardInterrupt):
            execute(base, SimpleNamespace(record=lambda _row: None), path, evaluator=suite)
        state = deepcopy(load_checkpoint(path, base.config))
        individual = state["population"][0]
        if corrupt == "shape":
            individual.screening_payload["fitness"] = [0.1]
        elif corrupt == "penalty":
            individual.screening_payload["constraint_violation"] = float("nan")
        elif corrupt == "evaluated":
            individual.evaluated.update(("F", "G", "H"))
        else:
            state["version"] = 1
        path.write_bytes(pickle.dumps(state))
        with pytest.raises(ValueError, match="partial screening evidence|compatible native checkpoint"):
            load_checkpoint(path, base.config)
