"""Benchmark evidence must remain reproducible and refuse misleading comparisons."""

from copy import deepcopy
from itertools import combinations, product
import json
import random
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.optimizer_benchmark.catalogue import (Case, PAIRWISE_FLAGS, catalogue,
    declared_gpu_support, pairwise_rows, select_cases)
from tools.optimizer_benchmark.workloads import array_digest, digest, freeze, load_manifest
from tools.optimizer_benchmark.report import numerical_comparison, summary, tuning_comparison, validate_policies
from tools.optimizer_benchmark.__main__ import modes, read_worker_report
from tools.optimizer_benchmark.worker import strict_json, instrument_replay, cohort_digest
from tools.optimizer_benchmark import worker


@pytest.fixture(autouse=True)
def restore_rng_states():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_pairwise_covers_all_pairs_and_states():
    rows = pairwise_rows()
    for left, right in combinations(PAIRWISE_FLAGS, 2):
        assert {(int(left in row), int(right in row)) for row in rows} == set(product((0, 1), repeat=2))
    assert len(rows) < 2 ** len(PAIRWISE_FLAGS)


def test_catalogue_has_direction_topology_baselines_and_unique_ids():
    cases = catalogue()
    assert len({c.name for c in cases}) == len(cases)
    for strategy, side, topology in product(("trailing_martingale", "ema_anchor"),
                                           ("long", "short", "both"), ("single", "fixed", "forager")):
        assert Case(strategy, side, topology) in cases


@pytest.mark.parametrize("case", [Case("trailing_grid_v7"), Case("ema_anchor", features=("position_enforcer",)),
                                  Case("trailing_martingale", features=("btc_collateral",))])
def test_capability_limits_are_explicit(case):
    assert declared_gpu_support(case)


def test_empty_case_filter_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        select_cases("smoke", ["does-not-exist"])


@pytest.fixture
def real_runtime():
    import passivbot_rust
    if getattr(passivbot_rust, "__is_stub__", False):
        pytest.skip("real Rust strategy metadata required")


@pytest.fixture
def frozen(tmp_path, real_runtime):
    directory = tmp_path / "workload"
    manifest = freeze(directory, patterns=["trailing_martingale/short/fixed/*"], candidates=8, bars=128)
    return directory, manifest


def test_population_reproduces_across_calls_and_preserves_rng_state(tmp_path, real_runtime):
    random.seed(91)
    np.random.seed(91)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    first = freeze(tmp_path / "one", patterns=["trailing_martingale/short/fixed/*"], candidates=8, bars=128)
    second = freeze(tmp_path / "two", patterns=["trailing_martingale/short/fixed/*"], candidates=8, bars=128)
    assert first == second
    assert random.getstate() == python_state
    observed = np.random.get_state()
    assert observed[0] == numpy_state[0] and np.array_equal(observed[1], numpy_state[1])
    assert observed[2:] == numpy_state[2:]
    third = freeze(tmp_path / "three", patterns=["trailing_martingale/short/fixed/*"], candidates=8, bars=128, seeds=[43])
    assert first["entries"][0]["vectors"] != third["entries"][0]["vectors"]


def test_population_obeys_bounds_steps_and_side_class(frozen):
    _directory, manifest = frozen
    entry = manifest["entries"][0]
    assert len({tuple(row) for row in entry["vectors"]}) == 8
    for vector in entry["vectors"]:
        for value, bound in zip(vector, entry["bounds"], strict=True):
            assert bound["low"] <= value <= bound["high"]
            if bound["step"]:
                index = (value - bound["low"]) / bound["step"]
                assert abs(index - round(index)) < 1e-7
    for config in entry["configs"]:
        assert config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 0
        assert config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] > 0
        assert not config["bot"]["short"]["hsl"]["enabled"]
        assert not config["bot"]["short"]["unstuck"]["enabled"]
        assert config["bot"]["short"]["entry_cooldown"]["base_duration_minutes"] == 0


def test_manifest_rejects_tampering_and_nonempty_destination(frozen):
    directory, manifest = frozen
    assert load_manifest(directory) == manifest
    manifest["entries"][0]["vectors"][0][0] += 1
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="integrity"):
        load_manifest(directory)
    with pytest.raises(ValueError, match="empty"):
        freeze(directory)


def test_dataset_hash_includes_shape_dtype_and_contents():
    values = np.arange(12, dtype=np.float64)
    assert array_digest(values) != array_digest(values.reshape(3, 4))
    assert array_digest(values) != array_digest(values.astype(np.float32))
    assert array_digest(values) != array_digest(values + 1)


def test_summary_counts_completed_work_over_total_wall_time():
    result = dict(status="completed", samples=[dict(seconds=1, completed=8, results=[], dispatches=[dict(candidates=4)]),
                                               dict(seconds=3, completed=8, results=[], dispatches=[dict(candidates=2)])])
    measured = summary(result)
    assert measured["candidates_per_second"] == 4
    assert measured["median_batch_seconds"] == 2
    assert measured["max_observed_runner_width"] == 4
    assert measured["single_candidate_latency_median_seconds"] is None
    assert summary(dict(status="bounded_stop", samples=result["samples"])) == dict(status="bounded_stop")


def result(value=1.0, liquidated=False):
    return dict(status="completed", samples=[dict(results=[dict(metrics={"adg_strategy_eq": value}, liquidated=liquidated)])])


def test_missing_policy_liquidation_and_nonfinite_never_pass():
    metric = ["adg_strategy_eq"]
    exact = {metric[0]: dict(absolute=0, relative=0)}
    assert numerical_comparison(result(), result(), metric, exact)["status"] == "passed"
    for observed, policy in ((result(), {}), (result(liquidated=None), exact),
                             (result(liquidated=True), exact), (result(float("inf")), exact)):
        assert numerical_comparison(result(), observed, metric, policy)["status"] != "passed"
    assert numerical_comparison(result(), None, metric, exact)["status"] == "unassessed"


@pytest.mark.parametrize("invalid", [["cpu:auto"], ["cpu:0"], ["gpu:-1"], ["gpu:4097"], ["unknown:1"]])
def test_invalid_execution_modes_are_rejected(invalid):
    with pytest.raises(ValueError):
        modes(invalid)


def test_nonfinite_json_evidence_is_retained():
    result = strict_json(dict(nan=float("nan"), positive=float("inf"), negative=-float("inf")))
    assert json.loads(json.dumps(result, allow_nan=False)) == dict(nan="NaN", positive="Infinity", negative="-Infinity")


def test_empty_candidate_results_are_unassessed():
    empty = dict(status="completed", samples=[dict(results=[])])
    assert numerical_comparison(empty, empty, ["adg_strategy_eq"], {})["status"] == "unassessed"


def test_numerical_tolerance_cannot_hide_a_constraint_flip():
    metric = "adg_strategy_eq"
    policy = {metric: dict(absolute=1, relative=0)}
    checks = [dict(metric=metric, mode="greater_than", bound=.5, penalty_weight=1)]
    comparison = numerical_comparison(result(.49), result(.51), [metric], policy, checks)
    assert comparison["rows"][0]["metrics"][metric]["status"] == "match"
    assert comparison["rows"][0]["limits"][0]["status"] == "flip"
    assert comparison["status"] != "passed"


def test_tuner_without_completed_windows_is_not_converged():
    automatic = dict(samples=[dict(seconds=1, completed=16, results=[], elapsed_seconds=i + 1) for i in range(3)])
    fixed = dict(status="completed", samples=[dict(seconds=1, completed=16, results=[])])
    comparison = tuning_comparison(automatic, [fixed])
    assert comparison["status"] == "insufficient_tuner_evidence"
    assert comparison["time_to_three_cohorts_within_five_percent_seconds"] == 3
    assert comparison["first_tuning_window_seconds"] is None


def test_boundaries_retain_random_population(tmp_path, real_runtime):
    manifest = freeze(tmp_path / "boundaries", patterns=["trailing_martingale/short/fixed/*"],
                      candidates=8, bars=128, boundary_candidates=True)
    entry = manifest["entries"][0]
    assert entry["boundary_candidates"] == 2
    assert len({tuple(row) for row in entry["vectors"]}) >= 3


def test_standalone_workers_survive_target_tools_package(tmp_path):
    # An older checkout can own a different regular `tools` package. Spawn must
    # still resolve the suite's worker and use the requested backend's modules.
    backend = tmp_path / "backend"
    (backend / "tools").mkdir(parents=True)
    (backend / "tools" / "__init__.py").write_text("")
    script = tmp_path / "spawn_check.py"
    script.write_text("""import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from optimizer_benchmark.worker import strict_json
if __name__ == '__main__':
    sys.path.insert(0, sys.argv[1])
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context('spawn')) as pool:
        assert pool.submit(strict_json, float('inf')).result() == 'Infinity'
""")
    suite_tools = Path(__file__).resolve().parents[2] / "src" / "tools"
    subprocess.run([sys.executable, str(script), str(backend)], check=True, timeout=20,
                   env=dict(os.environ, PYTHONPATH=str(suite_tools)))


def test_full_catalogue_freezes_without_device_import(tmp_path, real_runtime):
    manifest = freeze(tmp_path / "all-cases", profile="standard", candidates=2, bars=128)
    assert len(manifest["entries"]) == len(catalogue())
    assert {entry["case"]["name"] for entry in manifest["entries"]} == {case.name for case in catalogue()}


def test_lazy_native_runner_is_observed_after_materialization():
    class Runner:
        def run(self, params):
            return {}
    class LazyReplay:
        runners = {}
        fused_runner = None
        def _construct_runner(self, name, cls, kwargs):
            return cls(**kwargs)
    observations = []
    replay = LazyReplay()
    instrument_replay(replay, observations)
    runner = replay._construct_runner("fused", Runner, {})
    runner.run(np.zeros((3, 2)))
    assert len(observations) == 1
    assert observations[0]["candidates"] == 3


@pytest.mark.parametrize("policy", [dict(absolute=-1, relative=0), dict(absolute=float("inf"), relative=0),
                                   dict(absolute=True, relative=0), dict(absolute=0, relative=0, matching_infinity="false")])
def test_invalid_tolerances_are_rejected(policy):
    with pytest.raises(ValueError):
        validate_policies(dict(metric=policy))


@pytest.mark.parametrize("engine", ["gpu", "search-gpu"])
@pytest.mark.parametrize("pattern", ["trailing_grid_v7/*", "ema_anchor/*/position_enforcer/*",
                                    "trailing_martingale/*/btc_collateral/*"])
def test_worker_rejects_unsupported_gpu_work_before_dispatch(tmp_path, real_runtime, monkeypatch, engine, pattern):
    directory = tmp_path / "unsupported"
    manifest = freeze(directory, profile="standard", patterns=[pattern], candidates=2, bars=128)
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported GPU workload must not enter either GPU backend")
    monkeypatch.setattr(worker, "search_run", forbidden)
    monkeypatch.setattr(worker, "gpu_runs", forbidden)
    args = SimpleNamespace(harness_sha256=None, workload=directory,
                           entry=manifest["entries"][0]["id"], engine=engine)
    assert worker.run(args)["status"] == "unsupported"


@pytest.mark.parametrize("content", [None, '{"status":', "[]", "{}", b"\xff"])
@pytest.mark.parametrize("reason", [None, "timeout", "process_tree_memory_limit"])
def test_incomplete_worker_reports_keep_termination_evidence(tmp_path, content, reason):
    path = tmp_path / "result.json"
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif content is not None:
        path.write_text(content)
    termination = dict(termination_reason=reason, exit_code=-15 if reason else 1)
    outcome = read_worker_report(path, termination)
    assert outcome["status"] == ("bounded_stop" if reason else "error")
    assert outcome["process"] == termination
    assert outcome["report_error"]
    if reason:
        assert outcome["reason"] == reason


def test_report_read_failure_and_unsuccessful_exit_are_preserved(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text('{"status": "completed"}')
    termination = dict(termination_reason=None, exit_code=1)
    assert read_worker_report(path, termination)["status"] == "error"
    def unreadable(*args, **kwargs):
        raise PermissionError("unreadable report")
    monkeypatch.setattr(Path, "read_text", unreadable)
    outcome = read_worker_report(path, termination)
    assert outcome["status"] == "error" and "PermissionError" in outcome["report_error"]


def test_passive_baseline_and_trailing_variants_are_distinct(tmp_path, real_runtime):
    manifest = freeze(tmp_path / "orders", profile="standard", bars=128, candidates=2,
                      patterns=["trailing_martingale/both/forager/base/oscillating",
                                "trailing_martingale/both/forager/entry_trailing/*",
                                "trailing_martingale/both/forager/close_trailing/*"])
    entries = {entry["case"]["features"][0] if entry["case"]["features"] else "base": entry
               for entry in manifest["entries"]}
    assert set(entries) == {"base", "entry_trailing", "close_trailing"}
    for name, entry in entries.items():
        for config in entry["configs"]:
            for side in ("long", "short"):
                strategy = config["bot"][side]["strategy"]["trailing_martingale"]
                for leg in ("entry", "close"):
                    assert (strategy[leg]["retracement_base_pct"] > 0) == (name == f"{leg}_trailing")
    assert not any("passive" in case.features for case in catalogue())


@pytest.mark.parametrize("variation", [False, True])
def test_cpu_worker_hashes_repetitions_without_latency_noise(frozen, monkeypatch, variation):
    directory, manifest = frozen
    entry = manifest["entries"][0]
    calls = 0
    def candidate(_task):
        nonlocal calls
        calls += 1
        return dict(metrics={"adg_strategy_eq": 1.0 + (variation and calls > len(entry["configs"]))},
                    liquidated=False, latency_seconds=calls / 1000)
    monkeypatch.setattr(worker, "cpu_candidate", candidate)
    import importlib.metadata
    version = importlib.metadata.version
    def cpu_only_version(name):
        if name == "torch":
            raise importlib.metadata.PackageNotFoundError(name)
        return version(name)
    monkeypatch.setattr(importlib.metadata, "version", cpu_only_version)
    args = SimpleNamespace(harness_sha256=None, workload=directory, entry=entry["id"],
                           engine="cpu", workers=1, warmup=0, repeats=2)
    outcome = worker.run(args)
    assert outcome["dependency_versions"]["torch"] is None
    hashes = [sample["results_sha256"] for sample in outcome["samples"]]
    assert (len(set(hashes)) > 1) == variation
    comparison = numerical_comparison(outcome, outcome, ["adg_strategy_eq"],
                                     {"adg_strategy_eq": dict(absolute=0, relative=0)})
    assert comparison["status"] == ("repeat_variation_unassessed" if variation else "passed")


def test_cohort_digest_covers_terminal_status():
    rows = [dict(metrics={"adg_strategy_eq": 1.0}, liquidated=False)]
    assert cohort_digest(rows) != cohort_digest([dict(metrics=rows[0]["metrics"], liquidated=True)])


def test_phase_profiles_are_snapshotted_on_the_replay_owner():
    class Replay:
        profile_enabled = False
        runners = {}
        last_profile = {}
        def evaluate(self, rows):
            self.last_profile.clear()
            self.last_profile.update(wall_seconds=float(len(rows)))
            return rows
    replay, profiles, dispatches = Replay(), [], []
    instrument_replay(replay, dispatches, profiles=profiles)
    assert replay.profile_enabled
    replay.evaluate([1])
    replay.evaluate([1, 2])
    assert profiles == [dict(wall_seconds=1.0), dict(wall_seconds=2.0)]


def test_summary_aggregates_phases_without_losing_timing_scope():
    result = dict(status="completed", samples=[dict(seconds=5.0, completed=2, results=[],
        phase_profiles=[dict(timings_seconds={"pipeline_preparation": 1.0, "kernel_execution": 2.0})],
        outside_replay_seconds=2.0), dict(seconds=4.0, completed=2, results=[],
        phase_profiles=[dict(timings_seconds={"pipeline_preparation": 0.0, "kernel_execution": 3.0})],
        outside_replay_seconds=1.0)])
    measured = summary(result)
    assert measured["profiled_evaluations"] == 2
    assert measured["phase_seconds_total"] == {"pipeline_preparation": 1.0, "kernel_execution": 5.0}
    assert measured["outside_replay_seconds_total"] == 3.0
    assert measured["candidates_per_second"] == pytest.approx(4 / 9)


def test_phase_profile_rejects_search_modes(frozen):
    from tools.optimizer_benchmark.__main__ import run
    directory, _ = frozen
    with pytest.raises(ValueError, match="supports frozen replay"):
        run(SimpleNamespace(workload=directory, modes=["search-gpu:8"], phase_profile=True))


def test_gpu_instrumentation_hooks_are_restored_after_failed_setup(monkeypatch):
    pytest.importorskip("torch")
    from optimization.gpu.autotune import BatchController
    from optimization.gpu.service import MpsMulticoinProxy
    original_observe, original_init = BatchController.observe, MpsMulticoinProxy.__init__
    def failing(*_args):
        BatchController.observe = object()
        MpsMulticoinProxy.__init__ = object()
        raise RuntimeError("setup failed")
    monkeypatch.setattr(worker, "_gpu_runs", failing)
    with pytest.raises(RuntimeError, match="setup failed"):
        worker.gpu_runs(None, None, "8", 1, 1, 0, 100)
    assert BatchController.observe is original_observe
    assert MpsMulticoinProxy.__init__ is original_init


def test_frozen_gpu_profiling_is_explicit():
    replay = SimpleNamespace(profile_enabled=True, runners={})
    instrument_replay(replay, [])
    assert replay.profile_enabled is False


@pytest.mark.parametrize("raises", [False, True])
def test_search_host_profile_records_backend_work_and_restores_hook(raises):
    before = sys.getprofile()
    def backend_work():
        return sum(x * x for x in range(100))
    try:
        with worker.profile_search_host(True) as profile:
            backend_work()
            if raises:
                raise RuntimeError("backend failed")
    except RuntimeError:
        assert raises
    assert sys.getprofile() is before
    assert profile["wall_seconds"] >= 0
    assert profile["main_thread_cpu_seconds"] >= 0
    assert any(row["function"] == "backend_work" for row in profile["functions"])


def test_disabled_search_host_profile_does_not_create_profiler(monkeypatch):
    import cProfile
    def forbidden(*_args, **_kwargs):
        pytest.fail("unprofiled acceptance must not create a profiler")
    monkeypatch.setattr(cProfile, "Profile", forbidden)
    with worker.profile_search_host(False) as profile:
        pass
    assert profile == {}


def test_host_profile_rejects_frozen_replay(frozen):
    from tools.optimizer_benchmark.__main__ import run
    directory, _ = frozen
    with pytest.raises(ValueError, match="production search modes only"):
        run(SimpleNamespace(workload=directory, modes=["gpu:8"], phase_profile=False, host_profile=True))


def test_host_profile_leaves_an_existing_profiler_intact():
    def existing(_frame, _event, _arg):
        pass
    before = sys.getprofile()
    sys.setprofile(existing)
    try:
        with pytest.raises(RuntimeError, match="without an active profiler"):
            with worker.profile_search_host(True):
                pass
        assert sys.getprofile() is existing
    finally:
        sys.setprofile(before)


def test_search_transports_continuous_bounds_without_null_step(frozen, monkeypatch, caplog):
    from optimization import backends
    from optimization.bounds import Bound
    directory, manifest = frozen
    entry = manifest["entries"][0]
    assert any(spec["step"] is None for spec in entry["bounds"])
    with np.load(directory / entry["data_file"]) as data:
        arrays = tuple(data[key] for key in ("hlcvs", "btc", "timestamps"))
    class BackendReached(Exception):
        pass
    def inspect_bounds(**kwargs):
        transported = kwargs["config"]["optimize"]["bounds"]
        for spec in entry["bounds"]:
            bound = Bound.from_config(spec["key"], transported[spec["key"]])
            assert bound.step == spec["step"]
        raise BackendReached
    monkeypatch.setattr(backends, "get_backend_runner", lambda _name: inspect_bounds)
    args = SimpleNamespace(engine="search-cpu", workers=1, search_iters=64, width="8", dispatch_budget=100000)
    with pytest.raises(BackendReached):
        worker.search_run(entry, arrays, args)
    assert not any("step is null" in record.message for record in caplog.records)


@pytest.mark.parametrize("profiling", [None, "host_profile", "phase_profile_enabled"])
def test_profiled_timings_cannot_establish_an_acceptance_speedup(profiling):
    from tools.optimizer_benchmark.report import compare
    names = ["adg_strategy_eq", "drawdown_worst_strategy_eq"]
    result = dict(status="completed", samples=[dict(seconds=1.0, completed=1,
        results=[dict(metrics=dict(zip(names, [0.01, 0.1])), liquidated=False)])])
    other = deepcopy(result)
    if profiling:
        other[profiling] = {"wall_seconds": 1.0} if profiling == "host_profile" else True
    manifest = dict(entries=[dict(id="case", case=dict(name="case"), seed=7, metrics=names)])
    report = dict(backends={"before": {}, "after": {}},
        policies={key: dict(absolute=1e-6, relative=1e-4) for key in names},
        runs=[dict(entry="case", label=label, mode="cpu:1", result=row)
              for label, row in (("before", result), ("after", other))])
    row = compare(report, manifest)["rows"][0]
    assert row["speedup_second_over_first"] == 1.0
    assert row["equivalent_work_speedup"] == (None if profiling else 1.0)


def test_older_replay_pipeline_timing_remains_explicitly_combined():
    class Replay:
        runners = {}
        def evaluate(self, rows):
            self.last_profile = dict(timings_seconds=dict(kernel_execution=12.0), wall_seconds=12.0)
            return rows
    replay, profiles = Replay(), []
    instrument_replay(replay, [], profiles=profiles)
    replay.evaluate([])
    assert profiles[0]["pipeline_preparation_measured"] is False
    assert profiles[0]["timings_seconds"] == {"kernel_execution_and_pipeline_preparation": 12.0}
