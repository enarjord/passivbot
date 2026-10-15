"""Benchmark evidence must remain reproducible and refuse misleading comparisons."""

from copy import deepcopy
from itertools import combinations, product
import json
import random
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.optimizer_benchmark.catalogue import (Case, PAIRWISE_FLAGS, catalogue,
    declared_gpu_support, pairwise_rows, select_cases)
from tools.optimizer_benchmark.workloads import array_digest, digest, freeze, load_manifest
from tools.optimizer_benchmark.report import numerical_comparison, summary, tuning_comparison, validate_policies
from tools.optimizer_benchmark.__main__ import modes
from tools.optimizer_benchmark.worker import strict_json, instrument_replay


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
