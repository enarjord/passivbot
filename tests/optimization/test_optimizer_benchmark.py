"""Benchmark evidence must remain reproducible and refuse misleading comparisons."""

from copy import deepcopy
from itertools import combinations, product
import json
import multiprocessing
import random
import os
from pathlib import Path
import subprocess
import sys
import socket
import signal
import tempfile
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


@pytest.fixture
def blocked_worker_network(monkeypatch):
    # Register the original methods for restoration before process-scoped setup.
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    worker.block_network()
    worker.block_network()


@pytest.mark.parametrize("family", [socket.AF_INET, socket.AF_INET6])
def test_worker_rejects_network_families_before_connecting(blocked_worker_network, monkeypatch, family):
    def forbidden(*_args, **_kwargs):
        pytest.fail("network rejection must precede the original socket connect")
    monkeypatch.setattr(worker, "_SOCKET_CONNECT", forbidden)
    with pytest.raises(RuntimeError, match="forbids network connections"):
        socket.socket.connect(SimpleNamespace(family=family), ("localhost", 1))
    with pytest.raises(RuntimeError, match="forbids network connections"):
        socket.create_connection(("localhost", 1))


def test_worker_preserves_local_unix_ipc_after_repeated_setup(blocked_worker_network):
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("Unix-domain sockets unavailable")
    # Keep the socket path below macOS's short Unix-domain address limit.
    with tempfile.TemporaryDirectory(prefix="pb-bench-ipc-") as directory:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.settimeout(2)
            listener.bind(str(Path(directory) / "ipc"))
            listener.listen(1)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(2)
                client.connect(str(Path(directory) / "ipc"))
                connection, _address = listener.accept()
                with connection:
                    connection.settimeout(2)
                    client.sendall(b"ok")
                    assert connection.recv(2) == b"ok"


def _optimizer_pool_network_policy():
    """Probe child policy without attempting any real socket connection."""
    from optimization import problem
    statuses = []
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            socket.socket.connect(SimpleNamespace(family=family), ("localhost", 1))
        except RuntimeError as error:
            statuses.append(str(error) == "optimizer benchmark forbids network connections")
        except TypeError:
            # An unpatched original descriptor rejects this non-socket safely.
            statuses.append(False)
        else:
            statuses.append(False)
    return dict(
        network_families_blocked=statuses,
        create_connection_blocked=socket.create_connection is worker.deny_network,
        ignores_sigint=signal.getsignal(signal.SIGINT) == signal.SIG_IGN,
        evaluator_marker=problem._PYMOO_WORKER_EVALUATOR.marker,
    )


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_optimizer_pool_initializer_blocks_network_in_fresh_children(start_method):
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} multiprocessing unavailable")
    from optimization.problem import initialize_pymoo_worker

    context = multiprocessing.get_context(start_method)
    with context.Pool(processes=1, initializer=initialize_pymoo_worker,
                      initargs=(SimpleNamespace(marker="benchmark"), [], 2, False, 7,
                                worker.initialize_optimizer_worker)) as pool:
        policy = pool.apply_async(_optimizer_pool_network_policy).get(timeout=30)
    assert policy == dict(network_families_blocked=[True, True], create_connection_blocked=True,
                          ignores_sigint=True, evaluator_marker="benchmark")


def test_optimizer_worker_policy_precedes_existing_signal_setup(monkeypatch):
    import optimize
    calls = []
    monkeypatch.setattr(worker, "block_network", lambda: calls.append("network"))
    monkeypatch.setattr(optimize, "ignore_sigint_in_worker", lambda: calls.append("signal"))
    worker.initialize_optimizer_worker()
    assert calls == ["network", "signal"]


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


@pytest.mark.parametrize("case", [Case("ema_anchor", features=("position_enforcer",)),
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


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("scope", ["coin", "pside", "portfolio"])
def test_frozen_hsl_population_reconstructs_valid_search_bounds(tmp_path, real_runtime, strategy, scope):
    from config.hsl import validate_hsl_paths
    from config.optimize_bounds import set_flat_optimize_bound
    from optimization.shape import build_optimization_shape

    manifest = freeze(tmp_path / "scoped", profile="standard",
                      patterns=[f"{strategy}/both/forager/hsl_{scope}/*"],
                      candidates=3, bars=128)
    entry = manifest["entries"][0]
    hsl_paths = [tuple(spec["path"]) for spec in entry["bounds"] if "hsl" in spec["path"]]
    assert hsl_paths
    if scope == "portfolio":
        assert all(path[:3] == ("bot", "global", "hsl") for path in hsl_paths)
    else:
        assert {path[:3] for path in hsl_paths} == {("bot", side, "hsl") for side in ("long", "short")}
    for candidate in entry["configs"]:
        search = deepcopy(candidate)
        search["optimize"]["bounds"] = {}
        for spec in entry["bounds"]:
            endpoints = [spec["low"], spec["high"]]
            if spec["step"] is not None:
                endpoints.append(spec["step"])
            set_flat_optimize_bound(search["optimize"]["bounds"], strategy, spec["key"], endpoints)
        validate_hsl_paths(search)
        shape = build_optimization_shape(search)
        assert shape.key_paths == tuple((spec["key"], tuple(spec["path"])) for spec in entry["bounds"])
        if scope == "portfolio":
            assert all("hsl" not in search["optimize"]["bounds"].get(side, {}) for side in ("long", "short"))


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("scope", ["coin", "pside"])
def test_scope_projection_preserves_existing_side_hsl_populations(tmp_path, real_runtime, monkeypatch, strategy, scope):
    from config import hsl

    options = dict(profile="standard", patterns=[f"{strategy}/both/forager/hsl_{scope}/*"],
                   candidates=3, bars=128)
    projected = freeze(tmp_path / "projected", **options)
    # Retain the previous unprojected default-bound generation as the comparison.
    # Only the bounds template is replaced; fixture policy and sampling stay intact.
    original = hsl.generated_template
    def side_template(template, mode="coin"):
        config = original(template, mode)
        config["optimize"]["bounds"] = deepcopy(template["optimize"]["bounds"])
        return config
    monkeypatch.setattr(hsl, "generated_template", side_template)
    unprojected = freeze(tmp_path / "unprojected", **options)
    assert projected == unprojected


def test_explicit_portfolio_bounds_are_preserved_and_inactive_side_bounds_rejected(tmp_path, real_runtime):
    from config.schema import get_template_config

    options = dict(profile="standard", patterns=["ema_anchor/both/forager/hsl_portfolio/*"],
                   candidates=3, bars=128)
    explicit = {"global": {"hsl": {"drawdown_ratio_threshold": [0.11, 0.19, 0.01]}}}
    original = deepcopy(explicit)
    manifest = freeze(tmp_path / "explicit", bounds_config=explicit, **options)
    assert explicit == original
    specs = manifest["entries"][0]["bounds"]
    assert specs == [dict(key="hsl_red_threshold", path=["bot", "global", "hsl", "drawdown_ratio_threshold"],
                          low=0.11, high=0.19, step=0.01)]
    invalid = get_template_config()["optimize"]["bounds"]
    original = deepcopy(invalid)
    with pytest.raises(ValueError, match="inactive in portfolio mode"):
        freeze(tmp_path / "invalid", bounds_config=invalid, **options)
    assert invalid == original


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
    from config.authored import validate_authored_config
    from config.schema import get_template_config
    manifest = freeze(tmp_path / "all-cases", profile="standard", candidates=2, bars=128)
    assert len(manifest["entries"]) == len(catalogue())
    assert {entry["case"]["name"] for entry in manifest["entries"]} == {case.name for case in catalogue()}
    for entry in manifest["entries"]:
        authored = deepcopy(entry["configs"][0])
        authored["backtest"].pop("coins")  # Known ordered preparation metadata.
        validate_authored_config(authored, get_template_config())


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
@pytest.mark.parametrize("pattern", ["ema_anchor/*/position_enforcer/*",
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
    from optimization.gpu.service import SharedAccountReplay
    original_observe, original_init = BatchController.observe, SharedAccountReplay.__init__
    def failing(*_args):
        BatchController.observe = object()
        SharedAccountReplay.__init__ = object()
        raise RuntimeError("setup failed")
    monkeypatch.setattr(worker, "_gpu_runs", failing)
    with pytest.raises(RuntimeError, match="setup failed"):
        worker.gpu_runs(None, None, "8", 1, 1, 0, 100)
    assert BatchController.observe is original_observe
    assert SharedAccountReplay.__init__ is original_init


def test_tuning_and_cohort_clocks_exclude_the_same_setup(monkeypatch, tmp_path):
    from collections import deque
    from concurrent.futures import Future
    torch = pytest.importorskip("torch")
    from optimization.gpu import autotune, runtime, native, parameters, datasets
    from shared_arrays import SharedArrayManager

    clock = [0.0]
    monkeypatch.setattr(worker.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(runtime, "gpu_device", lambda *_: "mps")
    monkeypatch.setattr(runtime, "synchronize", lambda: None)
    for name in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
        monkeypatch.setattr(torch.mps, name, lambda: 0)
    monkeypatch.setattr(worker, "instrument_replay", lambda *_args, **_kwargs: None)
    def project(*_args):
        clock[0] += 100.0
        return {}
    monkeypatch.setattr(parameters, "prepare_candidate_parameters", project)
    monkeypatch.setattr(autotune.BatchController, "observe", lambda *_args, **_kwargs: True)
    controller = SimpleNamespace(width=1, samples=deque(maxlen=24), baseline=None,
                                 seconds=0.0, cooldown=0)
    monkeypatch.setattr(SharedArrayManager, "create_from", lambda self, a: (a, None))
    monkeypatch.setattr(SharedArrayManager, "cleanup", lambda *_: None)
    monkeypatch.setattr(datasets, "PreparedGpuDataset", lambda **kwargs: kwargs)
    class Service:
        def __init__(self, **kwargs):
            clock[0] += 200.0
            self._batch_tuner = SimpleNamespace(controllers={"fixture": controller})
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def register_dataset(self, *_): pass
        def submit(self, _request):
            clock[0] += 2.0
            autotune.BatchController.observe(controller, 1, 2.0)
            result = Future()
            result.set_result(SimpleNamespace(metrics={"score": 1.0}, liquidated=False))
            return result
    monkeypatch.setattr(native, "NativeBacktestService", Service)
    arrays = (np.zeros((2, 1, 4)), np.ones(2), np.arange(2))
    config = {"optimize": {"gpu": {}}, "backtest": {"coins": {"fake": ["BTC"]}}}
    entry = dict(id="fixture", configs=[config], metrics=["score"], markets={}, exchange="fake")
    report = worker.gpu_runs(entry, arrays, "auto", 1, 0, 0, 100)
    assert report["parameter_seconds"] == 100.0
    assert report["setup_seconds"] == 200.0
    assert report["first_tuning_window_seconds"] == 2.0
    assert report["first_use"]["elapsed_seconds"] == 2.0
    assert report["samples"][0]["elapsed_seconds"] == 2.0


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
        from optimization.config_adapter import _flatten_bounds_for_config
        config = kwargs["config"]
        assert "short" in config["optimize"]["bounds"]
        assert all(spec["key"] not in config["optimize"]["bounds"] for spec in entry["bounds"])
        transported = _flatten_bounds_for_config(config, config["optimize"]["bounds"])
        for spec in entry["bounds"]:
            bound = Bound.from_config(spec["key"], transported[spec["key"]])
            assert bound.step == spec["step"]
        raise BackendReached
    monkeypatch.setattr(backends, "get_backend_runner", lambda _name: inspect_bounds)
    args = SimpleNamespace(engine="search-cpu", workers=1, search_iters=64, width="8", dispatch_budget=100000)
    with pytest.raises(BackendReached):
        worker.search_run(entry, arrays, args)
    assert not any("step is null" in record.message for record in caplog.records)


@pytest.mark.parametrize("engine", ["search-cpu", "search-gpu"])
def test_search_matches_backend_signature_and_prepares_recording_contract(frozen, monkeypatch, engine):
    import inspect
    import optimize
    from optimization import backends
    from optimization.backends import gpu_backend, pymoo_backend

    directory, manifest = frozen
    entry = manifest["entries"][0]
    with np.load(directory / entry["data_file"]) as data:
        arrays = tuple(data[key] for key in ("hlcvs", "btc", "timestamps"))
    original_evaluator = optimize.Evaluator
    prepared = []
    def evaluator(*args, **kwargs):
        config = args[3]
        assert config[optimize.CONTRACT_CACHE_KEY] == optimize.build_evaluation_contract(config)
        prepared.append(config[optimize.CONTRACT_CACHE_KEY])
        return original_evaluator(*args, **kwargs)
    monkeypatch.setattr(optimize, "Evaluator", evaluator)
    runner = gpu_backend.run_backend if engine == "search-gpu" else pymoo_backend.run_backend
    class BackendReached(Exception):
        pass
    def inspect_call(**kwargs):
        inspect.signature(runner).bind(**kwargs)
        assert kwargs["config"][optimize.CONTRACT_CACHE_KEY] is prepared[0]
        assert kwargs["ignore_sigint_in_worker"] is worker.initialize_optimizer_worker
        assert not {"constraint_fitness_cls", "record_individual_result", "run_evolution"} & kwargs.keys()
        raise BackendReached
    monkeypatch.setattr(backends, "get_backend_runner", lambda _name: inspect_call)
    args = SimpleNamespace(engine=engine, workers=1, search_iters=8, width="8", dispatch_budget=100000)
    with pytest.raises(BackendReached):
        worker.search_run(entry, arrays, args)


@pytest.mark.parametrize("start_method", [None, "forkserver"])
def test_prepared_cpu_search_completes_and_verifies_saved_objectives(frozen, start_method):
    if start_method is not None and start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} multiprocessing unavailable")
    directory, manifest = frozen
    entry = manifest["entries"][0]
    # Network rejection belongs to the worker process, including the Rust
    # verification pass; it must not alter the pytest process's socket methods.
    output = directory / "cpu-search.json"
    root = Path(__file__).resolve().parents[2]
    invocation = [sys.executable, "-m", "tools.optimizer_benchmark.worker"]
    if start_method is not None:
        invocation = [sys.executable, "-c", (
            "import multiprocessing, runpy; "
            f"multiprocessing.set_start_method({start_method!r}); "
            "runpy.run_module('tools.optimizer_benchmark.worker', run_name='__main__', alter_sys=True)"
        )]
    subprocess.run([*invocation,
                    "--backend", str(root), "--workload", str(directory),
                    "--entry", entry["id"], "--engine", "search-cpu", "--workers", "1",
                    "--search-iters", "8", "--output", str(output)],
                   cwd=root, env=dict(os.environ, PYTHONPATH=str(root / "src")),
                   check=True, timeout=60)
    report = json.loads(output.read_text())
    assert report["status"] == "completed", report
    assert report["samples"][0]["completed"] == 8
    assert report["persisted_bytes"] > 0
    assert report["rust_reference_verification"]["status"] == "passed"


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


@pytest.mark.parametrize("selected_seed, expected_status", [(43, 0), (101, 2)])
def test_seed_filtered_run_preserves_frozen_population(tmp_path, real_runtime, monkeypatch,
                                                     selected_seed, expected_status):
    from tools.optimizer_benchmark import __main__ as cli

    directory = tmp_path / "frozen"
    manifest = freeze(directory, patterns=["trailing_martingale/short/fixed/*"],
                      candidates=2, bars=128, seeds=[7, 43])
    original = (directory / "manifest.json").read_bytes()
    dispatched = []
    original_run = cli.subprocess.run
    def preflight(command, **kwargs):
        if len(command) > 1 and command[1] == "-c":
            return SimpleNamespace(returncode=0, stdout="{}")
        return original_run(command, **kwargs)
    monkeypatch.setattr(cli.subprocess, "run", preflight)
    original_check_output = cli.subprocess.check_output
    def check_output(command, **kwargs):
        return "test-head" if command[0] == "git" else original_check_output(command, **kwargs)
    monkeypatch.setattr(cli.subprocess, "check_output", check_output)

    def bounded(command, **kwargs):
        dispatched.append(command[command.index("--entry") + 1])
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(dict(status="unsupported", reason="test worker")))
        return dict(exit_code=0, termination_reason=None, wall_seconds=0.0)

    monkeypatch.setattr(cli, "bounded_process", bounded)
    destination = tmp_path / "report"
    command = ["run", "--workload", str(directory), "--seed", str(selected_seed),
               "--backend", "before", str(tmp_path), sys.executable,
               "--backend", "after", str(tmp_path), sys.executable,
               "--modes", "cpu:1", "--output", str(destination)]
    if expected_status == 0:
        assert cli.main(command) == 0
    else:
        with pytest.raises(SystemExit) as error:
            cli.main(command)
        assert error.value.code == 2
    assert (directory / "manifest.json").read_bytes() == original
    expected_ids = [entry["id"] for entry in manifest["entries"]
                    if entry["seed"] == selected_seed]
    assert dispatched == [identity for identity in expected_ids for _ in range(2)]
    if expected_status == 0:
        report = json.loads((destination / "report.json").read_text())
        assert report["workload_sha256"] == manifest["sha256"]
        assert report["selection"]["entry_ids"] == expected_ids
        assert report["selection"]["seeds"] == [selected_seed]
    else:
        assert not destination.exists()


def test_retired_v7_is_absent_from_catalogue_and_rejected_by_public_freeze(tmp_path):
    assert not any(case.strategy == "trailing_grid_v7" for case in catalogue())
    with pytest.raises(ValueError, match="empty"):
        freeze(tmp_path / "retired", profile="standard", patterns=["trailing_grid_v7/*"], candidates=2, bars=128)


def test_retired_optional_entry_gate_is_not_a_workload_feature():
    from tools.optimizer_benchmark.workloads import apply_features
    assert not any("entry_gate_off" in case.features for case in catalogue())
    with pytest.raises(ValueError, match="retired benchmark feature"):
        apply_features({}, Case("trailing_martingale", features=("entry_gate_off",)), [], {}, 128)


@pytest.mark.parametrize("scope", ["coin", "pside", "portfolio"])
def test_hsl_features_preserve_independent_memories_and_fixed_default_genes(scope, real_runtime):
    from tools.gpu_parity import build_parser, fixture_inputs
    from tools.optimizer_benchmark.workloads import apply_features, get_path, sampled_population
    config, _candles, markets, _btc, _timestamps = fixture_inputs(build_parser().parse_args(
        ["--fixture", "trailing_martingale", "--hsl", "position_side" if scope == "pside" else scope,
         "--coins", "4", "--bars", "128"]))
    case = Case("trailing_martingale", features=(f"hsl_{scope}", "hsl_slow_memory", "hsl_market",
                                                "hsl_scaled", "hsl_zero_cooldown", "loss_gate"))
    apply_features(config, case, config["backtest"]["coins"]["binance"], markets, 128)
    policies = [config["bot"]["global"]["hsl"]] if scope == "portfolio" else [config["bot"][s]["hsl"] for s in ("long", "short")]
    for policy in policies:
        assert policy["memory_ema_span_minutes"] == 10.5
        assert policy["panic_close_order_type"] == "market"
        assert policy["scale_budget_with_excess_allowance"] is True
        assert policy["cooldown_minutes_after_red"] == 0.0
    assert config["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] == 1440.0
    assert config["bot"]["global"]["loss_gate"]["memory_ema_span_minutes"] == 1440.0
    assert config["bot"]["global"]["loss_gate"]["max_drawdown_ratio"] == .001
    population, _vectors, bounds = sampled_population(config, case, 3, 7)
    for spec in bounds:
        if spec["path"][-1] in {"memory_ema_span_minutes", "max_positions"}:
            assert spec["low"] == spec["high"] == get_path(config, spec["path"])
        if spec["path"][:3] == ("bot", "global", "hsl") and spec["path"][-1] == "drawdown_ratio_threshold":
            assert (spec["low"] < spec["high"]) == (scope == "portfolio")
    assert all(candidate["bot"]["long"]["risk"]["max_positions"] == 2 for candidate in population)
