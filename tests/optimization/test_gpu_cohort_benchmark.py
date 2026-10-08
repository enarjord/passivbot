from concurrent.futures import Future
import json
import math
from types import SimpleNamespace

import pytest

from tools import gpu_cohort_benchmark as benchmark


def options(*values):
    parser = benchmark.build_parser()
    args = parser.parse_args(values)
    benchmark.validate_args(parser, args)
    return args


@pytest.mark.parametrize("flags", [
    ("--hsl-red-threshold", "nan"), ("--hsl-ema-span-minutes", "0"),
    ("--hsl-cooldown-minutes", "-1"), ("--hsl-lookback-days", "91"),
    ("--price-shock", "4", "64", ".7"), ("--price-shock", "0", "10080", ".7"),
    ("--price-shock", "0", "64", "inf"),
])
def test_cohort_stress_options_are_checked_before_device_access(flags):
    with pytest.raises(SystemExit) as error:
        options(*flags)
    assert error.value.code == 2


@pytest.mark.parametrize("value", [1e300, (2**63 - 1) / 60_000])
def test_cohort_rejects_unsupported_cooldown_before_benchmark(monkeypatch, value):
    def forbidden(*a):
        pytest.fail("invalid timestamp policy must not initialize a benchmark/device")
    monkeypatch.setattr(benchmark, "run_benchmark", forbidden)
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--hsl", "unified", "--hsl-cooldown-minutes", str(value)])
    assert error.value.code == 2


def test_cohort_accepts_cooldown_just_inside_rust_timestamp_range():
    value = math.nextafter((2**63 - 1) / 60_000, 0)
    assert options("--hsl", "unified", "--hsl-cooldown-minutes", str(value)).hsl_cooldown_minutes == value


def test_cohort_uses_same_resolved_stress_fixture_as_parity_tool():
    import numpy as np
    flags = ["--hsl", "unified", "--bars", "128", "--coins", "2", "--sides", "both",
             "--hsl-red-threshold", ".002", "--hsl-ema-span-minutes", "2.5",
             "--hsl-cooldown-minutes", "10000", "--price-shock", "0", "64", ".7"]
    args = options("--candidates", "2", *flags)
    inputs, configs, parameters = benchmark._cohort(args, "ema_anchor", 7)
    reference = benchmark.gpu_parity.fixture_inputs(
        benchmark.gpu_parity.build_parser().parse_args(["--fixture", "ema_anchor", *flags]))
    np.testing.assert_array_equal(inputs[1], reference[1])
    assert benchmark.gpu_parity._identity(
        inputs[0], (inputs[1], inputs[3], inputs[4]), inputs[2], "binance"
    ) == benchmark.gpu_parity._identity(
        reference[0], (reference[1], reference[3], reference[4]), reference[2], "binance"
    )
    assert len(configs) == len(parameters) == 2
    assert all(c["bot"]["hsl"]["red_threshold"] == .002 for c in configs)
    assert args.price_shocks == [[0, 64, .7]]
    assert args.hsl_lookback_days == 1


@pytest.mark.parametrize("flags", [("--widths", "0"), ("--widths", "broken"),
                                   ("--warm-runs", "0"), ("--candidates", "129"),
                                   ("--adg-floor", "nan"), ("--drawdown-ceiling", "inf"),
                                   ("--limit", "fills_gap_p95_hours", "greater_than", "nan"),
                                   ("--limit", "fills_gap_p95_hours", "auto", "1"),
                                   ("--limit", "unknown", "greater_than", "1")])
def test_invalid_workload_is_rejected_before_device_access(flags):
    with pytest.raises(SystemExit) as error:
        options(*flags)
    assert error.value.code == 2


def test_metric_requests_include_limit_work_and_preserve_undefined_policies(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"adg_strategy_pnl_rebased": {"absolute": 0.001, "relative": 0.01}}))
    args = options("--metrics", "adg_strategy_pnl_rebased", "strategy_eq_recovery_days_p95",
                   "--limit", "volume_pct_per_day_avg_w", "greater_than", "2",
                   "--tolerances", str(path))
    assert args.metrics == [*benchmark.gpu_parity.DEFAULT_METRICS,
                            "strategy_eq_recovery_days_p95", "volume_pct_per_day_avg_w"]
    assert args.policies["adg_strategy_eq"] == benchmark.gpu_parity.MetricTolerance(0.001, 0.01)
    assert "strategy_eq_recovery_days_p95" not in args.policies
    assert "volume_pct_per_day_avg_w" not in args.policies


@pytest.mark.parametrize("policy", [[], {"adg_strategy_eq": {"absolute": -1, "relative": 0}},
                                    {"adg_strategy_eq": {"absolute": True, "relative": 0}},
                                    {"adg_strategy_eq": {"absolute": 0, "relative": float("nan")}}])
def test_malformed_metric_policies_are_rejected_before_device_access(tmp_path, policy):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(SystemExit) as error:
        options("--tolerances", str(path))
    assert error.value.code == 2


def test_order_disagreements_include_ties_and_front_changes():
    cpu = [dict(adg_strategy_eq=0.1, drawdown_worst_strategy_eq=0.2),
           dict(adg_strategy_eq=0.1, drawdown_worst_strategy_eq=0.1)]
    gpu = [dict(adg_strategy_eq=0.100001, drawdown_worst_strategy_eq=0.2), cpu[1]]
    report = benchmark._ranking(cpu, gpu)
    assert report["cpu_front"] == [1]
    assert report["gpu_front"] == [0, 1]
    assert report["pair_order_disagreements"] == {"adg_strategy_eq": 1, "drawdown_worst_strategy_eq": 0}
    assert report["cpu_adg_regret_of_gpu_max"] == 0


@pytest.mark.parametrize("value", [None, math.inf, math.nan])
def test_missing_or_nonfinite_objectives_are_unassessed(value):
    row = dict(adg_strategy_eq=value, drawdown_worst_strategy_eq=0)
    report = benchmark._ranking([row], [row])
    assert report == {"assessed": False, "reason": "nonfinite_or_missing_objective"}
    json.dumps(report, allow_nan=False)


def test_diagnostic_limits_use_canonical_feasibility_and_inclusive_boundaries():
    from optimization.gpu.parity import compare_limits

    checks = benchmark._checks(options("--adg-floor", "0", "--drawdown-ceiling", "0.01"))
    cpu = dict(adg_strategy_eq=0, drawdown_worst_strategy_eq=0.01)
    gpu = dict(adg_strategy_eq=0, drawdown_worst_strategy_eq=0.010000001)
    report = compare_limits(cpu, gpu, checks)
    assert report["cpu_feasible"] is True
    assert report["gpu_feasible"] is False
    assert [item["status"] for item in report["checks"]] == ["match", "flip"]
    assert benchmark._checks(options()) == []


def test_additional_diagnostic_limits_use_canonical_metric_feasibility():
    from optimization.gpu.parity import compare_limits

    checks = benchmark._checks(options("--limit", "fills_gap_p95_hours", "greater_than", "1"))
    report = compare_limits({"fills_gap_p95_hours": 1.0}, {"fills_gap_p95_hours": 1.1}, checks)
    assert report["cpu_feasible"] is True
    assert report["gpu_feasible"] is False
    assert report["checks"][0]["status"] == "flip"


@pytest.mark.parametrize("fault", [None, "identity", "metric", "liquidation", "rounding"])
def test_native_observation_binds_each_completion_and_keeps_batch_evidence(monkeypatch, fault):
    import optimization.gpu.native as native
    from optimization.gpu.executor import BacktestResult, ReplayResult

    closed = []
    class Service:
        def __init__(self, **kwargs):
            self._batch_policy = SimpleNamespace(controllers={}, observe=lambda *a, **kw: None)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closed.append(True)
        def register_dataset(self, *args):
            pass
        def submit(self, request):
            self._batch_policy.observe(request.dataset_id, 1, 0.001, backlog=0, closing=False)
            result = BacktestResult(request.request_id, "wrong" if fault == "identity" else request.dataset_id,
                                    {"metric": (math.nextafter(request.parameters["metric"], math.inf) if fault == "rounding" else
                                                request.parameters["metric"] + (1 if fault == "metric" else 0))},
                                    fault == "liquidation")
            future = Future()
            future.set_result(result)
            return future
    monkeypatch.setattr(native, "CudaBacktestService", Service)
    torch = SimpleNamespace(cuda=SimpleNamespace(
        memory_allocated=lambda: 2, memory_reserved=lambda: 4,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 6, max_memory_reserved=lambda: 8,
    ))
    parameters = [{"metric": value} for value in range(3)]
    reference = [ReplayResult(row, False) for row in parameters]
    if fault and fault != "rounding":
        with pytest.raises(RuntimeError, match="identity|differs") as error:
            benchmark._native_runs(torch, None, parameters, reference, 2, 1)
        if fault == "metric":
            assert "metric_differences=[('metric'" in str(error.value)
    else:
        report = benchmark._native_runs(torch, None, parameters, reference, 2, 1)
        assert report["matches_direct_gpu"]
        assert report["matches_direct_gpu_exactly"] is (fault != "rounding")
        assert bool(report["reduction_rounding"]) is (fault == "rounding")
        if fault == "rounding":
            assert report["reduction_rounding"]["metric"]["count"] == 6
            assert report["reduction_rounding"]["metric"]["max_float64_ulps"] == 1
        assert sum(row["count"] for row in report["successful_batches"]) == 6
        assert report["tuning"] is None
        assert len(report["runs"]) == 2
        assert report["torch_memory_bytes"]["peak_allocated"] == 6
    assert closed == [True]


def test_gpu_reference_comparison_accepts_only_reported_float64_rounding():
    expected = {"value": 1.0}
    actual = 1.0
    for _ in range(benchmark.MAX_FLOAT64_REDUCTION_ULPS):
        actual = math.nextafter(actual, math.inf)
    assert benchmark._metric_rounding(expected, {"value": actual}) == {
        "value": {"absolute_error": actual - 1, "float64_ulps": 8}}
    assert benchmark._metric_rounding(expected, {"value": math.nextafter(actual, math.inf)}) is None
    assert benchmark._metric_rounding(expected, {"value": 1 + 2**-23}) is None
    assert benchmark._metric_rounding(expected, {}) is None
    assert benchmark._metric_rounding(expected, {"value": math.nan}) is None
    assert benchmark._metric_rounding({"value": math.inf}, {"value": math.inf}) == {}
    assert benchmark._metric_rounding({"value": math.inf}, {"value": -math.inf}) is None


def test_failed_execution_is_structured_and_report_file_matches_stdout(monkeypatch, tmp_path, capsys):
    def fail(args):
        raise RuntimeError("CUDA unavailable")
    monkeypatch.setattr(benchmark, "run_benchmark", fail)
    report_path = tmp_path / "report.json"
    assert benchmark.main(["--report", str(report_path), "--compact",
        "--hsl", "unified", "--hsl-red-threshold", ".002",
        "--price-shock", "0", "64", ".7"]) == 2
    text = capsys.readouterr().out
    assert report_path.read_text() == text
    assert json.loads(text)["status"] == "execution_failed"
    assert json.loads(text)["recipe"]["hsl_red_threshold"] == .002
    assert json.loads(text)["recipe"]["price_shocks"] == [[0, 64, .7]]
    assert "report" not in json.loads(text)["recipe"]


def test_tuning_report_retains_consumed_windows_and_incomplete_remainder():
    from optimization.gpu.autotune import WINDOW
    from optimization.gpu.execution_tuning import ExecutionBatchTuner

    policy = ExecutionBatchTuner(initial=2)
    policy.constrain("cohort", 2)
    assert policy.width("cohort", 2) == 2
    batches = []
    evidence = benchmark._observe_batches(policy, batches)
    # Cold first use and underfilled work are successful but ineligible evidence.
    policy.observe("cohort", 1, 2.0, backlog=0, closing=False)
    policy.observe("cohort", 2, 2.0, backlog=4, closing=False)
    for _ in range(WINDOW):
        policy.observe("cohort", 2, 2.0, backlog=4, closing=False)
    controller = policy.controllers["cohort"]
    assert controller.width == 1
    assert len(controller.samples) == 0 and controller.seconds == 0
    assert evidence["samples"] == WINDOW
    assert evidence["seconds"] == WINDOW * 2.0
    assert evidence["completed_windows"] == [dict(
        width=2, samples=WINDOW, seconds=WINDOW * 2.0,
        median_candidates_per_second=1.0, resulting_width=1)]
    # The smaller trial loses throughput and rolls back; its window is retained too.
    for _ in range(WINDOW + 1):
        policy.observe("cohort", 1, 2.0, backlog=4, closing=False)
    assert controller.width == 2
    assert evidence["samples"] == WINDOW * 2
    assert evidence["completed_windows"][-1]["resulting_width"] == 2
    policy.observe("cohort", 2, 2.0, backlog=4, closing=False)
    assert evidence["samples"] == WINDOW * 2 + 1
    assert len(controller.samples) == 1 and controller.seconds == 2.0
    assert len(batches) == WINDOW * 2 + 4


def test_cli_dispatches_benchmark_help_without_full_dependency_gate(monkeypatch):
    import sys
    from passivbot_cli import main as cli

    observed = []
    def invoke(module):
        observed.append((module, list(sys.argv)))
        return True, 0
    def forbidden():
        pytest.fail("help must not inspect full runtime dependencies")
    monkeypatch.setattr(cli, "_invoke_module_main", invoke)
    monkeypatch.setattr(cli, "_missing_full_install_markers", forbidden)
    assert cli.main(["tool", "gpu-cohort-benchmark", "--help"]) == 0
    assert observed == [("tools.gpu_cohort_benchmark", ["passivbot tool gpu-cohort-benchmark", "--help"])]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_real_cuda_cohort_reports_serial_cpu_and_equivalent_service_metrics(strategy, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"strategy_eq_recovery_days_p95": {"absolute": 0, "relative": 0}}))
    args = options("--strategies", strategy, "--bars", "512", "--coins", "2",
                   "--candidates", "3", "--warm-runs", "1", "--widths", "1", "2", "auto",
                   "--metrics", "strategy_eq_recovery_days_p95", "volume_pct_per_day_avg_w", "adg_btc",
                   "--limit", "fills_gap_p95_hours", "greater_than", "1",
                   "--tolerances", str(path),
                   "--objective", "adg_strategy_eq", "max",
                   "--objective", "drawdown_worst_strategy_eq", "min",
                   "--objective", "fills_gap_p95_hours", "min")
    report = benchmark.run_benchmark(args)
    json.dumps(report, allow_nan=False)
    case = report["cases"][0]
    assert report["runtime"]["rust_source_fingerprint"]
    assert len(case["cpu_gpu_comparisons"]) == 3
    assert all(set(row["metrics"]) == set(args.metrics) for row in case["cpu_gpu_comparisons"])
    assert report["recipe"]["metrics"] == args.metrics
    assert "tolerances" not in report["recipe"] and "policies" not in report["recipe"]
    assert report["tolerance_policy"]["volume_pct_per_day_avg_w"] is None
    assert report["tolerance_policy"]["strategy_eq_recovery_days_p95"] == dict(
        absolute=0, relative=0, matching_infinity=False)
    assert all(row["metrics"]["adg_btc"]["cpu"] is not None for row in case["cpu_gpu_comparisons"])
    assert all(row["metrics"]["strategy_eq_recovery_days_p95"]["status"] in {"match", "mismatch"}
               for row in case["cpu_gpu_comparisons"])
    assert all(row["metrics"]["volume_pct_per_day_avg_w"]["status"] == "unassessed"
               for row in case["cpu_gpu_comparisons"])
    assert case["ranking"]["assessed"]
    assert case["ranking"]["objectives"] == args.objectives
    assert report["recipe"]["objectives"] == args.objectives
    assert set(case["ranking"]["cpu_regret_at_gpu_best"]) == set(args.objectives)
    assert len(case["diagnostic_limits"]["comparisons"]) == 3
    assert all(row["assessed"] for row in case["diagnostic_limits"]["comparisons"])
    for native in case["native"]:
        assert native["matches_direct_gpu_exactly"]
        assert sum(row["count"] for row in native["successful_batches"]) == 6
        assert len(native["timing"]["warm_seconds"]) == 1


def test_cuda_weighted_reduction_batch_shapes_preserve_raw_replay():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    from optimization.gpu.service import MpsMulticoinProxy

    args = options("--strategies", "trailing_martingale", "--seeds", "43",
                   "--metrics", "drawdown_worst_mean_1pct_strategy_eq",
                   "drawdown_worst_mean_1pct_ema_strategy_eq", "fills_gap_p95_hours",
                   "adg_strategy_eq_w", "strategy_eq_recovery_days_mean",
                   "strategy_eq_recovery_days_p95", "strategy_eq_recovery_days_mean_worst_1pct",
                   "volume_pct_per_day_avg_w")
    inputs, _configs, parameters = benchmark._cohort(args, "trailing_martingale", 43)
    config, candles, markets, btc, timestamps = inputs
    proxy = MpsMulticoinProxy(config=config, hlcvs=candles, mss=markets, btc=btc,
                             timestamps=timestamps, exchange="binance", batch_size=16,
                             needed_metrics=args.metrics,
                             max_dispatch_candidate_bars=benchmark.DISPATCH_BUDGET)
    keys = ("day_end_eq", "day_min_eq", "day_max_dd", "first_eq_ts", "last_eq_ts", "fill_count", "max_dd")
    captured = []
    original = proxy._compute_objectives

    def record(out, run, data, needed=None):
        captured.append({name: out[name].clone().cpu() for name in keys})
        return original(out, run, data, needed=needed)

    try:
        proxy._compute_objectives = record
        full = proxy.evaluate_results(parameters)
        split = proxy.evaluate_results(parameters[:1]) + proxy.evaluate_results(parameters[1:])
    finally:
        proxy._compute_objectives = original
    assert len(captured) == 3
    for name in keys:
        torch.testing.assert_close(captured[0][name],
                                   torch.cat([captured[1][name], captured[2][name]], dim=0),
                                   rtol=0, atol=0, equal_nan=True)
    for expected, observed in zip(full, split, strict=True):
        assert expected.liquidated == observed.liquidated
        assert benchmark._metric_rounding(expected.metrics, observed.metrics) is not None


def test_explicit_objectives_replace_default_and_request_canonical_metrics():
    args = options('--objective', 'adg_strategy_pnl_rebased', 'max',
                   '--objective', 'fills_gap_p95_hours', 'min')
    assert args.objectives == {'adg_strategy_eq': 'max', 'fills_gap_p95_hours': 'min'}
    assert args.metrics == [*benchmark.gpu_parity.DEFAULT_METRICS, 'fills_gap_p95_hours']
    assert options('--objective', 'fills_gap_p95_hours', 'min', '--objective', 'fills_gap_p95_hours', 'min').objectives == {'fills_gap_p95_hours': 'min'}


@pytest.mark.parametrize('flags', [('--objective', 'fills_gap_p95_hours', 'auto'),
    ('--objective', 'adg_strategy_eq', 'max', '--objective', 'adg_strategy_pnl_rebased', 'min')])
def test_bad_objective_direction_is_rejected(flags):
    with pytest.raises(SystemExit) as error:
        options(*flags)
    assert error.value.code == 2


def test_third_objective_exposes_front_change_hidden_by_default():
    cpu = [dict(adg_strategy_eq=.1, drawdown_worst_strategy_eq=.2, fills_gap_p95_hours=1),
           dict(adg_strategy_eq=.1, drawdown_worst_strategy_eq=.1, fills_gap_p95_hours=2)]
    gpu = [cpu[0], {**cpu[1], 'fills_gap_p95_hours': .5}]
    assert benchmark._ranking(cpu, gpu)['front_members_match']
    report = benchmark._ranking(cpu, gpu, {**benchmark.DEFAULT_OBJECTIVES, 'fills_gap_p95_hours': 'min'})
    assert report['cpu_front'] == [0, 1]
    assert report['gpu_front'] == [1]
    assert report['pair_order_disagreements']['fills_gap_p95_hours'] == 1
    assert report['cpu_regret_at_gpu_best']['fills_gap_p95_hours'] == 1
    assert report['gpu_best_candidates']['fills_gap_p95_hours'] == 1


@pytest.mark.parametrize('direction,selected,regret', [('min', 1, 1), ('max', 0, 1)])
def test_non_adg_objective_directions_and_axis_regret(direction, selected, regret):
    cpu = [{'fills_gap_p95_hours': 1}, {'fills_gap_p95_hours': 2}]
    gpu = [{'fills_gap_p95_hours': 2}, {'fills_gap_p95_hours': 1}]
    report = benchmark._ranking(cpu, gpu, {'fills_gap_p95_hours': direction})
    assert report['gpu_front'] == [selected]
    assert report['gpu_best_candidates'] == {'fills_gap_p95_hours': selected}
    assert report['cpu_regret_at_gpu_best'] == {'fills_gap_p95_hours': regret}
    assert 'cpu_adg_regret_of_gpu_max' not in report


@pytest.mark.parametrize('rows', [([], []), ([{'fills_gap_p95_hours': 1}], [])])
def test_empty_or_mismatched_cohorts_are_unassessed(rows):
    assert benchmark._ranking(*rows, {'fills_gap_p95_hours': 'min'}) == {'assessed':False, 'reason':'empty_or_mismatched_candidates'}


@pytest.mark.parametrize('value', [None, float('nan'), float('inf')])
def test_missing_extra_objective_is_unassessed(value):
    rows = [{'fills_gap_p95_hours': value}]
    assert benchmark._ranking(rows, rows, {'fills_gap_p95_hours': 'min'})['assessed'] is False


def test_fourth_objective_direction_changes_dominance():
    axes = {**benchmark.DEFAULT_OBJECTIVES, "fills_gap_p95_hours": "min",
            "adg_strategy_eq_w": "max"}
    cpu = [dict(adg_strategy_eq=.1, drawdown_worst_strategy_eq=.1,
                fills_gap_p95_hours=gap, adg_strategy_eq_w=weighted)
           for gap, weighted in [(1, 0), (2, 1), (1, 1)]]
    gpu = [*cpu[:2], {**cpu[2], "adg_strategy_eq_w": -1}]
    report = benchmark._ranking(cpu, gpu, axes)
    assert report["cpu_front"] == [2]
    assert report["gpu_front"] == [0, 1]
    assert report["pair_order_disagreements"]["adg_strategy_eq_w"] == 2
