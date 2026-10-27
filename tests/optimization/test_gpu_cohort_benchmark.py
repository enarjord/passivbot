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
    ("--hsl-drawdown-threshold-fraction", "nan"), ("--hsl-drawdown-fraction-ema-span-minutes", "0"),
    ("--hsl-cooldown-minutes", "-1"), ("--hsl-memory-ema-span-minutes", "0"),
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
        benchmark.main(["--hsl", "portfolio", "--hsl-cooldown-minutes", str(value)])
    assert error.value.code == 2


def test_cohort_accepts_cooldown_just_inside_rust_timestamp_range():
    value = math.nextafter((2**63 - 1) / 60_000, 0)
    assert options("--hsl", "portfolio", "--hsl-cooldown-minutes", str(value)).hsl_cooldown_minutes == value


@pytest.mark.parametrize("flags", [
    ("--hsl-drawdown-fraction-ema-span-minutes", "1e300"), ("--hsl-drawdown-threshold-fraction", "1e-300"),
    ("--hsl-drawdown-fraction-ema-span-minutes", "1e10"), ("--hsl-memory-ema-span-minutes", "1e10"),
    ("--price-shock", "0", "64", "1e40"),
    ("--price-shock", "0", "64", "1e-100"),
    ("--price-shock", "0", "64", "1e20", "--price-shock", "0", "64", "1e20"),
    ("--price-shock", "0", "64", "1e-20", "--price-shock", "0", "64", "1e-20",
     "--price-shock", "0", "64", "1e-20"),
])
def test_gpu_encoding_rejected_before_benchmark(monkeypatch, flags):
    def forbidden(*a):
        pytest.fail("invalid float32 recipe must not initialize CUDA or run backtests")
    monkeypatch.setattr(benchmark, "run_benchmark", forbidden)
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--bars", "128", "--coins", "1", "--hsl", "portfolio", *flags])
    assert error.value.code == 2


def test_gpu_policy_encoding_accepts_decaying_span_and_subnormal_threshold():
    args = options("--hsl-drawdown-fraction-ema-span-minutes", "65536.5",
                   "--hsl-drawdown-threshold-fraction", str(2**-149))
    assert args.hsl_drawdown_fraction_ema_span_minutes == 65536.5
    assert args.hsl_drawdown_threshold_fraction == 2**-149


def test_cohort_uses_same_resolved_stress_fixture_as_parity_tool():
    import numpy as np
    flags = ["--hsl", "portfolio", "--bars", "128", "--coins", "2", "--sides", "both",
             "--hsl-drawdown-threshold-fraction", ".002", "--hsl-drawdown-fraction-ema-span-minutes", "2.5",
             "--unstuck-memory-ema-span-minutes", "15.5", "--loss-gate-memory-ema-span-minutes", "31.25",
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
    for config in configs:
        assert config["bot"]["portfolio"]["hsl"]["scope"] == "portfolio"
        assert config["bot"]["portfolio"]["hsl"]["drawdown_threshold_fraction"] == .002
        assert config["bot"]["portfolio"]["hsl"]["drawdown_fraction_ema_span_minutes"] == 2.5
        assert config["bot"]["portfolio"]["unstuck"]["memory_ema_span_minutes"] == 15.5
        assert config["bot"]["portfolio"]["loss_gate"]["memory_ema_span_minutes"] == 31.25
    assert args.price_shocks == [[0, 64, .7]]
    assert args.unstuck_memory_ema_span_minutes == 15.5
    assert args.loss_gate_memory_ema_span_minutes == 31.25


@pytest.mark.parametrize("flags", [("--widths", "0"), ("--widths", "broken"),
                                   ("--warm-runs", "0"), ("--candidates", "129"),
                                   ("--cagr-floor", "nan"), ("--drawdown-ceiling", "inf"),
                                   ("--limit", "fills_gap_max_days", "greater_than", "nan"),
                                   ("--limit", "fills_gap_max_days", "auto", "1"),
                                   ("--limit", "unknown", "greater_than", "1")])
def test_invalid_workload_is_rejected_before_device_access(flags):
    with pytest.raises(SystemExit) as error:
        options(*flags)
    assert error.value.code == 2


def test_metric_requests_include_limit_work_and_preserve_undefined_policies(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"cagr_fraction_strategy_equity": {"absolute": 0.001, "relative": 0.01}}))
    args = options("--metrics", "cagr_fraction_strategy_equity", "peak_recovery_max_days_strategy_equity",
                   "--limit", "position_held_max_days", "greater_than", "2",
                   "--tolerances", str(path))
    assert args.metrics == [*benchmark.gpu_parity.DEFAULT_METRICS,
                            "peak_recovery_max_days_strategy_equity", "ulcer_index_fraction_strategy_equity", "position_held_max_days"]
    assert args.policies["cagr_fraction_strategy_equity"] == benchmark.gpu_parity.MetricTolerance(0.001, 0.01)
    assert "peak_recovery_max_days_strategy_equity" not in args.policies
    assert "position_held_max_days" not in args.policies


@pytest.mark.parametrize("policy", [[], {"cagr_fraction_strategy_equity": {"absolute": -1, "relative": 0}},
                                    {"cagr_fraction_strategy_equity": {"absolute": True, "relative": 0}},
                                    {"cagr_fraction_strategy_equity": {"absolute": 0, "relative": float("nan")}}])
def test_malformed_metric_policies_are_rejected_before_device_access(tmp_path, policy):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(SystemExit) as error:
        options("--tolerances", str(path))
    assert error.value.code == 2


def test_order_disagreements_include_ties_and_front_changes():
    cpu = [dict(cagr_fraction_strategy_equity=0.1, ulcer_index_fraction_strategy_equity=0.2),
           dict(cagr_fraction_strategy_equity=0.1, ulcer_index_fraction_strategy_equity=0.1)]
    gpu = [dict(cagr_fraction_strategy_equity=0.100001, ulcer_index_fraction_strategy_equity=0.2), cpu[1]]
    report = benchmark._ranking(cpu, gpu)
    assert report["cpu_front"] == [1]
    assert report["gpu_front"] == [0, 1]
    assert report["pair_order_disagreements"] == {"cagr_fraction_strategy_equity": 1, "ulcer_index_fraction_strategy_equity": 0}
    assert report["cpu_cagr_regret_of_gpu_max"] == 0


@pytest.mark.parametrize("value", [None, math.inf, math.nan])
def test_missing_or_nonfinite_objectives_are_unassessed(value):
    row = dict(cagr_fraction_strategy_equity=value, ulcer_index_fraction_strategy_equity=0)
    report = benchmark._ranking([row], [row])
    assert report == {"assessed": False, "reason": "nonfinite_or_missing_objective"}
    json.dumps(report, allow_nan=False)


def test_diagnostic_limits_use_canonical_feasibility_and_inclusive_boundaries():
    from optimization.gpu.parity import compare_limits

    checks = benchmark._checks(options("--cagr-floor", "0", "--drawdown-ceiling", "0.01"))
    cpu = dict(cagr_fraction_strategy_equity=0, drawdown_max_fraction_strategy_equity=0.01)
    gpu = dict(cagr_fraction_strategy_equity=0, drawdown_max_fraction_strategy_equity=0.010000001)
    report = compare_limits(cpu, gpu, checks)
    assert report["cpu_feasible"] is True
    assert report["gpu_feasible"] is False
    assert [item["status"] for item in report["checks"]] == ["match", "flip"]
    assert benchmark._checks(options()) == []


def test_additional_diagnostic_limits_use_canonical_metric_feasibility():
    from optimization.gpu.parity import compare_limits

    checks = benchmark._checks(options("--limit", "fills_gap_max_days", "greater_than", "1"))
    report = compare_limits({"fills_gap_max_days": 1.0}, {"fills_gap_max_days": 1.1}, checks)
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
        "--hsl", "portfolio", "--hsl-drawdown-threshold-fraction", ".002",
        "--price-shock", "0", "64", ".7"]) == 2
    text = capsys.readouterr().out
    assert report_path.read_text() == text
    assert json.loads(text)["status"] == "execution_failed"
    assert json.loads(text)["recipe"]["hsl_drawdown_threshold_fraction"] == .002
    assert json.loads(text)["recipe"]["price_shocks"] == [[0, 64, .7]]
    assert "report" not in json.loads(text)["recipe"]


def test_tuning_report_retains_consumed_windows_and_incomplete_remainder():
    from optimization.gpu.execution_tuning import EVIDENCE_SAMPLES as WINDOW, ExecutionBatchTuner

    policy = ExecutionBatchTuner(initial=2)
    policy.constrain("cohort", 2)
    assert policy.width("cohort", 2) == 2
    batches = []
    evidence = benchmark._observe_batches(policy, batches)
    # Each actual shape's cold first use is successful but ineligible evidence.
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
        median_candidates_per_second=1.0, candidates_per_second=1.0,
        rate_aggregation="per_count", resulting_width=1)]
    # The smaller trial loses throughput and rolls back; its window is retained too.
    # Count one was already seen before the initial width-two evidence window.
    for _ in range(WINDOW):
        policy.observe("cohort", 1, 2.0, backlog=4, closing=False)
    assert controller.width == 2
    assert evidence["samples"] == WINDOW * 2
    assert evidence["completed_windows"][-1]["resulting_width"] == 2
    policy.observe("cohort", 2, 2.0, backlog=4, closing=False)
    assert evidence["samples"] == WINDOW * 2 + 1
    assert len(controller.samples) == 1 and controller.seconds == 2.0
    assert len(batches) == WINDOW * 2 + 3


def test_tuning_report_counts_warm_partial_shapes_and_their_consumed_window():
    from optimization.gpu.execution_tuning import EVIDENCE_SAMPLES as WINDOW, ExecutionBatchTuner

    policy = ExecutionBatchTuner(initial=64)
    policy.constrain("cohort", 128)
    assert policy.width("cohort", 128) == 64
    batches = []
    evidence = benchmark._observe_batches(policy, batches)
    policy.observe("cohort", 63, 1000, backlog=0, closing=False)
    for _ in range(WINDOW):
        policy.observe("cohort", 63, 2, backlog=0, closing=False)
    assert evidence["samples"] == WINDOW
    assert evidence["seconds"] == WINDOW * 2
    assert evidence["completed_windows"] == [dict(
        width=64, samples=WINDOW, seconds=WINDOW * 2,
        median_candidates_per_second=31.5, candidates_per_second=31.5,
        rate_aggregation="per_count", resulting_width=32)]
    assert len(batches) == WINDOW + 1


def test_tuning_report_excludes_invalid_partial_observations():
    from optimization.gpu.execution_tuning import ExecutionBatchTuner

    policy = ExecutionBatchTuner(initial=4)
    policy.constrain("cohort", 16)
    policy.width("cohort", 16)
    evidence = benchmark._observe_batches(policy, [])
    policy.observe("cohort", 3, 1000, backlog=0, closing=False)
    for count, seconds in ((0, 1), (5, 1), (3, 0), (3, float("nan")), (3, float("inf"))):
        policy.observe("cohort", count, seconds, backlog=0, closing=False)
    assert evidence == dict(samples=0, seconds=0.0, completed_windows=[])
    policy.observe("cohort", 3, 2, backlog=0, closing=False)
    assert evidence["samples"] == 1 and evidence["seconds"] == 2


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


def test_cohort_reference_uses_the_native_factual_simulation_contract(monkeypatch):
    import sys

    class ReferencePrepared(Exception):
        pass

    def reference(**kwargs):
        # A service-equivalence check must compare the same simulation policy,
        # rather than reject factual replay against the retired HSL observer.
        assert kwargs.get("factual_hsl") is True
        raise ReferencePrepared

    monkeypatch.setitem(sys.modules, "optimization.gpu.service",
                        SimpleNamespace(SharedAccountReplay=reference))
    monkeypatch.setitem(sys.modules, "backtest", SimpleNamespace(
        build_backtest_payload=lambda *a, **kw: None,
        execute_backtest=lambda *a: (None, None, {"cagr_fraction_strategy_equity": 0.0}),
    ))
    monkeypatch.setattr(benchmark, "_cohort", lambda *a: (
        ({}, None, {}, None, None), [{}], [{}],
    ))
    args = SimpleNamespace(warm_runs=0, metrics=["cagr_fraction_strategy_equity"])
    with pytest.raises(ReferencePrepared):
        benchmark._measure(SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
                           args, "ema_anchor", 7)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("hsl", ["disabled", "portfolio"])
def test_real_cuda_cohort_reports_serial_cpu_and_equivalent_service_metrics(strategy, hsl, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"peak_recovery_max_days_strategy_equity": {"absolute": 0, "relative": 0}}))
    args = options("--strategies", strategy, "--bars", "512", "--coins", "2",
                   "--candidates", "3", "--warm-runs", "1", "--widths", "1", "2", "auto",
                   "--hsl", hsl, "--hsl-drawdown-threshold-fraction", ".002",
                   "--hsl-drawdown-fraction-ema-span-minutes", "2.5", "--hsl-cooldown-minutes", "1000",
                   "--price-shock", "0", "256", ".7", "--price-shock", "1", "320", "1.3",
                   "--metrics", "peak_recovery_max_days_strategy_equity", "position_held_max_days", "gross_exposure_max_ratio_strategy_equity",
                   "--limit", "fills_gap_max_days", "greater_than", "1",
                   "--tolerances", str(path),
                   "--objective", "cagr_fraction_strategy_equity", "max",
                   "--objective", "ulcer_index_fraction_strategy_equity", "min",
                   "--objective", "pain_index_fraction_strategy_equity", "min")
    report = benchmark.run_benchmark(args)
    json.dumps(report, allow_nan=False)
    case = report["cases"][0]
    assert report["runtime"]["rust_source_fingerprint"]
    assert report["measurement_scope"]["gpu_replay"] == "native_factual"
    assert len(case["cpu_gpu_comparisons"]) == 3
    assert all(set(row["metrics"]) == set(args.metrics) for row in case["cpu_gpu_comparisons"])
    assert report["recipe"]["metrics"] == args.metrics
    assert "tolerances" not in report["recipe"] and "policies" not in report["recipe"]
    assert report["tolerance_policy"]["position_held_max_days"] is None
    assert report["tolerance_policy"]["peak_recovery_max_days_strategy_equity"] == dict(
        absolute=0, relative=0, matching_infinity=False)
    assert all(row["metrics"]["gross_exposure_max_ratio_strategy_equity"]["cpu"] is not None for row in case["cpu_gpu_comparisons"])
    assert all(row["metrics"]["peak_recovery_max_days_strategy_equity"]["status"] in {"match", "mismatch"}
               for row in case["cpu_gpu_comparisons"])
    assert all(row["metrics"]["position_held_max_days"]["status"] == "unassessed"
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




def test_explicit_objectives_replace_default_and_request_canonical_metrics():
    args = options('--objective', 'cagr_fraction_strategy_equity', 'max',
                   '--objective', 'fills_gap_max_days', 'min')
    assert args.objectives == {'cagr_fraction_strategy_equity': 'max', 'fills_gap_max_days': 'min'}
    assert args.metrics == [*benchmark.gpu_parity.DEFAULT_METRICS, 'fills_gap_max_days']
    assert options('--objective', 'fills_gap_max_days', 'min', '--objective', 'fills_gap_max_days', 'min').objectives == {'fills_gap_max_days': 'min'}


@pytest.mark.parametrize('flags', [('--objective', 'fills_gap_max_days', 'auto'),
    ('--objective', 'cagr_fraction_strategy_equity', 'max', '--objective', 'cagr_fraction_strategy_equity', 'min')])
def test_bad_objective_direction_is_rejected(flags):
    with pytest.raises(SystemExit) as error:
        options(*flags)
    assert error.value.code == 2


def test_third_objective_exposes_front_change_hidden_by_default():
    cpu = [dict(cagr_fraction_strategy_equity=.1, ulcer_index_fraction_strategy_equity=.2, fills_gap_max_days=1),
           dict(cagr_fraction_strategy_equity=.1, ulcer_index_fraction_strategy_equity=.1, fills_gap_max_days=2)]
    gpu = [cpu[0], {**cpu[1], 'fills_gap_max_days': .5}]
    assert benchmark._ranking(cpu, gpu)['front_members_match']
    report = benchmark._ranking(cpu, gpu, {**benchmark.DEFAULT_OBJECTIVES, 'fills_gap_max_days': 'min'})
    assert report['cpu_front'] == [0, 1]
    assert report['gpu_front'] == [1]
    assert report['pair_order_disagreements']['fills_gap_max_days'] == 1
    assert report['cpu_regret_at_gpu_best']['fills_gap_max_days'] == 1
    assert report['gpu_best_candidates']['fills_gap_max_days'] == 1


@pytest.mark.parametrize('direction,selected,regret', [('min', 1, 1), ('max', 0, 1)])
def test_non_adg_objective_directions_and_axis_regret(direction, selected, regret):
    cpu = [{'fills_gap_max_days': 1}, {'fills_gap_max_days': 2}]
    gpu = [{'fills_gap_max_days': 2}, {'fills_gap_max_days': 1}]
    report = benchmark._ranking(cpu, gpu, {'fills_gap_max_days': direction})
    assert report['gpu_front'] == [selected]
    assert report['gpu_best_candidates'] == {'fills_gap_max_days': selected}
    assert report['cpu_regret_at_gpu_best'] == {'fills_gap_max_days': regret}
    assert 'cpu_cagr_regret_of_gpu_max' not in report


@pytest.mark.parametrize('rows', [([], []), ([{'fills_gap_max_days': 1}], [])])
def test_empty_or_mismatched_cohorts_are_unassessed(rows):
    assert benchmark._ranking(*rows, {'fills_gap_max_days': 'min'}) == {'assessed':False, 'reason':'empty_or_mismatched_candidates'}


@pytest.mark.parametrize('value', [None, float('nan'), float('inf')])
def test_missing_extra_objective_is_unassessed(value):
    rows = [{'fills_gap_max_days': value}]
    assert benchmark._ranking(rows, rows, {'fills_gap_max_days': 'min'})['assessed'] is False


def test_fourth_objective_direction_changes_dominance():
    axes = {**benchmark.DEFAULT_OBJECTIVES, "fills_gap_max_days": "min",
            "cagr_fraction_strategy_equity_ew": "max"}
    cpu = [dict(cagr_fraction_strategy_equity=.1, ulcer_index_fraction_strategy_equity=.1,
                fills_gap_max_days=gap, cagr_fraction_strategy_equity_ew=momentum)
           for gap, momentum in [(1, 0), (2, 1), (1, 1)]]
    gpu = [*cpu[:2], {**cpu[2], "cagr_fraction_strategy_equity_ew": -1}]
    report = benchmark._ranking(cpu, gpu, axes)
    assert report["cpu_front"] == [2]
    assert report["gpu_front"] == [0, 1]
    assert report["pair_order_disagreements"]["cagr_fraction_strategy_equity_ew"] == 2
