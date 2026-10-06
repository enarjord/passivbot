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


@pytest.mark.parametrize("flags", [("--widths", "0"), ("--widths", "broken"),
                                   ("--warm-runs", "0"), ("--candidates", "129"),
                                   ("--adg-floor", "nan"), ("--drawdown-ceiling", "inf")])
def test_invalid_workload_is_rejected_before_device_access(flags):
    with pytest.raises(SystemExit) as error:
        options(*flags)
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


@pytest.mark.parametrize("fault", [None, "identity", "metric", "liquidation"])
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
                                    {"metric": request.parameters["metric"] + (1 if fault == "metric" else 0)},
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
    if fault:
        with pytest.raises(RuntimeError, match="identity|differs"):
            benchmark._native_runs(torch, None, parameters, reference, 2, 1)
    else:
        report = benchmark._native_runs(torch, None, parameters, reference, 2, 1)
        assert report["matches_direct_gpu_exactly"]
        assert sum(row["count"] for row in report["successful_batches"]) == 6
        assert report["tuning"] is None
        assert len(report["runs"]) == 2
        assert report["torch_memory_bytes"]["peak_allocated"] == 6
    assert closed == [True]


def test_failed_execution_is_structured_and_report_file_matches_stdout(monkeypatch, tmp_path, capsys):
    def fail(args):
        raise RuntimeError("CUDA unavailable")
    monkeypatch.setattr(benchmark, "run_benchmark", fail)
    report_path = tmp_path / "report.json"
    assert benchmark.main(["--report", str(report_path), "--compact"]) == 2
    text = capsys.readouterr().out
    assert report_path.read_text() == text
    assert json.loads(text)["status"] == "execution_failed"


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
def test_real_cuda_cohort_reports_serial_cpu_and_equivalent_service_metrics(strategy):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    args = options("--strategies", strategy, "--bars", "512", "--coins", "2",
                   "--candidates", "3", "--warm-runs", "1", "--widths", "1", "2", "auto")
    report = benchmark.run_benchmark(args)
    json.dumps(report, allow_nan=False)
    case = report["cases"][0]
    assert report["runtime"]["rust_source_fingerprint"]
    assert len(case["cpu_gpu_comparisons"]) == 3
    assert all(set(row["metrics"]) == set(benchmark.gpu_parity.DEFAULT_METRICS)
               for row in case["cpu_gpu_comparisons"])
    assert case["ranking"]["assessed"]
    assert case["diagnostic_limits"]["comparisons"] is None
    for native in case["native"]:
        assert native["matches_direct_gpu_exactly"]
        assert sum(row["count"] for row in native["successful_batches"]) == 6
        assert len(native["timing"]["warm_seconds"]) == 1
