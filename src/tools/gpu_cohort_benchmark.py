"""Offline synthetic CPU/direct-GPU/native-service cohort measurements.

CPU timing includes serial payload preparation and simulation, not CPU optimizer
throughput. Native first-use timing follows direct GPU warmup; it is not a cold
compiler/cache measurement. This tool never runs inside an optimizer.
"""

import argparse
from concurrent.futures import as_completed
from contextlib import redirect_stdout
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

from optimization.gpu.parity import compare_limits, compare_metrics
from tools import gpu_parity

DISPATCH_BUDGET = 500_000_000


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", nargs="+", choices=("ema_anchor", "trailing_martingale"),
                        default=["ema_anchor", "trailing_martingale"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--sides", choices=("long", "short", "both"), default="both")
    parser.add_argument("--coins", type=int, default=4)
    parser.add_argument("--bars", type=int, default=10080)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--widths", nargs="+", default=["1", "4", "16", "auto"])
    parser.add_argument("--hsl", choices=("disabled", "coin", "pside", "unified"), default="disabled")
    parser.add_argument("--unstuck", action="store_true")
    parser.add_argument("--adg-floor", type=float)
    parser.add_argument("--drawdown-ceiling", type=float)
    parser.add_argument("--report")
    parser.add_argument("--compact", action="store_true")
    return parser


def validate_args(parser, args):
    for name, lower, upper in (("coins", 1, 64), ("bars", 61, 100000),
                               ("candidates", 2, 128), ("warm_runs", 1, 10)):
        if not lower <= getattr(args, name) <= upper:
            parser.error(f"--{name.replace('_', '-')} must be between {lower} and {upper}")
    widths = []
    for value in args.widths:
        if value == "auto":
            width = None
        else:
            try:
                width = int(value)
            except ValueError:
                parser.error("--widths requires positive integers or auto")
            if not 1 <= width <= 1024:
                parser.error("--widths must be between 1 and 1024, or auto")
        if width not in widths:
            widths.append(width)
    args.widths = widths
    args.strategies = list(dict.fromkeys(args.strategies))
    args.seeds = list(dict.fromkeys(args.seeds))
    if len(args.seeds) > 8:
        parser.error("at most eight fixture seeds are allowed")
    for value in (args.adg_floor, args.drawdown_ceiling):
        if value is not None and not math.isfinite(value):
            parser.error("diagnostic limit values must be finite")


def _timing_summary(samples, count):
    median = statistics.median(samples[1:])
    return dict(first_seconds=samples[0], warm_seconds=samples[1:],
                warm_median_seconds=median, warm_candidates_per_second=count / median)


def _front(rows):
    values = [(-row["adg_strategy_eq"], row["drawdown_worst_strategy_eq"]) for row in rows]
    return [i for i, row in enumerate(values) if not any(
        all(left <= right for left, right in zip(other, row)) and
        any(left < right for left, right in zip(other, row))
        for j, other in enumerate(values) if j != i)]


def _ranking(cpu, gpu):
    objectives = ("adg_strategy_eq", "drawdown_worst_strategy_eq")
    if any(row.get(name) is None or not math.isfinite(row[name])
           for rows in (cpu, gpu) for row in rows for name in objectives):
        return {"assessed": False, "reason": "nonfinite_or_missing_objective"}
    cpu_front, gpu_front = _front(cpu), _front(gpu)
    relation = lambda a, b: int(a > b) - int(a < b)
    flips = {name: sum(
        relation(cpu[i][name], cpu[j][name]) != relation(gpu[i][name], gpu[j][name])
        for i in range(len(cpu)) for j in range(i + 1, len(cpu))) for name in objectives}
    selected = max(range(len(gpu)), key=lambda i: gpu[i]["adg_strategy_eq"])
    return dict(assessed=True, objectives={objectives[0]: "max", objectives[1]: "min"},
                cpu_front=cpu_front, gpu_front=gpu_front, front_members_match=cpu_front == gpu_front,
                pair_order_disagreements=flips, pairs=len(cpu) * (len(cpu) - 1) // 2,
                gpu_max_adg_candidate=selected,
                cpu_adg_regret_of_gpu_max=max(row[objectives[0]] for row in cpu) - cpu[selected][objectives[0]])


def _checks(args):
    from limit_utils import expand_limit_checks

    entries = []
    for metric, mode, bound in (("adg_strategy_eq", "less_than", args.adg_floor),
                                ("drawdown_worst_strategy_eq", "greater_than", args.drawdown_ceiling)):
        if bound is not None:
            entries.append(dict(metric=metric, penalize_if=mode, value=bound))
    return expand_limit_checks(entries, {"adg_strategy_eq": -1.0, "drawdown_worst_strategy_eq": 1.0},
                               penalty_weight=1.0)


def _cohort(args, strategy, seed):
    from optimization.gpu.parameters import prepare_candidate_parameters

    options = ["--fixture", strategy, "--sides", args.sides, "--coins", str(args.coins),
               "--bars", str(args.bars), "--seed", str(seed), "--hsl", args.hsl]
    if args.unstuck:
        options.append("--unstuck")
    inputs = gpu_parity.fixture_inputs(gpu_parity.build_parser().parse_args(options))
    config, _candles, markets, _btc, _timestamps = inputs
    configs, parameters = [], []
    active = ("long", "short") if args.sides == "both" else (args.sides,)
    for index in range(args.candidates):
        candidate = deepcopy(config)
        for side in active:
            bot = candidate["bot"][side]["strategy"][strategy]
            if strategy == "trailing_martingale":
                bot = bot["entry"]
                bot["initial_qty_pct"] = 0.01 + index * 0.001
            else:
                bot["base_qty_pct"] = 0.01 + index * 0.001
            bot["ema_span_0"] = 5.0 + index * 1.25
        configs.append(candidate)
        parameters.append(prepare_candidate_parameters(candidate, markets, "binance"))
    return inputs, configs, parameters


def _native_runs(torch, dataset, parameters, reference, width, warm_runs):
    import numpy as np
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService

    samples, batches = [], []
    allocated_start = torch.cuda.memory_allocated()
    reserved_start = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    with CudaBacktestService(batch_size=width, max_pending=max(64, len(parameters)),
                             max_dispatch_candidate_bars=DISPATCH_BUDGET) as service:
        service.register_dataset("cohort", dataset)
        # Tool-only observation of the existing controller; retain its decisions
        # and the executor's successful-work timings without running extra work.
        original_observe = service._batch_policy.observe
        def observe(dataset_id, count, seconds, **kwargs):
            batches.append(dict(count=count, seconds=seconds))
            original_observe(dataset_id, count, seconds, **kwargs)
        service._batch_policy.observe = observe
        for iteration in range(warm_runs + 1):
            started = time.perf_counter()
            pending = {}
            for index, values in enumerate(parameters):
                submitted = time.perf_counter()
                request_id = f"{iteration}:{index}"
                future = service.submit(BacktestRequest(request_id, "cohort", values))
                pending[future] = index, submitted, request_id
            elapsed, latencies = [], []
            for future in as_completed(pending):
                index, submitted, request_id = pending[future]
                row = future.result()
                observed = time.perf_counter()
                if row.request_id != request_id or row.dataset_id != "cohort":
                    raise RuntimeError("native result identity differs from submitted request")
                if row.metrics != reference[index].metrics or row.liquidated != reference[index].liquidated:
                    raise RuntimeError("native result differs from direct GPU replay")
                elapsed.append(observed - started)
                latencies.append(observed - submitted)
            samples.append(dict(seconds=elapsed[-1], first_completion_seconds=elapsed[0],
                                caller_observed_latency_p50_seconds=float(np.percentile(latencies, 50)),
                                caller_observed_latency_p95_seconds=float(np.percentile(latencies, 95))))
        controller = service._batch_policy.controllers.get("cohort")
        tuning = (dict(final_width=controller.width, evidence_samples=len(controller.samples),
                       evidence_seconds=controller.seconds, seen_widths=sorted(controller.seen))
                  if controller is not None else None)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    return dict(requested_width="auto" if width is None else width, matches_direct_gpu_exactly=True,
                timing=_timing_summary([row["seconds"] for row in samples], len(parameters)),
                runs=samples, successful_batches=batches, tuning=tuning,
                torch_memory_bytes=dict(allocated_start=allocated_start, reserved_start=reserved_start,
                                        peak_allocated=peak_allocated, peak_reserved=peak_reserved))


def _measure(torch, args, strategy, seed):
    from backtest import build_backtest_payload, execute_backtest
    from config.metrics import resolve_metric_value
    from optimization.gpu.service import MpsMulticoinProxy

    inputs, configs, parameters = _cohort(args, strategy, seed)
    config, candles, markets, btc, timestamps = inputs
    cpu_samples, cpu = [], None
    for _ in range(args.warm_runs + 1):
        started = time.perf_counter()
        rows = []
        for candidate in configs:
            payload = build_backtest_payload(candles, markets, candidate, "binance", btc, timestamps,
                                             metrics_only=True, skip_btc_analysis=True)
            _, _, analysis = execute_backtest(payload, candidate)
            rows.append({name: resolve_metric_value(analysis, name) for name in gpu_parity.DEFAULT_METRICS})
        cpu_samples.append(time.perf_counter() - started)
        cpu = rows
    started = time.perf_counter()
    proxy = MpsMulticoinProxy(config=config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
                             exchange="binance", batch_size=len(parameters), needed_metrics=gpu_parity.DEFAULT_METRICS,
                             max_dispatch_candidate_bars=DISPATCH_BUDGET)
    prepare_seconds = time.perf_counter() - started
    direct_samples, reference = [], None
    for _ in range(args.warm_runs + 1):
        torch.cuda.synchronize()
        started = time.perf_counter()
        rows = proxy.evaluate_results(parameters)
        torch.cuda.synchronize()
        direct_samples.append(time.perf_counter() - started)
        if reference is not None and rows != reference:
            raise RuntimeError("repeated direct GPU results differ")
        reference = rows
    del proxy
    gc.collect()
    with gpu_parity._native_dataset(inputs, "binance", gpu_parity.DEFAULT_METRICS) as dataset:
        native = [_native_runs(torch, dataset, parameters, reference, width, args.warm_runs)
                  for width in args.widths]
    gpu = [dict(row.metrics) for row in reference]
    policies = {name: gpu_parity.DEFAULT_TOLERANCES[name] for name in gpu_parity.DEFAULT_METRICS}
    comparisons = [compare_metrics(left, right, policies)
                   for left, right in zip(cpu, gpu, strict=True)]
    checks = _checks(args)
    limits = [compare_limits(left, right, checks) for left, right in zip(cpu, gpu, strict=True)] if checks else None
    return dict(strategy=strategy, seed=seed,
                evaluation_id=gpu_parity._identity(config, (candles, btc, timestamps), markets, "binance"),
                candidate_parameters_sha256=hashlib.sha256(json.dumps(parameters, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                cpu_serial_prepare_and_execute=_timing_summary(cpu_samples, len(parameters)),
                direct_gpu_prepare_seconds=prepare_seconds,
                direct_gpu=_timing_summary(direct_samples, len(parameters)), native=native,
                cpu_gpu_comparisons=comparisons, ranking=_ranking(cpu, gpu),
                diagnostic_limits=dict(adg_floor=args.adg_floor, drawdown_ceiling=args.drawdown_ceiling,
                                       comparisons=limits))


def run_benchmark(args):
    import torch
    from rust_utils import check_and_maybe_compile, verify_loaded_runtime_extension

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        raise RuntimeError("cohort benchmark requires NVIDIA CUDA")
    if "passivbot_rust" not in sys.modules:
        check_and_maybe_compile(fail_on_stale=True)
    import passivbot_rust
    runtime = verify_loaded_runtime_extension()
    if runtime.get("skipped") or runtime["expected_source_fingerprint"] != runtime["runtime_compiled_source_stamp"]:
        raise RuntimeError("cohort benchmark requires a source-verified Rust extension")
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    import cupy
    return dict(schema_version=1, status="measured", measurement_scope=dict(
        cpu="serial_payload_preparation_and_simulation",
        direct_first_use_cache="not_cleared",
        native_first_use_cache="after_direct_runs_not_cleared",
        latency="caller_observed_since_submission", memory="torch_allocations_only",
        dispatch_candidate_bars=DISPATCH_BUDGET), recipe={
        key: value for key, value in vars(args).items() if key not in {"report", "compact"}},
        runtime=dict(rust_source_fingerprint=runtime["expected_source_fingerprint"],
                     rust_artifact_sha256=runtime["runtime_compiled_sha256"],
                     python_source_fingerprint=gpu_parity._source_fingerprint(),
                     torch=torch.__version__, cuda=torch.version.cuda, cupy=cupy.__version__,
                     gpu=device.name, gpu_total_memory_bytes=device.total_memory),
        cases=[_measure(torch, args, strategy, seed) for strategy in args.strategies for seed in args.seeds])


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    try:
        with redirect_stdout(sys.stderr):
            report = run_benchmark(args)
        code = 0
    except Exception as error:
        report = dict(schema_version=1, status="execution_failed",
                      error=dict(type=type(error).__name__, message=str(error)))
        code = 2
    rendered = json.dumps(report, allow_nan=False, indent=None if args.compact else 2, sort_keys=True)
    print(rendered)
    if args.report:
        try:
            Path(args.report).write_text(rendered + "\n")
        except OSError as error:
            print(f"report_save_failed: {error}", file=sys.stderr)
            return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
