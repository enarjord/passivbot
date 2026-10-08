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
# Identical float32 replay summaries can differ in the last float64 reduction
# bits across batch shapes. This does not admit a float32 ULP of replay drift.
MAX_FLOAT64_REDUCTION_ULPS = 8
DEFAULT_OBJECTIVES = {"adg_strategy_eq": "max", "drawdown_worst_strategy_eq": "min"}


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
    gpu_parity.add_fixture_stress_options(parser)
    parser.add_argument("--unstuck", action="store_true")
    parser.add_argument("--adg-floor", type=float)
    parser.add_argument("--drawdown-ceiling", type=float)
    parser.add_argument("--metrics", nargs="+", default=[],
                        help="Additional GPU metrics; ADG, drawdown and fills/day remain included")
    parser.add_argument("--objective", dest="objectives", action="append", nargs=2, default=[],
                        metavar=("METRIC", "DIRECTION"),
                        help="Repeat to replace the default ADG/max, drawdown/min ranking; DIRECTION is min or max")
    parser.add_argument("--tolerances", help="JSON per-metric comparison policies, as in gpu-parity")
    parser.add_argument("--limit", dest="limits", action="append", nargs=3, default=[],
                        metavar=("METRIC", "MODE", "VALUE"),
                        help="Repeatable diagnostic limit; MODE is less_than or greater_than")
    parser.add_argument("--report")
    parser.add_argument("--compact", action="store_true")
    return parser


def validate_args(parser, args):
    try:
        recipe = gpu_parity.resolve_fixture_args(args)
        for name in gpu_parity.STRESS_OPTIONS:
            setattr(args, name, getattr(recipe, name))
    except ValueError as error:
        parser.error(str(error))
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

    from config.metrics import canonicalize_metric_name

    try:
        objectives = {}
        for metric, direction in args.objectives:
            if direction not in {"min", "max"}:
                raise ValueError("--objective DIRECTION must be min or max")
            name = canonicalize_metric_name(metric)
            if name in objectives and objectives[name] != direction:
                raise ValueError(f"conflicting objective directions for {name}")
            objectives[name] = direction
        args.objectives = objectives or dict(DEFAULT_OBJECTIVES)
        limits = []
        for metric, mode, value in args.limits:
            if mode not in {"less_than", "greater_than"}:
                raise ValueError("--limit MODE must be less_than or greater_than")
            bound = float(value)
            if not math.isfinite(bound):
                raise ValueError("diagnostic limit values must be finite")
            limits.append(dict(metric=canonicalize_metric_name(metric), penalize_if=mode, value=bound))
        args.limits = limits
        checks = _checks(args)
        args.metrics = list(dict.fromkeys(canonicalize_metric_name(name) for name in
                            [*gpu_parity.DEFAULT_METRICS, *args.metrics, *args.objectives,
                             *(check["metric"] for check in checks)]))
        args.policies = dict(gpu_parity.DEFAULT_TOLERANCES)
        if args.tolerances:
            policies = json.loads(Path(args.tolerances).read_text())
            if not isinstance(policies, dict):
                raise ValueError("comparison policies must be a JSON object")
            for name, value in policies.items():
                args.policies[canonicalize_metric_name(name)] = gpu_parity.MetricTolerance(**value)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.price_shocks:
        # Candles depend on the seed, not the strategy. Check stressed seeds one
        # at a time before CUDA access; do not retain whole cohorts for preflight.
        try:
            for seed in args.seeds:
                gpu_parity._fixture_candles(argparse.Namespace(**vars(args), seed=seed))
        except ValueError as error:
            parser.error(str(error))


def _timing_summary(samples, count):
    median = statistics.median(samples[1:])
    return dict(first_seconds=samples[0], warm_seconds=samples[1:],
                warm_median_seconds=median, warm_candidates_per_second=count / median)


def _front(rows, objectives):
    values = [tuple(row[name] * (-1 if direction == "max" else 1)
                    for name, direction in objectives.items()) for row in rows]
    return [i for i, row in enumerate(values) if not any(
        all(left <= right for left, right in zip(other, row)) and
        any(left < right for left, right in zip(other, row))
        for j, other in enumerate(values) if j != i)]


def _ranking(cpu, gpu, objectives=None):
    objectives = dict(DEFAULT_OBJECTIVES if objectives is None else objectives)
    if not cpu or len(cpu) != len(gpu):
        return {"assessed": False, "reason": "empty_or_mismatched_candidates"}
    if any(row.get(name) is None or not math.isfinite(row[name])
           for rows in (cpu, gpu) for row in rows for name in objectives):
        return {"assessed": False, "reason": "nonfinite_or_missing_objective"}
    cpu_front, gpu_front = _front(cpu, objectives), _front(gpu, objectives)
    relation = lambda a, b: int(a > b) - int(a < b)
    flips = {name: sum(
        relation(cpu[i][name], cpu[j][name]) != relation(gpu[i][name], gpu[j][name])
        for i in range(len(cpu)) for j in range(i + 1, len(cpu))) for name in objectives}
    best, regret = {}, {}
    for name, direction in objectives.items():
        sign = -1 if direction == "max" else 1
        selected = min(range(len(gpu)), key=lambda i: sign * gpu[i][name])
        best[name] = selected
        regret[name] = sign * cpu[selected][name] - min(sign * row[name] for row in cpu)
    result = dict(assessed=True, objectives=objectives,
                  cpu_front=cpu_front, gpu_front=gpu_front, front_members_match=cpu_front == gpu_front,
                  pair_order_disagreements=flips, pairs=len(cpu) * (len(cpu) - 1) // 2,
                  gpu_best_candidates=best, cpu_regret_at_gpu_best=regret)
    if objectives.get("adg_strategy_eq") == "max":
        result.update(gpu_max_adg_candidate=best["adg_strategy_eq"],
                      cpu_adg_regret_of_gpu_max=regret["adg_strategy_eq"])
    return result


def _checks(args):
    from limit_utils import expand_limit_checks

    entries = list(args.limits)
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
    fixture_args = gpu_parity.build_parser().parse_args(options)
    for name in gpu_parity.STRESS_OPTIONS:
        setattr(fixture_args, name, getattr(args, name, None))
    inputs = gpu_parity.fixture_inputs(fixture_args)
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


def _observe_batches(policy, batches):
    """Preserve cumulative eligible evidence even after consumed windows reset."""
    original_observe = policy.observe
    evidence = dict(samples=0, seconds=0.0, completed_windows=[])

    def observe(dataset_id, count, seconds, **kwargs):
        batches.append(dict(count=count, seconds=seconds))
        controller = policy.controllers.get(dataset_id)
        eligible = (controller is not None and count == controller.width
                    and controller.width in controller.seen
                    and math.isfinite(seconds) and seconds > 0)
        if eligible:
            evidence["samples"] += 1
            evidence["seconds"] += seconds
            width = controller.width
            rates = [*controller.samples, count / seconds][-controller.samples.maxlen:]
            window_seconds = controller.seconds + seconds
        original_observe(dataset_id, count, seconds, **kwargs)
        if eligible and not controller.samples:
            evidence["completed_windows"].append(dict(
                width=width, samples=len(rates), seconds=window_seconds,
                median_candidates_per_second=statistics.median(rates),
                resulting_width=controller.width))

    policy.observe = observe
    return evidence


def _metric_rounding(reference, observed):
    """Return bounded machine-scale differences, or None for a mismatch."""
    if reference.keys() != observed.keys():
        return None
    rounding = {}
    for name, expected in reference.items():
        actual = observed[name]
        if actual == expected:
            continue
        if not (math.isfinite(expected) and math.isfinite(actual)):
            return None
        error = abs(actual - expected)
        ulps = error / max(math.ulp(expected), math.ulp(actual))
        if ulps > MAX_FLOAT64_REDUCTION_ULPS:
            return None
        rounding[name] = dict(absolute_error=error, float64_ulps=ulps)
    return rounding


def _native_runs(torch, dataset, parameters, reference, width, warm_runs):
    import numpy as np
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService

    samples, batches, rounding = [], [], {}
    allocated_start = torch.cuda.memory_allocated()
    reserved_start = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    with CudaBacktestService(batch_size=width, max_pending=max(64, len(parameters)),
                             max_dispatch_candidate_bars=DISPATCH_BUDGET) as service:
        service.register_dataset("cohort", dataset)
        # Tool-only observation of the existing controller; retain its decisions
        # and the executor's successful-work timings without running extra work.
        evidence = _observe_batches(service._batch_policy, batches)
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
                differences_allowed = _metric_rounding(reference[index].metrics, row.metrics)
                if differences_allowed is None or row.liquidated != reference[index].liquidated:
                    expected = reference[index]
                    names = sorted(set(expected.metrics) | set(row.metrics))
                    differences = [(name, expected.metrics.get(name), row.metrics.get(name))
                                   for name in names if name not in expected.metrics or name not in row.metrics
                                   or expected.metrics[name] != row.metrics[name]]
                    raise RuntimeError(
                        f"native result differs from direct GPU replay: request={request_id!r}, "
                        f"liquidation={expected.liquidated!r}/{row.liquidated!r}, "
                        f"metric_differences={differences[:8]!r}, count={len(differences)}")
                for name, difference in differences_allowed.items():
                    metric_evidence = rounding.setdefault(name, dict(count=0, max_absolute_error=0.0, max_float64_ulps=0.0))
                    metric_evidence["count"] += 1
                    metric_evidence["max_absolute_error"] = max(metric_evidence["max_absolute_error"], difference["absolute_error"])
                    metric_evidence["max_float64_ulps"] = max(metric_evidence["max_float64_ulps"], difference["float64_ulps"])
                elapsed.append(observed - started)
                latencies.append(observed - submitted)
            samples.append(dict(seconds=elapsed[-1], first_completion_seconds=elapsed[0],
                                caller_observed_latency_p50_seconds=float(np.percentile(latencies, 50)),
                                caller_observed_latency_p95_seconds=float(np.percentile(latencies, 95))))
        controller = service._batch_policy.controllers.get("cohort")
        tuning = (dict(final_width=controller.width, evidence_samples=evidence["samples"],
                       evidence_seconds=evidence["seconds"],
                       completed_windows=evidence["completed_windows"],
                       pending_window_samples=len(controller.samples),
                       pending_window_seconds=controller.seconds,
                       seen_widths=sorted(controller.seen))
                  if controller is not None else None)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    return dict(requested_width="auto" if width is None else width,
                matches_direct_gpu=True, matches_direct_gpu_exactly=not rounding,
                reduction_rounding=rounding,
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
                                             metrics_only=True,
                                             skip_btc_analysis=not any(name.endswith("_btc") for name in args.metrics))
            _, _, analysis = execute_backtest(payload, candidate)
            rows.append({name: value for name in args.metrics
                         if (value := resolve_metric_value(analysis, name)) is not None})
        cpu_samples.append(time.perf_counter() - started)
        cpu = rows
    started = time.perf_counter()
    proxy = MpsMulticoinProxy(config=config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
                             exchange="binance", batch_size=len(parameters), needed_metrics=args.metrics,
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
    with gpu_parity._native_dataset(inputs, "binance", args.metrics) as dataset:
        native = [_native_runs(torch, dataset, parameters, reference, width, args.warm_runs)
                  for width in args.widths]
    gpu = [dict(row.metrics) for row in reference]
    policies = {name: args.policies.get(name) for name in args.metrics}
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
                cpu_gpu_comparisons=comparisons, ranking=_ranking(cpu, gpu, args.objectives),
                diagnostic_limits=dict(adg_floor=args.adg_floor, drawdown_ceiling=args.drawdown_ceiling,
                                       limits=args.limits, comparisons=limits))


def run_benchmark(args):
    import torch
    from rust_utils import check_and_maybe_compile, verify_loaded_runtime_extension
    from optimization.gpu.metrics import validate_gpu_metric_names

    validate_gpu_metric_names(args.metrics)
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
    return dict(schema_version=1, status="measured",
                native_reference_policy=dict(max_float64_ulps=MAX_FLOAT64_REDUCTION_ULPS,
                                             metric_keys="exact", liquidation="exact"),
                measurement_scope=dict(
        cpu="serial_payload_preparation_and_simulation",
        direct_first_use_cache="not_cleared",
        native_first_use_cache="after_direct_runs_not_cleared",
        latency="caller_observed_since_submission", memory="torch_allocations_only",
        dispatch_candidate_bars=DISPATCH_BUDGET), recipe=_recipe(args),
        tolerance_policy={name: vars(args.policies[name]) if name in args.policies else None
                          for name in args.metrics},
        runtime=dict(rust_source_fingerprint=runtime["expected_source_fingerprint"],
                     rust_artifact_sha256=runtime["runtime_compiled_sha256"],
                     python_source_fingerprint=gpu_parity._source_fingerprint(),
                     torch=torch.__version__, cuda=torch.version.cuda, cupy=cupy.__version__,
                     gpu=device.name, gpu_total_memory_bytes=device.total_memory),
        cases=[_measure(torch, args, strategy, seed) for strategy in args.strategies for seed in args.seeds])


def _recipe(args):
    return {key: value for key, value in vars(args).items()
            if key not in {"report", "compact", "tolerances", "policies"}}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    try:
        with redirect_stdout(sys.stderr):
            report = run_benchmark(args)
        code = 0
    except Exception as error:
        report = dict(schema_version=1, status="execution_failed", recipe=_recipe(args),
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
