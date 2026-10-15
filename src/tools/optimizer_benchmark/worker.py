"""One isolated backend measurement. Invoked by the controller, never downloads."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import math
import multiprocessing
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback

from .workloads import array_digest, digest, harness_digest, load_manifest

_CPU_CONTEXT = None


def initialize_cpu_worker(specs, markets, exchange, metrics):
    from shared_arrays import attach_shared_array
    global _CPU_CONTEXT
    attachments = [attach_shared_array(spec) for spec in specs]
    for attachment in attachments:
        attachment.array.flags.writeable = False
    _CPU_CONTEXT = (attachments, markets, exchange, metrics)


def cpu_config(config):
    attachments, markets, exchange, metrics = _CPU_CONTEXT
    candles, btc, timestamps = [attachment.array for attachment in attachments]
    return cpu_candidate((config, candles, markets, btc, timestamps, exchange, metrics))


def deny_network(*_args, **_kwargs):
    raise RuntimeError("optimizer benchmark forbids network connections")


def strict_json(value):
    """Retain nonfinite evidence in standard JSON, without turning it into zero."""
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if hasattr(value, "tolist"):
        return strict_json(value.tolist())
    return value


def cohort_digest(rows):
    """Hash simulator outcomes, excluding nondeterministic per-call timings."""
    return digest(strict_json([dict(metrics=row["metrics"], liquidated=row["liquidated"])
                               for row in rows]))


class Resources:
    """Sample process-tree RSS (shared pages can be counted more than once)."""
    def __init__(self):
        self.rows, self.errors = [], []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        import psutil
        process = psutil.Process()
        while not self.stop.is_set():
            try:
                processes = [process, *process.children(recursive=True)]
                rss = 0
                for child in processes:
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        continue
                vm = psutil.virtual_memory()
                self.rows.append(dict(seconds=time.perf_counter(), tree_rss_bytes=rss,
                                      system_available_bytes=vm.available,
                                      swap_used_bytes=psutil.swap_memory().used))
            except (psutil.Error, OSError) as error:
                self.errors.append(str(error))
            self.stop.wait(0.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=2)

    def report(self):
        return dict(sample_interval_seconds=.2, samples=len(self.rows), errors=self.errors,
                    peak_process_tree_rss_bytes=max((r["tree_rss_bytes"] for r in self.rows), default=None),
                    minimum_system_available_bytes=min((r["system_available_bytes"] for r in self.rows), default=None),
                    swap_growth_bytes=(max(r["swap_used_bytes"] for r in self.rows) - self.rows[0]["swap_used_bytes"])
                    if self.rows else None,
                    scope="process tree RSS; shared pages may be counted repeatedly; sampled peaks can miss transients")


def cpu_candidate(task):
    socket.socket.connect = deny_network
    socket.create_connection = deny_network
    config, candles, markets, btc, timestamps, exchange, metrics = task
    from backtest import build_backtest_payload, execute_backtest
    from config.metrics import resolve_metric_value
    started = time.perf_counter()
    payload = build_backtest_payload(candles, markets, deepcopy(config), exchange, btc, timestamps,
                                     metrics_only=True, skip_btc_analysis=True)
    prepared = time.perf_counter()
    _fills, _equity, analysis = execute_backtest(payload, config)
    ended = time.perf_counter()
    liquidation = next((analysis[key] for key in ("liquidated", "liquidated_usd") if key in analysis), None)
    if not isinstance(liquidation, (bool, int)):
        raise ValueError("CPU backtest omitted actual liquidation status")
    return dict(metrics={name: resolve_metric_value(analysis, name) for name in metrics},
                liquidated=bool(liquidation), preparation_seconds=prepared - started,
                replay_seconds=ended - prepared, latency_seconds=ended - started)


def legacy_parameters(config, inputs, metrics):
    """Use the branch's own payload projection and replay encoders."""
    from backtest import build_backtest_payload
    from config.shared_bot import flatten_shared_bot_side
    from optimization.gpu import model, service
    from optimization.gpu.hsl import project_bot
    global_config = deepcopy(config)
    global_config["coin_overrides"] = {}
    candles, markets, btc, timestamps, exchange = inputs
    payload = build_backtest_payload(candles, markets, global_config, exchange, btc, timestamps,
                                     metrics_only=True, skip_btc_analysis=True)
    kind = config["live"]["strategy_kind"]
    result = {}
    keys = model.TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS if kind == "trailing_martingale" else model.EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    for side in ("long", "short"):
        bot = project_bot(payload, 0, side, global_config)
        values = dict(payload.strategy_params_list[0][side])
        if kind == "trailing_martingale":
            values = model.flatten_trailing_martingale_params(values, bot)
        weights = bot["forager_score_weights"]
        values.update(entry_cooldown_minutes=bot["risk_entry_cooldown_minutes"],
                      total_wallet_exposure_limit=bot["total_wallet_exposure_limit"],
                      n_positions=bot["n_positions"],
                      forager_volume_ema_span_1m=bot["filter_volume_ema_span_1m"],
                      forager_volatility_ema_span_1m=bot["filter_volatility_ema_span_1m"],
                      forager_volume_drop_pct=bot["filter_volume_drop_pct"])
        for name in ("volume", "volatility", "ema_readiness"):
            values[f"forager_score_weights_{name}"] = weights[name]
        risk = config["bot"][side]["risk"]
        values.update(service._single_coin_exposure_params(risk, side=side,
                      scale_hsl_budget=config["bot"][side]["hsl"]["scale_budget_with_excess_allowance"]))
        if kind == "trailing_martingale":
            values.update(service._position_exposure_enforcer_params(risk, side=side))
        values.update(service._total_exposure_enforcer_params(risk, side=side))
        values.update(service._unstuck_params(flatten_shared_bot_side(config["bot"][side])))
        values.update(service._hsl_params(project_bot(payload, 0, side, global_config, base=True),
                                         signal_mode=config["live"]["hsl_signal_mode"]))
        values.update(model.adaptive_params(flatten_shared_bot_side(config["bot"][side])))
        for key in keys:
            result[f"{side}_{key}"] = float(values[key])
    return result


def instrument_replay(replay, dispatches):
    def observe_runner(runner):
        if getattr(runner, "_optimizer_benchmark_observed", False):
            return
        original = runner.run
        def measured(params, *args, _original=original, **kwargs):
            arrays = list(params.values()) if isinstance(params, dict) else [params]
            count = int(arrays[0].shape[0])
            started = time.perf_counter()
            output = _original(params, *args, **kwargs)
            observation = dict(candidates=count, host_seconds=time.perf_counter() - started)
            if "liq_step" in output:
                observation["liquidation_flags"] = (output["liq_step"] >= 0).detach().cpu().tolist()
            dispatches.append(observation)
            return output
        runner.run = measured
        runner._optimizer_benchmark_observed = True
    runners = [getattr(replay, "runner", None), getattr(replay, "fused_runner", None),
               *getattr(replay, "runners", {}).values()]
    for runner in {id(r): r for r in runners if r is not None}.values():
        observe_runner(runner)
    if callable(getattr(replay, "_construct_runner", None)):
        construct = replay._construct_runner
        def observed_construct(*args, **kwargs):
            runner = construct(*args, **kwargs)
            observe_runner(runner)
            return runner
        replay._construct_runner = observed_construct


def gpu_runs(entry, arrays, width, repeats, warmup, tune_seconds, dispatch_budget):
    import torch
    from optimization.gpu.runtime import gpu_device, synchronize
    from optimization.gpu.service import MpsMulticoinProxy, MpsSingleCoinProxy
    device = gpu_device(torch)
    config = entry["configs"][0]
    candles, btc, timestamps = arrays
    metrics, markets, exchange = entry["metrics"], entry["markets"], entry["exchange"]
    native = (Path("src/optimization/gpu/native.py")).exists()
    dispatches, samples, tuning, tuning_windows = [], [], [], []
    from optimization.gpu.autotune import BatchController
    original_observe = BatchController.observe
    tuning_started = time.perf_counter()
    def observed_window(controller, count, seconds, **kwargs):
        previous_width = controller.width
        rates = [*controller.samples, count / seconds][-controller.samples.maxlen:] if seconds > 0 else []
        completed = original_observe(controller, count, seconds, **kwargs)
        if completed:
            tuning_windows.append(dict(elapsed_seconds=time.perf_counter() - tuning_started,
                previous_width=previous_width, next_width=controller.width,
                candidates_per_second=statistics.median(rates), baseline=controller.baseline))
        return completed
    BatchController.observe = observed_window
    prepared = time.perf_counter()
    if native:
        from optimization.gpu.parameters import prepare_candidate_parameters
        parameters = [prepare_candidate_parameters(c, markets, exchange) for c in entry["configs"]]
    else:
        parameters = [legacy_parameters(c, (candles, markets, btc, timestamps, exchange), metrics)
                      for c in entry["configs"]]
    parameter_seconds = time.perf_counter() - prepared
    prepared = time.perf_counter()
    if native:
        from optimization.gpu.datasets import PreparedGpuDataset
        from optimization.gpu.executor import BacktestRequest
        from optimization.gpu.native import NativeBacktestService
        from shared_arrays import SharedArrayManager
        manager = SharedArrayManager()
        specs = [manager.create_from(a)[0] for a in arrays]
        dataset = PreparedGpuDataset(config=config, markets=markets, exchange=exchange,
            hlcvs=specs[0], btc=specs[1], timestamps=specs[2],
            candle_coins=config["backtest"]["coins"][exchange], metrics=metrics)
        service = NativeBacktestService(batch_size=None if width == "auto" else int(width),
            max_pending=max(1024, len(parameters)), tuning_mode="refresh" if width == "auto" else "off",
            max_dispatch_candidate_bars=dispatch_budget)
        # Instrument actual runner calls on their owning worker, not queued requests.
        import optimization.gpu.service as module
        original_init = module.MpsMulticoinProxy.__init__
        def observed_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            instrument_replay(self, dispatches)
        module.MpsMulticoinProxy.__init__ = observed_init
        service.__enter__()
        service.register_dataset(entry["id"], dataset)
        def evaluate(round_index):
            submitted = time.perf_counter()
            if width == "1":
                rows, latencies = [], []
                for i, parameter in enumerate(parameters):
                    request_start = time.perf_counter()
                    result = service.submit(BacktestRequest(f"{round_index}:{i}", entry["id"], parameter)).result()
                    latencies.append(time.perf_counter() - request_start)
                    rows.append(dict(metrics=result.metrics, liquidated=result.liquidated))
                return rows, latencies
            futures = [service.submit(BacktestRequest(f"{round_index}:{i}", entry["id"], p))
                       for i, p in enumerate(parameters)]
            from concurrent.futures import as_completed
            by_future = {future: i for i, future in enumerate(futures)}
            rows, latencies = [None] * len(futures), []
            for future in as_completed(futures):
                result = future.result()
                rows[by_future[future]] = dict(metrics=result.metrics, liquidated=result.liquidated)
                latencies.append(time.perf_counter() - submitted)
            return rows, latencies
    else:
        proxy_type = MpsSingleCoinProxy if candles.shape[1] == 1 else MpsMulticoinProxy
        replay = proxy_type(config=config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
            exchange=exchange, batch_size=64 if width == "auto" else int(width), needed_metrics=set(metrics),
            max_dispatch_candidate_bars=dispatch_budget)
        instrument_replay(replay, dispatches)
        if width == "auto":
            from optimization.gpu.autotune import configure_batch_tuning
            auto_config = deepcopy(config)
            auto_config["optimize"]["gpu"]["batch_size"] = None
            configure_batch_tuning([replay], auto_config,
                dict(tuning_mode="refresh", exact_workers=1, screening={}))
        def evaluate(_round_index):
            if width == "1":
                rows, latencies = [], []
                for parameter in parameters:
                    request_start = time.perf_counter()
                    row = replay.evaluate([parameter])[0]
                    latencies.append(time.perf_counter() - request_start)
                    rows.append(dict(metrics=row, liquidated=None))
                return rows, latencies
            rows = replay.evaluate(parameters)
            return [dict(metrics=row, liquidated=None) for row in rows], []
    setup_seconds = time.perf_counter() - prepared
    started = time.perf_counter()
    cold = None
    try:
        round_index = 0
        while round_index < warmup + repeats or (width == "auto" and time.perf_counter() - started < tune_seconds):
            synchronize()
            sample_start = time.perf_counter()
            dispatch_start = len(dispatches)
            rows, latencies = evaluate(round_index)
            synchronize()
            seconds = time.perf_counter() - sample_start
            if not native:
                observed = dispatches[dispatch_start:]
                flags = [flag for row in observed for flag in row.get("liquidation_flags", [])]
                # Multiple directional runs cannot establish shared-account status.
                if len(flags) == len(rows) and sum(row["candidates"] for row in observed) == len(rows):
                    for row, flag in zip(rows, flags, strict=True):
                        row["liquidated"] = bool(flag)
            sample = dict(seconds=seconds, completed=len(rows), results=rows,
                          dispatches=dispatches[dispatch_start:], completion_latencies_seconds=latencies,
                          elapsed_seconds=time.perf_counter() - started,
                          results_sha256=cohort_digest(rows))
            if round_index == 0:
                cold = sample
            if round_index >= warmup:
                if samples:
                    sample["results"] = []  # First measured cohort retains full candidate evidence.
                samples.append(sample)
            if native and service._batch_tuner is not None:
                for dataset_id, controller in service._batch_tuner.controllers.items():
                    tuning.append(dict(elapsed_seconds=time.perf_counter() - started, dataset_id=dataset_id,
                        width=controller.width, eligible_samples=len(controller.samples),
                        pending_evidence_seconds=controller.seconds,
                        baseline=controller.baseline, cooldown=controller.cooldown))
            round_index += 1
            if round_index >= 10000:
                break  # Bound fast automatic runs even if the caller requests a long duration.
        if device == "mps":
            memory = dict(current_allocated_bytes=torch.mps.current_allocated_memory(),
                          driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                          recommended_working_set_bytes=torch.mps.recommended_max_memory())
        else:
            memory = dict(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    finally:
        BatchController.observe = original_observe
        if native:
            try:
                service.__exit__(None, None, None)
            finally:
                module.MpsMulticoinProxy.__init__ = original_init
                manager.cleanup()
    return dict(samples=samples, first_use=cold, parameter_seconds=parameter_seconds,
                setup_seconds=setup_seconds, device=device, device_memory=memory,
                engine="native_service" if native else "legacy_direct_replay",
                tuning_observations=tuning,
                tuning_windows=tuning_windows,
                first_tuning_window_seconds=tuning_windows[0]["elapsed_seconds"] if tuning_windows else None,
                tuning_assessment="requires completed evidence windows and matched fixed-width sweep" if width == "auto" else None,
                isolated_requests=width == "1",
                latency_scope="isolated serial requests" if width == "1" else
                    "request completion including queueing" if native else "cohort latency only")


def search_run(entry, arrays, args):
    """Exercise the production backend, scoring, checkpoint and result persistence."""
    import optimize
    from optimization.backends import get_backend_runner
    from optimization.random_seed import seed_rngs
    from shared_arrays import SharedArrayManager
    from .workloads import get_path

    config = deepcopy(entry["configs"][0])
    backend = "gpu" if args.engine == "search-gpu" else "pymoo"
    config["optimize"].update(backend=backend, population_size=len(entry["configs"]),
        iters=args.search_iters, seed=entry["seed"], n_cpus=args.workers, write_all_results=True)
    config["optimize"]["bounds"] = {spec["key"]: [spec["low"], spec["high"], spec["step"]]
                                      for spec in entry["bounds"]}
    config["optimize"]["scoring"] = [dict(metric="adg_strategy_eq", goal="max"),
                                        dict(metric="drawdown_worst_strategy_eq", goal="min")]
    config["optimize"]["limits"] = []
    config["optimize"]["pymoo"]["algorithm"] = "nsga2"
    config["optimize"]["gpu"].update(batch_size=None if args.width == "auto" else int(args.width),
        tuning_mode="refresh" if args.width == "auto" else "off", max_dispatch_candidate_bars=args.dispatch_budget,
        exact_workers=args.workers)
    seed_rngs(entry["seed"], context="frozen benchmark search")
    manager = SharedArrayManager()
    records = []
    exchange = entry["exchange"]
    try:
        specs = [manager.create_from(a)[0] for a in arrays[:2]]
        base = optimize.Evaluator({exchange: specs[0]}, {exchange: specs[1]}, {exchange: entry["markets"]},
            config, timestamps={exchange: arrays[2]}, shared_array_manager=manager)
        with tempfile.TemporaryDirectory(prefix="optimizer-benchmark-search-") as directory:
            recorder = optimize.ResultRecorder(results_dir=directory, sig_digits=base.sig_digits,
                flush_interval=1, scoring_keys=config["optimize"]["scoring"], compress=False,
                write_all_results=True, bounds=base.bounds)
            original = recorder.record
            started = time.perf_counter()
            def record(row):
                records.append(dict(elapsed_seconds=time.perf_counter() - started, result=row))
                original(row)
            recorder.record = record
            try:
                result = get_backend_runner(backend)(config=config, evaluator=base, evaluator_for_pool=base,
                    recorder=recorder, overrides_list=[], duplicate_counter={"count": 0},
                    starting_configs_path="frozen-population", constraint_fitness_cls=None,
                    ignore_sigint_in_worker=optimize.ignore_sigint_in_worker,
                    get_starting_configs=lambda _path: entry["configs"],
                    configs_to_individuals=lambda configs, _bounds, _digits, **_kwargs: [[get_path(c, path)
                        for _key, path in base.key_paths] for c in configs],
                    record_individual_result=optimize._record_individual_result, run_evolution=None,
                    build_config_fn=optimize.individual_to_config, overrides_fn=optimize.optimizer_overrides,
                    checkpoint_path=str(Path(directory) / "checkpoint.pkl"), resume=False,
                    **({"standalone_candle_coins": {exchange: tuple(config["backtest"]["coins"][exchange])}}
                       if Path("src/optimization/gpu/native.py").exists() and backend == "gpu" else {}))
                pool = result.get("pool")
                if pool is not None:
                    pool.close()
                    pool.join()
                recorder.flush()
                seconds = time.perf_counter() - started
                disk_bytes = sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())
            finally:
                recorder.close()
        from .report import numerical_comparison
        reference_start = time.perf_counter()
        names = ["adg_strategy_eq", "drawdown_worst_strategy_eq"]
        reference_rows, observed_rows = [], []
        for record_row in records:
            saved = record_row["result"]
            reference_rows.append(cpu_candidate((saved, arrays[0], entry["markets"], arrays[1], arrays[2],
                                                 exchange, names)))
            observed_rows.append(dict(metrics={name: saved["metrics"]["stats"][name]["mean"] for name in names},
                                      liquidated=saved["metrics"]["liquidated"]))
        verification = numerical_comparison(
            dict(status="completed", samples=[dict(results=reference_rows)]),
            dict(status="completed", samples=[dict(results=observed_rows)]), names,
            {"adg_strategy_eq": dict(absolute=1e-7, relative=1e-4),
             "drawdown_worst_strategy_eq": dict(absolute=1e-5, relative=1e-4)})
        verification["seconds_outside_search_timing"] = time.perf_counter() - reference_start
        return dict(engine="prepared_production_optimizer", backend=backend,
            samples=[dict(seconds=seconds, completed=len(records), results=records)],
            first_result_seconds=records[0]["elapsed_seconds"] if records else None,
            persisted_bytes=disk_bytes, requested_iterations=args.search_iters,
            rust_reference_verification=verification,
            scope="prepared single-scenario production optimizer including scoring/checkpoint/persistence; excludes market preparation",
            correctness="saved objective values checked against CPU Rust outside timed search; trajectories may differ")
    finally:
        manager.cleanup()


def run(args):
    if args.harness_sha256 and harness_digest() != args.harness_sha256:
        raise ValueError("harness source changed during campaign; start a fresh campaign")
    import numpy as np
    from rust_utils import verify_loaded_runtime_extension
    import passivbot_rust
    runtime = verify_loaded_runtime_extension()
    if not runtime.get("runtime_compiled_source_stamp") or runtime["runtime_compiled_source_stamp"] != runtime["expected_source_fingerprint"]:
        raise RuntimeError("benchmark requires a source-verified Rust extension")
    manifest = load_manifest(args.workload)
    entry = next(e for e in manifest["entries"] if e["id"] == args.entry)
    if args.engine in {"gpu", "search-gpu"} and entry["declared_gpu_limitation"]:
        return dict(status="unsupported", reason=entry["declared_gpu_limitation"], runtime=runtime)
    with np.load(Path(args.workload) / entry["data_file"], allow_pickle=False) as data:
        arrays = tuple(data[key] for key in ("hlcvs", "btc", "timestamps"))
    if array_digest(*arrays) != entry["data_sha256"]:
        raise ValueError("dataset integrity mismatch")
    started = time.perf_counter()
    with Resources() as resources:
        if args.engine.startswith("search-"):
            result = search_run(entry, arrays, args)
        elif args.engine == "cpu":
            tasks = [(c, arrays[0], entry["markets"], arrays[1], arrays[2], entry["exchange"], entry["metrics"])
                     for c in entry["configs"]]
            from shared_arrays import SharedArrayManager
            manager = SharedArrayManager()
            specs = [manager.create_from(array)[0] for array in arrays] if args.workers > 1 else []
            pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_cpu_worker, initargs=(specs, entry["markets"], entry["exchange"], entry["metrics"])) if args.workers > 1 else None
            samples, first = [], None
            try:
                for repeat in range(args.warmup + args.repeats):
                    tick = time.perf_counter()
                    rows = list(pool.map(cpu_config, entry["configs"])) if pool else [cpu_candidate(task) for task in tasks]
                    sample = dict(seconds=time.perf_counter() - tick, completed=len(rows), results=rows,
                                  results_sha256=cohort_digest(rows))
                    if first is None:
                        first = sample
                    if repeat >= args.warmup:
                        samples.append(sample)
            finally:
                if pool:
                    pool.shutdown()
                manager.cleanup()
            result = dict(samples=samples, first_use=first, engine="cpu_prepared_backtests",
                          transport="spawn workers over immutable shared market arrays; candidate serialization included" if pool else "serial",
                          latency_scope="payload preparation plus simulation")
        else:
            result = gpu_runs(entry, arrays, args.width, args.repeats, args.warmup,
                              args.tune_seconds, args.dispatch_budget)
    import importlib.metadata
    tracked = subprocess.check_output(["git", "ls-files", "-z", "src", "passivbot-rust" ]).decode().split("\0")
    import hashlib
    sources = hashlib.sha256()
    for name in sorted(name for name in tracked if name and Path(name).is_file()):
        sources.update(name.encode())
        sources.update(Path(name).read_bytes())
    result.update(status="completed", runtime=runtime, resources=resources.report(),
                  implementation_sources_sha256=sources.hexdigest(),
                  dependency_versions={name: importlib.metadata.version(name) for name in ("numpy", "torch", "pymoo", "psutil")},
                  measurement_seconds=time.perf_counter() - started,
                  workload_sha256=manifest["sha256"], candidate_id=entry["id"],
                  backend_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  backend_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
                  python_version=sys.version, numpy_version=np.__version__)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--harness-sha256")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--engine", choices=("cpu", "gpu", "search-cpu", "search-gpu"), required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--width", default="8")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tune-seconds", type=float, default=0)
    parser.add_argument("--dispatch-budget", type=int, default=50_000_000)
    parser.add_argument("--search-iters", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.backend).resolve() / "src"))
    socket.socket.connect = deny_network
    socket.create_connection = deny_network
    try:
        result = run(args)
    except Exception as error:
        result = dict(status="error", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    Path(args.output).write_text(json.dumps(strict_json(result), indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] in {"completed", "unsupported"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
