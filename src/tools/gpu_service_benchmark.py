"""Offline native CUDA suite resources, latency and default tuning evidence.

Synthetic inputs only; no downloads, CPU simulations or evolutionary search.
Global device samples include other processes and all GPUs.
"""
import argparse
from concurrent.futures import as_completed
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from threading import Event, Thread
import time


def rss_tree(pid):
    total = 0
    pending = [pid]
    seen = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            for line in Path(f"/proc/{current}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
            pending.extend(map(int, Path(f"/proc/{current}/task/{current}/children").read_text().split()))
        except (FileNotFoundError, ProcessLookupError):
            pass  # A sampled compiler child can exit between the two reads.
    return total


class Sampler:
    def __init__(self):
        self.stop = Event()
        self.rows = []
        self.errors = []
        self.directory = None
        self.smi = shutil.which("nvidia-smi")
        if self.smi is None and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
            self.smi = "/usr/lib/wsl/lib/nvidia-smi"
        self.thread = Thread(target=self.run, daemon=True)

    def run(self):
        try:
            self._sample()
        except Exception as error:
            self.errors.append(f"sampler_failed:{type(error).__name__}")

    def _sample(self):
        while not self.stop.is_set():
            started = time.perf_counter()
            row = {"seconds": started, "process_tree_rss_bytes": rss_tree(os.getpid()) if sys.platform.startswith("linux") else None}
            if self.smi:
                try:
                    value = subprocess.check_output([
                        self.smi, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
                    ], text=True, timeout=5).strip().splitlines()
                    devices = [line.split(",") for line in value]
                    row["global_device_used_bytes"] = sum(int(device[0].strip()) * 2**20 for device in devices)
                    row["global_gpu_utilization_max_pct"] = max(int(device[1].strip()) for device in devices)
                except (subprocess.SubprocessError, ValueError, IndexError) as error:
                    self.errors.append(type(error).__name__)
            directory = self.directory
            if directory is not None:
                try:
                    row["packing_disk_bytes"] = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
                except FileNotFoundError:
                    pass  # Shutdown can remove spill files during sampling.
            row["sampler_seconds"] = time.perf_counter() - started
            self.rows.append(row)
            self.stop.wait(max(0.01, 1.0 - row["sampler_seconds"]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["ema_anchor", "trailing_martingale"], required=True)
    parser.add_argument("--coins", type=int, default=12)
    parser.add_argument("--bars", type=int, default=5760)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3,
                        help="Minimum rounds per wider phase; first round is excluded from warm medians")
    parser.add_argument("--tuning-windows", type=int, default=0,
                        help="Require this many completed default tuner windows per scenario; auto may run longer")
    parser.add_argument("--max-rounds", type=int, default=256,
                        help="Bound automatic extension for --tuning-windows")
    parser.add_argument("--hsl", choices=["disabled", "unified"], default="disabled")
    parser.add_argument("--accumulation-delay", type=float, default=None,
                        help="Fixed accumulation seconds; omitted uses production adaptive accumulation, zero isolates width")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report", required=True)
    return parser


def validate_args(parser, args):
    for name, lower, upper in (("coins", 3, 64), ("bars", 2880, 100000),
                               ("candidates", 2, 128), ("rounds", 2, 256),
                               ("tuning_windows", 0, 8), ("max_rounds", 2, 256)):
        if not lower <= getattr(args, name) <= upper:
            parser.error(f"--{name.replace('_', '-')} must be between {lower} and {upper}")
    if args.accumulation_delay is not None and (not math.isfinite(args.accumulation_delay)
                                               or not 0 <= args.accumulation_delay <= 0.1):
        parser.error("--accumulation-delay must be finite and between zero and 0.1 seconds")
    if args.max_rounds < args.rounds:
        parser.error("--max-rounds must be at least --rounds")


def fixture_digest(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def scenario_metadata(config, markets, coins, timestamps, span):
    config = deepcopy(config)
    config["backtest"]["coins"]["binance"] = list(coins)
    start, stop = span
    config["backtest"]["start_date"] = datetime.fromtimestamp(int(timestamps[start]) / 1000, timezone.utc).isoformat()
    config["backtest"]["end_date"] = datetime.fromtimestamp((int(timestamps[stop - 1]) + 60_000) / 1000, timezone.utc).isoformat()
    markets = {coin: deepcopy(markets[coin]) for coin in coins}
    for market in markets.values():
        market.update(first_valid_index=0, last_valid_index=stop - start - 1)
    markets["__meta__"] = {"requested_start_ts": int(timestamps[start])}
    return config, markets


def enough_windows(evidence, datasets, required):
    return all(sum(row["dataset"] == name for row in evidence["completed_windows"]) >= required
               for name in datasets)


@contextmanager
def forbid_cpu_simulations(backtest):
    def forbidden(*unused, **keywords):
        raise RuntimeError("GPU service benchmark must not invoke CPU simulation")
    targets = [(backtest, "execute_backtest"), (backtest, "run_backtest"),
               (backtest.pbr, "run_backtest_bundle")]
    original = [getattr(owner, name) for owner, name in targets]
    try:
        for owner, name in targets:
            setattr(owner, name, forbidden)
        yield
    finally:
        for (owner, name), value in zip(targets, original):
            setattr(owner, name, value)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    import numpy as np
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension
    verified = verify_loaded_runtime_extension()
    runtime = {name: verified[name] for name in
               ("runtime_compiled_sha256", "runtime_compiled_source_stamp", "expected_source_fingerprint")}
    from tools.gpu_parity import build_parser as parity_parser, fixture_inputs, _native_dataset
    from optimization.gpu.datasets import PreparedGpuDataset
    from optimization.gpu.parameters import prepare_candidate_parameters
    inputs = fixture_inputs(parity_parser().parse_args([
        "--fixture", args.strategy, "--sides", "both", "--coins", str(args.coins),
        "--bars", str(args.bars), "--seed", "7", "--hsl", args.hsl,
        "--hsl-red-threshold", ".99", "--hsl-lookback-days", "1",
    ]))
    metrics = ("adg_strategy_eq", "drawdown_worst_strategy_eq", "fills_per_day",
               "backtest_completion_ratio", "hard_stop_time_in_red_pct",
               "strategy_eq_recovery_days_mean", "strategy_eq_recovery_days_p95",
               "strategy_eq_recovery_days_mean_worst_1pct", "volume_pct_per_day_avg_w",
               "adg_strategy_eq_w")
    recipe = dict(strategy=args.strategy, coins=args.coins, bars=args.bars,
                  candidates=args.candidates, rounds=args.rounds, hsl=args.hsl,
                  tuning_windows=args.tuning_windows, max_rounds=args.max_rounds, reference_rounds=2,
                  accumulation_delay="auto" if args.accumulation_delay is None else args.accumulation_delay,
                  metrics=metrics, seed=7, widths=[1, 8, "auto"],
                  scopes="native service request cohorts, not evolutionary search or CPU throughput")
    report = dict(recipe=recipe, runtime=runtime, phases=[], prepared=[])
    with _native_dataset(inputs, "binance", metrics) as master, ExitStack() as controls:
        datasets, candidates = {}, {}
        half = args.bars // 2
        count = max(2, args.coins // 3)
        selections = [("base", (0, args.bars), tuple(range(args.coins))),
                      ("early", (0, half), tuple(range(count))),
                      ("late", (half, args.bars), tuple(range(args.coins - count, args.coins)))]
        for name, span, indices in selections:
            selected = [master.candle_coins[i] for i in indices]
            config, markets = scenario_metadata(inputs[0], inputs[2], selected, inputs[4], span)
            dataset = PreparedGpuDataset(
                config=config, markets=markets, exchange="binance", metrics=metrics,
                hlcvs=master.hlcvs, btc=master.btc, timestamps=master.timestamps,
                candle_coins=master.candle_coins, coin_indices=indices, time_range=span,
            )
            datasets[name] = dataset
            rows = []
            for index in range(args.candidates):
                candidate = deepcopy(config)
                for side in ["long", "short"]:
                    bot = candidate["bot"][side]["strategy"][args.strategy]
                    if args.strategy == "trailing_martingale":
                        bot = bot["entry"]
                        bot["initial_qty_pct"] = .01 + index * .001
                    else:
                        bot["base_qty_pct"] = .01 + index * .001
                    bot["ema_span_0"] = 5. + index * 1.25
                rows.append(prepare_candidate_parameters(candidate, markets, "binance"))
            candidates[name] = rows
            report["prepared"].append(dict(name=name, coins=selected, rows=span))
        with master.attach() as shared:
            report["fixture_sha256"] = fixture_digest(shared)
        assert report["fixture_sha256"] == fixture_digest((inputs[1], inputs[3], inputs[4]))
        if args.prepare_only:
            assert not any(name in sys.modules for name in ["torch", "cupy", "optimization.gpu.mps_kernel"])
            Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
            print("Prepared three shared-array scenarios without simulation", flush=True)
            return 0
        import torch
        import backtest
        from optimization.gpu.native import CudaBacktestService
        from optimization.gpu.executor import BacktestRequest
        from tools.gpu_cohort_benchmark import _metric_rounding, _observe_batches
        from optimization.gpu.service import MpsMulticoinProxy
        controls.enter_context(forbid_cpu_simulations(backtest))
        original = MpsMulticoinProxy.evaluate_results
        references = {}
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        for width in [1, 8, None]:
            sampler = Sampler()
            owner_rows = []
            def observed(self, values):
                result = original(self, values)
                from optimization.gpu.residency import current_cuda_residency
                residency = current_cuda_residency()
                if residency._directory is not None:
                    sampler.directory = Path(residency._directory.name)
                owner_rows.append(dict(count=len(values), torch_allocated=torch.cuda.memory_allocated(),
                                       torch_reserved=torch.cuda.memory_reserved(),
                                       device_free=torch.cuda.mem_get_info()[0],
                                       packing_entries=len(residency._entries),
                                       resident_entries=sum(any(isinstance(value, torch.Tensor)
                                           for value in entry["data"].values()) for entry in residency._entries.values())))
                assert owner_rows[-1]["resident_entries"] == 1
                return result
            torch.cuda.reset_peak_memory_stats()
            phase = dict(width="auto" if width is None else width, rounds=[], batches=[],
                         sampling_availability=dict(process_tree_rss=sys.platform.startswith("linux"),
                                                    global_device=bool(sampler.smi)))
            sampler.thread.start()
            try:
                MpsMulticoinProxy.evaluate_results = observed
                with CudaBacktestService(batch_size=width, max_pending=max(128, 3 * args.candidates), max_batch_delay=args.accumulation_delay) as service:
                    for name, dataset in datasets.items():
                        service.register_dataset(name, dataset)
                    evidence = _observe_batches(service._batch_policy, phase["batches"])
                    previous_observe = service._batch_policy.observe
                    def record_dataset(dataset_id, count, seconds, **kwargs):
                        windows = len(evidence["completed_windows"])
                        previous_observe(dataset_id, count, seconds, **kwargs)
                        phase["batches"][-1]["dataset"] = dataset_id
                        for entry in evidence["completed_windows"][windows:]:
                            entry["dataset"] = dataset_id
                    service._batch_policy.observe = record_dataset
                    limit = 2 if width == 1 else args.max_rounds if width is None else args.rounds
                    for iteration in range(limit):
                        started = time.perf_counter()
                        cpu_started = time.process_time()
                        pending = {}
                        for index in range(args.candidates):
                            for name in datasets:
                                request_id = f"{iteration}:{name}:{index}"
                                submitted = time.perf_counter()
                                future = service.submit(BacktestRequest(request_id, name, candidates[name][index]))
                                pending[future] = (name, index, request_id, submitted)
                        latencies, completions, rounding = [], [], 0
                        for future in as_completed(pending):
                            name, index, request_id, submitted = pending[future]
                            row = future.result(timeout=600)
                            now = time.perf_counter()
                            assert (row.dataset_id, row.request_id) == (name, request_id)
                            assert set(row.metrics) == set(metrics), (request_id, sorted(row.metrics))
                            assert all(np.isfinite(value) for value in row.metrics.values())
                            key = name, index
                            if width != 1:
                                assert key in references, key
                            if key not in references:
                                references[key] = row
                            differences = _metric_rounding(references[key].metrics, row.metrics)
                            assert differences is not None, (key, references[key].metrics, row.metrics)
                            assert references[key].liquidated == row.liquidated
                            rounding += bool(differences)
                            latencies.append(now - submitted)
                            completions.append(now - started)
                        phase["rounds"].append(dict(seconds=completions[-1], first_completion=completions[0],
                                                    latency_p95=float(np.percentile(latencies, 95)),
                                                    whole_process_cpu_seconds=time.process_time() - cpu_started,
                                                    reduction_rounding_results=rounding))
                        print("Completed", phase["width"], iteration, phase["rounds"][-1]["seconds"], flush=True)
                        if width is None and iteration + 1 >= args.rounds and enough_windows(
                                evidence, datasets, args.tuning_windows):
                            break
                    phase["tuning"] = dict(evidence, controllers={name:dict(width=c.width, ceiling=c.ceiling,
                        pending_samples=len(c.samples), pending_seconds=c.seconds, seen_counts=sorted(c.seen))
                        for name,c in service._batch_policy.controllers.items()})
                    phase["torch_peak_allocated"] = torch.cuda.max_memory_allocated()
                    phase["torch_peak_reserved"] = torch.cuda.max_memory_reserved()
                phase["spill_removed_after_close"] = sampler.directory is None or not sampler.directory.exists()
                assert phase["spill_removed_after_close"]
            finally:
                MpsMulticoinProxy.evaluate_results = original
                sampler.stop.set()
                sampler.thread.join(timeout=10)
            if sampler.thread.is_alive():
                raise RuntimeError("Resource sampling thread did not stop")
            phase["tuning_requirement_met"] = width is not None or enough_windows(
                evidence, datasets, args.tuning_windows)
            with master.attach() as shared:
                phase["source_arrays_unchanged"] = fixture_digest(shared) == report["fixture_sha256"]
            assert phase["source_arrays_unchanged"]
            phase["owner_snapshots"] = owner_rows
            phase["resource_samples"] = sampler.rows
            phase["sampling_errors"] = sampler.errors
            phase["warm_median_seconds"] = statistics.median(row["seconds"] for row in phase["rounds"][1:])
            phase["warm_requests_per_second"] = 3 * args.candidates / phase["warm_median_seconds"]
            report["phases"].append(phase)
            Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
            gc.collect()
        report["limits"] = ["Global device snapshots include driver and unrelated display allocations.",
                            "One-second sampling may miss short peaks; Torch allocator peaks are separate.",
                            "Process CPU time includes worker/preparation/compilation, not isolated orchestrator cost.",
                            "No CPU reference or optimization-quality claim; results compare isolated GPU width one.",
                            "The benchmark does not change production tuner evidence windows.",
                            "Width-one reference uses two rounds; wider phases may have different sample counts.",
                            "Global device memory sums all GPUs and is not exclusive service VRAM."]
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        return 0 if all(phase["tuning_requirement_met"] for phase in report["phases"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
