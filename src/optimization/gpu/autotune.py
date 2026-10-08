"""Bounded, evidence-driven batch sizing using completed proxy work only.

This is execution policy, never evolutionary or drift policy. Cache loss changes
performance only; exceptions from evaluation are deliberately not intercepted.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from contextvars import ContextVar
from functools import lru_cache
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time
from optimization.progress import log_tokens

WINDOW = 24
MIN_SECONDS = 30.0
CACHE_VERSION = 2
_REPLAY_SAMPLES = ContextVar("gpu_tuning_replay_samples", default=None)


class ReplayEvidence(deque):
    def __init__(self):
        super().__init__(maxlen=128)
        self.kernel_seconds = 0.0


def record_replay_chunk(count, bars, total_bars, seconds, *, eligible=True):
    """Observe completed production dispatches; never launch calibration work."""
    samples = _REPLAY_SAMPLES.get()
    if (
        samples is not None and count > 0 and bars > 0 and total_bars > 0
        and math.isfinite(seconds) and seconds > 0
    ):
        samples.kernel_seconds += seconds
        if eligible:
            samples.append((count, seconds * total_bars / bars, seconds))


def is_auto(value):
    return value is None or (isinstance(value, str) and value.strip().lower() == "auto")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@lru_cache(maxsize=1)
def _implementation_identity():
    # Include the actual compiled Rust artifact and Python/kernel source contract.
    import passivbot_rust

    files = [Path(passivbot_rust.__file__)]
    if files[0].name == "__init__.py":
        files.extend(sorted(files[0].parent.glob("*.so")))
        files.extend(sorted(files[0].parent.glob("*.pyd")))
    files.extend(sorted(Path(__file__).parent.glob("*.py")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def hardware_identity(torch):
    from optimization.gpu.runtime import gpu_device

    device = gpu_device(torch)
    identity = dict(
        device=device,
        torch=str(torch.__version__),
        system=platform.platform(),
        cpu=platform.processor(),
        cpus=os.cpu_count(),
        implementation=_implementation_identity(),
    )
    if device == "cuda":
        if getattr(torch.version, "hip", None):
            raise RuntimeError("GPU auto-tuning requires the supported NVIDIA CUDA backend")
        import cupy

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        identity.update(
            name=props.name,
            memory=props.total_memory,
            processors=props.multi_processor_count,
            capability=[props.major, props.minor],
            cuda=torch.version.cuda,
            cupy=str(cupy.__version__),
        )
    else:
        identity.update(
            name=torch.backends.mps.get_name(), memory=torch.mps.recommended_max_memory()
        )
    return identity


def workload_contract(contract):
    # Candle values and absolute dates do not define an execution shape. Preserve
    # history length, validity/warmup indices, fixed parameters and feature flags.
    result = dict(contract)
    result["hlcvs"] = {k: v for k, v in contract.get("hlcvs", {}).items() if k != "sha256"}
    result.pop("timestamps", None)
    result["backtest"] = {
        k: v
        for k, v in contract.get("backtest", {}).items()
        if k not in {"requested_start_timestamp_ms", "first_timestamp_ms"}
    }
    if "btc_analysis" in result:
        result["btc_analysis"] = {k: v for k, v in result["btc_analysis"].items() if k != "sha256"}
    return result


class BatchController:
    """Bounded hill climbing over completed work, with median smoothing."""

    def __init__(
        self,
        ceiling,
        initial,
        *,
        save=lambda *args: None,
        can_grow=lambda: True,
        can_trial=lambda: True,
        allow_partial_batches=False,
    ):
        self.ceiling = max(1, int(ceiling))
        self.width = max(1, min(int(initial), self.ceiling))
        self.save = save
        self.can_grow = can_grow
        self.can_trial = can_trial
        self.allow_partial_batches = allow_partial_batches
        self.samples = deque(maxlen=WINDOW)
        self.seconds = 0.0
        self.seen = set()
        self.baseline = None
        self.direction = -1 if self.width == self.ceiling else 1
        self.cooldown = 0

    def observe(self, count, seconds, *, evidence_seconds=None):
        """Return true when a complete evidence window has been consumed."""
        # Retained proxy tuning requires full batches. The async service learns
        # actual dispatch shapes, including warm partial request cohorts.
        if (not 1 <= count <= self.width
                or (not self.allow_partial_batches and count != self.width)
                or not math.isfinite(seconds) or seconds <= 0):
            return
        if count not in self.seen:
            self.seen.add(count)
            return
        self.samples.append(count / seconds)
        self.seconds += seconds if evidence_seconds is None else evidence_seconds
        if len(self.samples) < WINDOW or self.seconds < MIN_SECONDS:
            return
        rate = statistics.median(self.samples)
        evidence_seconds = self.seconds
        self.samples.clear()
        self.seconds = 0.0
        if self.baseline is not None:
            old_width, old_rate = self.baseline
            self.baseline = None
            # Prefer a smaller allocation on a plateau; larger ones must pay off.
            accepted = rate >= old_rate * (0.98 if self.width < old_width else 1.05)
            if accepted:
                log_tokens("GPU auto-tune accepted |", [
                    f"batch={self.width}", f"candidates/s={rate:.3f}",
                    f"previous_candidates/s={old_rate:.3f}",
                    "reason=smaller_plateau" if self.width < old_width else "reason=throughput_gain",
                ])
                self.save(self.width, rate, evidence_seconds)
                self.cooldown = 1
            else:
                log_tokens("GPU auto-tune retained |", [
                    f"batch={old_width}", f"trial_batch={self.width}",
                    f"candidates/s={old_rate:.3f}", f"trial_candidates/s={rate:.3f}",
                    "reason=insufficient_gain",
                ])
                self.width = old_width
                self.direction *= -1
                self.cooldown = 3
            return True
        self.save(self.width, rate, evidence_seconds)
        if self.cooldown:
            self.cooldown -= 1
            return True
        if not self.can_trial():
            return True
        trial = (
            min(self.ceiling, self.width * 2) if self.direction > 0 else max(1, self.width // 2)
        )
        if trial == self.width:
            self.direction *= -1
            self.cooldown = 1
            return True
        if trial > self.width and not self.can_grow():
            if not self.allow_partial_batches or self.width == 1:
                self.cooldown = 1
                return True
            # A bounded producer may never queue enough work for growth. Try a
            # smaller allocation using subsequent real requests instead.
            self.direction = -1
            trial = max(1, self.width // 2)
        self.baseline = (self.width, rate)
        self.width = trial
        log_tokens("GPU auto-tune trial |", [
            f"batch={trial}", f"previous={self.baseline[0]}",
            f"rolling_candidates/s={rate:.3f}", "reason=throughput_probe",
        ])
        return True


class CalibrationCache:
    """Bounded advisory measurements; no market data, configs or search state."""

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir or "caches/gpu_autotune")
        self.cache_warning = False
        self.cache_write_disabled = False

    def _warn(self, error):
        if not self.cache_warning:
            logging.warning(
                "GPU auto-tune cache unavailable; continuing with in-memory tuning: %s", error
            )
            self.cache_warning = True

    def read(self, key, field, floor, ceiling):
        try:
            record = json.loads((self.cache_dir / (key + ".json")).read_text())
            width = record[field]
            if (
                record["version"] != CACHE_VERSION
                or type(width) is not int
                or not floor <= width <= ceiling
                or not math.isfinite(record["candidates_per_second"])
                or record["candidates_per_second"] <= 0
            ):
                raise ValueError("invalid calibration")
            return width
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._warn(error)
            return None

    def write(self, key, field, width, rate, seconds):
        if self.cache_write_disabled:
            return
        path = self.cache_dir / (key + ".json")
        temporary = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.cache_dir, suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(
                    dict(
                        version=CACHE_VERSION,
                        **{field: width},
                        candidates_per_second=rate,
                        evidence_seconds=seconds,
                        window=WINDOW,
                    ),
                    stream,
                )
            os.replace(temporary, path)
            # Bound the local advisory cache; never touch unrelated files.
            entries = sorted(
                self.cache_dir.glob("[0-9a-f]" * 64 + ".json"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            for stale in entries[128:]:
                stale.unlink(missing_ok=True)
        except OSError as error:
            self.cache_write_disabled = True
            self._warn(error)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    self._warn(error)


class ProxyBatchTuner(CalibrationCache):
    """Per-proxy workload classes; cache contains no configs or market data."""

    def __init__(self, proxy, *, mode, hardware, context, cache_dir=None):
        self.proxy = proxy
        self.mode = mode
        self.hardware = hardware
        self.worker_count = context.get("workers")
        self.identity = _digest(
            dict(
                hardware=hardware,
                context={k: v for k, v in context.items() if k != "workers"},
                execution=workload_contract(proxy.checkpoint_contract),
                overrides=getattr(proxy, "coin_override_contract", {}),
                metrics=sorted(proxy.needed_metrics),
            )
        )
        super().__init__(cache_dir)
        self.controllers = OrderedDict()
        self.allow_trial = lambda: True
        self.revision = 0

    def set_worker_count(self, workers):
        if self.worker_count == workers:
            return
        self.retire_trials()
        self.controllers.clear()
        self.worker_count = workers
        self.revision += 1

    def _headroom(self):
        torch = self.proxy._torch
        if self.hardware["device"] == "cuda":
            free, total = torch.cuda.mem_get_info()
            return free >= max(256 * 1024**2, total * 0.2)
        return torch.mps.driver_allocated_memory() < torch.mps.recommended_max_memory() * 0.7

    def retire_trials(self, *, except_key=None):
        """Return inactive workload trials to measured widths without caching them."""
        for key, controller in self.controllers.items():
            if key == except_key or controller.baseline is None:
                continue
            controller.width = controller.baseline[0]
            controller.baseline = None
            controller.samples.clear()
            controller.seconds = 0.0
            controller.cooldown = max(controller.cooldown, 1)
            self.revision += 1
            log_tokens("GPU auto-tune retained |", [
                f"batch={controller.width}", "reason=workload_phase_end",
            ])

    def controller(self, ceiling, demand, end_step):
        ceiling = min(ceiling, demand)
        key = _digest(
            [
                CACHE_VERSION,
                self.identity,
                self.worker_count,
                ceiling,
                demand,
                end_step,
                self.proxy.max_dispatch_candidate_bars,
            ]
        )
        # Trials can span repeated calls of the same class, but a different
        # demand/history class cannot inherit a gate from an inactive workload.
        self.retire_trials(except_key=key)
        if key in self.controllers:
            self.controllers.move_to_end(key)
            return self.controllers[key]
        # Bound the initial allocation. The existing dispatch plan is always the cap.
        # The work envelope does not bound batch-scaled replay/output buffers.
        # Retain the established smaller start when device headroom is low.
        initial = ceiling if self._headroom() else min(128, ceiling)
        source = "dispatch_plan" if initial == ceiling else "memory_headroom"
        if self.mode != "refresh":
            width = self.read(key, "batch_size", 1, ceiling)
            if width is not None and width <= initial:
                initial = width
                source = "cache"

        def save(width, rate, seconds):
            self.write(key, "batch_size", width, rate, seconds)

        logging.info(
            "GPU auto-tune starting | batch=%d ceiling=%d source=%s", initial, ceiling, source
        )
        controller = BatchController(
            ceiling,
            initial,
            save=save,
            can_grow=self._headroom,
            can_trial=lambda: self.allow_trial(),
        )
        self.controllers[key] = controller
        while len(self.controllers) > 8:
            self.controllers.popitem(last=False)
        return controller


def history_dispatch_ceiling(proxy, ceiling):
    """Bound replay plus reduction history before claiming an outer request batch."""
    single = getattr(proxy, "runner", None)
    fused = getattr(proxy, "fused_runner", None)
    runners = ([single] if single is not None else [fused] if fused is not None
               else list(getattr(proxy, "runners", {}).values()))
    for runner in runners:
        history_size = getattr(runner, "_history_bytes_per_candidate", None)
        history_bytes = history_size() if history_size is not None else 0
        if history_bytes:
            ceiling = min(ceiling, max(1, runner.hsl_scratch_budget_bytes // history_bytes))
    return ceiling


def proxy_batches(proxy, candidates, ceiling, *, end_step=None, clock=time.perf_counter):
    """Measure through the caller's completed host results, without extra GPU sync.

    A failed/interrupted yield never produces a sample. Shape changes happen only
    after the entire previous candidate replay, reductions and host copy finish.
    """
    # Limit the outer batch too: splitting a kernel and then concatenating its
    # full sample histories would exceed the budget again during reduction.
    ceiling = history_dispatch_ceiling(proxy, ceiling)
    tuner = getattr(proxy, "batch_tuner", None)
    controller = (
        tuner.controller(ceiling, len(candidates), end_step) if tuner and candidates else None
    )
    start = 0
    while start < len(candidates):
        width = controller.width if controller is not None else ceiling
        chunk = candidates[start : start + width]
        started = clock() if controller is not None else 0.0
        # Keep bounded timing evidence local until the entire replay, host copy,
        # and metric reductions succeed. Failed/abandoned evaluations cannot tune.
        samples = ReplayEvidence()
        token = _REPLAY_SAMPLES.set(samples if controller is not None else None)
        try:
            yield start, chunk
        finally:
            _REPLAY_SAMPLES.reset(token)
        if controller is not None:
            previous = controller.width
            previous_trial = getattr(controller, "baseline", None) is not None
            if samples and isinstance(controller, BatchController):
                # Charge packing, allocations, reductions and host-copy overhead
                # to each kernel sample so trials optimize end-to-end throughput.
                overhead = max(1.0, (clock() - started) / samples.kernel_seconds)
                for count, seconds, evidence_seconds in samples:
                    # Scratch-limited runners may split one outer batch. Convert
                    # their per-candidate rate to the outer controller's width.
                    consumed = controller.observe(
                        len(chunk), seconds * overhead * len(chunk) / count,
                        evidence_seconds=evidence_seconds,
                    )
                    if consumed:
                        # At most one decision/cooldown window per successful
                        # candidate batch. Correlated temporal samples must not
                        # exhaust the retry cooldown within a single replay.
                        break
            else:
                controller.observe(len(chunk), clock() - started)
            if (
                controller.width != previous
                or previous_trial != (getattr(controller, "baseline", None) is not None)
            ):
                tuner.revision = getattr(tuner, "revision", 0) + 1

        start += len(chunk)


class SingleCoinScratchPolicy:
    """Keep one completed single-coin replay's scratch resident per suite.

    Extra history capacity comes from current free memory, with room retained
    for CPU validators and GPU output/workspace allocations. It never changes
    candles, history capacity, candidate parameters, or the dispatch work cap.
    """

    def __init__(self, hardware):
        self.hardware = hardware
        self.active = None
        self.torch = None
        self.warned = False

    def release(self):
        if self.active is None:
            return
        self.active.release_replay_scratch()
        self.active = None
        if self.hardware["device"] == "cuda":
            self.torch.cuda.empty_cache()
        else:
            self.torch.mps.empty_cache()

    def activate(self, proxy):
        if self.active is proxy.runner:
            return
        torch = proxy._torch
        if self.active is not None:
            # Host metrics from the previous group are complete before another
            # proxy can enter this method; no returned GPU views remain live.
            self.release()
        self.torch = torch
        runner = proxy.runner
        if self.hardware["device"] == "cuda":
            available = int(torch.cuda.mem_get_info()[0])
        else:
            from optimization.gpu.exact_autotune import resource_snapshot, psutil

            device_available = max(
                0,
                torch.mps.recommended_max_memory() * 0.7
                - torch.mps.driver_allocated_memory(),
            )
            try:
                host_available = resource_snapshot()["available"]
            except (OSError, psutil.Error) as error:
                if not self.warned:
                    logging.warning(
                        "GPU auto-tune host memory hint unavailable: %s", error
                    )
                    self.warned = True
                host_available = runner.hsl_scratch_budget_bytes / 0.35
            available = min(host_available, device_available)
        budget = max(
            runner._hsl_bytes_per_candidate(), min(2 * 1024**3, int(available * 0.35))
        )
        # Never shrink an allocation still owned by an active replay.
        runner.hsl_scratch_budget_bytes = budget
        ceiling = min(
            proxy.auto_dispatch_ceiling,
            max(1, budget // runner._hsl_bytes_per_candidate()),
        )
        if ceiling < proxy.auto_dispatch_ceiling:
            # Stable memory classes keep ordinary RAM noise from resetting
            # rolling evidence/cache identity on each suite pass.
            ceiling = 1 << (ceiling.bit_length() - 1)
        previous_ceiling = getattr(proxy, "dispatch_batch_size", None)
        proxy.dispatch_batch_size = ceiling
        tuner = getattr(proxy, "batch_tuner", None)
        if tuner is not None and previous_ceiling != ceiling:
            tuner.revision += 1
        self.active = runner
        memory_class = (ceiling, budget.bit_length())
        changed = getattr(proxy, "_scratch_memory_class", None) != memory_class
        proxy._scratch_memory_class = memory_class
        (logging.info if changed else logging.debug)(
            "GPU auto-tune memory | scratch_mib=%d batch_ceiling=%d",
            budget // 1024**2,
            proxy.dispatch_batch_size,
        )


def configure_batch_tuning(proxies, config, options):
    requested = (config.get("optimize", {}).get("gpu") or {}).get("batch_size")
    if options["tuning_mode"] == "off" or not is_auto(requested):
        return
    hardware = hardware_identity(proxies[0]._torch)
    opt = config.get("optimize", {})
    backtest = config.get("backtest", {})
    context = dict(
        bounds=opt.get("bounds"),
        workers=options["exact_workers"] or opt.get("n_cpus"),
        screening=options["screening"],
        suite=backtest.get("suite_enabled", False),
        scenarios=backtest.get("scenarios", []),
    )
    logging.info(
        "GPU auto-tune enabled | device=%s name=%s window=%d min_seconds=%.0f mode=%s",
        hardware["device"],
        hardware["name"],
        WINDOW,
        MIN_SECONDS,
        options["tuning_mode"],
    )
    scratch_proxies = [
        p
        for p in proxies
        if getattr(getattr(p, "runner", None), "hsl_capacity", 0)
        and hasattr(p.runner, "release_replay_scratch")
    ]
    scratch_policy = SingleCoinScratchPolicy(hardware)
    for proxy in proxies:
        runner = getattr(proxy, "runner", None)
        if proxy in scratch_proxies:
            proxy.auto_dispatch_ceiling = proxy.dispatch_batch_size
            proxy.scratch_policy = scratch_policy
        if runner is not None and hasattr(runner, "tuning_chunk_bars"):
            # Hidden scratch splits otherwise leave long full-history kernels
            # without temporal evidence. Use the actual allocation ceiling and
            # gather a bounded window from production replay, including one cold
            # dispatch and its final remainder. No calibration replay is added.
            runner.tuning_chunk_bars = (
                min(32768, max(1, runner.n // (WINDOW + 3)))
                if runner.n > 65536
                else None
            )
            if runner.hsl_capacity:
                scratch_limit = (
                    runner.hsl_scratch_budget_bytes // runner._hsl_bytes_per_candidate()
                )
                proxy.dispatch_batch_size = min(
                    proxy.dispatch_batch_size, max(1, scratch_limit)
                )
            runner.interrupt_check = getattr(proxy, "interrupt_check", lambda: None)
        proxy.batch_tuner = ProxyBatchTuner(
            proxy, mode=options["tuning_mode"], hardware=hardware, context=context
        )
