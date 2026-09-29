"""Bounded, evidence-driven batch sizing using completed proxy work only.

This is execution policy, never evolutionary or drift policy. Cache loss changes
performance only; exceptions from evaluation are deliberately not intercepted.
"""

from __future__ import annotations

from collections import OrderedDict, deque
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

WINDOW = 24
MIN_SECONDS = 30.0
CACHE_VERSION = 1


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
    """Slow hill climbing with a rolling median, warm-up exclusion and cooldown."""

    def __init__(self, ceiling, initial, *, save=lambda *args: None, can_grow=lambda: True):
        self.ceiling = max(1, int(ceiling))
        self.width = max(1, min(int(initial), self.ceiling))
        self.save = save
        self.can_grow = can_grow
        self.samples = deque(maxlen=WINDOW)
        self.seconds = 0.0
        self.seen = set()
        self.baseline = None
        self.direction = 1
        self.cooldown = 0

    def observe(self, count, seconds):
        # Remainders and cold first use of each allocation shape are not evidence.
        if count != self.width or not math.isfinite(seconds) or seconds <= 0:
            return
        if self.width not in self.seen:
            self.seen.add(self.width)
            return
        self.samples.append(count / seconds)
        self.seconds += seconds
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
                logging.info(
                    "GPU auto-tune accepted | batch=%d candidates/s=%.3f", self.width, rate
                )
                self.save(self.width, rate, evidence_seconds)
                self.cooldown = 1
            else:
                logging.info(
                    "GPU auto-tune retained | batch=%d trial_batch=%d", old_width, self.width
                )
                self.width = old_width
                self.direction *= -1
                self.cooldown = 3
            return
        self.save(self.width, rate, evidence_seconds)
        if self.cooldown:
            self.cooldown -= 1
            return
        trial = (
            min(self.ceiling, self.width * 2) if self.direction > 0 else max(1, self.width // 2)
        )
        if trial == self.width:
            self.direction *= -1
            self.cooldown = 1
            return
        if trial > self.width and not self.can_grow():
            self.cooldown = 1
            return
        self.baseline = (self.width, rate)
        self.width = trial
        logging.info(
            "GPU auto-tune trial | batch=%d previous=%d rolling_candidates/s=%.3f",
            trial,
            self.baseline[0],
            rate,
        )


class ProxyBatchTuner:
    """Per-proxy workload classes; cache contains no configs or market data."""

    def __init__(self, proxy, *, mode, hardware, context, cache_dir=None):
        self.proxy = proxy
        self.mode = mode
        self.hardware = hardware
        self.identity = _digest(
            dict(
                hardware=hardware,
                context=context,
                execution=workload_contract(proxy.checkpoint_contract),
                overrides=getattr(proxy, "coin_override_contract", {}),
                metrics=sorted(proxy.needed_metrics),
            )
        )
        self.cache_dir = Path(cache_dir or "caches/gpu_autotune")
        self.controllers = OrderedDict()
        self.cache_warning = False
        self.cache_write_disabled = False

    def _warn(self, error):
        if not self.cache_warning:
            logging.warning(
                "GPU auto-tune cache unavailable; continuing with in-memory tuning: %s", error
            )
            self.cache_warning = True

    def _headroom(self):
        torch = self.proxy._torch
        if self.hardware["device"] == "cuda":
            free, total = torch.cuda.mem_get_info()
            return free >= max(256 * 1024**2, total * 0.2)
        return torch.mps.driver_allocated_memory() < torch.mps.recommended_max_memory() * 0.7

    def controller(self, ceiling, demand, end_step):
        ceiling = min(ceiling, demand)
        key = _digest(
            [
                CACHE_VERSION,
                self.identity,
                ceiling,
                demand,
                end_step,
                self.proxy.max_dispatch_candidate_bars,
            ]
        )
        if key in self.controllers:
            self.controllers.move_to_end(key)
            return self.controllers[key]
        # Bound the initial allocation. The existing dispatch plan is always the cap.
        initial = min(128, ceiling)
        source = "conservative"
        path = self.cache_dir / (key + ".json")
        if self.mode != "refresh":
            try:
                record = json.loads(path.read_text())
                width = record["batch_size"]
                if (
                    record["version"] != CACHE_VERSION
                    or type(width) is not int
                    or not 1 <= width <= ceiling
                    or not math.isfinite(record["candidates_per_second"])
                    or record["candidates_per_second"] <= 0
                ):
                    raise ValueError("invalid batch calibration")
                if width <= initial or self._headroom():
                    initial = width
                    source = "cache"
            except FileNotFoundError:
                pass
            except (OSError, ValueError, KeyError, TypeError) as error:
                self._warn(error)

        def save(width, rate, seconds):
            if self.cache_write_disabled:
                return
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
                            batch_size=width,
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

        logging.info(
            "GPU auto-tune starting | batch=%d ceiling=%d source=%s", initial, ceiling, source
        )
        controller = BatchController(ceiling, initial, save=save, can_grow=self._headroom)
        self.controllers[key] = controller
        while len(self.controllers) > 8:
            self.controllers.popitem(last=False)
        return controller


def proxy_batches(proxy, candidates, ceiling, *, end_step=None, clock=time.perf_counter):
    """Measure through the caller's completed host results, without extra GPU sync.

    A failed/interrupted yield never produces a sample. Shape changes happen only
    after the entire previous candidate replay, reductions and host copy finish.
    """
    tuner = getattr(proxy, "batch_tuner", None)
    controller = (
        tuner.controller(ceiling, len(candidates), end_step) if tuner and candidates else None
    )
    start = 0
    while start < len(candidates):
        width = controller.width if controller is not None else ceiling
        chunk = candidates[start : start + width]
        started = clock() if controller is not None else 0.0
        yield start, chunk
        if controller is not None:
            controller.observe(len(chunk), clock() - started)
        start += len(chunk)


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
    for proxy in proxies:
        proxy.batch_tuner = ProxyBatchTuner(
            proxy, mode=options["tuning_mode"], hardware=hardware, context=context
        )
