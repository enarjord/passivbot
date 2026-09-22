from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Callable


OPTIMIZER_PROGRESS_INTERVAL_SECONDS = 300.0


class _ProgressHeartbeat:
    """Emit bounded optimizer progress while asynchronous work is still pending."""

    def __init__(self, label: str | None, total: int | None, interval_seconds: float):
        self.label = label
        self.total = total
        self.interval_seconds = max(0.0, float(interval_seconds))
        self.started_at = time.monotonic()
        self.last_log_at = self.started_at

    def maybe_log(self, *, completed: int, pending: int, submitted: int) -> None:
        if not self.label:
            return
        now = time.monotonic()
        if now - self.last_log_at < self.interval_seconds:
            return
        elapsed = max(0.0, now - self.started_at)
        rate = completed / elapsed if completed > 0 and elapsed > 0.0 else 0.0
        remaining = max(0, self.total - completed) if self.total is not None else None
        eta = remaining / rate if remaining is not None and rate > 0.0 else None
        total_text = str(self.total) if self.total is not None else "?"
        eta_text = f"{eta:.1f}s" if eta is not None else "unknown"
        logging.info(
            "%s progress | completed=%d/%s pending=%d submitted=%d elapsed=%.1fs rate=%.3f/s eta=%s",
            self.label,
            completed,
            total_text,
            pending,
            submitted,
            elapsed,
            rate,
            eta_text,
        )
        self.last_log_at = now


def seed_memory_debug_enabled() -> bool:
    return os.environ.get("PASSIVBOT_OPTIMIZE_SEED_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def current_rss_mib() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def approx_object_size(obj: Any, *, sample_size: int = 32) -> int:
    if obj is None:
        return 0
    if hasattr(obj, "nbytes"):
        try:
            return int(obj.nbytes)
        except Exception:
            pass

    try:
        size = int(sys.getsizeof(obj))
    except Exception:
        return 0

    if isinstance(obj, (list, tuple)):
        sample = list(obj[:sample_size])
        if sample:
            sample_total = sum(sys.getsizeof(item) for item in sample)
            size += int((sample_total / len(sample)) * len(obj))
    elif isinstance(obj, dict):
        items = list(obj.items())[:sample_size]
        if items:
            sample_total = sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in items)
            size += int((sample_total / len(items)) * len(obj))
    return size


def log_seed_memory(stage: str, **details: Any) -> None:
    if not seed_memory_debug_enabled():
        return
    parts = [f"stage={stage}"]
    rss_mib = current_rss_mib()
    if rss_mib is not None:
        parts.append(f"rss_mib={rss_mib:.1f}")
    for key, value in details.items():
        parts.append(f"{key}={value}")
    logging.info("[seed-mem] %s", " | ".join(parts))


def load_starting_individuals(
    *,
    starting_configs_path: str | None,
    population_size: int,
    get_starting_configs: Callable[[str | None], list],
    configs_to_individuals: Callable[[list, Any, int | None], list],
    iter_starting_configs: Callable[[str | None], Any] | None = None,
    configs_to_individuals_streaming: Callable[..., tuple[list, int]] | None = None,
    optimization_shape=None,
    bounds,
    sig_digits: int | None,
) -> list:
    if iter_starting_configs is not None and configs_to_individuals_streaming is not None:
        starting_individuals, starting_config_count = configs_to_individuals_streaming(
            iter_starting_configs(starting_configs_path),
            bounds,
            sig_digits,
            optimization_shape=optimization_shape,
        )
        if starting_config_count:
            logging.info(
                "Loaded %d starting configs before quantization (population size=%d)",
                starting_config_count,
                population_size,
            )
            log_seed_memory(
                "starting_configs_streamed",
                count=starting_config_count,
            )
        else:
            logging.info("No starting configs provided; population will be random-initialized")
        log_seed_memory(
            "starting_individuals_built",
            count=len(starting_individuals),
            approx_bytes=approx_object_size(starting_individuals),
        )
        return starting_individuals

    starting_configs = get_starting_configs(starting_configs_path)
    if starting_configs:
        logging.info(
            "Loaded %d starting configs before quantization (population size=%d)",
            len(starting_configs),
            population_size,
        )
        log_seed_memory(
            "starting_configs_loaded",
            count=len(starting_configs),
            approx_bytes=approx_object_size(starting_configs),
        )
    else:
        logging.info("No starting configs provided; population will be random-initialized")
    if optimization_shape is None:
        starting_individuals = configs_to_individuals(starting_configs, bounds, sig_digits)
    else:
        starting_individuals = configs_to_individuals(
            starting_configs,
            bounds,
            sig_digits,
            optimization_shape=optimization_shape,
        )
    log_seed_memory(
        "starting_individuals_built",
        count=len(starting_individuals),
        approx_bytes=approx_object_size(starting_individuals),
    )
    return starting_individuals


def cancel_pending_async_results(pending: dict) -> None:
    for res in pending:
        try:
            res.cancel()
        except Exception:
            pass


def drain_async_results(
    pending: dict,
    *,
    poll_interval_seconds: float = 0.05,
    on_result: Callable[[Any, Any], None],
    on_interrupt: Callable[[dict], None] | None = None,
    pending_health_check: Callable[[], None] | None = None,
    progress_label: str | None = None,
    progress_interval_seconds: float = OPTIMIZER_PROGRESS_INTERVAL_SECONDS,
) -> int:
    completed = 0
    total = len(pending)
    heartbeat = _ProgressHeartbeat(progress_label, total, progress_interval_seconds)
    try:
        while pending:
            if pending_health_check is not None:
                pending_health_check()
            heartbeat.maybe_log(completed=completed, pending=len(pending), submitted=total)
            ready = [res for res in pending if res.ready()]
            if not ready:
                time.sleep(max(0.0, float(poll_interval_seconds)))
                continue
            for res in ready:
                context = pending.pop(res)
                payload = res.get()
                on_result(context, payload)
                completed += 1
    except KeyboardInterrupt:
        if on_interrupt is not None:
            on_interrupt(pending)
        raise
    return completed


def stream_async_results(
    items,
    *,
    submit: Callable[[Any], tuple[Any, Any]],
    on_result: Callable[[Any, Any], None],
    max_pending: int | None = None,
    poll_interval_seconds: float = 0.05,
    on_interrupt: Callable[[dict], None] | None = None,
    pending_health_check: Callable[[], None] | None = None,
    progress_label: str | None = None,
    progress_total: int | None = None,
    progress_interval_seconds: float = OPTIMIZER_PROGRESS_INTERVAL_SECONDS,
) -> int:
    max_pending = None if max_pending is None else max(1, int(max_pending))
    iterator = iter(items)
    pending: dict[Any, Any] = {}
    completed = 0
    submitted = 0
    exhausted = False
    heartbeat = _ProgressHeartbeat(progress_label, progress_total, progress_interval_seconds)
    try:
        while pending or not exhausted:
            while not exhausted and (max_pending is None or len(pending) < max_pending):
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                res, context = submit(item)
                pending[res] = context
                submitted += 1

            if not pending:
                continue

            if pending_health_check is not None:
                pending_health_check()
            heartbeat.maybe_log(
                completed=completed,
                pending=len(pending),
                submitted=submitted,
            )
            ready = [res for res in pending if res.ready()]
            if not ready:
                time.sleep(max(0.0, float(poll_interval_seconds)))
                continue

            for res in ready:
                context = pending.pop(res)
                payload = res.get()
                on_result(context, payload)
                completed += 1
    except KeyboardInterrupt:
        if on_interrupt is not None:
            on_interrupt(pending)
        raise
    return completed
