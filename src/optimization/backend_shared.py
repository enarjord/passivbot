from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Callable


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
) -> int:
    completed = 0
    try:
        while pending:
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
) -> int:
    max_pending = None if max_pending is None else max(1, int(max_pending))
    iterator = iter(items)
    pending: dict[Any, Any] = {}
    completed = 0
    exhausted = False
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

            if not pending:
                continue

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
