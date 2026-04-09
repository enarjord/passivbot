from __future__ import annotations

import logging
import time
from typing import Any, Callable


def load_starting_individuals(
    *,
    starting_configs_path: str | None,
    population_size: int,
    get_starting_configs: Callable[[str | None], list],
    configs_to_individuals: Callable[[list, Any, int | None], list],
    optimization_shape=None,
    bounds,
    sig_digits: int | None,
) -> list:
    starting_configs = get_starting_configs(starting_configs_path)
    if starting_configs:
        logging.info(
            "Loaded %d starting configs before quantization (population size=%d)",
            len(starting_configs),
            population_size,
        )
    else:
        logging.info("No starting configs provided; population will be random-initialized")
    if optimization_shape is None:
        return configs_to_individuals(starting_configs, bounds, sig_digits)
    return configs_to_individuals(
        starting_configs,
        bounds,
        sig_digits,
        optimization_shape=optimization_shape,
    )


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
