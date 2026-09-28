"""Bounded, task-local attribution for an EMA bundle; never a scheduling input."""

from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter


_current = ContextVar("ema_symbol_timing", default=None)
_SAMPLE_LIMIT = 8


@contextmanager
def measure(stage, *, requested_ms=None):
    row = _current.get()
    if row is None or not row["active"]:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        values = row["values"]
        values[stage + "_ms"] = values.get(stage + "_ms", 0.0) + (
            perf_counter() - started
        ) * 1000.0
        values[stage + "_calls"] = values.get(stage + "_calls", 0) + 1
        if requested_ms is not None:
            values[stage + "_requested_ms"] = (
                values.get(stage + "_requested_ms", 0.0) + requested_ms
            )


def timed(stage):
    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with measure(stage):
                return func(*args, **kwargs)

        return wrapped

    return decorate


def timed_async(stage):
    def decorate(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            with measure(stage):
                return await func(*args, **kwargs)

        return wrapped

    return decorate


def timed_async_entry(stage):
    """Time acquiring a context, excluding its body and release."""

    def decorate(func):
        @wraps(func)
        @asynccontextmanager
        async def wrapped(*args, **kwargs):
            async with AsyncExitStack() as stack:
                with measure(stage):
                    value = await stack.enter_async_context(func(*args, **kwargs))
                yield value

        return wrapped

    return decorate


class EmaBundleTimings:
    def __init__(self):
        self.started = perf_counter()
        self.symbol_count = 0
        self.symbol_elapsed_ms = 0.0
        self.stage_totals = {}
        self.slowest = []

    async def run_symbol(self, symbol, load):
        row = {"active": True, "values": {}}
        token = _current.set(row)
        started = perf_counter()
        try:
            return await load(symbol)
        finally:
            row["active"] = False
            _current.reset(token)
            elapsed_ms = (perf_counter() - started) * 1000.0
            self.symbol_count += 1
            self.symbol_elapsed_ms += elapsed_ms
            for key, value in row["values"].items():
                self.stage_totals[key] = self.stage_totals.get(key, 0) + value
            self.slowest.append({"symbol": symbol, "elapsed_ms": elapsed_ms, **row["values"]})
            self.slowest.sort(key=lambda item: (-item["elapsed_ms"], item["symbol"]))
            del self.slowest[_SAMPLE_LIMIT:]

    def summary(self):
        return {
            "elapsed_ms": (perf_counter() - self.started) * 1000.0,
            "symbol_count": self.symbol_count,
            "symbol_elapsed_ms": self.symbol_elapsed_ms,
            "stage_totals": dict(self.stage_totals),
            "slowest_symbols": [dict(row) for row in self.slowest],
            "symbols_omitted": max(0, self.symbol_count - len(self.slowest)),
        }
