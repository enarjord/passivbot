"""Bounded source-resolution acquisition for revised HSL's Rust estimator.

No resampling, price carrying or risk decisions occur here. Runtime activation is
separate; the caller owns background scheduling and current-state revalidation.
"""
import asyncio
from dataclasses import dataclass
import math

from ccxt.base.errors import BaseError as ExchangeError
from candlestick_manager import OhlcvFetchError
from live.diagnostic_safety import bounded_exception_type
from live.hsl_revised_inputs import CandleTape, capture_candles


@dataclass(frozen=True)
class Failure:
    timeframe: str
    stage: str
    error_type: str


@dataclass(frozen=True)
class Sources:
    tapes: tuple[CandleTape, ...]
    failures: tuple[Failure, ...]

    def payload(self):
        return [row for tape in self.tapes for row in tape.payload()]


async def acquire_sources(manager, symbol, *, start, end, timeout_seconds,
                          allow_remote_fetch=False):
    """Read the supported source resolutions across the whole requested window.

    Each independent resolution gets a bounded manager read. Expected I/O failure
    attempts a bounded cache-only read, preserving available history despite a
    transient fetch failure. Programming errors and task cancellation propagate.
    Exceptions are diagnostic type only; they never enter the source candle data.
    The manager may retain verified no-trade rows under its existing contract.
    Rust alone decides source completeness, causal clipping and interpolation.
    """
    if (type(start) is not int or type(end) is not int or start < 0 or end < start
            or end > 2**63-1 or type(allow_remote_fetch) is not bool
            or end - start > 90 * 86_400_000
            or not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("invalid revised HSL candle acquisition bounds")
    advertised = getattr(manager.exchange, "timeframes", None)
    supported = set(advertised) if isinstance(advertised, dict) and advertised else None
    ladder = [(tf, minutes) for tf, minutes in [("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60)]
              if supported is None or tf in supported]
    expected_errors = (TimeoutError, OSError, OhlcvFetchError, ExchangeError)

    async def one(timeframe, minutes):
        failures = []
        # A 1m source names its open; its instantaneous close at the left edge
        # belongs to the inclusive window. Request that preceding source bucket.
        # Rust excludes earlier closes and coarse buckets crossing the edge.
        async def read(remote):
            return await asyncio.wait_for(manager.get_candles(
                symbol, start_ts=max(0, start - 60_000) if minutes == 1 else start,
                end_ts=end, strict=False,
                timeframe=None if timeframe == "1m" else timeframe,
                allow_remote_fetch=remote, standardize=False,
            ), timeout=timeout_seconds)

        try:
            rows = await read(allow_remote_fetch)
        except expected_errors as exc:
            failures.append(Failure(timeframe, "fetch" if allow_remote_fetch else "cache",
                                    bounded_exception_type(exc)))
            if not allow_remote_fetch:
                return None, failures
            try:
                rows = await read(False)
            except expected_errors as exc:
                failures.append(Failure(timeframe, "cache", bounded_exception_type(exc)))
                return None, failures
        # Keep parsing outside the I/O exception boundary. Invalid producer shape
        # is a bug; missing historical numeric components are handled by capture.
        return capture_candles(rows, minutes=minutes, observed_at=manager._now_ms()), failures

    tasks = [asyncio.create_task(one(tf, minutes)) for tf, minutes in ladder]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        # Structured cleanup only: never leave fetches running after a fatal
        # producer error or cancellation, and never convert it to healthy data.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return Sources(tuple(tape for tape, _ in results if tape is not None),
                   tuple(failure for _, failures in results for failure in failures))
