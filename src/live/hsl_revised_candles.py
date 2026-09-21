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
    pending_reads: int

    def payload(self):
        return [row for tape in self.tapes for row in tape.payload()]


_IO_ERRORS = (TimeoutError, OSError, OhlcvFetchError, ExchangeError)


class CandleReadBusy(TimeoutError):
    """An earlier resistant read owns this source or the reader's bounded capacity."""


class CandleSourceReader:
    """One reusable reader per manager, holding I/O tasks but no trading state.

    The owner must reuse this instance across scopes/cycles. A timed-out coroutine
    may resist cancellation; retain it until completion, refuse another read of
    that source, and bound total live reads across symbols. Late rows are never
    published as the timed-out call's result. Normal manager cache updates remain
    ordinary historical observations for a later capture.
    """
    def __init__(self, manager, *, max_pending_reads=8):
        if type(max_pending_reads) is not int or max_pending_reads < 1:
            raise ValueError("max_pending_reads must be a positive integer")
        self.manager = manager
        self.max_pending_reads = max_pending_reads
        self._pending = {}
        self._fatal = None

    @property
    def pending_reads(self):
        return sum(not task.done() for task in self._pending.values())

    def _finished(self, key, task):
        if self._pending.get(key) is task:
            del self._pending[key]
        # Retrieve every late exception. Programming failures remain fatal on
        # the next acquisition rather than becoming silent background errors.
        if not task.cancelled():
            error = task.exception()
            if error is not None and not isinstance(error, _IO_ERRORS) and self._fatal is None:
                self._fatal = error

    def _raise_late_failure(self):
        if self._fatal is not None:
            raise self._fatal

    async def _read(self, symbol, timeframe, remote, timeout, **kwargs):
        for key, task in list(self._pending.items()):
            if task.done():
                self._finished(key, task)
        self._raise_late_failure()
        key = (symbol, timeframe, remote)
        if key in self._pending or len(self._pending) >= self.max_pending_reads:
            raise CandleReadBusy("candle source read still pending")
        task = asyncio.create_task(self.manager.get_candles(
            symbol, timeframe=None if timeframe == "1m" else timeframe,
            allow_remote_fetch=remote, standardize=False, strict=False, **kwargs))
        self._pending[key] = task
        task.add_done_callback(lambda done: self._finished(key, done))
        try:
            done, _ = await asyncio.wait((task,), timeout=timeout)
        except asyncio.CancelledError:
            task.cancel()
            raise
        if not done:
            task.cancel()
            raise TimeoutError("candle source read deadline exceeded")
        return task.result()

    async def acquire(self, symbol, *, start, end, timeout_seconds, allow_remote_fetch=False):
        self._raise_late_failure()
        result = await _acquire_sources(self, symbol, start=start, end=end,
            timeout_seconds=timeout_seconds, allow_remote_fetch=allow_remote_fetch)
        self._raise_late_failure()
        return result


async def _acquire_sources(reader, symbol, *, start, end, timeout_seconds,
                           allow_remote_fetch):
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
    manager = reader.manager
    advertised = getattr(manager.exchange, "timeframes", None)
    supported = set(advertised) if isinstance(advertised, dict) and advertised else None
    ladder = [(tf, minutes) for tf, minutes in [("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60)]
              if supported is None or tf in supported]

    async def one(timeframe, minutes):
        failures = []
        # A 1m source names its open; its instantaneous close at the left edge
        # belongs to the inclusive window. Request that preceding source bucket.
        # Rust excludes earlier closes and coarse buckets crossing the edge.
        async def read(remote):
            return await reader._read(symbol, timeframe, remote, timeout_seconds,
                start_ts=max(0, start - 60_000) if minutes == 1 else start, end_ts=end)

        try:
            rows = await read(allow_remote_fetch)
        except _IO_ERRORS as exc:
            failures.append(Failure(timeframe, "fetch" if allow_remote_fetch else "cache",
                                    bounded_exception_type(exc)))
            if not allow_remote_fetch:
                return None, failures
            try:
                rows = await read(False)
            except _IO_ERRORS as exc:
                failures.append(Failure(timeframe, "cache", bounded_exception_type(exc)))
                return None, failures
        # Keep parsing outside the I/O exception boundary. Invalid producer shape
        # is a bug; missing historical numeric components are handled by capture.
        return capture_candles(rows, minutes=minutes, observed_at=manager._now_ms()), failures

    tasks = [asyncio.create_task(one(tf, minutes)) for tf, minutes in ladder]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        # Cancel acquisition wrappers without awaiting cancellation-resistant
        # manager reads. The reusable reader tracks those within its fixed cap.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return Sources(tuple(tape for tape, _ in results if tape is not None),
                   tuple(failure for _, failures in results for failure in failures),
                   reader.pending_reads)
