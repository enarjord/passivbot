"""Best-effort WebSocket candle ingestion for flat forager candidates.

Proven-final public 1m rows are persisted through CandlestickManager's
canonical candle path. REST remains the complete fallback for startup basis,
historical coverage, gaps, prolonged silence, reconnect recovery, and periodic
integrity audits.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from config.access import get_optional_live_value
from live.diagnostic_safety import bounded_exception_type


_SUBSCRIPTION_RECONCILE_SECONDS = 5.0
_MAX_RECONNECT_DELAY_SECONDS = 30.0
_FAILURES_BEFORE_COOLDOWN = 5
_UNSTABLE_COOLDOWN_SECONDS = 300.0
_UNWATCH_TIMEOUT_SECONDS = 2.0
_UNWATCH_MANY_TIMEOUT_SECONDS = 3.0
_WATCHER_RETIRE_GRACE_SECONDS = 3.0
_WATCHER_CANCEL_GRACE_SECONDS = 1.0


def reconnect_delay_seconds(consecutive_failures: int) -> float:
    """Return bounded retry delay, cooling persistently unstable streams."""
    failures = max(1, int(consecutive_failures))
    if failures >= _FAILURES_BEFORE_COOLDOWN:
        return _UNSTABLE_COOLDOWN_SECONDS
    return min(
        _MAX_RECONNECT_DELAY_SECONDS,
        float(2 ** max(0, failures - 1)),
    )


def forager_ws_candles_enabled(bot: Any) -> bool:
    """Return whether the configured transport can maintain forager WS tasks.

    The reconciler intentionally starts even before a side enters forager mode;
    ``desired_forager_ws_symbols`` keeps its subscription set empty until then.
    """
    if not bool(
        get_optional_live_value(
            getattr(bot, "config", {}) or {},
            "enable_forager_ws_candles",
            True,
        )
    ):
        return False
    if not bool(getattr(bot, "ws_enabled", False)):
        return False
    try:
        if int(
            get_optional_live_value(
                getattr(bot, "config", {}) or {},
                "max_ohlcv_fetches_per_minute",
                30,
            )
            or 0
        ) <= 0:
            return False
    except (TypeError, ValueError):
        return False
    ccp = getattr(bot, "ccp", None)
    if ccp is None:
        return False
    capabilities = getattr(ccp, "has", {}) or {}
    return bool(capabilities.get("watchOHLCV")) and callable(
        getattr(ccp, "watch_ohlcv", None)
    )


def desired_forager_ws_symbols(bot: Any) -> set[str]:
    """Return flat candidate symbols eligible for a public 1m WS subscription."""
    if not forager_ws_candles_enabled(bot):
        return set()
    approved = getattr(bot, "approved_coins_minus_ignored_coins", {}) or {}
    is_forager_mode = getattr(bot, "is_forager_mode", None)
    candidates = set()
    for pside in ("long", "short"):
        if not callable(is_forager_mode) or not bool(is_forager_mode(pside)):
            continue
        candidates.update(
            str(symbol)
            for symbol in (approved.get(pside, set()) or set())
            if symbol
        )
    urgent_fn = getattr(bot, "_urgent_active_candle_symbols", None)
    urgent = set(urgent_fn() or []) if callable(urgent_fn) else set()
    return candidates - urgent


async def _sleep_unless_shutdown(bot: Any, delay_s: float, *, stage: str) -> None:
    sleep_fn = getattr(bot, "_sleep_unless_shutdown", None)
    if callable(sleep_fn):
        await sleep_fn(delay_s, stage=stage)
    else:
        await asyncio.sleep(delay_s)


def _abandoned_unsubscribe_tasks(bot: Any) -> set[asyncio.Future]:
    tasks = getattr(bot, "_forager_ws_abandoned_unsubscribe_tasks", None)
    if not isinstance(tasks, set):
        tasks = set()
        bot._forager_ws_abandoned_unsubscribe_tasks = tasks
    return tasks


def _consume_abandoned_unsubscribe(bot: Any, task: asyncio.Future) -> None:
    _abandoned_unsubscribe_tasks(bot).discard(task)
    if task.done() and not task.cancelled():
        try:
            task.exception()
        except BaseException:
            pass


def _cancel_and_track_unsubscribe(bot: Any, task: asyncio.Future) -> None:
    task.cancel()
    abandoned = _abandoned_unsubscribe_tasks(bot)
    abandoned.add(task)
    task.add_done_callback(
        lambda done_task: _consume_abandoned_unsubscribe(bot, done_task)
    )


async def _run_unsubscribe_with_hard_timeout(
    bot: Any, awaitable: Any, *, timeout: float
) -> bool:
    """Bound connector unsubscribe without awaiting cancellation-resistant work."""
    try:
        task = asyncio.ensure_future(awaitable)
    except Exception:
        return False
    try:
        done, _pending = await asyncio.wait(
            {task}, timeout=max(0.0, float(timeout))
        )
    except asyncio.CancelledError:
        _cancel_and_track_unsubscribe(bot, task)
        raise
    if task not in done:
        _cancel_and_track_unsubscribe(bot, task)
        return False
    try:
        task.result()
        return True
    except asyncio.CancelledError:
        return False
    except Exception:
        return False


async def _best_effort_unwatch(bot: Any, symbol: str) -> bool:
    ccp = getattr(bot, "ccp", None)
    unwatch = getattr(ccp, "un_watch_ohlcv", None)
    if not callable(unwatch):
        return False
    try:
        awaitable = unwatch(symbol, "1m")
    except asyncio.CancelledError:
        raise
    except Exception:
        return False
    return await _run_unsubscribe_with_hard_timeout(
        bot,
        awaitable,
        timeout=_UNWATCH_TIMEOUT_SECONDS,
    )


async def _best_effort_unwatch_many(bot: Any, symbols: list[str]) -> bool:
    ccp = getattr(bot, "ccp", None)
    unwatch = getattr(ccp, "un_watch_ohlcv_for_symbols", None)
    if not callable(unwatch) or not symbols:
        return False
    subscriptions = [[symbol, "1m"] for symbol in symbols]
    try:
        awaitable = unwatch(subscriptions)
    except asyncio.CancelledError:
        raise
    except Exception:
        return False
    return await _run_unsubscribe_with_hard_timeout(
        bot,
        awaitable,
        timeout=_UNWATCH_MANY_TIMEOUT_SECONDS,
    )


def _retiring_watcher_tasks(bot: Any) -> dict[str, set[asyncio.Task]]:
    retiring = getattr(bot, "_forager_ws_retiring_watchers", None)
    if not isinstance(retiring, dict):
        retiring = {}
        bot._forager_ws_retiring_watchers = retiring
    return retiring


def _watcher_is_retiring(bot: Any, symbol: str, task: asyncio.Task | None) -> bool:
    if task is None:
        return False
    return task in _retiring_watcher_tasks(bot).get(symbol, set())


def _mark_watcher_retiring(bot: Any, symbol: str, task: asyncio.Task) -> None:
    _retiring_watcher_tasks(bot).setdefault(symbol, set()).add(task)


def _clear_watcher_retiring(bot: Any, symbol: str, task: asyncio.Task) -> None:
    retiring = _retiring_watcher_tasks(bot)
    symbol_tasks = retiring.get(symbol)
    if not symbol_tasks:
        return
    symbol_tasks.discard(task)
    if not symbol_tasks:
        retiring.pop(symbol, None)


async def _retire_watcher(bot: Any, symbol: str, task: asyncio.Task) -> None:
    """Unsubscribe while the watcher still owns CCXT's subscription future.

    CCXT Pro uses ``UnsubscribeError`` to wake outstanding ``watch_ohlcv``
    futures after an unsubscribe acknowledgement.  Cancelling Passivbot's
    awaiting task first leaves that future orphaned, which asyncio later logs
    as ``Future exception was never retrieved``.  Mark retirement first, ask
    CCXT to unsubscribe, and let the watcher consume the wake-up exception.
    Cancellation remains the bounded fallback for transports which do not wake
    the watcher.
    """
    await _retire_watchers(bot, {symbol: task})


async def _retire_watchers(bot: Any, tasks: dict[str, asyncio.Task]) -> None:
    """Retire a watcher set through one bulk unsubscribe where available."""
    active = {symbol: task for symbol, task in tasks.items() if task is not None}
    if not active:
        return
    symbols = sorted(active)
    for symbol, task in active.items():
        _mark_watcher_retiring(bot, symbol, task)
    abandoned_tasks: dict[str, asyncio.Task] = {}

    def consume_done(done_tasks) -> None:
        for done_task in done_tasks:
            if done_task.done() and not done_task.cancelled():
                try:
                    done_task.exception()
                except BaseException:
                    pass

    try:
        unwatched = await _best_effort_unwatch_many(bot, symbols)
        if not unwatched:
            await asyncio.gather(
                *(_best_effort_unwatch(bot, symbol) for symbol in symbols),
                return_exceptions=True,
            )
        consume_done(task for task in active.values() if task.done())
        pending = [task for task in active.values() if not task.done()]
        if pending:
            done, still_pending = await asyncio.wait(
                pending, timeout=_WATCHER_RETIRE_GRACE_SECONDS
            )
            consume_done(done)
            for task in still_pending:
                task.cancel()
            if still_pending:
                cancelled, resistant = await asyncio.wait(
                    still_pending, timeout=_WATCHER_CANCEL_GRACE_SECONDS
                )
                consume_done(cancelled)
                resistant_set = set(resistant)
                abandoned_tasks = {
                    symbol: task
                    for symbol, task in active.items()
                    if task in resistant_set
                }
                if abandoned_tasks:
                    logging.warning(
                        "[candle] websocket watcher cancellation grace expired "
                        "| symbols=%d action=abandon_pending",
                        len(abandoned_tasks),
                    )
                    for symbol, task in abandoned_tasks.items():
                        task.add_done_callback(
                            lambda done_task, symbol=symbol: _clear_watcher_retiring(
                                bot, symbol, done_task
                            )
                        )
    finally:
        for symbol, task in active.items():
            if abandoned_tasks.get(symbol) is not task:
                _clear_watcher_retiring(bot, symbol, task)


async def watch_forager_ws_symbol(bot: Any, symbol: str) -> None:
    """Watch one symbol and pass only validated finalized rows to the manager."""
    consecutive_failures = 0
    watcher_task = asyncio.current_task()
    try:
        while not bool(getattr(bot, "stop_signal_received", False)):
            try:
                rows = await bot.ccp.watch_ohlcv(symbol, "1m")
                if _watcher_is_retiring(bot, symbol, watcher_task):
                    break
                ingest = getattr(bot.cm, "ingest_live_ws_ohlcv", None)
                if callable(ingest):
                    # Some venues return hundreds of cached rows on every
                    # update. Only the current/finalized tail can contribute
                    # to canonical ingestion; reconnect gaps deliberately use
                    # REST instead of replaying a large WS snapshot.
                    tail_rows = rows[-3:] if isinstance(rows, list) else rows
                    result = ingest(symbol, tail_rows)
                    if inspect.isawaitable(result):
                        await result
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if _watcher_is_retiring(bot, symbol, watcher_task):
                    break
                consecutive_failures += 1
                delay_s = reconnect_delay_seconds(consecutive_failures)
                clear_state = getattr(bot.cm, "clear_live_ws_ohlcv_state", None)
                if callable(clear_state):
                    clear_state(symbol)
                warning_state = getattr(bot, "_forager_ws_candle_warning_ms", None)
                if not isinstance(warning_state, dict):
                    warning_state = {}
                now_ms_fn = getattr(bot, "get_exchange_time", None)
                try:
                    now_ms = int(now_ms_fn()) if callable(now_ms_fn) else 0
                except Exception:
                    now_ms = 0
                last_warning_ms = int(warning_state.get(symbol, 0) or 0)
                if now_ms <= 0 or now_ms - last_warning_ms >= 300_000:
                    logging.warning(
                        "[candle] forager websocket unavailable | symbol=%s "
                        "error_type=%s retry=%.1fs action=rest_fallback",
                        symbol,
                        bounded_exception_type(exc),
                        delay_s,
                    )
                    if now_ms > 0:
                        warning_state[symbol] = now_ms
                        bot._forager_ws_candle_warning_ms = warning_state
                await _sleep_unless_shutdown(
                    bot,
                    delay_s,
                    stage="forager_ws_candle_reconnect",
                )
    finally:
        if not _watcher_is_retiring(bot, symbol, watcher_task):
            await _best_effort_unwatch(bot, symbol)


async def reconcile_forager_ws_tasks(
    bot: Any, tasks: dict[str, asyncio.Task]
) -> tuple[set[str], set[str]]:
    """Make per-symbol watcher tasks match the current flat forager universe."""
    desired = desired_forager_ws_symbols(bot)
    existing = set(tasks)
    removed = existing - desired
    added = desired - existing
    removed_tasks = {symbol: tasks[symbol] for symbol in sorted(removed)}
    if removed_tasks:
        await _retire_watchers(bot, removed_tasks)
        for symbol, retired_task in removed_tasks.items():
            if tasks.get(symbol) is retired_task:
                tasks.pop(symbol, None)
    clear_state = getattr(bot.cm, "clear_live_ws_ohlcv_state", None)
    if callable(clear_state):
        for symbol in sorted(removed):
            clear_state(symbol)
    for symbol in sorted(added):
        tasks[symbol] = asyncio.create_task(
            watch_forager_ws_symbol(bot, symbol),
            name=f"forager_ws_1m:{symbol}",
        )
    return added, removed


async def maintain_forager_ws_candles(bot: Any) -> None:
    """Maintain dynamic flat-candidate subscriptions until bot shutdown."""
    tasks: dict[str, asyncio.Task] = {}
    bot.WS_ohlcvs_1m_tasks = tasks
    logging.info(
        "[candle] starting finalized 1m websocket ingestion for flat forager candidates "
        "| persist=true rest_fallback=true"
    )
    try:
        while not bool(getattr(bot, "stop_signal_received", False)):
            await reconcile_forager_ws_tasks(bot, tasks)
            await _sleep_unless_shutdown(
                bot,
                _SUBSCRIPTION_RECONCILE_SECONDS,
                stage="forager_ws_candle_subscriptions",
            )
    except asyncio.CancelledError:
        raise
    finally:
        await _retire_watchers(bot, dict(tasks))
        tasks.clear()
