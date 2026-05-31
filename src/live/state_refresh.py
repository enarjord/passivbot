from __future__ import annotations

import asyncio
import logging
import sys

from config.access import get_optional_config_value
from utils import ts_to_date, utc_ms


def _utc_ms() -> int:
    passivbot_module = sys.modules.get("passivbot")
    time_fn = getattr(passivbot_module, "utc_ms", None)
    if callable(time_fn):
        return int(time_fn())
    return int(utc_ms())


async def refresh_authoritative_state(bot) -> bool:
    """Refresh authoritative account state before planning/execution."""
    if bot.stop_signal_received:
        return False
    bot._begin_authoritative_refresh_epoch()
    return await bot._refresh_authoritative_state_staged()


async def refresh_authoritative_state_staged(bot) -> bool:
    """Refresh live account state through the staged authoritative cohort."""
    bot._last_authoritative_block_reason = None
    bot._last_authoritative_pending_pnl_count = 0
    plan = bot._authoritative_staged_refresh_plan()
    snapshot = await bot._fetch_authoritative_state_staged_snapshot(plan)
    fetched_balance = snapshot.get("balance")
    fetched_positions = snapshot.get("positions")
    fetched_open_orders = snapshot.get("open_orders")
    pnls_ok = snapshot.get("pnls_ok", True)

    if "positions" in plan and fetched_positions in [None, False]:
        return False
    if "balance" in plan and fetched_balance in [None, False]:
        return False
    if "open_orders" in plan and fetched_open_orders in [None, False]:
        return False
    if "fills" in plan and not pnls_ok:
        bot._last_authoritative_block_reason = "pending_pnl"
        bot._last_authoritative_pending_pnl_count = int(
            snapshot.get("pending_pnl_count", 0) or 0
        )
        return False
    prepared_balance_snapshot = None
    if "balance" in plan:
        prepared_balance_snapshot = bot._prepare_balance_snapshot(fetched_balance)
        if prepared_balance_snapshot is None:
            return False

    fetched_positions_old = None
    fetched_positions_new = None
    if "open_orders" in plan:
        open_orders_ok = await bot._apply_open_orders_snapshot(
            fetched_open_orders,
            allow_followup_positions_refresh=False,
            reconcile_balance="balance" not in plan,
        )
        if not open_orders_ok:
            return False
    if "positions" in plan:
        fetched_positions_old, fetched_positions_new = bot._apply_positions_snapshot(
            fetched_positions
        )
    if "balance" in plan:
        bot._commit_balance_snapshot(prepared_balance_snapshot)
    if "positions" in plan:
        bot._record_authoritative_surface(
            "positions",
            bot._positions_signature(fetched_positions_new),
        )
        bot._update_entry_cooldown_position_delta_guard(
            sorted(bot.positions), now_ms=int(bot.get_exchange_time())
        )
    if "balance" in plan:
        if "open_orders" in plan:
            bot._reconcile_balance_after_staged_refresh()
            balance_source = bot._staged_balance_update_source()
        else:
            bot._reconcile_balance_after_positions_and_balance_refresh()
            balance_source = "REST"
        bot._record_authoritative_surface(
            "balance", round(float(bot.get_hysteresis_snapped_balance()), 12)
        )
        try:
            await bot.log_position_changes(fetched_positions_old, fetched_positions_new)
        except Exception as e:
            if not bot._log_noncritical_market_snapshot_error(
                "position-change diagnostics", e
            ):
                logging.error(f"error logging position changes {e}")
        await bot.handle_balance_update(source=balance_source)
    bot._finalize_authoritative_refresh_consistency(plan)
    return True


async def capture_balance_staged_snapshot(bot) -> tuple[object, float]:
    """Fetch a single balance payload and its normalized value for staged refresh."""
    if hasattr(bot, "capture_balance_snapshot"):
        return await bot.capture_balance_snapshot()
    balance = await bot.fetch_balance()
    return None, balance


async def capture_positions_staged_snapshot(bot) -> tuple[object, list[dict]]:
    """Fetch a single positions payload and its normalized value for staged refresh."""
    if hasattr(bot, "capture_positions_snapshot"):
        return await bot.capture_positions_snapshot()
    positions = await bot.fetch_positions()
    return None, positions


def authoritative_staged_refresh_plan(bot) -> set[str]:
    """Return the minimal staged authoritative surfaces needed this cycle."""
    pending = set(getattr(bot, "_authoritative_pending_confirmations", {}) or {})
    if pending == {"open_orders"}:
        plan = {"open_orders"}
        bot._authoritative_refresh_plan_surfaces = set(plan)
        return plan
    plan = {"balance", "positions", "open_orders", "fills"}
    if "fills" not in pending:
        if not bot._staged_fills_refresh_due():
            plan.discard("fills")
            logging.debug("[state] staged fills refresh deferred until next minute boundary")
        elif bot._staged_fills_can_prefetch_routine() and (
            bot._schedule_routine_fill_refresh_prefetch(reason="minute_boundary")
        ):
            plan.discard("fills")
            logging.debug("[state] staged routine fills refresh scheduled in background")
    bot._authoritative_refresh_plan_surfaces = set(plan)
    return plan


def staged_fills_refresh_due(bot) -> bool:
    """Return whether routine staged refresh must fetch fill events this cycle."""
    ledger = getattr(bot, "freshness_ledger", None)
    if ledger is None:
        return True
    fills_state = getattr(ledger, "surfaces", {}).get("fills")
    last_updated_ms = int(getattr(fills_state, "updated_ms", 0) or 0)
    if last_updated_ms <= 0:
        return True
    return int(_utc_ms()) // 60_000 > last_updated_ms // 60_000


def staged_fills_can_prefetch_routine(bot) -> bool:
    """Return whether routine fills may refresh outside the blocking staged cohort."""
    ledger = getattr(bot, "freshness_ledger", None)
    if ledger is None or int(ledger.surface_epoch("fills") or 0) < 1:
        return False
    last_updated_ms = int(ledger.surface_updated_ms("fills") or 0)
    if last_updated_ms <= 0:
        return False
    max_staleness_ms = 3 * 60 * 1000
    return int(_utc_ms()) - last_updated_ms <= max_staleness_ms


def schedule_routine_fill_refresh_prefetch(bot, *, reason: str) -> bool:
    """Schedule one routine fills refresh if none is already running."""
    if bot._shutdown_requested():
        return False
    if not hasattr(bot, "maintainers") or not isinstance(bot.maintainers, dict):
        bot.maintainers = {}
    key = "routine_fill_refresh"
    existing = bot.maintainers.get(key)
    if existing is not None and not existing.done():
        return True
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    bot.maintainers[key] = loop.create_task(
        bot._routine_fill_refresh_prefetch_task(reason=reason)
    )
    return True


async def routine_fill_refresh_prefetch_task(bot, *, reason: str) -> None:
    """Refresh routine fills outside the critical account-state refresh path."""
    try:
        started_ms = _utc_ms()
        ok = await bot.update_pnls(source=f"routine_prefetch:{reason}")
        elapsed_ms = int(max(0, _utc_ms() - started_ms))
        logging.debug(
            "[fills] routine prefetch complete | reason=%s ok=%s elapsed=%dms",
            reason,
            ok,
            elapsed_ms,
        )
    except asyncio.CancelledError:
        logging.debug("[shutdown] routine fills prefetch cancelled")
        raise
    except Exception as exc:
        if bot._shutdown_requested():
            logging.debug("[shutdown] routine fills prefetch stopped: %s", exc)
            return
        logging.warning(
            "[fills] routine prefetch failed; blocking/confirmation refresh will retry | reason=%s error_type=%s error=%s",
            reason,
            type(exc).__name__,
            exc,
        )


async def timed_authoritative_fetch(bot, surface: str, coro, timings_ms: dict[str, int]):
    """Measure one staged authoritative fetch while preserving exceptions."""
    del bot
    started = _utc_ms()
    try:
        return await coro
    finally:
        timings_ms[surface] = int(max(0, _utc_ms() - started))


def log_staged_refresh_timings(
    bot, plan: set[str], timings_ms: dict[str, int], wall_ms: int
) -> None:
    """Emit a compact timing line for slow staged refresh cohorts."""
    if not timings_ms:
        return
    sum_ms = int(sum(int(v) for v in timings_ms.values()))
    max_surface_ms = int(max(int(v) for v in timings_ms.values()))
    residual_ms = int(max(0, wall_ms - max_surface_ms))
    pending_confirmations = bool(
        getattr(bot, "_authoritative_pending_confirmations", {}) or {}
    )
    full_plan = {"balance", "fills", "open_orders", "positions"}
    routine_without_fills = {"balance", "open_orders", "positions"}
    plan_set = set(plan)
    unusual_plan = plan_set not in (full_plan, routine_without_fills)
    epoch_changed = set(getattr(bot, "_authoritative_refresh_epoch_changed", set()) or set())
    meaningful_surfaces = epoch_changed - {"balance"}
    if pending_confirmations and plan_set == {"open_orders"} and wall_ms < 2_000:
        meaningful_surfaces -= {"open_orders"}
    meaningful_change = bool(meaningful_surfaces)
    interesting = (
        (pending_confirmations and wall_ms >= 2_000)
        or meaningful_change
        or (unusual_plan and wall_ms >= 1_000)
        or wall_ms >= 10_000
    )
    if not interesting:
        if wall_ms < 1_000 and not (len(plan) > 1 and wall_ms >= 500):
            bot._record_staged_refresh_timing_summary(
                plan,
                timings_ms,
                wall_ms,
                sum_ms,
                max_surface_ms,
                residual_ms,
            )
            return
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    parts = [
        f"{surface}={int(timings_ms[surface])}ms" for surface in sorted(timings_ms)
    ]
    if residual_ms >= 500:
        parts.append(f"residual={residual_ms}ms")
        parts.append("residual_hint=scheduler_or_lock_wait")
    suffix = " | pending_confirmations=yes" if pending_confirmations else ""
    if log_level < logging.INFO:
        bot._record_staged_refresh_timing_summary(
            plan,
            timings_ms,
            wall_ms,
            sum_ms,
            max_surface_ms,
            residual_ms,
        )
    logging.log(
        log_level,
        "[state] staged refresh timings | plan=%s | wall=%dms | surface_sum=%dms | surface_max=%dms | parallel=%s | %s%s",
        ",".join(sorted(plan)),
        wall_ms,
        sum_ms,
        max_surface_ms,
        "yes" if len(timings_ms) > 1 else "no",
        " ".join(parts),
        suffix,
    )


def record_staged_refresh_timing_summary(
    bot,
    plan: set[str],
    timings_ms: dict[str, int],
    wall_ms: int,
    sum_ms: int,
    max_surface_ms: int,
    residual_ms: int,
) -> None:
    """Aggregate routine staged refresh timings into periodic operator summaries."""

    def update_stats(stats: dict[str, int], value: int) -> None:
        value = int(value)
        count = int(stats.get("count", 0))
        stats["count"] = count + 1
        stats["sum"] = int(stats.get("sum", 0)) + value
        stats["min"] = value if count == 0 else min(int(stats.get("min", value)), value)
        stats["max"] = value if count == 0 else max(int(stats.get("max", value)), value)

    def format_stats(stats: dict[str, int]) -> str:
        count = max(1, int(stats.get("count", 0)))
        mean = int(round(int(stats.get("sum", 0)) / count))
        return f"{int(stats.get('min', 0))}/{mean}/{int(stats.get('max', 0))}ms"

    now = _utc_ms()
    plan_key = ",".join(sorted(plan))
    summaries = getattr(bot, "_staged_refresh_timing_summaries", None)
    if not isinstance(summaries, dict):
        summaries = {}
        bot._staged_refresh_timing_summaries = summaries
    summary = summaries.get(plan_key)
    if not isinstance(summary, dict):
        summary = {
            "first_ms": now,
            "wall": {},
            "surface_sum": {},
            "surface_max": {},
            "residual": {},
            "surfaces": {},
        }
        summaries[plan_key] = summary
    update_stats(summary["wall"], wall_ms)
    update_stats(summary["surface_sum"], sum_ms)
    update_stats(summary["surface_max"], max_surface_ms)
    update_stats(summary["residual"], residual_ms)
    surfaces = summary.get("surfaces")
    if not isinstance(surfaces, dict):
        surfaces = {}
        summary["surfaces"] = surfaces
    for surface, value in timings_ms.items():
        stats = surfaces.setdefault(surface, {})
        update_stats(stats, int(value))
    count = int(summary["wall"].get("count", 0))
    first_ms = int(summary.get("first_ms", now))
    if count < 60 and now - first_ms < 15 * 60 * 1000:
        return
    surface_parts = [
        f"{surface}={format_stats(stats)}" for surface, stats in sorted(surfaces.items())
    ]
    logging.info(
        "[state] staged refresh timing summary | plan=%s | count=%d since=%s | wall=%s | surface_sum=%s | surface_max=%s | residual=%s | %s",
        plan_key,
        count,
        ts_to_date(first_ms),
        format_stats(summary["wall"]),
        format_stats(summary["surface_sum"]),
        format_stats(summary["surface_max"]),
        format_stats(summary["residual"]),
        " ".join(surface_parts),
    )
    summaries.pop(plan_key, None)


async def log_staged_refresh_progress_until(
    bot,
    plan: set[str],
    timings_ms: dict[str, int],
    tasks: dict[str, asyncio.Task],
    wall_started_ms: int,
) -> None:
    """Log visible progress when a required staged refresh surface is still pending."""
    try:
        threshold_s = float(
            get_optional_config_value(
                bot.config, "logging.staged_refresh_slow_surface_seconds", 10.0
            )
        )
    except Exception:
        threshold_s = 10.0
    if threshold_s <= 0.0 or not tasks:
        return
    interval_s = max(5.0, threshold_s)
    logged_pending: set[tuple[str, ...]] = set()
    try:
        await bot._sleep_unless_shutdown(
            threshold_s, stage="staged_refresh_progress"
        )
        while not bot._shutdown_requested():
            pending = tuple(
                sorted(name for name, task in tasks.items() if not task.done())
            )
            if not pending:
                return
            elapsed_ms = int(max(0, _utc_ms() - wall_started_ms))
            completed = " ".join(
                f"{surface}={int(timings_ms[surface])}ms"
                for surface in sorted(timings_ms)
                if surface not in pending
            )
            key = pending
            level = logging.INFO if key not in logged_pending else logging.DEBUG
            logged_pending.add(key)
            logging.log(
                level,
                "[state] staged refresh still waiting | plan=%s | pending=%s | elapsed=%dms%s",
                ",".join(sorted(plan)),
                ",".join(pending),
                elapsed_ms,
                f" | completed={completed}" if completed else "",
            )
            await bot._sleep_unless_shutdown(
                interval_s, stage="staged_refresh_progress"
            )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logging.debug("[state] staged refresh progress logger stopped: %s", exc)


async def fetch_authoritative_state_staged_snapshot(bot, plan: set[str]) -> dict:
    """Fetch staged authoritative components concurrently without mutating live state."""
    wall_started = _utc_ms()
    timings_ms = {}
    exchange_task = asyncio.create_task(
        bot.capture_authoritative_state_staged_snapshot(plan, timings_ms)
    )
    progress_task = asyncio.create_task(
        bot._log_staged_refresh_progress_until(
            plan,
            timings_ms,
            {"exchange_staged_capture": exchange_task},
            wall_started,
        )
    )
    try:
        exchange_snapshot = await exchange_task
    except asyncio.CancelledError:
        if not progress_task.done():
            progress_task.cancel()
        await asyncio.gather(progress_task, return_exceptions=True)
        wall_ms = int(max(0, _utc_ms() - wall_started))
        bot._log_staged_refresh_timings(plan, timings_ms, wall_ms)
        raise
    except Exception:
        if not progress_task.done():
            progress_task.cancel()
        await asyncio.gather(progress_task, return_exceptions=True)
        wall_ms = int(max(0, _utc_ms() - wall_started))
        bot._log_staged_refresh_timings(plan, timings_ms, wall_ms)
        raise
    finally:
        if not progress_task.done():
            progress_task.cancel()
            await asyncio.gather(progress_task, return_exceptions=True)
    if exchange_snapshot is not None:
        wall_ms = int(max(0, _utc_ms() - wall_started))
        bot._log_staged_refresh_timings(plan, timings_ms, wall_ms)
        exchange_snapshot.setdefault("plan", set(plan))
        exchange_snapshot.setdefault("pnls_ok", True)
        return exchange_snapshot
    tasks = {}
    if "balance" in plan:
        tasks["balance"] = asyncio.create_task(
            bot._timed_authoritative_fetch(
                "balance", bot._capture_balance_staged_snapshot(), timings_ms
            )
        )
    if "positions" in plan:
        tasks["positions"] = asyncio.create_task(
            bot._timed_authoritative_fetch(
                "positions", bot._capture_positions_staged_snapshot(), timings_ms
            )
        )
    if "open_orders" in plan:
        tasks["open_orders"] = asyncio.create_task(
            bot._timed_authoritative_fetch(
                "open_orders", bot.fetch_open_orders(), timings_ms
            )
        )
    if "fills" in plan:
        tasks["fills"] = asyncio.create_task(
            bot._timed_authoritative_fetch(
                "fills", bot.update_pnls(source="staged_blocking"), timings_ms
            )
        )
    progress_task = asyncio.create_task(
        bot._log_staged_refresh_progress_until(plan, timings_ms, tasks, wall_started)
    )
    try:
        keys = list(tasks)
        results = await asyncio.gather(*[tasks[key] for key in keys])
    except asyncio.CancelledError:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(
            *tasks.values(),
            return_exceptions=True,
        )
        if not progress_task.done():
            progress_task.cancel()
        await asyncio.gather(progress_task, return_exceptions=True)
        wall_ms = int(max(0, _utc_ms() - wall_started))
        bot._log_staged_refresh_timings(plan, timings_ms, wall_ms)
        raise
    except Exception:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(
            *tasks.values(),
            return_exceptions=True,
        )
        if not progress_task.done():
            progress_task.cancel()
        await asyncio.gather(progress_task, return_exceptions=True)
        wall_ms = int(max(0, _utc_ms() - wall_started))
        bot._log_staged_refresh_timings(plan, timings_ms, wall_ms)
        raise
    finally:
        if not progress_task.done():
            progress_task.cancel()
            await asyncio.gather(progress_task, return_exceptions=True)
    wall_ms = int(max(0, _utc_ms() - wall_started))
    bot._log_staged_refresh_timings(plan, timings_ms, wall_ms)
    out = {"plan": set(plan), "pnls_ok": True}
    for key, result in zip(keys, results):
        if key == "balance":
            _raw_balance, fetched_balance = result
            out["balance"] = fetched_balance
        elif key == "positions":
            _raw_positions, fetched_positions = result
            out["positions"] = fetched_positions
        elif key == "open_orders":
            out["open_orders"] = result
        elif key == "fills":
            out["pnls_ok"] = result
            out["pending_pnl_count"] = int(
                getattr(bot, "_last_fill_refresh_pending_pnl_count", 0) or 0
            )
    return out
