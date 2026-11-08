from __future__ import annotations

import asyncio
import logging
import sys
from typing import Iterable

from live.console_health import observe_ordinary_defer
from live.freshness import ACCOUNT_SURFACES
from live.market_snapshot import MarketSnapshot
from live.events import run_diagnostic_step
from live.planning_snapshot import PlanningSnapshot, PlanningSnapshotUnavailable
from utils import utc_ms


class StagedPlannerInputsUnavailable(RuntimeError):
    """Required live planner inputs failed readiness before snapshot capture."""

    def __init__(self, message: str, *, context: str, details: dict):
        invalid, invalid_counts = _group_invalid_details(
            dict(row, surface=surface)
            for surface, rows in details["invalid"].items()
            for row in rows
        )
        self.details = {
            "context": context,
            "epoch": details["epoch"],
            "required": details["required"],
            "min_epochs": details["min_epochs"],
            "missing": details["missing"],
            "invalid": invalid,
            "invalid_counts": invalid_counts,
            "defer_reason": "staged_planner_inputs_not_fresh",
        }
        super().__init__(message)


def _utc_ms() -> int:
    passivbot_module = sys.modules.get("passivbot")
    time_fn = getattr(passivbot_module, "utc_ms", None)
    if callable(time_fn):
        return int(time_fn())
    return int(utc_ms())


def staged_planner_required_surfaces(
    bot, *, include_market_snapshot: bool = True
) -> frozenset[str]:
    """Return live input surfaces required before staged order planning may proceed."""
    surfaces = set(ACCOUNT_SURFACES)
    now = int(bot.get_exchange_time())
    start = bot._pnls_lookback_start_ms()
    required, _ = bot._required_fill_history_start_ms(now, pnl_start_ms=start)
    if not required:
        # HSL can estimate history. Other enabled fill/PnL
        # consumers retain the canonical strict surface; trailing input
        # availability is scoped separately by the strategy reader.
        surfaces.discard("fills")
    if include_market_snapshot:
        surfaces.add("market_snapshot")
    return frozenset(surfaces)


def staged_planner_surface_min_epochs(
    bot, required: set[str] | frozenset[str]
) -> dict[str, int]:
    """Return minimum acceptable ledger epoch per staged planner input surface."""
    current_epoch = int(bot._ensure_freshness_ledger().epoch)
    pending = dict(getattr(bot, "_authoritative_pending_confirmations", {}) or {})
    min_epochs: dict[str, int] = {}
    for surface in required:
        if surface in ACCOUNT_SURFACES:
            min_epochs[surface] = max(1, int(pending.get(surface, 0) or 0))
        else:
            min_epochs[surface] = current_epoch
    return min_epochs


def staged_planner_precondition_state(
    bot,
    *,
    include_market_snapshot: bool = True,
    symbols: Iterable[str] | None = None,
) -> tuple[bool, dict]:
    """Return staged planner input-completeness state for the current planning pass."""
    ledger = bot._ensure_freshness_ledger()
    required = bot._staged_planner_required_surfaces(
        include_market_snapshot=include_market_snapshot
    )
    min_epochs = bot._staged_planner_surface_min_epochs(required)
    missing = []
    invalid: dict[str, list] = {}
    for surface in required:
        epoch = ledger.surface_epoch(surface)
        minimum = min_epochs[surface]
        if epoch < minimum:
            missing.append(surface)
            invalid[surface] = [
                {"reason": "surface_epoch_too_old", "epoch": epoch, "min_epoch": minimum}
            ]
    if (
        include_market_snapshot
        and "market_snapshot" in required
        and "market_snapshot" not in missing
    ):
        expected_market_symbols = tuple(
            sorted(dict.fromkeys(str(symbol) for symbol in (symbols or []) if symbol))
        )
        if expected_market_symbols:
            snapshot_invalid = bot._market_snapshot_signature_invalid(
                expected_market_symbols
            )
            if snapshot_invalid:
                missing.append("market_snapshot")
                invalid["market_snapshot"] = snapshot_invalid
    return not missing, {
        "missing": sorted(set(missing)),
        "required": sorted(required),
        "epoch": int(ledger.epoch),
        "min_epochs": min_epochs,
        "invalid": invalid,
    }


def assert_staged_planner_preconditions(
    bot,
    *,
    include_market_snapshot: bool = True,
    context: str = "planning",
    symbols: Iterable[str] | None = None,
) -> None:
    """Hard-fail before Rust planning if staged live inputs are incomplete."""
    ok, details = bot._staged_planner_precondition_state(
        include_market_snapshot=include_market_snapshot,
        symbols=symbols,
    )
    if ok:
        return
    missing = ",".join(details["missing"])
    required = ",".join(details["required"])
    raise StagedPlannerInputsUnavailable(
        f"staged planner precondition failed before {context}: "
        f"missing current-epoch surfaces={missing} epoch={details['epoch']} "
        f"required={required}",
        context=context,
        details=details,
    )


def staged_execution_ready_state(
    bot,
    *,
    include_market_snapshot: bool = True,
    context: str = "planning",
    symbols: Iterable[str] | None = None,
) -> tuple[bool, dict]:
    """Return whether live execution may proceed without raising on transient staleness."""
    ok, details = bot._staged_planner_precondition_state(
        include_market_snapshot=include_market_snapshot,
        symbols=symbols,
    )
    details = dict(details)
    details["context"] = context
    details["defer_reason"] = "staged_planner_inputs_not_fresh"
    return ok, details


def format_staged_execution_defer_message(bot, details: dict) -> str:
    """Return an operator-facing staged-defer summary."""
    missing = tuple(details.get("missing", ()))
    invalid = details.get("invalid") or {}
    context = str(details.get("context") or "planning")
    dependency = ",".join(missing) if missing else "unknown"
    parts = [
        "staged planning deferred",
        f"context={context}",
        f"dependency={dependency}",
        "scope=planner_cycle",
        "will_retry=automatic",
        f"epoch={int(details.get('epoch', 0) or 0)}",
    ]
    if missing:
        parts.append("action=refresh_required_surfaces_then_retry")
    if invalid:
        summaries = []
        for surface, items in invalid.items():
            if not isinstance(items, list) or not items:
                summaries.append(str(surface))
                continue
            first = items[0]
            if not isinstance(first, dict):
                summaries.append(f"{surface}:{type(first).__name__}")
                continue
            reason = first.get("reason") or "invalid"
            symbol = first.get("symbol")
            suffix = f":{bot._log_symbol(symbol)}" if symbol else ""
            summaries.append(f"{surface}:{reason}{suffix}")
        if summaries:
            parts.append("details=" + ",".join(summaries[:4]))
    return " | ".join(parts)


def _group_invalid_details(items: Iterable[dict]) -> tuple[dict, dict]:
    invalid: dict[str, list] = {}
    counts: dict[str, int] = {}
    for item in items:
        surface = item["surface"]
        counts[surface] = counts.get(surface, 0) + 1
        rows = invalid.setdefault(surface, [])
        if len(rows) < 8:
            row = dict(item)
            if "symbols" in row:
                symbols = row["symbols"]
                row["symbols"] = symbols[:12]
                row["symbols_count"] = len(symbols)
            rows.append(row)
    return invalid, counts


def handle_staged_execution_precondition_error(
    bot, exc: RuntimeError
) -> tuple[bool, dict]:
    """Classify staged precondition failures as safe live-loop defers."""
    if isinstance(exc, StagedPlannerInputsUnavailable):
        return True, exc.details
    if isinstance(exc, PlanningSnapshotUnavailable):
        invalid, counts = _group_invalid_details(exc.invalid_details)
        # The mutable provider may already be fresh. Preserve the failure of
        # the exact immutable cohort that was rejected instead of re-reading it.
        return True, {
            "context": exc.context,
            "epoch": exc.epoch,
            "required": list(exc.required_surfaces),
            "min_epochs": dict(exc.min_epochs),
            "missing": sorted(invalid),
            "invalid": invalid,
            "invalid_counts": counts,
            "defer_reason": "captured_planning_snapshot_unavailable",
        }
    message = str(exc)
    transient_prefixes = (
        "staged planner precondition failed",
        "planning snapshot invalid before capture",
    )
    if not message.startswith(transient_prefixes):
        return False, {}
    ok, details = bot._staged_execution_ready_state(
        include_market_snapshot=True,
        context="rust order calculation",
    )
    if ok:
        details["missing"] = []
    details["exception"] = message
    return True, details


def log_staged_execution_defer(bot, details: dict) -> None:
    """Log a throttled non-trading defer while staged inputs settle."""
    run_diagnostic_step(
        "record ordinary planning defer", lambda: observe_ordinary_defer(bot, details)
    )
    missing = tuple(details.get("missing", ()))
    required = tuple(details.get("required", ()))
    context = str(details.get("context") or "planning")
    invalid = details.get("invalid") or {}
    log_key = (context, missing, required, tuple(sorted(invalid)))
    now_ms = _utc_ms()
    last_log_ms = int(getattr(bot, "_staged_execution_defer_last_log_ms", 0) or 0)
    throttle_ms = 15_000
    if (
        log_key == getattr(bot, "_staged_execution_defer_last_log_key", None)
        and now_ms - last_log_ms < throttle_ms
    ):
        return
    bot._staged_execution_defer_last_log_key = log_key
    bot._staged_execution_defer_last_log_ms = now_ms
    emitter = getattr(bot, "_emit_planning_unavailable_diagnostic", None)
    if callable(emitter):
        run_diagnostic_step(
            "emit planning_unavailable diagnostic",
            lambda: emitter(details),
        )
    logging.info("[state] %s", bot._format_staged_execution_defer_message(details))
    if invalid:
        logging.debug("[state] staged execution deferred details | invalid=%s", invalid)


async def defer_staged_execution_cycle(bot, details: dict, loop_start_ms: int) -> None:
    """Skip trading for this loop while staged planner inputs settle."""
    bot._log_staged_execution_defer(details)
    bot._last_loop_duration_ms = _utc_ms() - loop_start_ms
    bot._maybe_log_health_summary()
    bot._maybe_log_trailing_status()
    run_diagnostic_step("unstuck status", lambda: bot._maybe_log_unstuck_status())
    bot._set_log_silence_watchdog_context(phase="runtime", stage="flush_snapshot")
    await bot._monitor_flush_snapshot()
    bot._set_log_silence_watchdog_context(
        phase="runtime", stage="staged_precondition_delay"
    )
    await asyncio.sleep(
        bot._authoritative_confirmation_retry_delay_seconds(details=details)
    )


def build_staged_planning_snapshot(
    bot, symbols: Iterable[str], market_snapshots: dict[str, MarketSnapshot]
) -> PlanningSnapshot:
    """Capture and validate the exact staged data set handed to Rust."""
    ordered_symbols = tuple(
        sorted(dict.fromkeys(str(symbol) for symbol in symbols if symbol))
    )
    ok, details = bot._staged_planner_precondition_state(
        include_market_snapshot=True, symbols=ordered_symbols
    )
    if not ok:
        raise StagedPlannerInputsUnavailable(
            "planning snapshot invalid before capture",
            context="rust order calculation",
            details=details,
        )
    ledger = bot._ensure_freshness_ledger()
    required = bot._staged_planner_required_surfaces(include_market_snapshot=True)
    min_epochs = bot._staged_planner_surface_min_epochs(required)
    packet_getter = getattr(bot, "_planning_data_packets", None)
    data_packets = (
        run_diagnostic_step(
            "collect staged planning data packet metadata",
            lambda: packet_getter(required),
            default={},
        )
        if callable(packet_getter)
        else {}
    )
    if not isinstance(data_packets, dict):
        data_packets = {}
    snapshot = PlanningSnapshot.capture(
        account_invalidation_generation=int(
            getattr(bot, "_account_invalidation_generation", 0) or 0
        ),
        ts_ms=_utc_ms(),
        exchange=str(getattr(bot, "exchange", "")),
        user=str(bot.config_get(["live", "account", "user"]) or ""),
        ledger=ledger,
        required_surfaces=required,
        min_epochs=min_epochs,
        symbols=ordered_symbols,
        market_snapshots=market_snapshots,
        market_snapshot_max_age_ms=bot._live_market_snapshot_max_age_ms(),
        data_packets=data_packets,
    )
    snapshot.raise_if_invalid(now_ms=_utc_ms(), context="rust order calculation")
    emitter = getattr(bot, "_emit_snapshot_built_diagnostic", None)
    if callable(emitter):
        run_diagnostic_step(
            "emit snapshot.built diagnostic",
            lambda: emitter(snapshot, context="rust_order_calculation"),
        )
    return snapshot


def build_protective_planning_snapshot(
    bot, symbols: Iterable[str], market_snapshots: dict[str, MarketSnapshot]
) -> PlanningSnapshot:
    """Capture the reduced live data contract used for protective panic execution."""
    ordered_symbols = tuple(
        sorted(dict.fromkeys(str(symbol) for symbol in symbols if symbol))
    )
    required = frozenset({"positions", "open_orders", "market_snapshot"})
    ledger = bot._ensure_freshness_ledger()
    current_epoch = int(ledger.epoch)
    required_epoch = max(1, current_epoch)
    min_epochs = {surface: required_epoch for surface in required}
    missing = sorted(
        surface
        for surface in required
        if ledger.surface_epoch(surface) < int(min_epochs.get(surface, 0) or 0)
    )
    invalid: dict[str, list] = {}
    if "market_snapshot" not in missing and ordered_symbols:
        market_invalid = bot._market_snapshot_signature_invalid(ordered_symbols)
        if market_invalid:
            missing.append("market_snapshot")
            invalid["market_snapshot"] = market_invalid
    if missing:
        raise RuntimeError(
            "protective planning snapshot invalid before capture: "
            f"missing current-epoch surfaces={sorted(set(missing))} "
            f"epoch={current_epoch} "
            f"required={sorted(required)}"
            f" invalid={invalid}"
        )
    packet_getter = getattr(bot, "_planning_data_packets", None)
    data_packets = (
        run_diagnostic_step(
            "collect protective planning data packet metadata",
            lambda: packet_getter(required),
            default={},
        )
        if callable(packet_getter)
        else {}
    )
    if not isinstance(data_packets, dict):
        data_packets = {}
    snapshot = PlanningSnapshot.capture(
        account_invalidation_generation=int(
            getattr(bot, "_account_invalidation_generation", 0) or 0
        ),
        ts_ms=_utc_ms(),
        exchange=str(getattr(bot, "exchange", "")),
        user=str(bot.config_get(["live", "account", "user"]) or ""),
        ledger=ledger,
        required_surfaces=required,
        min_epochs=min_epochs,
        symbols=ordered_symbols,
        market_snapshots=market_snapshots,
        market_snapshot_max_age_ms=bot._live_market_snapshot_max_age_ms(),
        data_packets=data_packets,
    )
    snapshot.raise_if_invalid(
        now_ms=_utc_ms(), context="protective panic order calculation"
    )
    emitter = getattr(bot, "_emit_snapshot_built_diagnostic", None)
    if callable(emitter):
        run_diagnostic_step(
            "emit snapshot.built diagnostic",
            lambda: emitter(snapshot, context="protective_panic_order_calculation"),
        )
    return snapshot


def current_planning_snapshot_invalid_for_creations(
    bot,
    symbols: Iterable[str],
    *,
    snapshot=None,
) -> list[dict]:
    """Return reasons the supplied/current planning snapshot is unsafe for creations."""
    if snapshot is None:
        snapshot = getattr(bot, "_current_planning_snapshot", None)
    ordered_symbols = tuple(
        sorted(dict.fromkeys(str(symbol) for symbol in symbols if symbol))
    )
    if snapshot is None:
        return [
            {
                "surface": "planning_snapshot",
                "reason": "missing_current_snapshot",
                "symbols": list(ordered_symbols),
            }
        ]
    invalid = list(snapshot.invalid_details(now_ms=_utc_ms()))
    snapshot_symbols = set(snapshot.symbols)
    missing_order_symbols = [
        symbol for symbol in ordered_symbols if symbol not in snapshot_symbols
    ]
    if missing_order_symbols:
        invalid.append(
            {
                "surface": "planning_snapshot",
                "reason": "creation_symbols_not_in_snapshot",
                "symbols": missing_order_symbols,
            }
        )
    return invalid
