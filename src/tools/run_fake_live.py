from __future__ import annotations

import argparse
import ast
import asyncio
import json
import math
import logging
import os
import sys
import weakref
from types import MethodType
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from candlestick_manager import CANDLE_DTYPE, sanitize_remote_fetch_diagnostic
from config import load_prepared_config
from config.overrides import parse_overrides
from config.risk_memory import risk_bootstrap_horizon_minutes
from config.hsl import require_live_runtime_support
from config.runtime_compile import compile_runtime_config
from exchanges.fake import FakeCCXTClient, load_fake_scenario
from fill_events_manager import FillEvent, FillEventCache
from live.event_bus import LiveEvent, LiveEventContext, LiveEventPipeline
from live.state_refresh import AuthoritativeSurfaceUnavailable
from logging_setup import configure_logging
import passivbot as passivbot_mod
from passivbot import setup_bot, shutdown_bot
from procedures import ensure_parent_directory

MAX_CAPTURED_LIVE_EVENTS = 2_000


class _BoundedLiveEventSink:
    def __init__(self, max_retained: int = MAX_CAPTURED_LIVE_EVENTS) -> None:
        self.max_retained = max(1, int(max_retained))
        self.events: deque[LiveEvent] = deque(maxlen=self.max_retained)
        self.total = 0

    def write(self, event: LiveEvent) -> LiveEvent:
        self.total += 1
        self.events.append(event)
        return event


def _build_output_dir(root: str | None, scenario: dict) -> Path:
    base = Path(root) if root else Path("artifacts") / "fake_live"
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scenario_name = str(
        scenario.get("name")
        or Path(str(scenario.get("_scenario_path", "scenario"))).stem
    )
    return base / f"{stamp}_{scenario_name}"


def _dump_json(path: Path, data: Any) -> None:
    ensure_parent_directory(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _summarize_remote_calls(call_log: List[dict]) -> dict:
    by_method: Dict[str, int] = {}
    by_step: Dict[str, Dict[str, int]] = {}
    by_category: Dict[str, int] = {}
    by_step_category: Dict[str, Dict[str, int]] = {}
    max_per_step_by_method: Dict[str, int] = {}
    ohlcv_calls: List[dict] = []
    account_state_methods = {
        "fetch_balance",
        "fetch_positions",
        "fetch_open_orders",
        "fetch_my_trades",
    }
    market_data_methods = {
        "fetch_ticker",
        "fetch_tickers",
        "fetch_ohlcv",
        "fetch_time",
        "load_markets",
    }
    order_write_methods = {
        "create_order",
        "cancel_order",
    }
    for entry in call_log:
        method = str(entry.get("method") or "unknown")
        step_key = str(
            entry.get("step_index")
            if entry.get("step_index") is not None
            else "unknown"
        )
        if method in account_state_methods:
            category = "account_state"
        elif method in market_data_methods:
            category = "market_data"
        elif method in order_write_methods:
            category = "order_write"
        elif method.startswith("set_"):
            category = "account_config"
        else:
            category = "other"
        by_method[method] = by_method.get(method, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1
        step_bucket = by_step.setdefault(step_key, {})
        step_bucket[method] = step_bucket.get(method, 0) + 1
        category_bucket = by_step_category.setdefault(step_key, {})
        category_bucket[category] = category_bucket.get(category, 0) + 1
        if method == "fetch_ohlcv":
            ohlcv_calls.append(
                {
                    "step_index": entry.get("step_index"),
                    "symbol": entry.get("symbol"),
                    "timeframe": entry.get("timeframe"),
                    "since": entry.get("since"),
                    "until": entry.get("until"),
                    "limit": entry.get("limit"),
                    "rows": entry.get("rows"),
                }
            )
    for step_bucket in by_step.values():
        for method, count in step_bucket.items():
            max_per_step_by_method[method] = max(
                max_per_step_by_method.get(method, 0), count
            )
    return {
        "total_calls": len(call_log),
        "by_category": dict(sorted(by_category.items())),
        "by_method": dict(sorted(by_method.items())),
        "by_step": {
            key: dict(sorted(value.items())) for key, value in sorted(by_step.items())
        },
        "by_step_category": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_step_category.items())
        },
        "max_per_step_by_method": dict(sorted(max_per_step_by_method.items())),
        "ohlcv_calls": ohlcv_calls,
    }


def _install_candle_remote_fetch_trace(bot) -> tuple[List[dict], callable]:
    if not hasattr(bot, "cm"):
        return [], lambda: None
    existing_cb = getattr(bot.cm, "_remote_fetch_callback", None)
    events: List[dict] = []

    def traced(payload: Dict[str, Any]) -> None:
        safe_payload = sanitize_remote_fetch_diagnostic(payload)
        item = dict(safe_payload)
        item["event_index"] = len(events)
        events.append(item)
        if existing_cb is not None:
            existing_cb(safe_payload)

    bot.cm._remote_fetch_callback = traced
    return events, lambda: setattr(bot.cm, "_remote_fetch_callback", existing_cb)


def _attach_file_logging(path: Path) -> logging.Handler:
    ensure_parent_directory(path)
    root = logging.getLogger()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    if root.handlers and root.handlers[0].formatter is not None:
        handler.setFormatter(root.handlers[0].formatter)
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(handler)
    return handler


def _install_live_event_capture(
    bot,
) -> tuple[_BoundedLiveEventSink, Callable[[], None]]:
    """Attach a fake-live artifact sink without replacing normal event sinks."""
    pipeline = getattr(bot, "_live_event_pipeline", None)
    capture_sink = _BoundedLiveEventSink()
    owns_pipeline = pipeline is None
    if owns_pipeline:
        pipeline = LiveEventPipeline(
            context=LiveEventContext(
                exchange=getattr(bot, "exchange", None),
                user=getattr(bot, "user", None),
                bot_id=getattr(bot, "bot_id", None),
            ),
            structured_sinks=(capture_sink,),
        )
        bot._live_event_pipeline = pipeline
        existing_sinks = ()
    else:
        if not callable(getattr(pipeline, "flush", None)):
            raise RuntimeError(
                "fake-live event pipeline does not support capture flushing"
            )
        existing_sinks = tuple(getattr(pipeline, "structured_sinks", ()))
        pipeline.structured_sinks = (*existing_sinks, capture_sink)

    def restore() -> None:
        if getattr(bot, "_live_event_pipeline", None) is not pipeline:
            return
        if owns_pipeline:
            pipeline.close(timeout=2.0)
            bot._live_event_pipeline = None
        else:
            pipeline.structured_sinks = existing_sinks

    return capture_sink, restore


def _serialize_live_event_capture(
    capture_sink: _BoundedLiveEventSink,
) -> tuple[list[dict], dict]:
    retained = len(capture_sink.events)
    return [event.to_dict() for event in capture_sink.events], {
        "total": capture_sink.total,
        "retained": retained,
        "truncated": capture_sink.total - retained,
        "max_retained": capture_sink.max_retained,
    }


def _extract_hsl_trace(bot) -> Dict[str, dict]:
    from live import hsl_diagnostics
    from utils import utc_ms

    return {"hsl": hsl_diagnostics.snapshot(bot, now_ms=int(utc_ms()))}


def _coerce_numeric_assertion(spec: Any) -> Dict[str, float]:
    if isinstance(spec, (int, float)):
        return {"eq": float(spec)}
    if not isinstance(spec, dict):
        raise TypeError(f"Unsupported numeric assertion spec: {spec!r}")
    result: Dict[str, float] = {}
    for key in ("eq", "min", "max", "approx", "tolerance"):
        if key in spec:
            result[key] = float(spec[key])
    return result


def _assert_numeric(name: str, actual: float, spec: Any) -> None:
    parsed = _coerce_numeric_assertion(spec)
    if "eq" in parsed and actual != parsed["eq"]:
        raise AssertionError(f"{name}: expected {parsed['eq']} got {actual}")
    if "min" in parsed and actual < parsed["min"]:
        raise AssertionError(f"{name}: expected >= {parsed['min']} got {actual}")
    if "max" in parsed and actual > parsed["max"]:
        raise AssertionError(f"{name}: expected <= {parsed['max']} got {actual}")
    if "approx" in parsed:
        tolerance = parsed.get("tolerance", 1e-9)
        if abs(actual - parsed["approx"]) > tolerance:
            raise AssertionError(
                f"{name}: expected {parsed['approx']} +/- {tolerance} got {actual}"
            )


def _assert_value(name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, dict) and any(
        key in expected for key in ("eq", "min", "max", "approx", "tolerance")
    ):
        _assert_numeric(name, float(actual), expected)
        return
    if isinstance(expected, dict) and "contains" in expected:
        needle = str(expected["contains"])
        if needle not in str(actual):
            raise AssertionError(
                f"{name}: expected to contain {needle!r}, got {actual!r}"
            )
        return
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r} got {actual!r}")


def _get_path_value(root: Any, path: str) -> Any:
    current = root
    for segment in [part for part in str(path).split(".") if part]:
        if isinstance(current, list):
            current = current[int(segment)]
        elif isinstance(current, dict):
            current = current[segment]
        else:
            raise KeyError(
                f"Cannot descend into {segment!r} on non-container value {current!r}"
            )
    return current


def _apply_path_assertions(group: str, root: Any, specs: Dict[str, Any]) -> None:
    for path, expected in specs.items():
        actual = _get_path_value(root, path)
        _assert_value(f"{group}[{path}]", actual, expected)


def _positions_map(fake_client: FakeCCXTClient) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for row in fake_client.export_positions():
        result[f"{row['symbol']}|{row['position_side']}"] = float(row["size"])
    return result


def _apply_assertions(
    bot,
    fake_client: FakeCCXTClient,
    scenario: dict,
    *,
    step_summaries: List[dict] | None = None,
    log_text: str = "",
    remote_calls: List[dict] | None = None,
    remote_call_summary: dict | None = None,
    candle_remote_fetches: List[dict] | None = None,
) -> None:
    assertions = scenario.get("assertions") or {}
    if not assertions:
        return
    state = fake_client.export_state()
    hsl_trace = _extract_hsl_trace(bot)

    if "fill_count" in assertions:
        _assert_numeric(
            "fill_count", float(len(fake_client.fills)), assertions["fill_count"]
        )
    if "final_balance" in assertions:
        _assert_numeric(
            "final_balance",
            float(fake_client.balance_total),
            assertions["final_balance"],
        )
    if "last_prices" in assertions:
        current_prices = fake_client.get_current_step()["prices"]
        for symbol, expected in assertions["last_prices"].items():
            if symbol not in current_prices:
                raise AssertionError(f"last_prices: missing symbol {symbol}")
            _assert_numeric(
                f"last_price[{symbol}]", float(current_prices[symbol]), expected
            )
    if "final_positions" in assertions:
        actual_positions = _positions_map(fake_client)
        for key, expected in assertions["final_positions"].items():
            actual = float(actual_positions.get(key, 0.0))
            _assert_numeric(f"final_position[{key}]", actual, expected)
    if "halted_psides" in assertions:
        raise ValueError(
            "halted_psides is a retired legacy HSL assertion; use hsl_paths with scoped action diagnostics"
        )
    if "state_paths" in assertions:
        _apply_path_assertions("state_paths", state, assertions["state_paths"])
    if "hsl_paths" in assertions:
        _apply_path_assertions("hsl_paths", hsl_trace, assertions["hsl_paths"])
    if "summary_paths" in assertions:
        summary_root = {
            "step_count": len(step_summaries or []),
            "last": (step_summaries or [None])[-1],
            "steps": step_summaries or [],
        }
        _apply_path_assertions(
            "summary_paths", summary_root, assertions["summary_paths"]
        )
    if "remote_call_paths" in assertions:
        remote_root = {
            "calls": remote_calls or [],
            "summary": remote_call_summary
            or _summarize_remote_calls(remote_calls or []),
            "candle_fetches": candle_remote_fetches or [],
        }
        _apply_path_assertions(
            "remote_call_paths",
            remote_root,
            assertions["remote_call_paths"],
        )
    if "log_contains" in assertions:
        for fragment in assertions["log_contains"]:
            if str(fragment) not in log_text:
                raise AssertionError(f"log_contains missing fragment: {fragment!r}")


def _install_fake_user_override(
    config: dict, scenario_path: str, user: str | None
) -> tuple[str, callable]:
    config.setdefault("live", {})
    fake_user = user or str(
        config["live"].get("account", {}).get("user") or "fake_runner"
    )
    config["live"]["account"]["user"] = fake_user
    fake_user_info = {
        "exchange": "fake",
        "quote": str(config["live"].get("quote") or "USDT"),
        "fake_scenario_path": scenario_path,
    }
    original = passivbot_mod.load_user_info

    def patched(requested_user: str):
        if requested_user == fake_user:
            return dict(fake_user_info)
        return original(requested_user)

    passivbot_mod.load_user_info = patched
    return fake_user, lambda: setattr(passivbot_mod, "load_user_info", original)


def _prime_fake_fill_cache(
    bot, fake_client: FakeCCXTClient, cache_root: Path | None = None
) -> Path:
    root = cache_root or Path("caches") / "fill_events"
    cache_path = root / str(bot.exchange) / str(bot.user)
    cache_path.mkdir(parents=True, exist_ok=True)
    for path in cache_path.glob("*.json"):
        path.unlink()
    metadata_path = cache_path / "metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()
    cache = FillEventCache(cache_path)
    events = [
        FillEvent.from_dict(event) for event in fake_client.get_fill_events(None, None)
    ]
    cache.save(events)
    if events:
        cache.update_metadata_from_events(events)
    coverage_start_ms, history_scope = _fake_fill_cache_coverage(
        bot, fake_client, events
    )
    if coverage_start_ms is not None:
        cache.mark_covered_start(coverage_start_ms)
    if history_scope is not None:
        cache.set_history_scope(history_scope)
    elif coverage_start_ms is not None:
        cache.set_history_scope("window")
    return cache_path


def _fake_fill_cache_coverage(
    bot, fake_client: FakeCCXTClient, events: list[FillEvent]
) -> tuple[int | None, str | None]:
    """Return the earliest fake-history start covered by the staged cache."""
    config = getattr(bot, "config", None)
    live = config.get("live") if isinstance(config, dict) else None
    if isinstance(live, dict) and "risk_bootstrap_max_lookback_minutes" in live.get(
        "startup", {}
    ):
        now_ms = int(fake_client.now_ms)
        minutes, _ = risk_bootstrap_horizon_minutes(config)
        minutes = max(minutes, require_live_runtime_support(config))
        start_ms = max(0, now_ms - int(math.floor(minutes * 60_000 + 0.5)))
        # The fake cache was populated from the complete observed fixture
        # ledger. A bootstrap reaching timestamp zero proves that origin.
        if start_ms == 0:
            return None, "all"
        return start_ms, "window"
    if events:
        return min(int(event.timestamp) for event in events), "window"
    return None, None


def _prime_fake_candles(bot, fake_client: FakeCCXTClient) -> None:
    if not hasattr(bot, "cm"):
        return
    # Fake scenarios advance exchange time without advancing wall-clock time.
    # MarketSnapshotProvider TTLs are wall-clock based, so clear cached tickers
    # before each fake step to keep HSL/upnl checks aligned with scenario prices.
    provider = getattr(bot, "market_snapshot_provider", None)
    if provider is not None and hasattr(provider, "_cache"):
        provider._cache.clear()
    for symbol in fake_client.symbols:
        rows = fake_client._candles_by_symbol.get(symbol, [])[
            : fake_client.current_index + 1
        ]
        arr = np.zeros(len(rows), dtype=CANDLE_DTYPE)
        for idx, row in enumerate(rows):
            arr[idx]["ts"] = int(row[0])
            arr[idx]["o"] = float(row[1])
            arr[idx]["h"] = float(row[2])
            arr[idx]["l"] = float(row[3])
            arr[idx]["c"] = float(row[4])
            arr[idx]["bv"] = float(row[5])
        bot.cm._cache[symbol] = arr
        bot.cm._ema_cache.pop(symbol, None)
        bot.cm._current_close_cache.pop(symbol, None)
        bot.cm._tf_range_cache.pop(symbol, None)


class _FakeHslFactClocks:
    """Acquisition-owned transport clocks; never trading decisions or UTC TTLs."""

    def __init__(self):
        self._receipts = {}

    def record(self, kind, receipt, observed_utc, event_ms):
        if type(event_ms) is not int or event_ms <= 0:
            raise ValueError("invalid fake fact observation time")
        fields = tuple(observed_utc)
        if any(type(t) is not int or t <= 0 for t in fields):
            raise ValueError("invalid fake fact UTC field")
        key = kind, id(receipt)
        previous = self._receipts.get(key)
        mapping = dict.fromkeys(fields, event_ms)
        if previous is not None:
            if previous[0]() is not receipt or previous[1] != mapping:
                raise ValueError("cannot retime a fake fact receipt")
            return
        reference = weakref.ref(receipt, lambda _ref: self._receipts.pop(key, None))
        self._receipts[key] = reference, mapping

    def project(self, kind, receipt, observed_utc):
        row = self._receipts.get((kind, id(receipt)))
        if (type(observed_utc) is not int or row is None
                or row[0]() is not receipt or observed_utc not in row[1]):
            raise ValueError("unknown fake fact observation clock")
        return row[1][observed_utc]


def _install_runtime_overrides(bot, scenario: dict) -> None:
    if hasattr(bot, "cca") and isinstance(bot.cca, FakeCCXTClient):
        fake_client = bot.cca
        from live import position_fill_sync

        settle_offset = [0.0]
        settle_clock = lambda: fake_client.now_ms / 1000.0 + settle_offset[0]
        position_fill_sync.state(bot).clock = settle_clock
        from live import hsl_live

        # Retry cadence follows scenario time, while acquisition timestamps,
        # TTLs and bounded I/O waits remain on the actual observation clock.
        instance = hsl_live.owner(bot)
        instance._schedule_clock = settle_clock
        # Synthetic events use scenario time while acquisition/TTL use UTC.
        # This is an explicit harness receipt-phase estimate, not execution proof.
        bot._hsl_observation_clock_offset = (
            lambda observed_utc: int(fake_client.now_ms) - observed_utc
        )
        instance._observation_clock_offset = bot._hsl_observation_clock_offset
        # UTC freshness and scenario causality are distinct. Record each
        # successful account receipt once; later captures cannot move it across
        # a fill boundary merely because a wall-clock millisecond elapsed.
        account_clocks = {}
        record_surface = bot._record_authoritative_surface

        def record_fake_surface(surface, signature):
            changed = record_surface(surface, signature)
            if surface in {"balance", "positions"}:
                ledger = bot._ensure_freshness_ledger()
                state = ledger.surfaces[surface]
                account_clocks[(id(ledger), surface, state.revision)] = (
                    state.updated_ms, int(fake_client.now_ms)
                )
            return changed

        def project_account_clock(surface, revision, observed_utc):
            ledger = bot._ensure_freshness_ledger()
            fact = account_clocks.get((id(ledger), surface, revision))
            if fact is None or fact[0] != observed_utc:
                raise ValueError("unknown fake account observation clock")
            return fact[1]

        bot._record_authoritative_surface = record_fake_surface
        bot._hsl_account_clock_projector = project_account_clock
        instance._account_clock_projector = project_account_clock
        fact_clocks = _FakeHslFactClocks()
        bot._hsl_fact_clock_projector = fact_clocks.project
        bot._hsl_candle_receipt_observer = lambda tape: fact_clocks.record(
            "candle", tape, (c.available_at for c in tape.candles), int(fake_client.now_ms)
        )
        provider = getattr(bot, "market_snapshot_provider", None)
        if provider is not None:
            snapshot_from_ticker = provider._snapshot_from_ticker

            def observed_snapshot(*args, **kwargs):
                snapshot = snapshot_from_ticker(*args, **kwargs)
                if snapshot is not None:
                    fact_clocks.record("quote", snapshot, (snapshot.fetched_ms,), int(fake_client.now_ms))
                return snapshot

            provider._snapshot_from_ticker = observed_snapshot
            get_cached = provider.get_cached

            def cached_snapshot(symbol, *, now_ms, max_age_ms):
                snapshot = get_cached(symbol, now_ms=now_ms, max_age_ms=max_age_ms)
                if snapshot is not None:
                    captured = fact_clocks.project("quote", snapshot, snapshot.fetched_ms)
                    age = int(fake_client.now_ms) - captured
                    if age < 0:
                        raise ValueError("fake cached quote precedes its receipt")
                    if age > max_age_ms:
                        return None
                return snapshot

            provider.get_cached = cached_snapshot

        retry_bar, retry_started = [None], [0.0]

        def retry_limit():
            bar = int(fake_client.now_ms)
            if retry_bar[0] != bar:
                retry_bar[0], retry_started[0] = bar, settle_clock()
            return retry_started[0] + position_fill_sync.MAX_WAIT_SECONDS

        def advance_to(deadline):
            settle_offset[0] += max(0.0, min(deadline, retry_limit()) - settle_clock())

        def advance_readiness(instance):
            limit = retry_limit()
            sync = position_fill_sync.state(bot)
            deadlines = [
                min(p.changed + position_fill_sync.SETTLE_SECONDS,
                    p.first + position_fill_sync.MAX_WAIT_SECONDS)
                for p in sync.pending.values()
                if not sync.fetch_ready()
            ]
            fill_task = getattr(instance, "_fill_task", None)
            if fill_task is None or fill_task.done():
                required = instance.required_fill_facts()
                failed = fill_task is not None and fill_task.result() is False
                if failed or (required is not None and not required[1]):
                    deadlines.append(instance._next_history)
            upcoming = [t for t in deadlines if settle_clock() < t <= limit]
            if upcoming:
                advance_to(min(upcoming))

        bot._advance_fake_hsl_readiness = advance_readiness
        bot.get_exchange_time = lambda: int(fake_client.now_ms)
        update_pnls = getattr(bot, "update_pnls", None)
        if callable(update_pnls):
            original_update_pnls = update_pnls

            async def update_fake_pnls(self, *, source="direct", since_ms=None):
                started_bar = int(fake_client.now_ms)
                previous = getattr(self, "_hsl_fill_observation", None)
                sync = position_fill_sync.state(self)
                if sync.clock is settle_clock and not sync.fetch_ready():
                    # Simulate the finite settling wait inside this market bar.
                    # Do not trap the execution owner on a frozen scenario clock,
                    # or consume wall-clock seconds for every deterministic pass.
                    deadline = min(
                        min(
                            p.changed + position_fill_sync.SETTLE_SECONDS,
                            p.first + position_fill_sync.MAX_WAIT_SECONDS,
                        )
                        for p in sync.pending.values()
                    )
                    advance_to(deadline)
                # Cache refresh watermarks use wall time, while scenarios can
                # replay any date. Fetch the finite fake tape explicitly so a
                # wall-clock overlap cannot skip scenario closing fills.
                ready = await original_update_pnls(
                    source=source, since_ms=0 if since_ms is None else since_ms
                )
                receipt = getattr(self, "_hsl_fill_observation", None)
                if receipt is not None and receipt is not previous:
                    if int(fake_client.now_ms) != started_bar:
                        raise ValueError("fake fill acquisition crossed an unsettled market bar")
                    fact_clocks.record("fill_start", receipt, (receipt.interval[0],), started_bar)
                    fact_clocks.record("fill_end", receipt, (receipt.interval[1],), started_bar)
                return ready

            bot.update_pnls = MethodType(update_fake_pnls, bot)
        if hasattr(bot, "cm"):
            bot.cm._now_ms_callback = lambda: int(fake_client.now_ms)


def _refuse_unsettled_fake_bar(result, summaries, snapshot_dir):
    reason = (
        "input_acquisition_pending" if result.get("input_acquisition_pending", False)
        else "confirmation_pending" if result.get("confirmation_pending", False)
        else None
    )
    if reason is None:
        return
    if snapshot_dir is not None:
        _dump_json(snapshot_dir.parent / "incomplete.json", {
            "reason": reason, "step_summaries": summaries,
            "market_clock_advanced_after_pending": False,
        })
    raise RuntimeError(f"fake market bar cannot advance with {reason.replace('_', ' ')}")


def _fake_quote_input_tasks(bot, instance):
    provider = getattr(bot, "market_snapshot_provider", None)
    return [task for task in (
        *instance._quote_tasks.values(), *instance._discard_quote_tasks,
        getattr(provider, "_fetch_task", None),
        *getattr(provider, "_symbol_fetch_tasks", {}).values(),
    ) if task is not None]


def _fake_input_tasks(bot, instance):
    reader = getattr(instance, "_reader", None)
    return list(dict.fromkeys(task for task in (
        getattr(instance, "_fill_task", None),
        getattr(instance, "_source_task", None),
        *_fake_quote_input_tasks(bot, instance),
        *getattr(reader, "_pending", {}).values(),
    ) if task is not None))


def _raise_fake_input_failures(bot, instance, tasks):
    from live.market_snapshot import MarketSnapshotUnavailable
    from ccxt.base.errors import NetworkError

    quote_tasks = set(instance._quote_tasks.values()) | instance._discard_quote_tasks
    direct = quote_tasks | {
        getattr(instance, "_fill_task", None), getattr(instance, "_source_task", None)
    }
    for task in tasks:
        if not task.done() or task not in direct or (task.cancelled() and task in quote_tasks):
            continue
        try:
            task.result()
        except (NetworkError, MarketSnapshotUnavailable):
            if task not in quote_tasks:
                raise
            # Same scoped quote-outage boundary as Owner.acquire_quotes;
            # never manufacture a successful quote or swallow a producer bug.
    # Cancellation-resistant underlying reads retain their existing owner's
    # result/error contract; they still block advancing the fake market bar.
    provider = getattr(bot, "market_snapshot_provider", None)
    reader = getattr(instance, "_reader", None)
    for consumer in (getattr(provider, "raise_pending_failure", None),
                     getattr(reader, "_raise_late_failure", None)):
        if consumer is not None:
            consumer()


async def _settle_fake_startup_inputs(bot, summaries, snapshot_dir):
    from live.hsl_live import owner

    instance = owner(bot)
    # Skipping the initial trading cycle does not permit acquisitions started
    # during bot preparation to straddle the first scenario-clock advance.
    for _ in range(8):
        tasks = _fake_input_tasks(bot, instance)
        _raise_fake_input_failures(bot, instance, tasks)
        pending = [task for task in tasks if not task.done()]
        if not pending:
            break
        await asyncio.wait(pending, timeout=0.25)
        _raise_fake_input_failures(bot, instance, tasks)
    tasks = _fake_input_tasks(bot, instance)
    _raise_fake_input_failures(bot, instance, tasks)
    _refuse_unsettled_fake_bar(
        dict(input_acquisition_pending=any(not task.done() for task in tasks)),
        summaries, snapshot_dir,
    )


async def _run_fake_bot(
    bot,
    fake_client: FakeCCXTClient,
    max_steps: int | None,
    *,
    snapshot_dir: Path | None = None,
    run_initial_cycle: bool = True,
) -> List[dict]:
    summaries: List[dict] = []
    steps_run = 0

    if run_initial_cycle:
        _prime_fake_candles(bot, fake_client)
        result = await _run_fake_cycle(bot)
        summaries.append(
            {
                "step_index": int(fake_client.current_index),
                "timestamp": int(fake_client.now_ms),
                "result": str(result),
                "fills": len(fake_client.fills),
                "open_orders": len(fake_client.open_orders),
                "positions": fake_client.export_positions(),
            }
        )
        if snapshot_dir is not None:
            _dump_json(
                snapshot_dir / f"step_{int(fake_client.current_index):04d}.json",
                {
                    "summary": summaries[-1],
                    "state": fake_client.export_state(),
                    "hsl_trace": _extract_hsl_trace(bot),
                },
            )
        steps_run += 1
        _refuse_unsettled_fake_bar(result, summaries, snapshot_dir)

    while fake_client.has_next_step():
        if max_steps is not None and steps_run >= max_steps:
            break
        if not run_initial_cycle and steps_run == 0:
            await _settle_fake_startup_inputs(bot, summaries, snapshot_dir)
        fake_client.advance_time()
        _prime_fake_candles(bot, fake_client)
        result = await _run_fake_cycle(bot)
        summaries.append(
            {
                "step_index": int(fake_client.current_index),
                "timestamp": int(fake_client.now_ms),
                "result": str(result),
                "fills": len(fake_client.fills),
                "open_orders": len(fake_client.open_orders),
                "positions": fake_client.export_positions(),
            }
        )
        if snapshot_dir is not None:
            _dump_json(
                snapshot_dir / f"step_{int(fake_client.current_index):04d}.json",
                {
                    "summary": summaries[-1],
                    "state": fake_client.export_state(),
                    "hsl_trace": _extract_hsl_trace(bot),
                },
            )
        steps_run += 1
        _refuse_unsettled_fake_bar(result, summaries, snapshot_dir)

    return summaries


async def _run_fake_cycle(bot):
    return await _run_fake_cycle_ready(bot)


def _fake_confirmation_pending(bot, instance):
    pending = getattr(bot, "_authoritative_pending_confirmations", {})
    account_minima = {surface: pending[surface] for surface in
                      ("balance", "positions", "open_orders") if surface in pending}
    if account_minima:
        ledger = bot._ensure_freshness_ledger()
        if any(ledger.surfaces[surface].epoch < minimum
               for surface, minimum in account_minima.items()):
            return True
    # Stored optional fill debt does not make fills a required ordinary input.
    required = instance.required_fill_facts()
    return required is not None and not required[1]


async def _run_fake_cycle_ready(bot):
    from live import hsl_live

    instance = hsl_live.owner(bot)

    def inputs():
        return _fake_input_tasks(bot, instance)

    def raise_finished_failures(tasks):
        _raise_fake_input_failures(bot, instance, tasks)

    # Settle fast fake reads/planning at one scenario timestamp. An outage
    # remains pending after a bounded number of production passes; never
    # await arbitrary ordinary work to completion before protection.
    for attempt in range(8):
        advance = getattr(bot, "_advance_fake_hsl_readiness", None)
        if advance is not None:
            advance(instance)
        # poll_inputs() runs at the start of cycle(). A read completing
        # during that cycle still needs a subsequent consuming pass.
        before = inputs()
        raise_finished_failures(before)
        completed_before = {task for task in before if task.done()}
        result = await instance.cycle()
        input_tasks = inputs()
        raise_finished_failures(input_tasks)
        inputs_consumed = all(task in completed_before for task in input_tasks)
        in_flight = any(not task.done() for task in input_tasks)
        confirmation_pending = (
            result["updated"] and _fake_confirmation_pending(bot, instance)
        )
        if ((not result["updated"] and not in_flight)
                or (result["updated"] and result["ordinary_executed"]
                    and inputs_consumed and not confirmation_pending)):
            return dict(result, engine="hsl", passes=attempt + 1)
        # A completed ordinary plan does not settle newly scheduled history
        # reads. Give them the same bounded opportunity before advancing the
        # scenario clock; the next production pass observes their results.
        pending = [
            task
            for task in (instance._ordinary, *input_tasks)
            if task is not None and not task.done()
        ]
        if pending:
            await asyncio.wait(pending, timeout=0.25)
            raise_finished_failures(input_tasks)
        # A wrapper may start a cancellation-resistant provider/reader child
        # during the wait, then finish itself. Inspect current ownership before
        # deciding whether the final pass may advance the fake market clock.
        input_tasks = inputs()
        raise_finished_failures(input_tasks)
    return dict(
        result, engine="hsl", passes=8, preparation_pending=True,
        input_acquisition_pending=any(not task.done() for task in input_tasks),
        confirmation_pending=(
            result["updated"] and _fake_confirmation_pending(bot, instance)
        ),
    )


def _load_run_artifacts(output_dir: Path) -> dict[str, Any]:
    def _load(name: str, default):
        path = output_dir / name
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    log_path = output_dir / "fake_live.log"
    return {
        "step_summaries": _load("step_summaries.json", []),
        "fake_exchange_state": _load("fake_exchange_state.json", {}),
        "fills": _load("fills.json", []),
        "positions": _load("positions.json", []),
        "hsl_trace": _load("hsl_trace.json", {}),
        "run_metadata": _load("run_metadata.json", {}),
        "remote_calls": _load("remote_calls.json", []),
        "remote_call_summary": _load("remote_call_summary.json", {}),
        "candle_remote_fetches": _load("candle_remote_fetches.json", []),
        "live_events": _load("live_events.json", []),
        "log_text": log_path.read_text(encoding="utf-8") if log_path.exists() else "",
    }


def _canonical_fill(fill: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in fill.items():
        if key in {"clientOrderId", "id", "order"}:
            continue
        if key == "info" and isinstance(value, dict):
            normalized[key] = {
                info_key: info_value
                for info_key, info_value in value.items()
                if info_key not in {"clientOrderId"}
            }
            continue
        normalized[key] = value
    return normalized


def _canonical_hsl_trace(trace: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(trace))
    hsl = normalized.get("hsl")
    if isinstance(hsl, dict):
        # Preserve native actions, exchange-time lifecycle and quality evidence;
        # only observation wall-clock bookkeeping varies between identical runs.
        for key in ("captured_at_ms", "input_expires_at_ms", "age_ms"):
            hsl.pop(key, None)
    for pside_state in normalized.values():
        if not isinstance(pside_state, dict):
            continue
        stop_event = pside_state.get("last_stop_event")
        if isinstance(stop_event, dict):
            stop_event.pop("triggered_at", None)
            stop_event.pop("user", None)
    return normalized


def _canonicalize_artifact(name: str, payload: Any) -> Any:
    if name == "step_summaries":
        # The saved report retains scheduler diagnostics. Replay equivalence
        # compares outcomes, not how many async owner passes settled a step.
        # Readiness, pending work, protection and all exchange facts stay strict.
        normalized = []
        for step in payload:
            step = dict(step)
            result = step.get("result")
            if isinstance(result, str):
                try:
                    parsed = ast.literal_eval(result)
                except (ValueError, SyntaxError):
                    parsed = None
                if isinstance(parsed, dict) and parsed.get("engine") == "hsl":
                    parsed.pop("passes", None)
                    step["result"] = parsed
            normalized.append(step)
        return normalized
    if name == "fills":
        return sorted(
            [_canonical_fill(fill) for fill in payload],
            key=lambda x: (
                x.get("timestamp"),
                x.get("symbol"),
                x.get("position_side"),
                x.get("side"),
                x.get("price"),
                x.get("amount"),
                x.get("pnl"),
                x.get("reduceOnly"),
            ),
        )
    if name == "positions":
        return sorted(
            payload,
            key=lambda x: (
                x.get("symbol"),
                x.get("position_side"),
                x.get("entry_price"),
                x.get("size"),
            ),
        )
    if name == "fake_exchange_state":
        normalized = dict(payload)
        if isinstance(normalized.get("fills"), list):
            normalized["fills"] = _canonicalize_artifact("fills", normalized["fills"])
        if isinstance(normalized.get("positions"), list):
            normalized["positions"] = _canonicalize_artifact(
                "positions", normalized["positions"]
            )
        if isinstance(normalized.get("open_orders"), list):
            normalized["open_orders"] = sorted(
                [
                    {
                        key: value
                        for key, value in order.items()
                        if key not in {"clientOrderId", "id"}
                    }
                    for order in normalized["open_orders"]
                ],
                key=lambda x: (
                    x.get("symbol"),
                    x.get("position_side"),
                    x.get("side"),
                    x.get("price"),
                    x.get("amount"),
                    x.get("qty"),
                ),
            )
        return normalized
    if name == "hsl_trace":
        return _canonical_hsl_trace(payload)
    return payload


def _compare_run_artifacts(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    keys = ("step_summaries", "fake_exchange_state", "fills", "positions", "hsl_trace")
    diffs = []
    for key in keys:
        left_normalized = _canonicalize_artifact(key, left.get(key))
        right_normalized = _canonicalize_artifact(key, right.get(key))
        if left_normalized != right_normalized:
            diffs.append(
                {
                    "artifact": key,
                    "left": left_normalized,
                    "right": right_normalized,
                }
            )
    left_summary = left.get("remote_call_summary") or {}
    right_summary = right.get("remote_call_summary") or {}
    left_by_method = left_summary.get("by_method") or {}
    right_by_method = right_summary.get("by_method") or {}
    remote_call_delta_by_method = {
        method: int(right_by_method.get(method, 0)) - int(left_by_method.get(method, 0))
        for method in sorted(set(left_by_method) | set(right_by_method))
    }
    return {
        "match": not diffs,
        "diff_count": len(diffs),
        "diffs": diffs,
        "remote_call_summary": {
            "left": left_summary,
            "right": right_summary,
        },
        "remote_call_delta": int(right_summary.get("total_calls") or 0)
        - int(left_summary.get("total_calls") or 0),
        "remote_call_delta_by_method": remote_call_delta_by_method,
    }


async def _run_fake_case(
    *,
    config_path: str,
    scenario_path: str,
    user: str | None,
    max_steps: int | None,
    output_dir: Path,
    log_level: int,
    snapshot_each_step: bool,
    enforce_assertions: bool = True,
) -> Path:
    config = load_prepared_config(
        config_path,
        verbose=False,
        target="live",
    )
    config = parse_overrides(config, verbose=False)
    config = compile_runtime_config(config, runtime="live")
    config.setdefault("live", {})
    config["live"]["fake_scenario_path"] = scenario_path

    scenario = load_fake_scenario(scenario_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "fake_live.log"
    log_handler = _attach_file_logging(log_path)
    _, restore_user_override = _install_fake_user_override(config, scenario_path, user)
    bot = None
    bot_shutdown = False
    restore_candle_trace = lambda: None
    restore_live_event_capture = lambda: None

    try:
        bot = setup_bot(config)
        if bot.exchange != "fake":
            raise ValueError(
                f"Config user resolved to exchange '{bot.exchange}', expected 'fake' for fake harness"
            )
        bot.debug_mode = True
        if not isinstance(bot.cca, FakeCCXTClient):
            raise TypeError("Fake harness expected bot.cca to be FakeCCXTClient")
        live_event_sink, restore_live_event_capture = _install_live_event_capture(bot)
        candle_remote_fetches, restore_candle_trace = (
            _install_candle_remote_fetch_trace(bot)
        )
        _prime_fake_fill_cache(bot, bot.cca)
        _prime_fake_candles(bot, bot.cca)
        _install_runtime_overrides(bot, scenario)
        await bot.start_bot()
        bot.debug_mode = False
        snapshot_dir = (output_dir / "snapshots") if snapshot_each_step else None
        if snapshot_dir is not None:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
        step_summaries = await _run_fake_bot(
            bot,
            bot.cca,
            max_steps,
            snapshot_dir=snapshot_dir,
            run_initial_cycle=bool(scenario.get("run_initial_cycle", True)),
        )
        pipeline = getattr(bot, "_live_event_pipeline", None)
        if pipeline is None or not pipeline.flush(timeout=2.0):
            raise RuntimeError("fake-live structured-event capture did not flush")
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        remote_calls = bot.cca.export_request_log()
        remote_call_summary = _summarize_remote_calls(remote_calls)
        if enforce_assertions:
            _apply_assertions(
                bot,
                bot.cca,
                scenario,
                step_summaries=step_summaries,
                log_text=log_text,
                remote_calls=remote_calls,
                remote_call_summary=remote_call_summary,
                candle_remote_fetches=candle_remote_fetches,
            )

        fake_exchange_state = bot.cca.export_state()
        fills = list(bot.cca.fills)
        positions = bot.cca.export_positions()
        hsl_trace = _extract_hsl_trace(bot)
        await bot.shutdown_gracefully()
        bot_shutdown = True
        live_events, live_event_capture = _serialize_live_event_capture(live_event_sink)

        _dump_json(output_dir / "step_summaries.json", step_summaries)
        _dump_json(output_dir / "fake_exchange_state.json", fake_exchange_state)
        _dump_json(output_dir / "fills.json", fills)
        _dump_json(output_dir / "positions.json", positions)
        _dump_json(output_dir / "hsl_trace.json", hsl_trace)
        _dump_json(
            output_dir / "run_metadata.json",
            {
                "assertions_enforced": bool(enforce_assertions),
                "user": str(
                    config.get("live", {}).get("account", {}).get("user") or ""
                ),
                "scenario_path": str(scenario_path),
                "live_event_capture": live_event_capture,
            },
        )
        _dump_json(output_dir / "remote_calls.json", remote_calls)
        _dump_json(output_dir / "remote_call_summary.json", remote_call_summary)
        _dump_json(output_dir / "candle_remote_fetches.json", candle_remote_fetches)
        _dump_json(output_dir / "live_events.json", live_events)
        return output_dir
    finally:
        try:
            if bot is not None and not bot_shutdown:
                await shutdown_bot(bot)
        finally:
            try:
                restore_candle_trace()
            except Exception:
                pass
            try:
                restore_live_event_capture()
            except Exception:
                pass
            restore_user_override()
            logging.getLogger().removeHandler(log_handler)
            log_handler.close()


async def _async_main(args: argparse.Namespace) -> int:
    configure_logging(debug=args.log_level)
    scenario = load_fake_scenario(args.scenario)
    output_dir = _build_output_dir(args.output_dir, scenario)

    run_dir = await _run_fake_case(
        config_path=args.config,
        scenario_path=args.scenario,
        user=args.user,
        max_steps=args.max_steps,
        output_dir=output_dir,
        log_level=args.log_level,
        snapshot_each_step=args.snapshot_each_step,
    )
    print(str(run_dir))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run passivbot against the fake exchange harness"
    )
    parser.add_argument("config", help="Passivbot config path")
    parser.add_argument("scenario", help="Fake scenario path (HJSON or JSON)")
    parser.add_argument(
        "--user", default=None, help="Override live.account.user from the config"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum execution cycles to run, including the initial boot cycle",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory root (default: artifacts/fake_live)",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=1,
        help="Logging level 0-3 (warning/info/debug/trace)",
    )
    parser.add_argument(
        "--snapshot-each-step",
        action="store_true",
        help="Write a JSON snapshot after each execution cycle",
    )
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
