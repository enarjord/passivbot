import asyncio
import contextlib
import inspect
import json
import logging
import signal
import sys
import time
import types
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from passivbot_exceptions import FatalBotException
from live.event_bus import (
    DEFAULT_ROUTES,
    EventTags,
    EventTypes,
    ListEventSink,
    LiveEvent,
    LiveEventPipeline,
    ReasonCodes,
)

# Stub passivbot_rust before importing passivbot to avoid native dependency during unit test.
sys.modules.setdefault(
    "passivbot_rust",
    types.SimpleNamespace(
        qty_to_cost=lambda *args, **kwargs: 0.0,
        round_dynamic=lambda x, y=None: x,
        calc_order_price_diff=lambda *args, **kwargs: 0.0,
        hysteresis=lambda x, y, z: x,
    ),
)

from passivbot import Passivbot
import passivbot as passivbot_module
from config import get_template_config, prepare_config
from freshness_ledger import ACCOUNT_SURFACES, LIVE_STATE_SURFACES, FreshnessLedger
from live.planning_availability import PlanningAvailability
from logging_setup import DEFAULT_DATEFMT, DEFAULT_FORMAT_WITH_PREFIX
from market_snapshot import MarketSnapshot
from planning_snapshot import (
    PlanningMarketSnapshot,
    PlanningSnapshot,
    PlanningSurfaceStamp,
)
from exchanges.binance import BinanceBot
from live.balance_composition import (
    balance_composition_signature,
    normalize_okx_balance_composition,
)


class _SafeRiskCache:
    def get_known_gaps(self):
        return []

    def get_covered_start_ms(self):
        return 1

    def get_history_scope(self):
        return "all"

    def load_metadata(self):
        return {
            "known_gaps": [],
            "covered_start_ms": 1,
            "history_scope": "all",
            "oldest_event_ts": 1,
        }


def test_repeated_shutdown_signal_forces_immediate_exit(monkeypatch):
    bot = SimpleNamespace(stop_signal_received=True, _shutdown_in_progress=True)
    monkeypatch.setattr(passivbot_module, "bot", bot)
    monkeypatch.setattr(passivbot_module.logging, "shutdown", lambda: None)

    def fake_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(passivbot_module.os, "_exit", fake_exit)

    with pytest.raises(SystemExit) as exc_info:
        passivbot_module.signal_handler(signal.SIGINT, None)

    assert exc_info.value.code == 130


def test_binance_execute_to_exchange_accepts_staged_prepare_cycle_kwarg():
    signature = inspect.signature(BinanceBot.execute_to_exchange)

    assert "prepare_cycle" in signature.parameters
    assert signature.parameters["prepare_cycle"].kind is inspect.Parameter.KEYWORD_ONLY


def test_market_snapshot_ticker_strategy_defaults_to_symbols_for_bitget():
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "bitget"
    bot.config = {"live": {}}

    assert bot._market_snapshot_ticker_strategy() == "symbols"


def test_market_snapshot_ticker_strategy_uses_symbols_for_hyperliquid_hip3_labels():
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "hyperliquid"
    bot.config = {"live": {}}

    assert bot._market_snapshot_ticker_strategy() == "symbols"


def test_market_snapshot_ticker_strategy_defaults_to_symbols_for_kucoin():
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "kucoin"
    bot.config = {"live": {}}

    assert bot._market_snapshot_ticker_strategy() == "symbols"


def test_market_snapshot_ticker_strategy_respects_explicit_override():
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "bitget"
    bot.config = {"live": {"market_snapshot_ticker_strategy": "bulk"}}

    assert bot._market_snapshot_ticker_strategy() == "bulk"


def test_noncritical_market_snapshot_error_emits_skipped_event():
    sink = ListEventSink()
    bot = Passivbot.__new__(Passivbot)
    bot._live_event_current_cycle_id = "cy_market_diag"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    exc = RuntimeError(
        "missing live market snapshots for upnl api_key=secret-token"
    )

    assert bot._log_noncritical_market_snapshot_error("upnl diagnostics", exc) is True
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.MARKET_SNAPSHOT_DIAGNOSTIC_SKIPPED
    ]
    assert len(events) == 1
    event = events[0]
    assert event.level == "warning"
    assert event.component == "market.snapshot"
    assert event.tags == ("market", "snapshot", "degraded")
    assert event.cycle_id == "cy_market_diag"
    assert event.status == "skipped"
    assert event.reason_code == ReasonCodes.MARKET_SNAPSHOT_DIAGNOSTIC_SKIPPED
    assert event.data["context"] == "upnl diagnostics"
    assert event.data["error_type"] == "RuntimeError"
    assert "secret-token" not in event.data["error"]
    assert DEFAULT_ROUTES[EventTypes.MARKET_SNAPSHOT_DIAGNOSTIC_SKIPPED].console is False
    assert DEFAULT_ROUTES[EventTypes.MARKET_SNAPSHOT_DIAGNOSTIC_SKIPPED].text is False
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_init_pnls_quarantines_and_rebuilds_unsupported_legacy_cache(monkeypatch):
    managers = []

    class _LegacyManager:
        def __init__(self, **_kwargs):
            self._events = []
            self.refresh_calls = []
            self.history_scope = None
            self.quarantine_reason = None
            managers.append(self)

        async def ensure_loaded(self):
            raise passivbot_module.FillEventCacheContractError("legacy contract")

        async def run_doctor(self, *, auto_repair: bool = False):
            assert auto_repair is True
            return {
                "legacy_contract": True,
                "unsupported_legacy_contract": True,
                "action": "rebuild_cache",
                "anomaly_events": 1,
                "repaired": False,
            }

        def quarantine_cache_for_rebuild(self, *, reason: str):
            self.quarantine_reason = reason
            return "/tmp/fills.backup"

        async def refresh(self, *, start_ms=None, end_ms=None):
            self.refresh_calls.append((start_ms, end_ms))

        def set_history_scope(self, scope: str):
            self.history_scope = scope

    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "hyperliquid"
    bot.user = "vps_user"
    bot.config = {"live": {"pnls_max_lookback_days": 1.0}}
    bot._pnls_initialized = False
    bot.get_exchange_time = lambda: 1_700_086_400_000
    sink = ListEventSink()
    bot._live_event_current_cycle_id = "cy_init_fills_legacy"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )

    monkeypatch.delenv("PASSIVBOT_FILL_EVENTS_DOCTOR", raising=False)
    monkeypatch.setattr(passivbot_module, "_extract_symbol_pool", lambda *_args: [])
    monkeypatch.setattr(passivbot_module, "_build_fetcher_for_bot", lambda *_args: object())
    monkeypatch.setattr(passivbot_module, "FillEventsManager", _LegacyManager)

    await Passivbot.init_pnls(bot)

    manager = managers[0]
    assert bot._pnls_initialized is True
    assert manager.quarantine_reason == "legacy_pnl_contract"
    assert manager.refresh_calls == [(1_700_000_000_000, None)]
    assert manager.history_scope == "window"
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.FILLS_REFRESH_SUMMARY
    ]
    events_by_reason = {event.reason_code: event for event in events}
    assert ReasonCodes.FILL_CACHE_DOCTOR_REPORT in events_by_reason
    assert ReasonCodes.FILL_CACHE_QUARANTINED in events_by_reason
    assert ReasonCodes.FILL_CACHE_REBUILD_STARTED in events_by_reason
    assert ReasonCodes.FILL_CACHE_READY in events_by_reason
    doctor_event = events_by_reason[ReasonCodes.FILL_CACHE_DOCTOR_REPORT]
    assert doctor_event.cycle_id == "cy_init_fills_legacy"
    assert doctor_event.status == "succeeded"
    assert doctor_event.data["source"] == "startup"
    assert doctor_event.data["refresh_mode"] == "doctor"
    assert doctor_event.data["doctor_mode"] == "auto"
    assert doctor_event.data["doctor_action"] == "rebuild_cache"
    assert doctor_event.data["auto_repair"] is True
    assert doctor_event.data["anomaly_events"] == 1
    assert doctor_event.data["repaired"] is False
    quarantine_event = events_by_reason[ReasonCodes.FILL_CACHE_QUARANTINED]
    assert quarantine_event.level == "warning"
    assert quarantine_event.status == "succeeded"
    assert quarantine_event.data["refresh_mode"] == "cache_quarantine"
    assert quarantine_event.data["quarantine_created"] is True
    assert quarantine_event.data["quarantine_reason"] == "legacy_pnl_contract"
    assert "/tmp/fills.backup" not in json.dumps(quarantine_event.data)
    rebuild_event = events_by_reason[ReasonCodes.FILL_CACHE_REBUILD_STARTED]
    assert rebuild_event.status == "succeeded"
    assert rebuild_event.data["refresh_mode"] == "cache_rebuild"
    assert rebuild_event.data["history_scope"] == "window"
    assert rebuild_event.data["start_ms"] == 1_700_000_000_000
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_init_pnls_emits_quarantine_event_when_doctor_repairs_legacy_cache(
    monkeypatch,
):
    managers = []

    class _RepairingManager:
        def __init__(self, **_kwargs):
            self._events = []
            self.refresh_calls = []
            self.history_scope = None
            managers.append(self)

        async def ensure_loaded(self):
            raise passivbot_module.FillEventCacheContractError("legacy contract")

        async def run_doctor(self, *, auto_repair: bool = False):
            assert auto_repair is True
            return {
                "legacy_contract": True,
                "action": "quarantine_legacy_files",
                "anomaly_events": 1,
                "repaired": True,
                "legacy_files_quarantined": 1,
            }

        async def refresh(self, *, start_ms=None, end_ms=None):
            self.refresh_calls.append((start_ms, end_ms))

        def set_history_scope(self, scope: str):
            self.history_scope = scope

    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "hyperliquid"
    bot.user = "vps_user"
    bot.config = {"live": {"pnls_max_lookback_days": 1.0}}
    bot._pnls_initialized = False
    bot.get_exchange_time = lambda: 1_700_086_400_000
    sink = ListEventSink()
    bot._live_event_current_cycle_id = "cy_init_fills_doctor_repair"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )

    monkeypatch.delenv("PASSIVBOT_FILL_EVENTS_DOCTOR", raising=False)
    monkeypatch.setattr(passivbot_module, "_extract_symbol_pool", lambda *_args: [])
    monkeypatch.setattr(passivbot_module, "_build_fetcher_for_bot", lambda *_args: object())
    monkeypatch.setattr(passivbot_module, "FillEventsManager", _RepairingManager)

    await Passivbot.init_pnls(bot)

    manager = managers[0]
    assert manager.refresh_calls == [(1_700_000_000_000, None)]
    assert manager.history_scope == "window"
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.FILLS_REFRESH_SUMMARY
    ]
    events_by_reason = {event.reason_code: event for event in events}
    assert ReasonCodes.FILL_CACHE_DOCTOR_REPORT in events_by_reason
    assert ReasonCodes.FILL_CACHE_QUARANTINED in events_by_reason
    assert ReasonCodes.FILL_CACHE_REBUILD_STARTED in events_by_reason
    quarantine_event = events_by_reason[ReasonCodes.FILL_CACHE_QUARANTINED]
    assert quarantine_event.level == "warning"
    assert quarantine_event.status == "succeeded"
    assert quarantine_event.data["refresh_mode"] == "cache_quarantine"
    assert quarantine_event.data["legacy_files_quarantined"] == 1
    assert quarantine_event.data["quarantine_created"] is True
    assert quarantine_event.data["quarantine_reason"] == "legacy_pnl_contract"
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_init_pnls_emits_cache_ready_event(monkeypatch):
    managers = []

    class _Manager:
        def __init__(self, **_kwargs):
            self._events = [object(), object(), object()]
            self.history_scope = "all"
            managers.append(self)

        async def ensure_loaded(self):
            return None

        def get_history_scope(self):
            return self.history_scope

    sink = ListEventSink()
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "binance"
    bot.user = "binance_01"
    bot.bot_id = "bot_1"
    bot.config = {"live": {"pnls_max_lookback_days": "all"}}
    bot._pnls_initialized = False
    bot._live_event_current_cycle_id = "cy_init_fills"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )

    monkeypatch.delenv("PASSIVBOT_FILL_EVENTS_DOCTOR", raising=False)
    monkeypatch.setattr(passivbot_module, "_extract_symbol_pool", lambda *_args: [])
    monkeypatch.setattr(passivbot_module, "_build_fetcher_for_bot", lambda *_args: object())
    monkeypatch.setattr(passivbot_module, "FillEventsManager", _Manager)

    await Passivbot.init_pnls(bot)

    assert bot._pnls_initialized is True
    assert len(managers) == 1
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.FILLS_REFRESH_SUMMARY
    ]
    assert len(events) == 1
    event = events[0]
    assert event.cycle_id == "cy_init_fills"
    assert event.status == "succeeded"
    assert event.reason_code == ReasonCodes.FILL_CACHE_READY
    assert event.data["source"] == "startup"
    assert event.data["refresh_mode"] == "cache_load"
    assert event.data["event_count_after"] == 3
    assert event.data["history_scope"] == "all"
    route = DEFAULT_ROUTES[EventTypes.FILLS_REFRESH_SUMMARY]
    assert route.console is False
    assert route.text is False

    assert bot._live_event_pipeline.close(timeout=2.0) is True


def _counted_staged_account_refresh_bot(
    *,
    balance: float = 100.0,
    positions: list[dict] | None = None,
    open_orders: list[dict] | None = None,
    pending_confirmations: dict[str, int] | None = None,
) -> tuple[Passivbot, dict[str, int]]:
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "fake"
    bot.stop_signal_received = False
    bot.balance_override = None
    bot._balance_override_logged = False
    bot.previous_hysteresis_balance = balance
    bot.balance_hysteresis_snap_pct = 0.02
    bot.balance_raw = balance
    bot.balance = balance
    bot._exchange_reported_balance_raw = balance
    bot.fetched_positions = list(positions or [])
    bot.positions = {}
    bot.open_orders = {}
    bot.fetched_open_orders = list(open_orders or [])
    bot.active_symbols = []
    bot.state_change_detected_by_symbol = set()
    bot.execution_scheduled = False
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot._authoritative_refresh_epoch = 0
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()
    bot._authoritative_pending_confirmations = dict(pending_confirmations or {})
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot.recent_order_cancellations = []
    bot.recent_order_executions = []
    bot.log_position_changes = AsyncMock()
    bot.handle_balance_update = AsyncMock()
    bot.order_matches_bot_cancellation = lambda order: False
    bot.order_was_recently_cancelled = lambda order: 0.0
    bot.order_matches_recent_execution = lambda order, max_age_ms=180_000: False
    bot.log_order_action = lambda *args, **kwargs: None
    bot._detect_foreign_passivbot_orders = AsyncMock()
    bot._reconcile_balance_after_positions_and_balance_refresh = lambda: False
    bot._reconcile_balance_after_open_orders_refresh = lambda: False
    bot._pnl_history_coverage_ready_for_risk = lambda: True

    counts = {
        "fetch_balance": 0,
        "fetch_positions": 0,
        "fetch_open_orders": 0,
        "update_pnls": 0,
    }

    async def counted_fetch_balance():
        counts["fetch_balance"] += 1
        return balance

    async def counted_fetch_positions():
        counts["fetch_positions"] += 1
        return list(positions or [])

    async def counted_fetch_open_orders():
        counts["fetch_open_orders"] += 1
        return list(open_orders or [])

    async def counted_update_pnls(**_kwargs):
        counts["update_pnls"] += 1
        bot._record_authoritative_surface("fills", ())
        return True

    bot.fetch_balance = counted_fetch_balance
    bot.fetch_positions = counted_fetch_positions
    bot.fetch_open_orders = counted_fetch_open_orders
    bot.update_pnls = counted_update_pnls

    # Seed the previous signatures so steady-state request-count tests do not
    # create follow-up confirmations just because the fake bot has no history.
    bot._authoritative_surface_signatures = {
        "balance": round(float(balance), 12),
        "positions": Passivbot._positions_signature(bot, list(positions or [])),
        "open_orders": Passivbot._open_orders_signature(bot, list(open_orders or [])),
        "fills": (),
    }
    return bot, counts


@pytest.mark.asyncio
async def test_staged_account_refresh_emits_data_packet_diagnostics():
    symbol = "BTC/USDT:USDT"
    bot, _counts = _counted_staged_account_refresh_bot(
        balance=123.45,
        positions=[
            {
                "symbol": symbol,
                "position_side": "long",
                "size": 0.1,
                "price": 100.0,
            }
        ],
        open_orders=[
            {
                "id": "entry-1",
                "symbol": symbol,
                "side": "buy",
                "position_side": "long",
                "qty": 0.01,
                "amount": 0.01,
                "price": 99.0,
                "timestamp": 1,
                "reduce_only": False,
            }
        ],
    )
    _disable_entry_cooldown_delta_guard_for_staged_refresh_test(bot)
    events = []
    bot._monitor_record_event = (
        lambda kind, tags, payload=None, **kwargs: events.append(
            {"kind": kind, "tags": tuple(tags), "payload": dict(payload or {}), **kwargs}
        )
    )

    assert await bot.refresh_authoritative_state() is True

    packet_events = [
        event for event in events if event["kind"] == "data_packet.updated"
    ]
    assert {event["payload"]["kind"] for event in packet_events} == {
        "balance",
        "positions",
        "open_orders",
    }
    by_kind = {event["payload"]["kind"]: event["payload"] for event in packet_events}
    assert by_kind["balance"]["revision"] == 1
    assert by_kind["balance"]["scope"] == "global"
    assert by_kind["balance"]["freshness"]["status"] == "fresh"
    assert by_kind["balance"]["quality"] == "ok"
    assert by_kind["balance"]["response_received_ts_ms"] >= by_kind["balance"][
        "call_started_ts_ms"
    ]
    assert "raw_hash" in by_kind["positions"]
    assert "raw" not in by_kind["positions"]
    assert by_kind["open_orders"]["coverage"]["row_count"] == 1


@pytest.mark.asyncio
async def test_data_packet_capture_failure_does_not_block_authoritative_fetch():
    from live import state_refresh

    def failing_recorder(*_args, **_kwargs):
        raise RuntimeError("metadata recorder failed")

    async def fetched_payload():
        return ("raw-balance", 123.45)

    bot = SimpleNamespace(_capture_live_data_packet_fetch_metadata=failing_recorder)
    timings_ms = {}

    result = await state_refresh.timed_authoritative_fetch(
        bot, "balance", fetched_payload(), timings_ms
    )

    assert result == ("raw-balance", 123.45)
    assert timings_ms["balance"] >= 0


def test_diagnostic_event_emit_failure_is_noncritical():
    from live.events import DiagnosticEvent, emit_diagnostic_event

    def failing_monitor(*_args, **_kwargs):
        raise RuntimeError("monitor unavailable")

    bot = SimpleNamespace(_monitor_record_event=failing_monitor)
    event = DiagnosticEvent.build(
        "snapshot.built", ("diagnostic",), {"snapshot_id": "x"}
    )

    assert emit_diagnostic_event(bot, event) is None


def test_authoritative_surface_record_survives_data_packet_finalize_failure():
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_refresh_epoch = 1
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()

    def failing_finalize(*_args, **_kwargs):
        raise RuntimeError("diagnostic finalize failed")

    bot._finalize_live_data_packet_metadata_inner = failing_finalize

    assert bot._record_authoritative_surface("balance", ("balance", "fresh")) is True
    assert bot._authoritative_surface_signatures["balance"] == ("balance", "fresh")
    assert bot.freshness_ledger.surface_epoch("balance") == 1


@pytest.mark.asyncio
async def test_staged_orchestrator_market_snapshot_fetch_uses_headroom_ttl():
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "bitget"
    bot.config = {"live": {}}
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {}
    symbol = "BTC/USDT:USDT"
    seen_max_age_ms = []

    async def fake_get_snapshots(symbols, max_age_ms=None):
        seen_max_age_ms.append(max_age_ms)
        return {
            symbol: MarketSnapshot(
                symbol=symbol,
                bid=100.0,
                ask=101.0,
                last=100.5,
                fetched_ms=passivbot_module.utc_ms() - 4_000,
                source="test",
            )
        }

    bot.market_snapshot_provider = SimpleNamespace(get_snapshots=fake_get_snapshots)
    bot._begin_authoritative_refresh_epoch()

    snapshots = await bot._get_orchestrator_market_snapshots([symbol])

    assert seen_max_age_ms == [5_000]
    assert snapshots[symbol].last == 100.5
    assert (
        bot.freshness_ledger.surface_epoch("market_snapshot")
        == bot._authoritative_refresh_epoch
    )
    assert bot._market_snapshot_signature_invalid([symbol]) == []


@pytest.mark.asyncio
async def test_hyperliquid_live_market_snapshot_uses_symbol_fallback_for_hip3():
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "hyperliquid"
    bot.symbol_ids = {}
    bot.market_snapshot_provider = SimpleNamespace(
        get_snapshots=AsyncMock(return_value={})
    )

    async def fake_fetch(*args, **kwargs):
        return {}

    async def fake_fetch_tickers_for_symbols(symbols):
        assert symbols == ["XYZ-SILVER/USDC:USDC"]
        return {"XYZ-SILVER/USDC:USDC": {"bid": 73.45, "ask": 73.46, "last": 73.455}}

    bot.cca = SimpleNamespace(fetch=fake_fetch)
    bot._hl_info_url = lambda: "https://example.invalid/info"
    bot.fetch_tickers_for_symbols = fake_fetch_tickers_for_symbols

    snapshots = await bot._get_live_market_snapshots(
        ["XYZ-SILVER/USDC:USDC"], context="test"
    )

    snap = snapshots["XYZ-SILVER/USDC:USDC"]
    assert snap.source == "hyperliquid_symbol_tickers"
    assert snap.last == pytest.approx(73.455)


@pytest.mark.asyncio
async def test_log_position_changes_batches_market_snapshot_request(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.inverse = False
    bot.c_mults = {
        "BTC/USDT:USDT": 1.0,
        "ETH/USDT:USDT": 1.0,
    }
    bot.pside_int_map = {"long": 1, "short": -1}
    bot.get_raw_balance = lambda: 1_000.0
    bot.bp = lambda pside, key, symbol: 1.0 if key == "wallet_exposure_limit" else 0.0
    bot.bot_value = lambda pside, key: 10.0

    calls = []

    async def _get_live_last_prices(symbols, **kwargs):
        calls.append((list(symbols), kwargs))
        return {
            "BTC/USDT:USDT": 50_000.0,
            "ETH/USDT:USDT": 3_000.0,
        }

    bot._get_live_last_prices = _get_live_last_prices
    monkeypatch.setattr(
        passivbot_module.pbr,
        "qty_to_cost",
        lambda qty, price, c_mult: abs(qty) * price * c_mult,
    )
    monkeypatch.setattr(
        passivbot_module.pbr, "calc_pprice_diff_int", lambda *args: 0.0, raising=False
    )

    await bot.log_position_changes(
        [],
        [
            {
                "symbol": "BTC/USDT:USDT",
                "position_side": "long",
                "size": 0.01,
                "price": 49_000.0,
            },
            {
                "symbol": "ETH/USDT:USDT",
                "position_side": "long",
                "size": 0.2,
                "price": 2_900.0,
            },
        ],
    )

    assert len(calls) == 1
    assert calls[0][0] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert calls[0][1]["context"] == "position_change_log"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pside", "old_size", "new_size", "expected_action"),
    [
        ("long", 0.1, 0.3, "added"),
        ("long", 0.3, 0.1, "reduced"),
        ("short", -0.1, -0.3, "added"),
        ("short", -0.3, -0.1, "reduced"),
    ],
)
async def test_log_position_changes_classifies_signed_exposure_changes(
    monkeypatch, caplog, pside, old_size, new_size, expected_action
):
    bot = Passivbot.__new__(Passivbot)
    bot.inverse = False
    bot.c_mults = {"XMR/USDT:USDT": 1.0}
    bot.pside_int_map = {"long": 1, "short": -1}
    bot.get_raw_balance = lambda: 1_000.0
    bot.bp = lambda pside, key, symbol: 1.0 if key == "wallet_exposure_limit" else 0.0
    bot.bot_value = lambda pside, key: 10.0
    sink = ListEventSink()
    bot.exchange = "binance"
    bot.user = "binance_01"
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_pos"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._current_live_event_cycle_id = Passivbot._current_live_event_cycle_id.__get__(
        bot, Passivbot
    )
    bot._emit_live_event = Passivbot._emit_live_event.__get__(bot, Passivbot)
    bot._emit_position_changed_event = (
        Passivbot._emit_position_changed_event.__get__(bot, Passivbot)
    )

    async def _get_live_last_prices(symbols, **kwargs):
        return {symbol: 0.0 for symbol in symbols}

    bot._get_live_last_prices = _get_live_last_prices
    monkeypatch.setattr(
        passivbot_module.pbr,
        "qty_to_cost",
        lambda qty, price, c_mult: abs(qty) * price * c_mult,
    )
    monkeypatch.setattr(
        passivbot_module.pbr, "calc_pprice_diff_int", lambda *args: 0.0, raising=False
    )

    with caplog.at_level(logging.INFO):
        await bot.log_position_changes(
            [
                {
                    "symbol": "XMR/USDT:USDT",
                    "position_side": pside,
                    "size": old_size,
                    "price": 100.0,
                }
            ],
            [
                {
                    "symbol": "XMR/USDT:USDT",
                    "position_side": pside,
                    "size": new_size,
                    "price": 100.0,
                }
            ],
        )

    pos_logs = [record.getMessage() for record in caplog.records if "[pos]" in record.getMessage()]
    assert len(pos_logs) == 1
    assert " ".join(pos_logs[0].split()).startswith(f"[pos] {expected_action} ")
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[-1]
    assert event.event_type == EventTypes.POSITION_CHANGED
    assert event.cycle_id == "cy_pos"
    assert event.symbol == "XMR/USDT:USDT"
    assert event.pside == pside
    assert event.reason_code == expected_action
    assert event.data["old_size"] == pytest.approx(old_size)
    assert event.data["new_size"] == pytest.approx(new_size)
    assert event.data["size_delta"] == pytest.approx(new_size - old_size)
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_log_position_changes_suppresses_legacy_table_when_event_console_active(
    monkeypatch, caplog
):
    bot = Passivbot.__new__(Passivbot)
    bot.inverse = False
    bot.c_mults = {"XMR/USDT:USDT": 1.0}
    bot.pside_int_map = {"long": 1, "short": -1}
    bot.get_raw_balance = lambda: 1_000.0
    bot.bp = lambda pside, key, symbol: 1.0 if key == "wallet_exposure_limit" else 0.0
    bot.bot_value = lambda pside, key: 10.0
    bot.live_event_console_enabled = True
    sink = ListEventSink()
    bot.exchange = "binance"
    bot.user = "binance_01"
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_pos"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._current_live_event_cycle_id = Passivbot._current_live_event_cycle_id.__get__(
        bot, Passivbot
    )
    bot._emit_live_event = Passivbot._emit_live_event.__get__(bot, Passivbot)
    bot._emit_position_changed_event = (
        Passivbot._emit_position_changed_event.__get__(bot, Passivbot)
    )

    async def _get_live_last_prices(symbols, **kwargs):
        return {symbol: 0.0 for symbol in symbols}

    bot._get_live_last_prices = _get_live_last_prices
    monkeypatch.setattr(
        passivbot_module.pbr,
        "qty_to_cost",
        lambda qty, price, c_mult: abs(qty) * price * c_mult,
    )
    monkeypatch.setattr(
        passivbot_module.pbr, "calc_pprice_diff_int", lambda *args: 0.0, raising=False
    )

    with caplog.at_level(logging.INFO):
        await bot.log_position_changes(
            [
                {
                    "symbol": "XMR/USDT:USDT",
                    "position_side": "long",
                    "size": 0.1,
                    "price": 100.0,
                }
            ],
            [
                {
                    "symbol": "XMR/USDT:USDT",
                    "position_side": "long",
                    "size": 0.3,
                    "price": 100.0,
                }
            ],
        )

    assert not any("[pos]" in record.getMessage() for record in caplog.records)
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[-1]
    assert event.event_type == EventTypes.POSITION_CHANGED
    assert event.reason_code == "added"
    assert event.data["old_size"] == pytest.approx(0.1)
    assert event.data["new_size"] == pytest.approx(0.3)
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_shutdown_gracefully_awaits_cancelled_maintainers():
    bot = Passivbot.__new__(Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.monitor_publisher = None

    seen = {"maintainer_cancelled": False, "ccp_closed": False, "cca_closed": False}

    async def _maintainer():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            seen["maintainer_cancelled"] = True
            raise

    class _Closer:
        def __init__(self, key):
            self.key = key

        async def close(self):
            seen[self.key] = True

    maintainer_task = asyncio.create_task(_maintainer())
    bot.maintainers = {"watch_orders": maintainer_task}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp = _Closer("ccp_closed")
    bot.cca = _Closer("cca_closed")

    await asyncio.sleep(0)
    await bot.shutdown_gracefully()

    assert seen == {
        "maintainer_cancelled": True,
        "ccp_closed": True,
        "cca_closed": True,
    }
    assert maintainer_task.done() is True


@pytest.mark.asyncio
async def test_shutdown_gracefully_awaits_cancelled_maintainers():
    bot = Passivbot.__new__(Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.monitor_publisher = None

    seen = {"maintainer_cancelled": False, "ccp_closed": False, "cca_closed": False}

    async def _maintainer():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            seen["maintainer_cancelled"] = True
            raise

    class _Closer:
        def __init__(self, key):
            self.key = key

        async def close(self):
            seen[self.key] = True

    maintainer_task = asyncio.create_task(_maintainer())
    bot.maintainers = {"watch_orders": maintainer_task}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp = _Closer("ccp_closed")
    bot.cca = _Closer("cca_closed")

    await asyncio.sleep(0)
    await bot.shutdown_gracefully()

    assert seen == {
        "maintainer_cancelled": True,
        "ccp_closed": True,
        "cca_closed": True,
    }
    assert maintainer_task.done() is True


@pytest.mark.asyncio
async def test_shutdown_gracefully_emits_stage_events():
    bot = Passivbot.__new__(Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    events = []

    def emit_event(event_type, *args, **kwargs):
        events.append((event_type, kwargs))
        return object()

    bot._emit_live_event = emit_event
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.monitor_publisher = None
    bot.maintainers = {}
    bot.WS_ohlcvs_1m_tasks = {}
    bot._execution_loop_task = None
    bot._execution_loop_stopped = None
    bot.ccp = None
    bot.cca = None
    bot._live_event_pipeline = None

    await bot.shutdown_gracefully()

    stages = [
        kwargs["reason_code"]
        for event_type, kwargs in events
        if event_type == EventTypes.BOT_SHUTDOWN_STAGE
    ]
    assert stages == [
        "requested",
        "monitor_snapshot_flushed",
        "event_pipeline_closing",
    ]
    assert events[0][0] == EventTypes.BOT_STOPPING
    assert any(event_type == EventTypes.BOT_STOPPED for event_type, _ in events)
    assert all(
        "elapsed_s" in kwargs["data"]
        for event_type, kwargs in events
        if event_type == EventTypes.BOT_SHUTDOWN_STAGE
    )


@pytest.mark.asyncio
async def test_shutdown_gracefully_emits_slow_stage_after_threshold():
    bot = Passivbot.__new__(Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._shutdown_degraded_after_seconds = 0.0
    events = []

    def emit_event(event_type, *args, **kwargs):
        events.append((event_type, kwargs))
        return object()

    bot._emit_live_event = emit_event
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.monitor_publisher = None
    bot.maintainers = {}
    bot.WS_ohlcvs_1m_tasks = {}
    bot._execution_loop_task = None
    bot._execution_loop_stopped = None
    bot.ccp = None
    bot.cca = None
    bot._live_event_pipeline = None

    await bot.shutdown_gracefully()

    slow_events = [
        kwargs
        for event_type, kwargs in events
        if event_type == EventTypes.BOT_SHUTDOWN_STAGE
        and kwargs["reason_code"] == "shutdown_slow"
    ]
    assert len(slow_events) == 1
    assert slow_events[0]["status"] == "degraded"
    assert slow_events[0]["data"]["threshold_s"] == 0.0


@pytest.mark.asyncio
async def test_shutdown_gracefully_times_out_stuck_maintainer_before_closing_sessions(
    caplog,
):
    bot = Passivbot.__new__(Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._shutdown_maintainer_grace_seconds = 0.01
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.monitor_publisher = None

    seen = []
    keep_running = asyncio.Event()
    cancel_count = 0

    async def _stuck_maintainer():
        nonlocal cancel_count
        while True:
            try:
                await keep_running.wait()
            except asyncio.CancelledError:
                cancel_count += 1
                seen.append("maintainer_cancel_requested")
                if cancel_count >= 2:
                    raise
                task = asyncio.current_task()
                if task is not None and hasattr(task, "uncancel"):
                    task.uncancel()

    class _Closer:
        def __init__(self, key):
            self.key = key

        async def close(self):
            seen.append(self.key)

    maintainer_task = asyncio.create_task(_stuck_maintainer())
    bot.maintainers = {"watch_orders": maintainer_task}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp = _Closer("ccp_closed")
    bot.cca = _Closer("cca_closed")

    await asyncio.sleep(0)
    with caplog.at_level(logging.WARNING):
        await bot.shutdown_gracefully()

    assert "ccp_closed" in seen
    assert "cca_closed" in seen
    assert any("timed out" in record.message for record in caplog.records)

    maintainer_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(maintainer_task, timeout=0.1)


@pytest.mark.asyncio
async def test_shutdown_gracefully_waits_for_execution_loop_before_closing_sessions():
    bot = Passivbot.__new__(Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.monitor_publisher = None

    seen = []
    execution_loop_stopped = asyncio.Event()

    async def _active_execution_loop():
        await asyncio.sleep(0.01)
        seen.append("execution_loop_stopped")
        execution_loop_stopped.set()

    class _Closer:
        def __init__(self, key):
            self.key = key

        async def close(self):
            seen.append(self.key)

    execution_task = asyncio.create_task(_active_execution_loop())
    bot._execution_loop_task = execution_task
    bot._execution_loop_task_is_inline = False
    bot._execution_loop_stopped = execution_loop_stopped
    bot.maintainers = {}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp = _Closer("ccp_closed")
    bot.cca = _Closer("cca_closed")

    await bot.shutdown_gracefully()
    await execution_task

    assert seen == ["execution_loop_stopped", "ccp_closed", "cca_closed"]


@pytest.mark.asyncio
async def test_shutdown_gracefully_cancels_stuck_execution_loop_before_closing_sessions():
    bot = Passivbot.__new__(Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._shutdown_execution_grace_seconds = 0.01
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.monitor_publisher = None

    seen = []
    execution_loop_stopped = asyncio.Event()

    async def _stuck_execution_loop():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            seen.append("execution_loop_cancelled")
            raise

    class _Closer:
        def __init__(self, key):
            self.key = key

        async def close(self):
            seen.append(self.key)

    execution_task = asyncio.create_task(_stuck_execution_loop())
    bot._execution_loop_task = execution_task
    bot._execution_loop_task_is_inline = False
    bot._execution_loop_stopped = execution_loop_stopped
    bot.maintainers = {}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp = _Closer("ccp_closed")
    bot.cca = _Closer("cca_closed")

    await bot.shutdown_gracefully()

    assert execution_task.cancelled() is True
    assert execution_loop_stopped.is_set() is True
    assert seen == ["execution_loop_cancelled", "ccp_closed", "cca_closed"]


@pytest.mark.asyncio
async def test_shutdown_gracefully_bounds_execution_loop_cancel_grace_before_closing_sessions():
    bot = Passivbot.__new__(Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._shutdown_execution_grace_seconds = 0.0
    bot._shutdown_execution_cancel_grace_seconds = 0.01
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.monitor_publisher = None

    seen = []
    events = []
    execution_loop_stopped = asyncio.Event()

    def emit_event(event_type, *args, **kwargs):
        events.append((event_type, kwargs))
        return object()

    async def _stubborn_execution_loop():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            seen.append("execution_loop_cancelled")
            await asyncio.sleep(3600)

    class _Closer:
        def __init__(self, key):
            self.key = key

        async def close(self):
            seen.append(self.key)

    execution_task = asyncio.create_task(_stubborn_execution_loop())
    bot._emit_live_event = emit_event
    bot._execution_loop_task = execution_task
    bot._execution_loop_task_is_inline = False
    bot._execution_loop_stopped = execution_loop_stopped
    bot.maintainers = {}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp = _Closer("ccp_closed")
    bot.cca = _Closer("cca_closed")

    await bot.shutdown_gracefully()

    timeout_events = [
        kwargs
        for event_type, kwargs in events
        if event_type == EventTypes.BOT_SHUTDOWN_STAGE
        and kwargs["reason_code"] == "execution_loop_cancel_timeout"
    ]
    assert len(timeout_events) == 1
    assert timeout_events[0]["status"] == "degraded"
    assert timeout_events[0]["data"]["cancel_timeout_s"] == 0.01
    assert execution_loop_stopped.is_set() is True
    assert seen[-2:] == ["ccp_closed", "cca_closed"]
    assert execution_task.done() is True


@pytest.mark.asyncio
async def test_shutdown_gracefully_does_not_cancel_inline_execution_task_on_timeout():
    bot = Passivbot.__new__(Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._shutdown_execution_grace_seconds = 0.01
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.monitor_publisher = None

    seen = []
    execution_loop_stopped = asyncio.Event()

    async def _inline_like_execution_loop():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            seen.append("unexpected_cancel")
            raise

    class _Closer:
        def __init__(self, key):
            self.key = key

        async def close(self):
            seen.append(self.key)

    execution_task = asyncio.create_task(_inline_like_execution_loop())
    bot._execution_loop_task = execution_task
    bot._execution_loop_task_is_inline = True
    bot._execution_loop_stopped = execution_loop_stopped
    bot.maintainers = {}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp = _Closer("ccp_closed")
    bot.cca = _Closer("cca_closed")

    await bot.shutdown_gracefully()

    assert execution_task.cancelled() is False
    assert execution_loop_stopped.is_set() is True
    assert seen == ["ccp_closed", "cca_closed"]
    execution_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution_task


def test_asyncio_runtime_exception_handler_suppresses_ping_timeout_callback(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "kucoin"
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._asyncio_ws_callback_last_log_ms = 0

    with caplog.at_level(logging.WARNING):
        handled = bot._handle_asyncio_runtime_exception(
            {
                "message": "Exception in callback Client.receive_loop",
                "exception": Exception("kucoinfutures ping timeout"),
            }
        )

    assert handled is True
    assert any(
        "websocket callback ping timeout" in record.message for record in caplog.records
    )
    assert (
        bot._handle_asyncio_runtime_exception(
            {
                "message": "Exception in callback unrelated",
                "exception": Exception("unexpected transport failure"),
            }
        )
        is False
    )


def test_asyncio_runtime_exception_handler_suppresses_ccxt_transport_callback(caplog):
    class ClientConnectionResetError(Exception):
        pass

    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "kucoin"
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._asyncio_ws_callback_last_log_ms = 0

    with caplog.at_level(logging.WARNING):
        handled = bot._handle_asyncio_runtime_exception(
            {
                "message": "Exception in callback Client.receive_loop",
                "exception": ClientConnectionResetError("connection reset by peer"),
            }
        )

    assert handled is True
    assert any(
        "websocket callback ClientConnectionResetError" in r.message
        for r in caplog.records
    )


def test_asyncio_runtime_exception_handler_suppresses_unretrieved_transport_future(caplog):
    class RequestTimeout(Exception):
        pass

    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "binance"
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._asyncio_ws_callback_last_log_ms = 0

    with caplog.at_level(logging.WARNING):
        handled = bot._handle_asyncio_runtime_exception(
            {
                "message": "Future exception was never retrieved",
                "exception": RequestTimeout("Connection timeout"),
            }
        )

    assert handled is True
    assert any("websocket callback RequestTimeout" in r.message for r in caplog.records)
    assert (
        bot._handle_asyncio_runtime_exception(
            {
                "message": "Future exception was never retrieved",
                "exception": RuntimeError("strategy failure"),
            }
        )
        is False
    )


def test_candle_fetch_concurrency_is_conservative_for_history_replay():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "hyperliquid"

    assert bot._candle_fetch_concurrency(context="history_replay") == 1

    bot.exchange = "kucoin"
    assert bot._candle_fetch_concurrency(context="history_replay") == 1

    bot.exchange = "gateio"
    assert bot._candle_fetch_concurrency(context="history_replay") == 1

    bot.config = {"live": {"warmup_concurrency": 7}}
    assert bot._candle_fetch_concurrency(context="history_replay") == 7


def test_hsl_extract_fill_events_normalization():
    bot = Passivbot.__new__(Passivbot)
    bot.c_mults = {"BTC/USDT:USDT": 2.0}
    bot.get_symbol_id_inv = (
        lambda sym: "BTC/USDT:USDT" if sym == "BTCUSDT" else ""
    )

    events = bot._hsl_extract_fill_events(
        [
            {
                "timestamp": 1_800_000_061_000,
                "symbol": "BTCUSDT",
                "position_side": "long",
                "side": "buy",
                "qty": 1.0,
                "price": 100.0,
                "pnl": 0.0,
            },
            {
                "timestamp": 1_800_000_060_000,
                "symbol": "BTC/USDT:USDT",
                "pside": "short",
                "side": "buy",
                "amount": 2.0,
                "avgPrice": 50.0,
                "pnl": -1.0,
            },
            {
                "timestamp": 1_800_000_062_000,
                "symbol": "BTC/USDT:USDT",
                "qty_signed": -0.5,
                "price": 10.0,
            },
            # Missing timestamp, missing symbol, zero qty, zero price: skipped.
            {"symbol": "BTC/USDT:USDT", "qty": 1.0, "price": 1.0},
            {"timestamp": 1_800_000_063_000, "symbol": "", "qty": 1.0, "price": 1.0},
            {"timestamp": 1_800_000_064_000, "symbol": "BTC/USDT:USDT", "qty": 0.0, "price": 1.0},
            {"timestamp": 1_800_000_065_000, "symbol": "BTC/USDT:USDT", "qty": 1.0, "price": 0.0},
        ]
    )

    assert [event["timestamp"] for event in events] == [1_800_000_060_000, 1_800_000_061_000, 1_800_000_062_000]
    by_ts = {event["timestamp"]: event for event in events}
    # Exchange-id symbols normalize through get_symbol_id_inv.
    assert by_ts[1_800_000_061_000]["symbol"] == "BTC/USDT:USDT"
    assert by_ts[1_800_000_061_000]["action"] == "increase"
    assert by_ts[1_800_000_061_000]["c_mult"] == 2.0
    # qty/price fallbacks (amount/avgPrice); short buy reduces.
    assert by_ts[1_800_000_060_000]["qty"] == 2.0
    assert by_ts[1_800_000_060_000]["price"] == 50.0
    assert by_ts[1_800_000_060_000]["pside"] == "short"
    assert by_ts[1_800_000_060_000]["action"] == "decrease"
    # Signed qty decides the action and qty is stored unsigned.
    assert by_ts[1_800_000_062_000]["qty"] == 0.5
    assert by_ts[1_800_000_062_000]["action"] == "decrease"
    assert all("fee" in event and "pb_order_type" in event for event in events)


@pytest.mark.asyncio
async def test_balance_equity_history_paces_replay_candle_fetches(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "kucoin"
    bot.user = "test_user"
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: 1.0 if key == "pnls_max_lookback_days" else None
    base_ts = 1_800_000_000_000
    bot.get_exchange_time = lambda: base_ts + 120_000
    bot.get_raw_balance = lambda: 100.0
    bot.get_symbol_id_inv = lambda symbol: symbol
    # Coin-mode price replay only fetches candles for held or panic symbols,
    # so hold a long position in every replayed symbol.
    bot.positions = {
        symbol: {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
        for symbol in ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT")
    }
    bot._pnls_manager = None
    bot.inverse = False
    bot._candle_fetch_concurrency = lambda *, context="runtime": 2
    bot._get_fetch_delay_seconds = lambda: 0.0
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._live_event_current_cycle_id = "cy_hsl_history_build"
    bot._emit_live_event = Passivbot._emit_live_event.__get__(bot, Passivbot)
    bot.c_mults = {
        "BTC/USDT:USDT": 1.0,
        "ETH/USDT:USDT": 1.0,
        "SOL/USDT:USDT": 1.0,
    }
    monkeypatch.setattr(
        passivbot_module, "compute_psize_pprice", lambda *args, **kwargs: None
    )

    class _CM:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.calls = []

        async def get_candles(self, symbol, **kwargs):
            self.calls.append((symbol, kwargs.get("timeframe")))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.01)
                return np.array(
                    [
                        (base_ts, 99.0, 101.0, 98.0, 100.0, 1.0),
                        (base_ts + 60_000, 100.0, 102.0, 99.0, 101.0, 1.0),
                        (base_ts + 120_000, 101.0, 103.0, 100.0, 102.0, 1.0),
                    ],
                    dtype=passivbot_module.CANDLE_DTYPE,
                )
            finally:
                self.active -= 1

    cm = _CM()
    bot.cm = cm
    fill_events = [
        {
            "timestamp": base_ts,
            "symbol": symbol,
            "position_side": "long",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        }
        for symbol in bot.c_mults
    ]

    await bot.get_balance_equity_history(
        fill_events=fill_events,
        current_balance=100.0,
        hsl_replay_signal_mode="coin",
    )

    assert cm.max_active == 2
    assert sorted(
        symbol for symbol, timeframe in cm.calls if timeframe == "1m"
    ) == sorted(bot.c_mults)
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [event for event in sink.events if event.event_type == EventTypes.HSL_REPLAY_PROGRESS]
    batch_events = [
        event
        for event in events
        if event.reason_code
        not in {
            ReasonCodes.HSL_PRICE_HISTORY_SYMBOL_FETCH_STARTED,
            ReasonCodes.HSL_PRICE_HISTORY_SYMBOL_FETCH_COMPLETED,
        }
    ]
    symbol_started_events = [
        event
        for event in events
        if event.reason_code == ReasonCodes.HSL_PRICE_HISTORY_SYMBOL_FETCH_STARTED
    ]
    symbol_completed_events = [
        event
        for event in events
        if event.reason_code == ReasonCodes.HSL_PRICE_HISTORY_SYMBOL_FETCH_COMPLETED
    ]
    assert [event.reason_code for event in batch_events] == [
        ReasonCodes.HSL_HISTORY_INPUTS_LOADED,
        ReasonCodes.HSL_PRICE_HISTORY_FETCH_STARTED,
        ReasonCodes.HSL_PRICE_HISTORY_FETCH_COMPLETED,
        ReasonCodes.HSL_TIMELINE_REPLAY_STARTED,
        ReasonCodes.HSL_TIMELINE_REPLAY_COMPLETED,
    ]
    assert {event.cycle_id for event in events} == {"cy_hsl_history_build"}
    assert {event.data["signal_mode"] for event in events} == {"coin"}
    assert batch_events[0].data["fill_events"] == 3
    assert batch_events[1].data["price_replay_symbols"] == 3
    assert batch_events[2].data["priced_symbols"] == 3
    assert batch_events[3].data["history_minutes"] >= 3
    assert (
        batch_events[4].data["timeline_rows"]
        == batch_events[3].data["history_minutes"]
    )
    assert batch_events[4].data["timeline_replay_elapsed_s"] is not None
    assert {event.symbol for event in symbol_started_events} == set(bot.c_mults)
    assert {event.symbol for event in symbol_completed_events} == set(bot.c_mults)
    assert {event.status for event in symbol_started_events} == {"started"}
    assert {event.status for event in symbol_completed_events} == {"succeeded"}
    assert {event.data["timeframe"] for event in symbol_completed_events} == {"1m"}
    assert {event.data["rows"] for event in symbol_completed_events} == {3}
    assert all(
        event.data["stage"] == "price_history_symbol_fetch_completed"
        for event in symbol_completed_events
    )
    assert all("elapsed_s" in event.data for event in symbol_completed_events)
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_pside_timeline_synthesis_rejects_inconsistent_pside_pnl():
    # Codex P1: the trust boundary must reject a v5 account series whose
    # account-level pnl and per-pside pnl describe different realized streams.
    account_arrays = passivbot_module.pb_hsl._hsl_replay_account_series_arrays(
        [
            passivbot_module.pb_hsl._hsl_replay_account_series_row(
                ts=60_000, pnl=10.0, pnl_long=3.0, pnl_short=0.0
            )
        ]
    )
    with pytest.raises(ValueError, match="pnl_pside_sum_mismatch"):
        passivbot_module.pb_hsl._hsl_replay_pside_timeline_rows_from_cache(
            {}, account_arrays, current_balance=100.0
        )
    # A consistent series passes.
    consistent = passivbot_module.pb_hsl._hsl_replay_account_series_arrays(
        [
            passivbot_module.pb_hsl._hsl_replay_account_series_row(
                ts=60_000, pnl=3.0, pnl_long=3.0, pnl_short=0.0
            )
        ]
    )
    rows = passivbot_module.pb_hsl._hsl_replay_pside_timeline_rows_from_cache(
        {}, consistent, current_balance=100.0
    )
    assert rows[0]["realized_pnl"] == rows[0]["realized_pnl_long"] == 3.0


def test_pside_timeline_synthesis_uses_qty_step_flat_epsilon():
    account_arrays = passivbot_module.pb_hsl._hsl_replay_account_series_arrays(
        [
            passivbot_module.pb_hsl._hsl_replay_account_series_row(
                ts=60_000, pnl=0.0, pnl_long=0.0, pnl_short=0.0
            )
        ]
    )
    pair_arrays = {
        ("long", "BTC/USDT:USDT"): passivbot_module.pb_hsl._hsl_replay_matrix_arrays(
            [
                passivbot_module.pb_hsl._hsl_replay_matrix_row(
                    pside="long",
                    ts=60_000,
                    price=100.0,
                    psize=0.004,
                    pprice=100.0,
                    pnl=0.0,
                    c_mult=1.0,
                )
            ]
        )
    }

    rows = passivbot_module.pb_hsl._hsl_replay_pside_timeline_rows_from_cache(
        pair_arrays,
        account_arrays,
        current_balance=100.0,
        qty_step_by_pair={("long", "BTC/USDT:USDT"): 0.01},
    )

    assert rows[0]["is_flat"] is True
    assert rows[0]["is_flat_long"] is True


@pytest.mark.asyncio
async def test_authoritative_pside_timeline_uses_qty_step_flat_epsilon(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "kucoin"
    bot.user = "test_user"
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    base_ts = 1_800_000_000_000
    ts_now = base_ts + 60_000
    bot.get_exchange_time = lambda: ts_now
    bot.get_raw_balance = lambda: 100.0
    bot.get_symbol_id_inv = lambda symbol: symbol
    symbol = "BTC/USDT:USDT"
    bot.positions = {
        symbol: {
            "long": {"size": 0.004, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot.qty_steps = {symbol: 0.01}
    bot._pnls_manager = None
    bot.inverse = False
    bot._candle_fetch_concurrency = lambda *, context="runtime": 2
    bot._get_fetch_delay_seconds = lambda: 0.0
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[ListEventSink()],
        monitor_sinks=[],
    )
    bot._live_event_current_cycle_id = "cy_hsl_authoritative_flat_epsilon"
    bot._emit_live_event = Passivbot._emit_live_event.__get__(bot, Passivbot)
    bot.c_mults = {symbol: 1.0}
    monkeypatch.setattr(
        passivbot_module, "compute_psize_pprice", lambda *args, **kwargs: None
    )

    class _CM:
        async def get_candles(self, sym, **kwargs):
            assert sym == symbol
            return np.array(
                [
                    (base_ts, 99.0, 101.0, 98.0, 100.0, 1.0),
                    (base_ts + 60_000, 100.0, 102.0, 99.0, 101.0, 1.0),
                ],
                dtype=passivbot_module.CANDLE_DTYPE,
            )

    bot.cm = _CM()
    fill_events = [
        {
            "timestamp": base_ts,
            "symbol": symbol,
            "position_side": "long",
            "side": "buy",
            "qty": 0.01,
            "price": 100.0,
            "pnl": 0.0,
        },
        {
            "timestamp": base_ts + 1_000,
            "symbol": symbol,
            "position_side": "long",
            "side": "sell",
            "qty": 0.006,
            "price": 100.0,
            "pnl": 0.0,
        },
    ]

    history = await bot.get_balance_equity_history(
        fill_events=fill_events,
        current_balance=100.0,
        hsl_replay_signal_mode="unified",
    )

    first = history["timeline"][0]
    assert first["is_flat"] is True
    assert first["is_flat_long"] is True
    assert first["is_flat_short"] is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_pside_timeline_synthesis_matches_authoritative_rows(monkeypatch):
    # Trust boundary for the future pside/unified reuse gate: rows synthesized
    # from the persisted cache arrays must equal the authoritative timeline
    # field-for-field on every aggregate field the pside/unified initializer
    # consumes, for every covered minute.
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "kucoin"
    bot.user = "test_user"
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    base_ts = 1_800_000_000_000
    ts_now = base_ts + 180_000
    bot.get_exchange_time = lambda: ts_now
    bot.get_raw_balance = lambda: 100.5
    bot.get_symbol_id_inv = lambda symbol: symbol
    sym_long = "BTC/USDT:USDT"
    sym_short = "ETH/USDT:USDT"
    bot.positions = {
        sym_long: {
            "long": {"size": 0.5, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        },
        sym_short: {
            "long": {"size": 0.0, "price": 0.0},
            "short": {"size": 1.0, "price": 50.0},
        },
    }
    bot._pnls_manager = None
    bot.inverse = False
    bot._candle_fetch_concurrency = lambda *, context="runtime": 2
    bot._get_fetch_delay_seconds = lambda: 0.0
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[ListEventSink()],
        monitor_sinks=[],
    )
    bot._live_event_current_cycle_id = "cy_hsl_pside_parity"
    bot._emit_live_event = Passivbot._emit_live_event.__get__(bot, Passivbot)
    bot.c_mults = {sym_long: 1.0, sym_short: 1.0}
    monkeypatch.setattr(
        passivbot_module, "compute_psize_pprice", lambda *args, **kwargs: None
    )

    candles = {
        sym_long: np.array(
            [
                (base_ts, 99.0, 101.0, 98.0, 100.0, 1.0),
                (base_ts + 60_000, 100.0, 102.0, 99.0, 101.0, 1.0),
                (base_ts + 120_000, 101.0, 103.0, 100.0, 102.0, 1.0),
                (base_ts + 180_000, 102.0, 104.0, 101.0, 103.0, 1.0),
            ],
            dtype=passivbot_module.CANDLE_DTYPE,
        ),
        sym_short: np.array(
            [
                (base_ts, 49.0, 51.0, 48.0, 50.0, 1.0),
                (base_ts + 60_000, 50.0, 52.0, 49.0, 49.0, 1.0),
                (base_ts + 120_000, 49.0, 51.0, 48.0, 48.0, 1.0),
                (base_ts + 180_000, 48.0, 50.0, 47.0, 47.0, 1.0),
            ],
            dtype=passivbot_module.CANDLE_DTYPE,
        ),
    }

    class _CM:
        async def get_candles(self, sym, **kwargs):
            return candles[sym]

    bot.cm = _CM()
    fill_events = [
        {
            "timestamp": base_ts,
            "symbol": sym_long,
            "position_side": "long",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        },
        {
            "timestamp": base_ts + 30_000,
            "symbol": sym_short,
            "position_side": "short",
            "side": "sell",
            "qty": 1.0,
            "price": 50.0,
            "pnl": 0.0,
        },
        {
            # Long partial close, realized +1.0 inside the second minute.
            "timestamp": base_ts + 90_000,
            "symbol": sym_long,
            "position_side": "long",
            "side": "sell",
            "qty": 0.5,
            "price": 102.0,
            "pnl": 1.0,
        },
        {
            # Short add, realized -0.5 fee-like loss inside the third minute.
            "timestamp": base_ts + 150_000,
            "symbol": sym_short,
            "position_side": "short",
            "side": "buy",
            "qty": 0.0,
            "price": 48.0,
            "pnl": -0.5,
        },
    ]

    history = await bot.get_balance_equity_history(
        fill_events=fill_events,
        current_balance=100.5,
        hsl_replay_signal_mode="unified",
    )

    matrices = history["hsl_replay_matrices"]
    assert set(matrices.get("long", {})) == {sym_long}
    assert set(matrices.get("short", {})) == {sym_short}
    pair_arrays = {
        (pside, sym): passivbot_module.pb_hsl._hsl_replay_matrix_arrays(rows)
        for pside, by_sym in matrices.items()
        for sym, rows in by_sym.items()
    }
    account_arrays = passivbot_module.pb_hsl._hsl_replay_account_series_arrays(
        history["hsl_replay_account_series"]
    )

    synthesized = passivbot_module.pb_hsl._hsl_replay_pside_timeline_rows_from_cache(
        pair_arrays,
        account_arrays,
        current_balance=100.5,
    )

    authoritative = {int(row["timestamp"]): row for row in history["timeline"]}
    contract_fields = (
        "balance",
        "realized_pnl",
        "realized_pnl_long",
        "realized_pnl_short",
        "unrealized_pnl_long",
        "unrealized_pnl_short",
        "is_flat",
        "is_flat_long",
        "is_flat_short",
    )
    assert synthesized, "synthesis must produce rows"
    compared = 0
    for row in synthesized:
        auth = authoritative.get(int(row["timestamp"]))
        if auth is None:
            continue
        compared += 1
        assert set(row) == {"timestamp", *contract_fields}
        for field in contract_fields:
            if isinstance(auth[field], bool):
                assert row[field] == auth[field], (row["timestamp"], field)
            else:
                assert row[field] == pytest.approx(auth[field], abs=1e-9), (
                    row["timestamp"],
                    field,
                )
    assert compared >= 4
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_balance_equity_history_first_minute_realized_pnl_attributes_pside(
    monkeypatch,
):
    # Codex P1 regression on schema v5: with lookback "all" the record window
    # starts at the first minute; a realized fill inside that minute must land
    # in pnl_long/pnl_short exactly like in pnl (the per-pside baseline must be
    # seeded alongside the balance baseline, pre-fill).
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "kucoin"
    bot.user = "test_user"
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    base_ts = 1_800_000_000_000
    ts_now = base_ts + 120_000
    bot.get_exchange_time = lambda: ts_now
    bot.get_raw_balance = lambda: 101.0
    bot.get_symbol_id_inv = lambda symbol: symbol
    symbol = "BTC/USDT:USDT"
    bot.positions = {
        symbol: {
            "long": {"size": 0.5, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot._pnls_manager = None
    bot.inverse = False
    bot._candle_fetch_concurrency = lambda *, context="runtime": 2
    bot._get_fetch_delay_seconds = lambda: 0.0
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[ListEventSink()],
        monitor_sinks=[],
    )
    bot._live_event_current_cycle_id = "cy_hsl_first_minute_pnl"
    bot._emit_live_event = Passivbot._emit_live_event.__get__(bot, Passivbot)
    bot.c_mults = {symbol: 1.0}
    monkeypatch.setattr(
        passivbot_module, "compute_psize_pprice", lambda *args, **kwargs: None
    )

    class _CM:
        async def get_candles(self, sym, **kwargs):
            return np.array(
                [
                    (base_ts, 99.0, 101.0, 98.0, 100.0, 1.0),
                    (base_ts + 60_000, 100.0, 102.0, 99.0, 101.0, 1.0),
                    (base_ts + 120_000, 101.0, 103.0, 100.0, 102.0, 1.0),
                ],
                dtype=passivbot_module.CANDLE_DTYPE,
            )

    bot.cm = _CM()
    fill_events = [
        {
            "timestamp": base_ts,
            "symbol": symbol,
            "position_side": "long",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        },
        {
            # Realized close inside the FIRST minute of the record window.
            "timestamp": base_ts + 1_000,
            "symbol": symbol,
            "position_side": "long",
            "side": "sell",
            "qty": 0.5,
            "price": 102.0,
            "pnl": 1.0,
        },
    ]

    history = await bot.get_balance_equity_history(
        fill_events=fill_events,
        current_balance=101.0,
        hsl_replay_signal_mode="unified",
    )

    account_rows = history["hsl_replay_account_series"]
    assert account_rows, "account series must be collected"
    first = account_rows[0]
    assert first["pnl"] == pytest.approx(1.0)
    assert first["pnl_long"] == pytest.approx(1.0)
    assert first["pnl_short"] == 0.0
    for row in account_rows[1:]:
        assert row["pnl"] == 0.0
        assert row["pnl_long"] == 0.0
        assert row["pnl_short"] == 0.0
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_balance_equity_history_builds_replay_matrices_for_held_pairs(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "kucoin"
    bot.user = "test_user"
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: 1.0 if key == "pnls_max_lookback_days" else None
    base_ts = 1_800_000_000_000
    ts_now = base_ts + 120_000
    bot.get_exchange_time = lambda: ts_now
    bot.get_raw_balance = lambda: 100.0
    bot.get_symbol_id_inv = lambda symbol: symbol
    symbol = "BTC/USDT:USDT"
    bot.positions = {
        symbol: {
            "long": {"size": 0.5, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot._pnls_manager = None
    bot.inverse = False
    bot._candle_fetch_concurrency = lambda *, context="runtime": 2
    bot._get_fetch_delay_seconds = lambda: 0.0
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[ListEventSink()],
        monitor_sinks=[],
    )
    bot._live_event_current_cycle_id = "cy_hsl_history_build"
    bot._emit_live_event = Passivbot._emit_live_event.__get__(bot, Passivbot)
    bot.c_mults = {symbol: 1.0}
    monkeypatch.setattr(
        passivbot_module, "compute_psize_pprice", lambda *args, **kwargs: None
    )

    class _CM:
        async def get_candles(self, sym, **kwargs):
            return np.array(
                [
                    (base_ts, 99.0, 101.0, 98.0, 100.0, 1.0),
                    (base_ts + 60_000, 100.0, 102.0, 99.0, 101.0, 1.0),
                    (base_ts + 120_000, 101.0, 103.0, 100.0, 102.0, 1.0),
                ],
                dtype=passivbot_module.CANDLE_DTYPE,
            )

    bot.cm = _CM()
    fill_events = [
        {
            "timestamp": base_ts,
            "symbol": symbol,
            "position_side": "long",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        },
        {
            "timestamp": base_ts + 90_000,
            "symbol": symbol,
            "position_side": "long",
            "side": "sell",
            "qty": 0.5,
            "price": 101.0,
            "pnl": 0.5,
        },
    ]

    history = await bot.get_balance_equity_history(
        fill_events=fill_events,
        current_balance=100.0,
        hsl_replay_signal_mode="coin",
    )

    matrices = history["hsl_replay_matrices"]
    assert set(matrices) == {"long"}
    assert set(matrices["long"]) == {symbol}
    rows = matrices["long"][symbol]
    assert [row["ts"] for row in rows] == [base_ts, base_ts + 60_000, base_ts + 120_000]
    assert [row["price"] for row in rows] == [100.0, 101.0, 102.0]
    assert [row["psize"] for row in rows] == [1.0, 0.5, 0.5]
    assert [row["pprice"] for row in rows] == [100.0, 100.0, 100.0]
    # Raw minute pnl: the partial close realized +0.5 inside the second minute.
    assert [row["pnl"] for row in rows] == [0.0, 0.5, 0.0]
    assert [row["upnl"] for row in rows] == [0.0, 0.5, 1.0]

    account_rows = history["hsl_replay_account_series"]
    assert len(account_rows) >= 3
    account_ts = [row["ts"] for row in account_rows]
    assert account_ts == sorted(account_ts)
    assert all(b - a == 60_000 for a, b in zip(account_ts, account_ts[1:]))
    assert account_ts[-3:] == [base_ts, base_ts + 60_000, base_ts + 120_000]
    # The partial close realized +0.5 inside the second-to-last minute; every
    # other recorded minute has zero account-level realized pnl.
    assert [row["pnl"] for row in account_rows[-3:]] == [0.0, 0.5, 0.0]
    assert sum(row["pnl"] for row in account_rows) == pytest.approx(0.5)
    # Schema v5: per-minute per-pside realized deltas. The fixture's only
    # realized event is the long partial close (+0.5).
    assert [row["pnl_long"] for row in account_rows[-3:]] == [0.0, 0.5, 0.0]
    assert all(row["pnl_short"] == 0.0 for row in account_rows)
    assert sum(row["pnl_long"] for row in account_rows) == pytest.approx(0.5)

    coverage = history["hsl_replay_matrix_coverage"]
    assert coverage["candle_covered_start_ms"] <= base_ts
    assert coverage["candle_covered_start_ms"] % 60_000 == 0
    assert coverage["candle_covered_end_ms"] == base_ts + 120_000
    assert coverage["fill_covered_start_ms"] <= base_ts
    assert coverage["fill_covered_end_ms"] == ts_now
    # Caller-provided fill events carry no pnls-manager coverage proof.
    assert coverage["fill_history_scope"] == "unknown"
    assert coverage["fill_coverage_proven"] is False
    assert coverage["fill_coverage_reason"] == "external_fill_events"

    # pside/unified signal modes now build the same held-pair matrices and
    # account series (write-only cache population; the cache config digest
    # includes the signal mode, so caches never cross modes).
    for other_mode in ("unified", "pside"):
        history_other = await bot.get_balance_equity_history(
            fill_events=fill_events,
            current_balance=100.0,
            hsl_replay_signal_mode=other_mode,
        )
        assert history_other["hsl_replay_matrices"] == matrices, other_mode
        assert history_other["hsl_replay_account_series"] == account_rows, other_mode

    # A replay fetch with no signal mode must not build matrices.
    history_no_mode = await bot.get_balance_equity_history(
        fill_events=fill_events,
        current_balance=100.0,
    )
    assert history_no_mode["hsl_replay_matrices"] == {}
    assert history_no_mode["hsl_replay_account_series"] == []

    # A failing matrix row build must drop the pair's cache, not the replay.
    def raising_row_builder(**kwargs):
        raise ValueError("synthetic matrix row failure")

    monkeypatch.setattr(
        passivbot_module.pb_hsl, "_hsl_replay_matrix_row", raising_row_builder
    )
    history_degraded = await bot.get_balance_equity_history(
        fill_events=fill_events,
        current_balance=100.0,
        hsl_replay_signal_mode="coin",
    )
    assert history_degraded["hsl_replay_matrices"] == {}
    assert history_degraded["hsl_replay_account_series"] == []
    assert len(history_degraded["timeline"]) == len(history["timeline"])
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_balance_equity_history_records_pnls_manager_coverage_proof(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "kucoin"
    bot.user = "test_user"
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: 1.0 if key == "pnls_max_lookback_days" else None
    base_ts = 1_800_000_000_000
    ts_now = base_ts + 120_000
    bot.get_exchange_time = lambda: ts_now
    bot.get_raw_balance = lambda: 100.0
    bot.get_symbol_id_inv = lambda symbol: symbol
    symbol = "BTC/USDT:USDT"
    bot.positions = {
        symbol: {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot.inverse = False
    bot._candle_fetch_concurrency = lambda *, context="runtime": 2
    bot._get_fetch_delay_seconds = lambda: 0.0
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[ListEventSink()],
        monitor_sinks=[],
    )
    bot._live_event_current_cycle_id = "cy_hsl_history_build"
    bot._emit_live_event = Passivbot._emit_live_event.__get__(bot, Passivbot)
    bot.c_mults = {symbol: 1.0}
    monkeypatch.setattr(
        passivbot_module, "compute_psize_pprice", lambda *args, **kwargs: None
    )

    class _CM:
        async def get_candles(self, sym, **kwargs):
            return np.array(
                [
                    (base_ts, 99.0, 101.0, 98.0, 100.0, 1.0),
                    (base_ts + 60_000, 100.0, 102.0, 99.0, 101.0, 1.0),
                    (base_ts + 120_000, 101.0, 103.0, 100.0, 102.0, 1.0),
                ],
                dtype=passivbot_module.CANDLE_DTYPE,
            )

    bot.cm = _CM()

    class _StubEvent:
        def to_dict(self):
            return {
                "timestamp": base_ts,
                "symbol": symbol,
                "position_side": "long",
                "side": "buy",
                "qty": 1.0,
                "price": 100.0,
                "pnl": 0.0,
            }

    class _StubCache:
        def load_metadata(self):
            return {"covered_start_ms": 1, "oldest_event_ts": base_ts}

        def get_covered_start_ms(self):
            return 1

        def get_history_scope(self):
            return "all"

    class _StubPnlsManager:
        cache = _StubCache()

        def get_events(self):
            return [_StubEvent()]

        def get_history_scope(self):
            return "all"

    bot._pnls_manager = _StubPnlsManager()

    history = await bot.get_balance_equity_history(
        current_balance=100.0,
        hsl_replay_signal_mode="coin",
    )

    coverage = history["hsl_replay_matrix_coverage"]
    assert coverage["fill_history_scope"] == "all"
    assert coverage["fill_coverage_proven"] is True
    assert coverage["fill_coverage_reason"] == "full_history"
    assert set(history["hsl_replay_matrices"]) == {"long"}
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_balance_equity_history_clamps_finite_replay_to_lookback_after_state_seed(
    monkeypatch,
):
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"pnls_max_lookback_days": 1.0}}
    bot.exchange = "kucoin"
    bot.user = "test_user"
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: bot.config["live"][key]
    day_ms = 86_400_000
    now_ms = 1_800_000_000_000
    lookback_start_ms = now_ms - day_ms
    lookback_start_minute = (lookback_start_ms // 60_000) * 60_000
    old_open_ts = lookback_start_ms - 5 * day_ms
    symbol = "XLM/USDT:USDT"
    old_flat_symbol = "AAVE/USDT:USDT"
    bot.get_exchange_time = lambda: now_ms
    bot.get_raw_balance = lambda: 1000.0
    bot.get_symbol_id_inv = lambda symbol: symbol
    bot.positions = {
        symbol: {
            "long": {"size": 1.0, "price": 100.0},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot._pnls_manager = None
    bot.inverse = False
    bot._candle_fetch_concurrency = lambda *, context="runtime": 1
    bot._get_fetch_delay_seconds = lambda: 0.0
    bot.c_mults = {symbol: 1.0, old_flat_symbol: 1.0}
    monkeypatch.setattr(
        passivbot_module, "compute_psize_pprice", lambda *args, **kwargs: None
    )

    class _CM:
        def __init__(self):
            self.calls = []

        async def get_candles(self, symbol_, **kwargs):
            self.calls.append((symbol_, dict(kwargs)))
            assert symbol_ == symbol
            assert kwargs["start_ts"] == lookback_start_minute
            assert kwargs["end_ts"] == (now_ms // 60_000) * 60_000
            return np.array(
                [
                    (lookback_start_minute, 109.0, 111.0, 108.0, 110.0, 1.0),
                    (lookback_start_minute + 60_000, 110.0, 112.0, 109.0, 111.0, 1.0),
                    ((now_ms // 60_000) * 60_000, 111.0, 113.0, 110.0, 112.0, 1.0),
                ],
                dtype=passivbot_module.CANDLE_DTYPE,
            )

    bot.cm = _CM()
    history = await bot.get_balance_equity_history(
        fill_events=[
            {
                "timestamp": old_open_ts,
                "symbol": symbol,
                "position_side": "long",
                "side": "buy",
                "qty": 1.0,
                "price": 100.0,
                "pnl": 0.0,
            },
            {
                "timestamp": old_open_ts,
                "symbol": old_flat_symbol,
                "position_side": "long",
                "side": "buy",
                "qty": 1.0,
                "price": 50.0,
                "pnl": 0.0,
            },
            {
                "timestamp": old_open_ts + 60_000,
                "symbol": old_flat_symbol,
                "position_side": "long",
                "side": "sell",
                "qty": 1.0,
                "price": 49.0,
                "pnl": -1.0,
            },
        ],
        current_balance=1000.0,
        hsl_replay_signal_mode="coin",
    )

    assert [symbol_ for symbol_, _kwargs in bot.cm.calls] == [symbol]
    assert history["timeline"]
    assert history["timeline"][0]["timestamp"] == lookback_start_minute
    assert len(history["timeline"]) == int(
        ((now_ms // 60_000) * 60_000 - lookback_start_minute) // 60_000
    ) + 1
    assert history["timeline"][0]["unrealized_pnl_by_coin_pside"][symbol][
        "long"
    ] == pytest.approx(10.0)
    assert history["timeline"][-1]["unrealized_pnl_by_coin_pside"][symbol][
        "long"
    ] == pytest.approx(12.0)
    assert old_flat_symbol not in history["metadata"]["symbols_covered"]
    assert history["metadata"]["events_used"] == 3


@pytest.mark.asyncio
async def test_balance_equity_history_skips_unsupported_closed_historical_symbols(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "gateio"
    bot.user = "test_user"
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: 1.0 if key == "pnls_max_lookback_days" else None
    base_ts = 1_800_000_000_000
    bot.get_exchange_time = lambda: base_ts + 120_000
    bot.get_raw_balance = lambda: 100.0
    bot.get_symbol_id_inv = lambda symbol: symbol
    bot.positions = {}
    bot._pnls_manager = None
    bot.inverse = False
    bot._candle_fetch_concurrency = lambda *, context="runtime": 2
    bot._get_fetch_delay_seconds = lambda: 0.0
    bot.c_mults = {"BTC/USDT:USDT": 1.0}
    monkeypatch.setattr(
        passivbot_module, "compute_psize_pprice", lambda *args, **kwargs: None
    )

    class _CM:
        def __init__(self):
            self.calls = []

        async def get_candles(self, symbol, **kwargs):
            self.calls.append(symbol)
            return np.array(
                [
                    (base_ts, 99.0, 101.0, 98.0, 100.0, 1.0),
                    (base_ts + 60_000, 100.0, 102.0, 99.0, 101.0, 1.0),
                    (base_ts + 120_000, 101.0, 103.0, 100.0, 102.0, 1.0),
                ],
                dtype=passivbot_module.CANDLE_DTYPE,
            )

    bot.cm = _CM()
    fill_events = [
        {
            "timestamp": base_ts,
            "symbol": "BTC/USDT:USDT",
            "position_side": "long",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        },
        {
            "timestamp": base_ts,
            "symbol": "TON/USDT:USDT",
            "position_side": "long",
            "side": "buy",
            "qty": 1.0,
            "price": 5.0,
            "pnl": 0.0,
        },
        {
            "timestamp": base_ts + 60_000,
            "symbol": "TON/USDT:USDT",
            "position_side": "long",
            "side": "sell",
            "qty": 1.0,
            "price": 4.9,
            "pnl": -0.1,
        },
    ]

    history = await bot.get_balance_equity_history(
        fill_events=fill_events, current_balance=100.0
    )

    assert bot.cm.calls == ["BTC/USDT:USDT"]
    assert [event["symbol"] for event in history["fill_events"]] == [
        "BTC/USDT:USDT",
        "TON/USDT:USDT",
        "TON/USDT:USDT",
    ]
    assert history["metadata"]["symbols_covered"] == ["BTC/USDT:USDT"]
    assert history["metadata"]["missing_price_symbols"] == []


@pytest.mark.asyncio
async def test_balance_equity_history_emits_zero_coin_upnl_for_flat_realized_symbol(
    monkeypatch,
):
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "gateio"
    bot.user = "test_user"
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: 1.0 if key == "pnls_max_lookback_days" else None
    base_ts = 1_800_000_000_000
    bot.get_exchange_time = lambda: base_ts + 120_000
    bot.get_raw_balance = lambda: 95.0
    bot.get_symbol_id_inv = lambda symbol: symbol
    bot.positions = {}
    bot._pnls_manager = None
    bot.inverse = False
    bot._candle_fetch_concurrency = lambda *, context="runtime": 2
    bot._get_fetch_delay_seconds = lambda: 0.0
    symbol = "M/USDT:USDT"
    bot.c_mults = {symbol: 1.0}
    monkeypatch.setattr(
        passivbot_module, "compute_psize_pprice", lambda *args, **kwargs: None
    )

    class _CM:
        async def get_candles(self, symbol, **kwargs):
            return np.array(
                [
                    (base_ts, 99.0, 101.0, 98.0, 100.0, 1.0),
                    (base_ts + 60_000, 100.0, 102.0, 99.0, 101.0, 1.0),
                    (base_ts + 120_000, 101.0, 103.0, 100.0, 102.0, 1.0),
                ],
                dtype=passivbot_module.CANDLE_DTYPE,
            )

    bot.cm = _CM()
    fill_events = [
        {
            "timestamp": base_ts,
            "symbol": symbol,
            "position_side": "long",
            "side": "buy",
            "qty": 1.0,
            "price": 100.0,
            "pnl": 0.0,
        },
        {
            "timestamp": base_ts + 60_000,
            "symbol": symbol,
            "position_side": "long",
            "side": "sell",
            "qty": 1.0,
            "price": 95.0,
            "pnl": -5.0,
        },
    ]

    history = await bot.get_balance_equity_history(
        fill_events=fill_events, current_balance=95.0
    )

    flat_rows = [
        row
        for row in history["timeline"]
        if row["timestamp"] >= base_ts + 60_000
    ]
    assert flat_rows
    for row in flat_rows:
        assert row["realized_pnl_by_coin_pside"][symbol]["long"] == pytest.approx(-5.0)
        assert row["unrealized_pnl_by_coin_pside"][symbol]["long"] == pytest.approx(0.0)
        assert row["unrealized_pnl_by_coin_pside"][symbol]["short"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_start_bot_treats_shutdown_cancelled_warmup_as_clean_stop(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "bybit"
    bot.user = "test_user"
    bot.quote = "USDT"
    bot.start_time_ms = 1_000_000
    bot.config = {"live": {}}
    bot.debug_mode = False
    bot.stop_signal_received = False
    bot._shutdown_in_progress = False
    bot._bot_ready = False
    bot.user_info = {"exchange": "bybit"}
    bot._log_startup_banner = lambda: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("shutdown cancellation should not be recorded as startup error")
    )
    bot._monitor_flush_snapshot = AsyncMock()
    stop_events = []
    bot._monitor_emit_stop = lambda *args, **kwargs: stop_events.append((args, kwargs))
    bot.init_markets = AsyncMock()
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False

    async def _format(*args, **kwargs):
        return None

    async def _warmup():
        bot.stop_signal_received = True
        raise asyncio.CancelledError("shutdown during warmup")

    monkeypatch.setattr(passivbot_module, "format_approved_ignored_coins", _format)
    bot.warmup_trading_ready_candles = _warmup

    await bot.start_bot()

    assert bot.stop_signal_received is True
    assert bot._bot_ready is False
    assert stop_events
    assert stop_events[-1][0][0] == "startup_aborted"
    assert stop_events[-1][1]["payload"]["stage"] == "warmup_trading_ready_candles"


@pytest.mark.asyncio
async def test_start_bot_treats_hsl_value_error_as_terminal_startup_failure(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "gateio"
    bot.user = "test_user"
    bot.quote = "USDT"
    bot.start_time_ms = 1_000_000
    bot.config = {"live": {"boot_stagger_seconds": 0}}
    bot.debug_mode = False
    bot.stop_signal_received = False
    bot._shutdown_in_progress = False
    bot._bot_ready = False
    bot.user_info = {"exchange": "gateio"}
    bot._log_startup_banner = lambda: None
    monitor_errors = []
    stop_events = []
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: monitor_errors.append((args, kwargs))
    bot._monitor_flush_snapshot = AsyncMock()
    bot._monitor_emit_stop = lambda *args, **kwargs: stop_events.append((args, kwargs))
    bot.init_markets = AsyncMock()
    bot.warmup_trading_ready_candles = AsyncMock()
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: True
    bot._equity_hard_stop_signal_mode = lambda *args, **kwargs: "coin"
    bot._equity_hard_stop_start_coin_history_replay = AsyncMock(
        side_effect=ValueError("missing unrealized_pnl_by_coin_pside")
    )

    async def _format(*args, **kwargs):
        return None

    monkeypatch.setattr(passivbot_module, "format_approved_ignored_coins", _format)

    with pytest.raises(FatalBotException, match="terminal startup validation failure"):
        await bot.start_bot()

    assert monitor_errors
    assert (
        monitor_errors[-1][1]["payload"]["stage"]
        == "equity_hard_stop_initialize_coin_from_history"
    )
    assert stop_events[-1][0][0] == "startup_error"
    assert stop_events[-1][1]["payload"] == {
        "stage": "equity_hard_stop_initialize_coin_from_history",
        "error_type": "ValueError",
    }


def test_coin_hsl_status_logs_distance_only_for_open_position(caplog, monkeypatch):
    sink = ListEventSink()

    class FakeBot:
        _emit_live_event = Passivbot._emit_live_event
        _equity_hard_stop_emit_coin_status = Passivbot._equity_hard_stop_emit_coin_status
        _hsl_coin_state = Passivbot._hsl_coin_state

        def __init__(self, *, has_position: bool, debug_profiles=()):
            self.exchange = "bybit"
            self.user = "bybit_01"
            self.bot_id = "bot_coin_hsl"
            self._live_event_current_cycle_id = "cy_hsl_coin"
            self.live_event_debug_profiles = tuple(debug_profiles)
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )
            self.has_position = has_position
            self._equity_hard_stop_coin = {
                "long": {
                    "NEAR/USDT:USDT": {
                        "last_status_log_ms": 0,
                        "cooldown_until_ms": 90_000,
                        "last_stop_event": {
                            "stop_event_timestamp_ms": 7_000,
                        },
                        "pending_stop_event": {
                            "stop_event_timestamp_ms": 8_000,
                        },
                        "pending_red_since_ms": 6_000,
                        "red_trigger_event_emitted": True,
                        "cooldown_intervention_active": True,
                    }
                },
                "short": {},
            }

        def _equity_hard_stop_has_open_position_symbol(self, pside, symbol):
            return self.has_position

    metrics = {
        "timestamp_ms": 10_000,
        "tier": "yellow",
        "red_threshold": 0.10,
        "drawdown_score": 0.06,
        "drawdown_raw": 0.05,
        "drawdown_ema": 0.06,
        "slot_budget": 500.0,
        "realized_pnl": -10.0,
        "peak_realized_pnl": 20.0,
        "unrealized_pnl": -15.0,
    }
    bot = FakeBot(has_position=True)

    with caplog.at_level(logging.INFO):
        bot._equity_hard_stop_emit_coin_status("long", "NEAR/USDT:USDT", metrics)

    messages = [record.message for record in caplog.records]
    assert any("[risk] HSL[long:NEAR/USDT:USDT] status" in msg for msg in messages)
    assert any("dist_to_red=0.040000" in msg for msg in messages)
    assert any("slot_budget=500.000000" in msg for msg in messages)
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[-1]
    assert event.event_type == EventTypes.HSL_STATUS
    assert event.symbol == "NEAR/USDT:USDT"
    assert event.pside == "long"
    assert event.data["dist_to_red"] == pytest.approx(0.04)
    assert "debug_profile" not in event.data
    assert "debug" not in event.data

    sink.events.clear()
    debug_bot = FakeBot(has_position=True, debug_profiles=("hsl",))
    debug_bot._equity_hard_stop_emit_coin_status(
        "long",
        "NEAR/USDT:USDT",
        metrics | {"timestamp_ms": 40_000},
    )

    assert debug_bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[-1]
    assert event.event_type == EventTypes.HSL_STATUS
    assert event.data["debug_profile"] == "hsl"
    debug = event.data["debug"]
    assert debug["event_type"] == EventTypes.HSL_STATUS
    assert debug["reason_code"] == "yellow"
    assert debug["status"] == "succeeded"
    assert debug["tier"] == "yellow"
    assert "dist_to_red" in debug["data_keys"]
    assert debug["state"] == {
        "halted": False,
        "no_restart_latched": False,
        "cooldown_until_present": True,
        "pending_red": True,
        "has_pending_stop_event": True,
        "has_last_stop_event": True,
        "red_trigger_event_emitted": True,
        "cooldown_intervention_active": True,
        "cooldown_repanic_reset_pending": False,
        "cooldown_unresolved_residue": False,
        "pnl_reset_timestamp_present": False,
    }

    caplog.clear()
    sink.events.clear()
    flat_bot = FakeBot(has_position=False)
    with caplog.at_level(logging.INFO):
        flat_bot._equity_hard_stop_emit_coin_status(
            "long",
            "NEAR/USDT:USDT",
            metrics | {"timestamp_ms": 20_000},
        )

    assert not [
        record
        for record in caplog.records
        if "[risk] HSL[long:NEAR/USDT:USDT] status" in record.message
    ]
    assert flat_bot._live_event_pipeline.flush(timeout=2.0) is True
    assert sink.events[-1].event_type == EventTypes.HSL_STATUS

    import passivbot_hsl as hsl_mod

    def fail_info(*args, **kwargs):
        raise RuntimeError("logging failed")

    caplog.clear()
    sink.events.clear()
    monkeypatch.setattr(hsl_mod.logging, "info", fail_info)
    failing_log_bot = FakeBot(has_position=True)
    failing_log_bot._equity_hard_stop_emit_coin_status(
        "long",
        "NEAR/USDT:USDT",
        metrics | {"timestamp_ms": 30_000},
    )

    assert failing_log_bot._live_event_pipeline.flush(timeout=2.0) is True
    assert sink.events[-1].event_type == EventTypes.HSL_STATUS
    assert sink.events[-1].data["dist_to_red"] == pytest.approx(0.04)


def test_startup_timing_marks_log_once(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    now_ms = [1_000]

    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now_ms[0])

    Passivbot._startup_timing_begin(bot)
    now_ms[0] = 2_500
    with caplog.at_level(logging.INFO):
        Passivbot._startup_timing_mark(bot, "account")
        now_ms[0] = 4_000
        Passivbot._startup_timing_mark(bot, "account")
        now_ms[0] = 9_000
        Passivbot._startup_timing_mark(bot, "active-candle", details="symbols=2")

    messages = [
        record.message
        for record in caplog.records
        if "startup timing" in record.message
    ]
    assert messages == [
        "[boot] startup timing | account-ready=1.50s | since_previous=1.50s",
        "[boot] startup timing | active-candle-ready=8.00s | since_previous=6.50s | symbols=2",
    ]


def test_background_warmup_uses_low_priority_candle_concurrency():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "bitget"
    bot.max_n_concurrent_ohlcvs_1m_updates = 8

    assert bot._candle_fetch_concurrency(context="runtime") == 8
    assert bot._candle_fetch_concurrency(context="background warmup") == 1


def test_ws_reconnect_warning_logs_are_throttled(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "kucoin"
    now_ms = [1_000]

    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now_ms[0])

    with caplog.at_level(logging.DEBUG):
        for reconnect_no in range(1, 6):
            Passivbot._log_ws_reconnect(
                bot,
                reconnect_no=reconnect_no,
                retry_delay_s=1.0,
                reason="connection_lost",
                exc=TimeoutError("ping timeout"),
            )

    reconnect_records = [
        record
        for record in caplog.records
        if "[ws] kucoin: connection lost" in record.message
    ]
    assert [record.levelno for record in reconnect_records] == [
        logging.WARNING,
        logging.WARNING,
        logging.WARNING,
        logging.DEBUG,
        logging.DEBUG,
    ]


def test_ws_reconnect_emits_bounded_structured_event(monkeypatch, caplog):
    sink = ListEventSink()
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "kucoin"
    bot._live_event_current_cycle_id = "cy_ws"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    now_ms = [1_000]
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now_ms[0])

    with caplog.at_level(logging.DEBUG):
        Passivbot._log_ws_reconnect(
            bot,
            reconnect_no=1,
            retry_delay_s=1.25,
            reason="time_sync",
            exc=TimeoutError("api_key=SECRET ping timeout"),
        )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.WEBSOCKET_RECONNECT
    ]
    assert len(events) == 1
    event = events[0]
    assert event.level == "warning"
    assert event.component == "exchange.websocket"
    assert event.tags == (EventTags.EXCHANGE, EventTags.WEBSOCKET)
    assert event.cycle_id == "cy_ws"
    assert event.status == "degraded"
    assert event.reason_code == ReasonCodes.WEBSOCKET_RECONNECT
    assert event.data == {
        "reconnect_no": 1,
        "retry_delay_ms": 1_250,
        "reason": "time_sync",
        "rate_limited": False,
        "warning_visible": True,
        "traceback_emitted": True,
        "error_type": "TimeoutError",
    }
    assert "SECRET" not in json.dumps(event.to_dict())
    route = DEFAULT_ROUTES[EventTypes.WEBSOCKET_RECONNECT]
    assert route.structured is True
    assert route.monitor is True
    assert route.console is False
    assert route.text is False

    Passivbot._log_ws_reconnect(
        bot,
        reconnect_no=4,
        retry_delay_s=2.5,
        reason="rate_limited",
        rate_limited=True,
    )
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    rate_limit_event = sink.events[-1]
    assert rate_limit_event.event_type == EventTypes.WEBSOCKET_RECONNECT
    assert rate_limit_event.level == "debug"
    assert rate_limit_event.data == {
        "reconnect_no": 4,
        "retry_delay_ms": 2_500,
        "reason": "rate_limited",
        "rate_limited": True,
        "warning_visible": False,
        "traceback_emitted": False,
    }
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_background_candle_warmup_marks_full_warmup_ready(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    now_ms = [10_000]
    calls = []

    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now_ms[0])

    async def warmup_candles_staggered(**kwargs):
        calls.append(kwargs)
        now_ms[0] = 14_250

    bot.warmup_candles_staggered = warmup_candles_staggered
    Passivbot._startup_timing_begin(bot)

    with caplog.at_level(logging.INFO):
        await Passivbot._background_candle_warmup_task(bot)

    assert calls == [{"context": "background warmup"}]
    assert any("full-warmup-ready=4.25s" in record.message for record in caplog.records)


def _set_pnl_lookback(bot, *, lookback_days: float, now_ms: int) -> None:
    bot.config = {"live": {"pnls_max_lookback_days": float(lookback_days)}}
    bot.get_exchange_time = lambda: now_ms


def test_handle_order_update_logs_summary_and_dedupes(caplog, monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.execution_scheduled = False
    bot._authoritative_refresh_epoch = 0
    now = [1000.0]

    monkeypatch.setattr(time, "time", lambda: now[0])

    batch = [
        {"symbol": "SOL/USDC:USDC", "status": "open"},
        {"symbol": "BTC/USDC:USDC", "status": "canceled"},
    ]

    with caplog.at_level(logging.INFO):
        bot.handle_order_update(batch)
        bot.handle_order_update(batch)
        now[0] += 5.1
        bot.handle_order_update(batch)

    ws_logs = [record.message for record in caplog.records if "[ws]" in record.message]
    assert ws_logs == [
        "[ws] order update detected | cause=replace_hint | events=2 | symbols=BTC,SOL | statuses=canceled,open | scheduling refresh",
        "[ws] order update detected | cause=replace_hint | events=2 | symbols=BTC,SOL | statuses=canceled,open | scheduling refresh",
    ]
    assert bot.execution_scheduled is True
    assert bot._authoritative_pending_confirmations == {
        "balance": 1,
        "positions": 1,
        "open_orders": 1,
        "fills": 1,
    }


def test_handle_order_update_ignores_empty_batches(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.execution_scheduled = False

    with caplog.at_level(logging.INFO):
        bot.handle_order_update([])

    assert bot.execution_scheduled is False
    assert not [record.message for record in caplog.records if "[ws]" in record.message]


def test_handle_order_update_suppresses_recent_self_echoes(caplog, monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.execution_scheduled = False
    bot._authoritative_pending_confirmations = {}
    now_ms = 1_000_000
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now_ms)
    bot.recent_order_executions = [
        {
            "symbol": "BTC/USDC:USDC",
            "side": "buy",
            "position_side": "long",
            "price": 70000.0,
            "qty": 0.001,
            "execution_timestamp": now_ms,
        }
    ]
    bot.recent_order_cancellations = [
        {
            "symbol": "ETH/USDC:USDC",
            "side": "sell",
            "position_side": "long",
            "price": 2500.0,
            "qty": 0.01,
            "execution_timestamp": now_ms,
        }
    ]
    batch = [
        {
            "symbol": "BTC/USDC:USDC",
            "side": "buy",
            "position_side": "long",
            "price": 70000.0,
            "qty": 0.001,
            "status": "open",
        },
        {
            "symbol": "ETH/USDC:USDC",
            "side": "sell",
            "position_side": "long",
            "price": 2500.0,
            "qty": 0.01,
            "status": "canceled",
        },
    ]

    with caplog.at_level(logging.INFO):
        bot.handle_order_update(batch)

    assert bot.execution_scheduled is True
    assert bot._authoritative_pending_confirmations == {}
    assert not [record.message for record in caplog.records if "[ws]" in record.message]


def test_handle_order_update_cancel_hint_requests_full_confirmation(
    caplog, monkeypatch
):
    bot = Passivbot.__new__(Passivbot)
    bot.execution_scheduled = False
    bot._authoritative_refresh_epoch = 7
    bot.state_change_detected_by_symbol = set()
    bot.recent_order_cancellations = []

    monkeypatch.setattr(time, "time", lambda: 1000.0)
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 1_000_000)

    with caplog.at_level(logging.DEBUG):
        bot.handle_order_update(
            [
                {
                    "symbol": "BTC/USDT:USDT",
                    "status": "canceled",
                    "id": "manual-cancel",
                }
            ]
        )

    assert bot.execution_scheduled is True
    assert bot.state_change_detected_by_symbol == {"BTC/USDT:USDT"}
    assert bot._authoritative_pending_confirmations == {
        "balance": 8,
        "positions": 8,
        "open_orders": 8,
        "fills": 8,
    }
    assert any("cause=cancel_hint" in record.message for record in caplog.records)
    assert any(
        "account-critical refresh requested" in record.message
        and "reason=order_ws_cancel_hint" in record.message
        for record in caplog.records
    )


def test_handle_order_update_fill_hint_requests_full_confirmation(caplog, monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.execution_scheduled = False
    bot._authoritative_pending_confirmations = {"open_orders": 1}
    bot._authoritative_refresh_epoch = 4
    now = [1000.0]

    monkeypatch.setattr(time, "time", lambda: now[0])

    with caplog.at_level(logging.INFO):
        bot.handle_order_update(
            [{"symbol": "PARA-TOTAL2/USDC:USDC", "status": "closed"}]
        )

    assert bot.execution_scheduled is True
    assert bot._authoritative_pending_confirmations == {
        "open_orders": 5,
        "balance": 5,
        "positions": 5,
        "fills": 5,
    }
    assert any("cause=fill_hint" in record.message for record in caplog.records)


def test_log_staged_refresh_timings_logs_only_for_slow_refreshes(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._authoritative_pending_confirmations = {}
    bot._authoritative_refresh_epoch_changed = set()

    with caplog.at_level(logging.DEBUG):
        bot._log_staged_refresh_timings({"open_orders"}, {"open_orders": 250}, 250)
        bot._log_staged_refresh_timings(
            {"balance", "positions", "open_orders", "fills"},
            {"balance": 250, "positions": 300, "open_orders": 200, "fills": 400},
            650,
        )
        bot._log_staged_refresh_timings(
            {"balance", "positions", "open_orders"},
            {"balance": 500, "positions": 600, "open_orders": 700},
            1700,
        )
        bot._log_staged_refresh_timings(
            {"balance", "positions", "open_orders", "fills"},
            {"balance": 2500, "positions": 3000, "open_orders": 2000, "fills": 4000},
            8500,
        )
        bot._authoritative_refresh_epoch_changed = {"positions"}
        bot._log_staged_refresh_timings(
            {"balance", "positions", "open_orders", "fills"},
            {"balance": 1200, "positions": 1700, "open_orders": 900, "fills": 1300},
            4100,
        )
        bot._authoritative_refresh_epoch_changed = set()
        bot._log_staged_refresh_timings(
            {"balance", "positions", "open_orders", "fills"},
            {"balance": 2500, "positions": 3000, "open_orders": 2000, "fills": 4000},
            10500,
        )

    state_logs = [
        (record.levelname, record.message)
        for record in caplog.records
        if "[state]" in record.message
    ]
    assert state_logs == [
        (
            "DEBUG",
            "[state] staged refresh timings | plan=balance,fills,open_orders,positions | wall=650ms | surface_sum=1150ms | surface_max=400ms | parallel=yes | balance=250ms fills=400ms open_orders=200ms positions=300ms",
        ),
        (
            "DEBUG",
            "[state] staged refresh timings | plan=balance,open_orders,positions | wall=1700ms | surface_sum=1800ms | surface_max=700ms | parallel=yes | balance=500ms open_orders=700ms positions=600ms residual=1000ms residual_hint=scheduler_or_lock_wait",
        ),
        (
            "DEBUG",
            "[state] staged refresh timings | plan=balance,fills,open_orders,positions | wall=8500ms | surface_sum=11500ms | surface_max=4000ms | parallel=yes | balance=2500ms fills=4000ms open_orders=2000ms positions=3000ms residual=4500ms residual_hint=scheduler_or_lock_wait",
        ),
        (
            "DEBUG",
            "[state] staged refresh timings | plan=balance,fills,open_orders,positions | wall=4100ms | surface_sum=5100ms | surface_max=1700ms | parallel=yes | balance=1200ms fills=1300ms open_orders=900ms positions=1700ms residual=2400ms residual_hint=scheduler_or_lock_wait",
        ),
        (
            "INFO",
            "[state] staged refresh timings | plan=balance,fills,open_orders,positions | wall=10500ms | surface_sum=11500ms | surface_max=4000ms | parallel=yes | balance=2500ms fills=4000ms open_orders=2000ms positions=3000ms residual=6500ms residual_hint=scheduler_or_lock_wait",
        ),
    ]


def test_log_staged_refresh_timings_keeps_meaningful_change_structured_at_info(caplog):
    sink = ListEventSink()
    bot = Passivbot.__new__(Passivbot)
    bot._live_event_current_cycle_id = "cy_refresh"
    bot._live_event_pipeline = LiveEventPipeline(structured_sinks=[sink], monitor_sinks=[])
    bot._authoritative_pending_confirmations = {}
    bot._authoritative_refresh_epoch_changed = {"positions"}

    with caplog.at_level(logging.DEBUG):
        bot._log_staged_refresh_timings(
            {"balance", "positions", "open_orders", "fills"},
            {"balance": 1200, "positions": 1700, "open_orders": 900, "fills": 1300},
            4100,
        )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    timing_logs = [
        record
        for record in caplog.records
        if "[state] staged refresh timings" in record.message
    ]
    assert len(timing_logs) == 1
    assert timing_logs[0].levelno == logging.DEBUG
    event = sink.events[0]
    assert event.event_type == EventTypes.STATE_REFRESH_TIMING
    assert event.level == "info"
    assert event.component == "state.refresh"
    assert event.tags == ("state", "refresh", "account")
    assert event.cycle_id == "cy_refresh"
    assert event.status == "succeeded"
    assert event.reason_code == ReasonCodes.STAGED_REFRESH_TIMING
    assert event.data["plan"] == ["balance", "fills", "open_orders", "positions"]
    assert event.data["timings_ms"] == {
        "balance": 1200,
        "fills": 1300,
        "open_orders": 900,
        "positions": 1700,
    }
    assert event.data["wall_ms"] == 4100
    assert event.data["surface_sum_ms"] == 5100
    assert event.data["surface_max_ms"] == 1700
    assert event.data["residual_ms"] == 2400
    assert event.data["meaningful_change"] is True
    assert event.data["epoch_changed"] == ["positions"]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_log_staged_refresh_timings_skips_structured_debug_sample():
    sink = ListEventSink()
    bot = Passivbot.__new__(Passivbot)
    bot._live_event_current_cycle_id = "cy_refresh_debug"
    bot._live_event_pipeline = LiveEventPipeline(structured_sinks=[sink], monitor_sinks=[])
    bot._authoritative_pending_confirmations = {}
    bot._authoritative_refresh_epoch_changed = set()

    bot._log_staged_refresh_timings(
        {"balance", "positions", "open_orders", "fills"},
        {"balance": 250, "positions": 300, "open_orders": 200, "fills": 400},
        650,
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert sink.events == []
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_routine_completed_candle_defers_are_debug_with_periodic_summary(
    caplog, monkeypatch
):
    now_ms = {"value": 0}
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now_ms["value"])
    bot = Passivbot.__new__(Passivbot)
    details = {
        "missing": ["completed_candles"],
        "required": ["balance", "completed_candles", "positions"],
        "epoch": 7,
        "context": "market snapshot refresh",
        "invalid": {
            "completed_candles": [
                {
                    "reason": "signature_mismatch",
                    "mismatch_type": "completed_candle_target_changed",
                    "expected_count": 2,
                    "stamped_count": 2,
                    "changed_count": 2,
                    "changed_symbols": ["TON/USDT:USDT", "ZEC/USDT:USDT"],
                }
            ]
        },
    }

    with caplog.at_level(logging.DEBUG):
        bot._log_staged_execution_defer(details)
        now_ms["value"] = 31 * 60_000
        bot._log_staged_execution_defer(details)

    individual = [
        record
        for record in caplog.records
        if "staged planning deferred: completed candle target changed or missing"
        in record.message
    ]
    assert individual
    assert all(record.levelno == logging.DEBUG for record in individual)
    assert any(
        "staged planning deferred summary" in record.message
        and "reason=completed_candle_target_changed" in record.message
        and record.levelno == logging.INFO
        for record in caplog.records
    )


def test_order_plan_summary_is_interesting_only_for_large_or_clipped_waves():
    bot = Passivbot.__new__(Passivbot)

    assert (
        bot._order_plan_summary_is_interesting(
            total_pre_cancel=1,
            total_cancel=1,
            total_pre_create=1,
            total_create=1,
            total_skipped=0,
        )
        is False
    )
    assert (
        bot._order_plan_summary_is_interesting(
            total_pre_cancel=6,
            total_cancel=3,
            total_pre_create=5,
            total_create=2,
            total_skipped=4,
        )
        is True
    )
    assert (
        bot._order_plan_summary_is_interesting(
            total_pre_cancel=10,
            total_cancel=0,
            total_pre_create=10,
            total_create=0,
            total_skipped=10,
        )
        is False
    )


def test_order_wave_summary_logs_elapsed_lifecycle(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._order_wave_seq = 0
    bot._order_wave_last_summary_key = None
    clock = iter([1_000_000, 1_002_500])
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: next(clock))

    wave = bot._begin_order_wave(
        [{"symbol": "BTC/USDT:USDT"}],
        [{"symbol": "ETH/USDT:USDT"}],
    )
    wave["cancel_posted"] = 1
    wave["create_posted"] = 1
    wave["cancel_ms"] = 400
    wave["create_ms"] = 600

    with caplog.at_level(logging.INFO):
        bot._log_order_wave_summary(wave)

    assert any(
        "[order] wave complete | id=1 | elapsed_ms=2500 | cancel 1->1 | create 1->1"
        in record.message
        and "symbols=BTC,ETH" in record.message
        for record in caplog.records
    )


def test_order_wave_summary_uses_event_console_when_available(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._order_wave_seq = 0
    bot._order_wave_last_summary_key = None
    bot.live_event_console_enabled = True
    bot._live_event_pipeline = object()
    emitted = []
    bot._emit_order_wave_completed_event = lambda wave, **kwargs: emitted.append(
        (wave, kwargs)
    )
    clock = iter([1_000_000, 1_002_500])
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: next(clock))

    wave = bot._begin_order_wave(
        [{"symbol": "BTC/USDT:USDT"}],
        [{"symbol": "ETH/USDT:USDT"}],
    )
    wave["cancel_posted"] = 1
    wave["create_posted"] = 1

    with caplog.at_level(logging.INFO):
        bot._log_order_wave_summary(wave)

    assert emitted
    assert emitted[0][1]["elapsed_ms"] == 2500
    assert emitted[0][1]["level"] == "info"
    assert not any("[order] wave complete" in record.message for record in caplog.records)


def test_order_wave_settlement_logs_authoritative_confirmation(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._order_wave_seq = 0
    bot._pending_order_waves = []
    times = iter([1_000_000, 1_000_500, 1_003_000, 1_003_000])
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: next(times))

    wave = bot._begin_order_wave(
        [{"symbol": "BTC/USDT:USDT"}],
        [{"symbol": "ETH/USDT:USDT"}],
    )
    bot._authoritative_refresh_epoch = 4
    bot._authoritative_pending_confirmations = {}
    bot._order_wave_in_progress = wave
    bot._request_authoritative_confirmation({"open_orders"})
    bot._order_wave_in_progress = None
    wave["cancel_posted"] = 1
    wave["create_posted"] = 1
    bot._track_order_wave_confirmation(wave)

    bot._authoritative_refresh_epoch = 5
    bot._authoritative_refresh_epoch_fresh = {"open_orders"}
    bot._authoritative_refresh_epoch_changed = {"open_orders", "positions"}

    with caplog.at_level(logging.INFO):
        blocked, details = bot._authoritative_execution_barrier_state()

    assert blocked is False
    assert details["missing"] == []
    assert bot._pending_order_waves == []
    assert bot._authoritative_pending_confirmations == {}
    assert any(
        "[order] wave settled | id=1 | elapsed_ms=3000 | confirm_ms=2500"
        in record.message
        and "confirmed=open_orders" in record.message
        and "changed=open_orders,positions" in record.message
        and "symbols=BTC,ETH" in record.message
        for record in caplog.records
    )


def test_order_wave_settlement_uses_event_console_when_available(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._order_wave_seq = 0
    bot._pending_order_waves = []
    bot.live_event_console_enabled = True
    bot._live_event_pipeline = object()
    emitted = []
    bot._emit_execution_confirmation_satisfied_event = (
        lambda **kwargs: emitted.append(kwargs)
    )
    times = iter([1_000_000, 1_000_500, 1_003_000, 1_003_000])
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: next(times))

    wave = bot._begin_order_wave(
        [{"symbol": "BTC/USDT:USDT"}],
        [{"symbol": "ETH/USDT:USDT"}],
    )
    bot._authoritative_refresh_epoch = 4
    bot._authoritative_pending_confirmations = {}
    bot._order_wave_in_progress = wave
    bot._request_authoritative_confirmation({"open_orders"})
    bot._order_wave_in_progress = None
    wave["cancel_posted"] = 1
    wave["create_posted"] = 1
    bot._track_order_wave_confirmation(wave)

    with caplog.at_level(logging.INFO):
        bot._log_settled_order_waves(
            current_epoch=5,
            fresh_surfaces={"open_orders"},
            changed_surfaces=["open_orders", "positions"],
        )

    assert emitted
    assert emitted[0]["elapsed_ms"] == 3000
    assert emitted[0]["confirm_ms"] == 2500
    assert emitted[0]["level"] == "info"
    assert bot._pending_order_waves == []
    assert not any("[order] wave settled" in record.message for record in caplog.records)


def test_order_wave_settlement_demotes_quick_open_orders_confirmation(
    monkeypatch, caplog
):
    bot = Passivbot.__new__(Passivbot)
    bot._pending_order_waves = [
        {
            "id": 7,
            "started_ms": 1_000_000,
            "posted_ms": 1_000_500,
            "cancel_posted": 1,
            "create_posted": 1,
            "symbols": ["BTC/USDT:USDT"],
            "confirmations": {"open_orders": 5},
        }
    ]
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 1_003_000)

    with caplog.at_level(logging.DEBUG):
        bot._log_settled_order_waves(
            current_epoch=5,
            fresh_surfaces={"open_orders"},
            changed_surfaces=["open_orders"],
        )

    settled_records = [
        record for record in caplog.records if "[order] wave settled" in record.message
    ]
    assert len(settled_records) == 1
    assert settled_records[0].levelno == logging.DEBUG
    assert bot._pending_order_waves == []


@pytest.mark.asyncio
async def test_update_pnls_routine_empty_refresh_timing_demoted_to_debug(
    monkeypatch, caplog
):
    bot = Passivbot.__new__(Passivbot)
    cached_events = [
        SimpleNamespace(timestamp=1_700_000_000_000, id="fill-1", source_ids=["fill-1"])
    ]

    class _Manager:
        def __init__(self, events):
            self._events = list(events)
            self.refresh = AsyncMock()
            self.refresh_latest = AsyncMock()
            self.history_scope = "all"

        def get_events(self):
            return list(self._events)

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    times = iter([1_000_000, 1_030_000])
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: next(times, 1_030_000))
    bot.stop_signal_received = False
    bot.config = {
        "live": {
            "fills_recent_overlap_minutes": 10.0,
            "pnls_max_lookback_days": "all",
        }
    }
    bot._authoritative_pending_confirmations = {}
    bot._pnls_manager = _Manager(cached_events)
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    bot.get_exchange_time = lambda: 1_700_000_060_000
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: None
    bot.logging_level = 0
    bot._health_rate_limits = 0

    with caplog.at_level(logging.DEBUG):
        result = await bot.update_pnls(source="routine_prefetch:minute_boundary")

    assert result is True
    fill_timing_records = [
        record for record in caplog.records if "[fills] refresh timing" in record.message
    ]
    assert len(fill_timing_records) == 1
    assert fill_timing_records[0].levelno == logging.DEBUG
    assert "elapsed=30000ms" in fill_timing_records[0].message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "add_new_fill"),
    [
        ("staged_blocking", False),
        ("routine_prefetch:minute_boundary", True),
    ],
    ids=["staged_source", "new_fill"],
)
async def test_update_pnls_completed_refresh_timing_trigger_cases_stay_debug(
    monkeypatch, caplog, source, add_new_fill
):
    bot = Passivbot.__new__(Passivbot)
    cached_event = SimpleNamespace(
        timestamp=1_700_000_000_000,
        id="fill-1",
        source_ids=["fill-1"],
    )
    refreshed_events = [cached_event]
    if add_new_fill:
        refreshed_events.append(
            SimpleNamespace(
                timestamp=1_700_000_060_000,
                id="fill-2",
                source_ids=["fill-2"],
            )
        )

    class _Manager:
        def __init__(self):
            self._events = [cached_event]
            self.history_scope = "all"

        async def refresh_latest(self, **kwargs):
            self._events = list(refreshed_events)

        def get_events(self):
            return list(self._events)

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    times = iter([1_000_000, 1_001_000])
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: next(times, 1_001_000))
    bot.stop_signal_received = False
    bot.config = {
        "live": {
            "fills_recent_overlap_minutes": 10.0,
            "pnls_max_lookback_days": "all",
        }
    }
    bot._authoritative_pending_confirmations = {}
    bot._pnls_manager = _Manager()
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    bot.get_exchange_time = lambda: 1_700_000_060_000
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: None
    refresh_summaries = []
    bot._emit_fills_refresh_summary_event = lambda **kwargs: refresh_summaries.append(
        kwargs
    )
    bot.logging_level = 0
    bot._health_rate_limits = 0

    with caplog.at_level(logging.DEBUG):
        result = await bot.update_pnls(source=source)

    assert result is True
    fill_timing_records = [
        record for record in caplog.records if "[fills] refresh timing" in record.message
    ]
    assert len(fill_timing_records) == 1
    assert fill_timing_records[0].levelno == logging.DEBUG
    assert f"source={source}" in fill_timing_records[0].message
    assert f"new={int(add_new_fill)}" in fill_timing_records[0].message
    assert len(refresh_summaries) == 1
    assert refresh_summaries[0]["level"] == "info"


def test_min_effective_cost_blocks_are_aggregated(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._min_effective_cost_last_log_ms = {}
    bot._min_effective_cost_log_interval_ms = 60 * 60 * 1000
    bot._min_effective_cost_summary_last_log_ms = 0
    bot._min_effective_cost_summary_log_interval_ms = 60 * 60 * 1000
    bot.is_pside_enabled = lambda _pside: True
    idx_to_symbol = {idx: f"SYM{idx}/USDT:USDT" for idx in range(5)}
    out = {
        "diagnostics": {
            "min_effective_cost_blocks": [
                {
                    "symbol_idx": idx,
                    "pside": "long",
                    "balance": 1000.0,
                    "effective_limit": 0.1,
                    "entry_initial_qty_pct": 0.01,
                    "projected_initial_cost": 1.0,
                    "effective_min_cost": 10.0,
                }
                for idx in range(5)
            ]
        }
    }

    with caplog.at_level(logging.INFO):
        bot._log_min_effective_cost_blocks(out, idx_to_symbol)

    infos = [record.message for record in caplog.records if record.levelno == logging.INFO]
    warnings = [
        record.message for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert sum("initial entry blocked by min effective cost | SYM" in msg for msg in infos) == 3
    assert not warnings
    assert any("blocked=5 detailed=3 suppressed=0" in msg for msg in infos)


def test_forager_selection_summary_formatter_exact_format():
    message = Passivbot._format_forager_selection_summary(
        pside="long",
        slots_to_fill=1,
        selected_symbols=("DOGE", "SOL", "ETH"),
        incumbent_symbols=("DOGE",),
        score_hysteresis_pct=0.005,
        reason="selection_changed",
        top_scores=[("SOL", 1, 0.625), ("DOGE", 2, 0.623)],
        hysteresis_events=[("DOGE", "SOL", 0.002, True)],
    )

    assert message == (
        "[forager] long selection | slots=1 selected=DOGE,SOL,ETH incumbents=DOGE "
        "hyst=0.005 reason=selection_changed events=kept:DOGE<->SOL gap=0.002 "
        "top=1:SOL=0.625,2:DOGE=0.623"
    )


def test_forager_selection_summary_formatter_bounds_hostile_tokens_for_live_console():
    hostile = "bad, field;|\n\t" + ("x" * 300)
    message = Passivbot._format_forager_selection_summary(
        pside=hostile,
        slots_to_fill=10**20,
        selected_symbols=tuple(hostile for _ in range(10)),
        incumbent_symbols=tuple(hostile for _ in range(10)),
        score_hysteresis_pct=0.005,
        reason=hostile,
        top_scores=[(hostile, 10**20, 0.625) for _ in range(10)],
        hysteresis_events=[(hostile, hostile, 0.002, True) for _ in range(10)],
    )
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 0, "%s", (message,), None
    )
    record.created = 0.0
    record.log_prefix = "hyperliquid"
    formatter = logging.Formatter(DEFAULT_FORMAT_WITH_PREFIX, datefmt=DEFAULT_DATEFMT)
    formatter.converter = time.gmtime
    rendered = formatter.format(record)

    assert "\n" not in message
    assert "\r" not in message
    assert "\t" not in message
    assert hostile not in message
    assert "selected=bad__...,bad__...,bad__...,+7" in message
    assert "incumbents=bad__...,bad__...,bad__...,+7" in message
    assert len(message) <= Passivbot.FORAGER_SELECTION_CONSOLE_MESSAGE_MAX_LEN
    assert Passivbot.FORAGER_SELECTION_CONSOLE_MESSAGE_MAX_LEN == 240 - (
        len(rendered) - len(message)
    )
    assert len(rendered) <= 240


def test_forager_selection_diagnostics_preserves_transition_and_cadence(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._forager_selection_info_interval_ms = 1_000
    bot._forager_selection_debug_interval_ms = 1_000
    now = [1_000]
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now[0])
    base = {
        "diagnostics": {
            "forager_selections": [
                {
                    "pside": "long",
                    "slots_to_fill": 1,
                    "score_hysteresis_pct": 0.005,
                    "selected_symbol_indices": [0],
                    "incumbent_symbol_indices": [0],
                    "top_scores": [
                        {"symbol_idx": 0, "rank": 1, "score": 0.625, "selected": True},
                        {"symbol_idx": 1, "rank": 2, "score": 0.623, "selected": False},
                    ],
                    "hysteresis_events": [
                        {
                            "incumbent_symbol_idx": 0,
                            "challenger_symbol_idx": 1,
                            "score_gap": 0.002,
                            "kept_incumbent": True,
                        }
                    ],
                }
            ]
        }
    }
    replacement = deepcopy(base)
    selection = replacement["diagnostics"]["forager_selections"][0]
    selection["selected_symbol_indices"] = [1]
    selection["incumbent_symbol_indices"] = [1]
    selection["hysteresis_events"][0]["kept_incumbent"] = False
    idx_to_symbol = {0: "SOL/USDT:USDT", 1: "DOGE/USDT:USDT"}

    with caplog.at_level(logging.DEBUG):
        Passivbot._log_forager_selection_diagnostics(bot, base, idx_to_symbol)
        now[0] = 1_001
        Passivbot._log_forager_selection_diagnostics(bot, replacement, idx_to_symbol)
        now[0] = 2_001
        Passivbot._log_forager_selection_diagnostics(bot, replacement, idx_to_symbol)

    info_messages = [
        record.message
        for record in caplog.records
        if record.levelno == logging.INFO and "[forager] long selection" in record.message
    ]
    debug_messages = [
        record.message
        for record in caplog.records
        if record.levelno == logging.DEBUG and "[forager] long score detail" in record.message
    ]
    assert [message.split("reason=", 1)[1].split(" ", 1)[0] for message in info_messages] == [
        "selection_changed",
        "hysteresis_replacement",
        "periodic",
    ]
    assert "events=replaced:SOL<->DOGE gap=0.002" in info_messages[1]
    assert len(debug_messages) == 3


def test_forager_selection_compact_info_preserves_structured_and_debug_detail(caplog):
    bot = Passivbot.__new__(Passivbot)
    sink = ListEventSink()
    bot.exchange = "binance"
    bot.user = "binance_01"
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_forager_compact"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink], monitor_sinks=[]
    )
    long_coin = "X" * 80
    challenger_coin = "Y" * 80
    symbol = f"{long_coin}/USDT:USDT"
    challenger = f"{challenger_coin}/USDT:USDT"
    out = {
        "diagnostics": {
            "forager_selections": [
                {
                    "pside": "long",
                    "slots_to_fill": 1,
                    "score_hysteresis_pct": 0.005,
                    "selected_symbol_indices": [0],
                    "incumbent_symbol_indices": [0],
                    "top_scores": [
                        {
                            "symbol_idx": 0,
                            "rank": 1,
                            "score": 0.625,
                            "volume_component": 0.4,
                            "ema_readiness_component": 0.5,
                            "volatility_component": 1.0,
                            "selected": True,
                            "incumbent": True,
                        }
                    ],
                    "hysteresis_events": [
                        {
                            "incumbent_symbol_idx": 0,
                            "challenger_symbol_idx": 1,
                            "score_gap": 0.002,
                            "kept_incumbent": True,
                        }
                    ],
                }
            ]
        }
    }

    with caplog.at_level(logging.DEBUG):
        Passivbot._log_forager_selection_diagnostics(
            bot, out, {0: symbol, 1: challenger}
        )

    info_message = next(
        record.message
        for record in caplog.records
        if record.levelno == logging.INFO and "[forager] long selection" in record.message
    )
    debug_message = next(
        record.message
        for record in caplog.records
        if record.levelno == logging.DEBUG and "[forager] long score detail" in record.message
    )
    assert long_coin not in info_message
    assert long_coin in debug_message
    assert "vol=0.400" in debug_message
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = next(
        event for event in sink.events if event.event_type == EventTypes.FORAGER_SELECTION
    )
    assert event.data["selected_symbols"] == [symbol]
    assert event.data["incumbent_symbols"] == [symbol]
    assert event.data["top_scores"] == [
        {
            "symbol": symbol,
            "rank": 1,
            "score": 0.625,
            "volume_component": 0.4,
            "ema_readiness_component": 0.5,
            "volatility_component": 1.0,
            "selected": True,
            "incumbent": True,
        }
    ]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_forager_selection_diagnostics_log_scores_and_hysteresis(caplog):
    bot = Passivbot.__new__(Passivbot)
    sink = ListEventSink()
    bot.exchange = "binance"
    bot.user = "binance_01"
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_forager"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    out = {
        "diagnostics": {
            "forager_selections": [
                {
                    "pside": "long",
                    "slots_to_fill": 1,
                    "score_hysteresis_pct": 0.005,
                    "selected_symbol_indices": [1],
                    "incumbent_symbol_indices": [1],
                    "top_scores": [
                        {
                            "symbol_idx": 0,
                            "rank": 1,
                            "score": 0.625,
                            "volume_component": 0.4,
                            "ema_readiness_component": 0.5,
                            "volatility_component": 1.0,
                            "selected": False,
                            "incumbent": False,
                        },
                        {
                            "symbol_idx": 1,
                            "rank": 2,
                            "score": 0.623,
                            "volume_component": 0.3,
                            "ema_readiness_component": 0.6,
                            "volatility_component": 0.9,
                            "selected": True,
                            "incumbent": True,
                        },
                    ],
                    "hysteresis_events": [
                        {
                            "incumbent_symbol_idx": 1,
                            "incumbent_score": 0.623,
                            "challenger_symbol_idx": 0,
                            "challenger_score": 0.625,
                            "score_gap": 0.002,
                            "kept_incumbent": True,
                        }
                    ],
                }
            ]
        }
    }
    idx_to_symbol = {0: "SOL/USDT:USDT", 1: "DOGE/USDT:USDT"}

    with caplog.at_level(logging.DEBUG):
        Passivbot._log_forager_selection_diagnostics(bot, out, idx_to_symbol)
        Passivbot._log_forager_selection_diagnostics(bot, out, idx_to_symbol)

    messages = [record.message for record in caplog.records]
    info_messages = [
        record.message
        for record in caplog.records
        if record.levelno == logging.INFO
        and "[forager] long selection" in record.message
    ]
    assert len(info_messages) == 1
    assert any("selected=DOGE" in msg for msg in messages)
    assert any("kept:DOGE<->SOL" in msg for msg in messages)
    assert not any("DOGE/USDT:USDT" in msg for msg in messages)
    assert any("[forager] long score detail" in msg for msg in messages)
    assert any("vol=0.400" in msg for msg in messages)
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.FORAGER_SELECTION
    ]
    assert len(events) == 2
    assert events[0].cycle_id == "cy_forager"
    assert events[0].level == "info"
    assert events[0].data["source"] == "rust_orchestrator"
    assert events[0].data["selected_symbols"] == ["DOGE/USDT:USDT"]
    assert events[0].data["incumbent_symbols"] == ["DOGE/USDT:USDT"]
    assert events[0].data["hysteresis_event_count"] == 1
    assert events[0].data["top_scores"][0]["symbol"] == "SOL/USDT:USDT"
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_forager_selection_diagnostics_demotes_rank_only_changes(caplog):
    bot = Passivbot.__new__(Passivbot)
    base = {
        "diagnostics": {
            "forager_selections": [
                {
                    "pside": "long",
                    "slots_to_fill": 2,
                    "score_hysteresis_pct": 0.005,
                    "selected_symbol_indices": [0, 1],
                    "incumbent_symbol_indices": [0, 1],
                    "top_scores": [
                        {"symbol_idx": 0, "rank": 1, "score": 0.625, "selected": True},
                        {"symbol_idx": 1, "rank": 2, "score": 0.623, "selected": True},
                    ],
                    "hysteresis_events": [],
                }
            ]
        }
    }
    rank_changed = deepcopy(base)
    rank_changed["diagnostics"]["forager_selections"][0][
        "selected_symbol_indices"
    ] = [1, 0]
    rank_changed["diagnostics"]["forager_selections"][0]["top_scores"] = [
        {"symbol_idx": 1, "rank": 1, "score": 0.626, "selected": True},
        {"symbol_idx": 0, "rank": 2, "score": 0.624, "selected": True},
    ]
    idx_to_symbol = {0: "SOL/USDT:USDT", 1: "DOGE/USDT:USDT"}

    with caplog.at_level(logging.DEBUG):
        Passivbot._log_forager_selection_diagnostics(bot, base, idx_to_symbol)
        Passivbot._log_forager_selection_diagnostics(bot, rank_changed, idx_to_symbol)

    info_messages = [
        record.message
        for record in caplog.records
        if record.levelno == logging.INFO
        and "[forager] long selection" in record.message
    ]
    debug_messages = [
        record.message
        for record in caplog.records
        if record.levelno == logging.DEBUG
        and "[forager] long score detail" in record.message
    ]
    assert len(info_messages) == 1
    assert len(debug_messages) == 2


def test_forager_selection_diagnostics_demotes_no_incumbent_selection_churn(caplog):
    bot = Passivbot.__new__(Passivbot)
    base = {
        "diagnostics": {
            "forager_selections": [
                {
                    "pside": "long",
                    "slots_to_fill": 1,
                    "score_hysteresis_pct": 0.02,
                    "selected_symbol_indices": [0],
                    "incumbent_symbol_indices": [],
                    "top_scores": [
                        {"symbol_idx": 0, "rank": 1, "score": 0.625, "selected": True},
                        {"symbol_idx": 1, "rank": 2, "score": 0.623, "selected": False},
                    ],
                    "hysteresis_events": [],
                }
            ]
        }
    }
    changed = deepcopy(base)
    changed["diagnostics"]["forager_selections"][0]["selected_symbol_indices"] = [1]
    changed["diagnostics"]["forager_selections"][0]["top_scores"] = [
        {"symbol_idx": 1, "rank": 1, "score": 0.626, "selected": True},
        {"symbol_idx": 0, "rank": 2, "score": 0.624, "selected": False},
    ]
    idx_to_symbol = {0: "SOL/USDT:USDT", 1: "DOGE/USDT:USDT"}

    with caplog.at_level(logging.DEBUG):
        Passivbot._log_forager_selection_diagnostics(bot, base, idx_to_symbol)
        Passivbot._log_forager_selection_diagnostics(bot, changed, idx_to_symbol)

    info_messages = [
        record.message
        for record in caplog.records
        if record.levelno == logging.INFO
        and "[forager] long selection" in record.message
    ]
    assert len(info_messages) == 1
    assert any(
        "selection changed without incumbents/events" in record.message
        for record in caplog.records
        if record.levelno == logging.DEBUG
    )


def test_active_candle_incomplete_publish_lag_is_info_throttled(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._active_candle_incomplete_last_log_ms = {}
    times = iter([1_000_000, 1_060_000, 1_400_001])
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: next(times))
    ordered = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    missing = [
        {
            "symbol": "BTC/USDT:USDT",
            "reason": "missing_latest_completed_1m",
            "missing_candles": 1,
        }
    ]

    with caplog.at_level(logging.DEBUG):
        bot._log_active_candle_refresh_incomplete(ordered, missing)
        bot._log_active_candle_refresh_incomplete(ordered, missing)
        bot._log_active_candle_refresh_incomplete(ordered, missing)

    levels = [
        record.levelname
        for record in caplog.records
        if "active completed-candle refresh incomplete" in record.message
    ]
    assert levels == ["INFO", "DEBUG", "DEBUG"]
    assert any("likely_publish_lag=yes" in record.message for record in caplog.records)


def test_active_candle_incomplete_actionable_gap_still_warns(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._active_candle_incomplete_last_log_ms = {}
    times = iter([1_000_000, 1_060_000, 1_400_001])
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: next(times))
    ordered = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    missing = [
        {
            "symbol": "BTC/USDT:USDT",
            "reason": "missing_latest_completed_1m",
            "missing_candles": 2,
        }
    ]

    with caplog.at_level(logging.DEBUG):
        bot._log_active_candle_refresh_incomplete(ordered, missing)
        bot._log_active_candle_refresh_incomplete(ordered, missing)
        bot._log_active_candle_refresh_incomplete(ordered, missing)

    levels = [
        record.levelname
        for record in caplog.records
        if "active completed-candle refresh incomplete" in record.message
    ]
    assert levels == ["WARNING", "DEBUG", "WARNING"]


def test_candle_health_missing_trailing_1m_gap_is_not_actionable_during_grace():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"candle_health_trailing_grace_ms": 60_000}}

    assert (
        bot._candle_health_missing_is_actionable(
            "1m",
            True,
            1,
            {"last_cached_age_ms": 60_000, "refresh_age_ms": 30_000},
        )
        is False
    )
    assert (
        bot._candle_health_missing_is_actionable(
            "1m",
            True,
            1,
            {"last_cached_age_ms": 60_000, "refresh_age_ms": 61_000},
        )
        is True
    )
    assert (
        bot._candle_health_missing_is_actionable(
            "1m",
            True,
            2,
            {"last_cached_age_ms": 120_000, "refresh_age_ms": 30_000},
        )
        is True
    )
    assert (
        bot._candle_health_missing_is_actionable(
            "1h",
            True,
            1,
            {"last_cached_age_ms": 3_600_000, "refresh_age_ms": 30_000},
        )
        is True
    )


def test_candle_health_summary_formatter_exact_format():
    message = Passivbot._format_candle_health_summary(
        symbol_count=3,
        unhealthy_surface_count=3,
        stale_count=2,
        synthetic_count=4,
        worst_missing=6,
        unhealthy_samples=[
            ("BTC", "1m", 2, 3),
            ("ETH", "1h", 1, None),
            ("SOL", "15m", 6, None),
        ],
    )

    assert message == (
        "[candle] health: symbols=3 unhealthy_surfaces=3 stale=2 synthetic=4 "
        "worst_missing=6 | BTC 1m missing=2 tail=3; ETH 1h missing=1; SOL 15m missing=6"
    )


def test_candle_health_summary_formatter_bounds_hostile_labels_for_live_console():
    hostile = "bad label;|\n\t" + ("x" * 100)
    message = Passivbot._format_candle_health_summary(
        symbol_count=10**20,
        unhealthy_surface_count=10,
        stale_count=10**20,
        synthetic_count=10**20,
        worst_missing=10**20,
        unhealthy_samples=[(hostile, hostile, 10**20, 10**20) for _ in range(10)],
    )
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 0, "%s", (message,), None
    )
    record.created = 0.0
    record.log_prefix = "hyperliquid"
    formatter = logging.Formatter(DEFAULT_FORMAT_WITH_PREFIX, datefmt=DEFAULT_DATEFMT)
    formatter.converter = time.gmtime
    rendered = formatter.format(record)

    assert "\n" not in message
    assert "\r" not in message
    assert "\t" not in message
    assert "+9 more" in message
    assert len(message) <= Passivbot.CANDLE_HEALTH_CONSOLE_MESSAGE_MAX_LEN
    assert Passivbot.CANDLE_HEALTH_CONSOLE_MESSAGE_MAX_LEN == 240 - (
        len(rendered) - len(message)
    )
    assert len(rendered) <= 240


def test_candle_health_summary_projection_preserves_debug_payload_and_transition(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.active_symbols = ["BTC/USDT:USDT"]
    bot.open_orders = {}
    bot.positions = {}
    bot.config = {"live": {}}
    report = {
        "symbols": {
            "BTC/USDT:USDT": {
                "timeframes": {
                    "1m": {
                        "coverage_ok": False,
                        "required": True,
                        "missing_candles": 2,
                        "last_cached_ts": 0,
                        "end_ts": 60_000,
                        "runtime_synthetic_count": 3,
                        "open_tail_gap": True,
                        "tail_gap_candles": 2,
                    }
                }
            }
        }
    }
    calls = []

    def build_report(symbols):
        calls.append(set(symbols))
        return report

    monkeypatch.setattr(bot, "build_required_candle_health_report", build_report)

    with caplog.at_level(logging.DEBUG):
        bot._maybe_log_candle_health_summary()
        bot._maybe_log_candle_health_summary()

    info_messages = [
        record.message
        for record in caplog.records
        if record.levelno == logging.INFO and record.message.startswith("[candle] health:")
    ]
    debug_records = [
        record
        for record in caplog.records
        if record.message.startswith("[candle] health diagnostics |")
    ]
    assert calls == [{"BTC/USDT:USDT"}, {"BTC/USDT:USDT"}]
    assert info_messages == [
        "[candle] health: symbols=1 unhealthy_surfaces=1 stale=0 synthetic=3 "
        "worst_missing=2 | BTC 1m missing=2 tail=2"
    ]
    assert len(debug_records) == 2
    assert debug_records[0].args[1] == {
        "BTC/USDT:USDT": {
            "1m": {
                "ok": False,
                "required": True,
                "missing": 2,
                "last_cached_ts": 0,
                "synthetic": 3,
                "open_tail_gap": True,
                "tail_gap_candles": 2,
            }
        }
    }


def test_memory_snapshot_is_interesting_only_initially_or_on_large_change():
    bot = Passivbot.__new__(Passivbot)

    assert bot._memory_snapshot_is_interesting(prev=None, pct_change=None) is True
    assert (
        bot._memory_snapshot_is_interesting(prev={"rss": 100}, pct_change=10.0) is False
    )
    assert (
        bot._memory_snapshot_is_interesting(prev={"rss": 100}, pct_change=24.999)
        is False
    )
    assert (
        bot._memory_snapshot_is_interesting(prev={"rss": 100}, pct_change=25.0)
        is True
    )
    assert (
        bot._memory_snapshot_is_interesting(prev={"rss": 100}, pct_change=-25.0)
        is True
    )
    assert (
        bot._memory_snapshot_is_interesting(prev={"rss": 100}, pct_change=30.0) is True
    )


def _memory_snapshot_bot():
    bot = Passivbot.__new__(Passivbot)
    bot.cm = SimpleNamespace(
        _cache={
            "BTC/USDT:USDT": np.zeros((3, 6), dtype=np.float64),
            "ETH/USDT:USDT": np.zeros((2, 6), dtype=np.float64),
        },
        _tf_range_cache={
            "BTC/USDT:USDT": {("1m", 1): (np.zeros((4, 6), dtype=np.float64),)},
        },
    )
    bot._mem_log_prev = None
    bot.live_event_console_enabled = False
    bot._live_event_pipeline = None
    return bot


def test_memory_snapshot_event_falls_back_to_compact_info_and_keeps_debug_detail(
    monkeypatch, caplog
):
    bot = _memory_snapshot_bot()
    emitted = []
    bot._emit_memory_snapshot_event = lambda **kwargs: emitted.append(kwargs)
    monkeypatch.setattr(
        passivbot_module, "_get_process_rss_bytes", lambda: 198 * 1024 * 1024
    )

    with caplog.at_level(logging.DEBUG):
        bot._log_memory_snapshot(now_ms=1_000)

    assert len(emitted) == 1
    assert emitted[0]["rss_bytes"] == 198 * 1024 * 1024
    assert len(emitted[0]["cache_samples"]) == 2
    assert len(emitted[0]["timeframe_cache_samples"]) == 1
    info_messages = [
        record.message
        for record in caplog.records
        if record.levelno == logging.INFO and record.message.startswith("[memory]")
    ]
    assert info_messages == [
        "[memory] rss=198.0MiB cache=0.0MiB/2sym tf=0.0MiB/1rng"
    ]
    assert all(len(message) <= 240 for message in info_messages)
    assert any("cm_top=BTC" in record.message for record in caplog.records)
    assert bot._mem_log_prev == {
        "timestamp": 1_000,
        "rss": 198 * 1024 * 1024,
        "cm_cache_bytes": 240,
    }


def test_memory_snapshot_structured_console_owns_info_and_routine_stays_debug(
    monkeypatch, caplog
):
    bot = _memory_snapshot_bot()
    emitted = []
    bot._emit_memory_snapshot_event = lambda **kwargs: emitted.append(kwargs)
    bot.live_event_console_enabled = True
    bot._live_event_pipeline = SimpleNamespace(console_sink=object())
    monkeypatch.setattr(
        passivbot_module, "_get_process_rss_bytes", lambda: 198 * 1024 * 1024
    )

    with caplog.at_level(logging.DEBUG):
        bot._log_memory_snapshot(now_ms=1_000)

    assert len(emitted) == 1
    assert not any(
        record.levelno == logging.INFO and record.message.startswith("[memory]")
        for record in caplog.records
    )
    assert any(
        record.levelno == logging.DEBUG and record.message.startswith("[memory]")
        for record in caplog.records
    )

    caplog.clear()
    bot._mem_log_prev = {"timestamp": 1_000, "rss": 200 * 1024 * 1024}
    monkeypatch.setattr(
        passivbot_module, "_get_process_rss_bytes", lambda: 220 * 1024 * 1024
    )
    with caplog.at_level(logging.DEBUG):
        bot._log_memory_snapshot(now_ms=2_000)

    assert len(emitted) == 1
    assert not any(
        record.levelno == logging.INFO and record.message.startswith("[memory]")
        for record in caplog.records
    )
    assert any(
        record.levelno == logging.DEBUG and record.message.startswith("[memory]")
        for record in caplog.records
    )


def test_memory_snapshot_event_failure_isolated_from_snapshot_state(monkeypatch, caplog):
    bot = _memory_snapshot_bot()

    def fail_emit(**kwargs):
        del kwargs
        raise OSError("event sink unavailable")

    bot._emit_memory_snapshot_event = fail_emit
    monkeypatch.setattr(
        passivbot_module, "_get_process_rss_bytes", lambda: 198 * 1024 * 1024
    )

    with caplog.at_level(logging.DEBUG):
        bot._log_memory_snapshot(now_ms=1_000)

    assert bot._mem_log_prev["timestamp"] == 1_000
    assert bot._mem_log_prev["rss"] == 198 * 1024 * 1024
    assert any(
        "failed to emit memory snapshot event" in record.message
        for record in caplog.records
    )


def test_unstuck_status_logs_info_on_change_then_hourly(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._unstuck_last_log_ms = 0
    bot._unstuck_log_interval_ms = 5 * 60 * 1000
    bot._unstuck_unchanged_info_log_interval_ms = 60 * 60 * 1000
    bot._unstuck_operator_visible_baseline_by_pside = {}
    bot._unstuck_operator_visible_ms_by_pside = {}
    bot._monitor_runtime_unstuck_hints = {
        "long": {
            "next_symbol": "BTC/USDT:USDT",
            "next_target_price": 101_000.0,
            "next_target_distance_ratio": 0.005,
            "next_unstuck_trigger_distance_ratio": 0.0125,
        }
    }
    captured_events = []
    bot._emit_unstuck_status_event = lambda **kwargs: captured_events.append(kwargs)
    now = [5 * 60 * 1000]
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now[0])

    state_by_pside = {
        "long": {
            "status": "ok",
            "allowance": -41.01,
            "peak": 16400.0,
            "pct_from_peak": -0.3,
        },
        "short": {"status": "disabled"},
    }

    def _calc(pside):
        return state_by_pside[pside]

    bot._calc_unstuck_allowance_for_logging = _calc

    with caplog.at_level(logging.INFO):
        bot._maybe_log_unstuck_status()
        state_by_pside["long"] = {
            "status": "ok",
            "allowance": -41.03,
            "peak": 16550.0,
            "pct_from_peak": -0.2,
        }
        bot._monitor_runtime_unstuck_hints["long"] = {
            "next_symbol": "ETH/USDT:USDT",
            "next_target_price": 2_500.0,
            "next_target_distance_ratio": -0.01,
            "next_unstuck_trigger_distance_ratio": 0.02,
        }
        now[0] += 5 * 60 * 1000
        bot._maybe_log_unstuck_status()
        state_by_pside["long"] = {
            "status": "ok",
            "allowance": -41.20,
            "peak": 16580.0,
            "pct_from_peak": -0.2,
        }
        now[0] += 5 * 60 * 1000
        bot._maybe_log_unstuck_status()
        now[0] += 60 * 60 * 1000
        bot._maybe_log_unstuck_status()

    unstuck_logs = [
        record.message for record in caplog.records if "[unstuck]" in record.message
    ]
    assert len(unstuck_logs) == 3
    assert len(captured_events) == 4
    assert [event["changed"] for event in captured_events] == [True, True, False, False]
    assert [event["operator_visible"] for event in captured_events] == [
        True,
        True,
        False,
        True,
    ]
    assert captured_events[0]["side_statuses"]["long"]["allowance"] == pytest.approx(
        -41.01
    )
    assert captured_events[0]["side_statuses"]["long"]["next_symbol"] == "BTC/USDT:USDT"
    assert captured_events[1]["side_statuses"]["long"]["next_symbol"] == "ETH/USDT:USDT"
    assert captured_events[1]["side_statuses"]["long"]["next_target_distance_ratio"] == pytest.approx(
        -0.01
    )
    assert captured_events[0]["side_statuses"]["short"]["status"] == "disabled"


def test_unstuck_status_suppresses_legacy_log_when_event_console_active(
    monkeypatch, caplog
):
    bot = Passivbot.__new__(Passivbot)
    bot.live_event_console_enabled = True
    bot._live_event_pipeline = object()
    bot._unstuck_last_log_ms = 0
    bot._unstuck_log_interval_ms = 5 * 60 * 1000
    bot._unstuck_unchanged_info_log_interval_ms = 60 * 60 * 1000
    bot._unstuck_operator_visible_baseline_by_pside = {}
    bot._unstuck_operator_visible_ms_by_pside = {}
    bot._monitor_runtime_unstuck_hints = {}
    captured_events = []
    bot._emit_unstuck_status_event = lambda **kwargs: captured_events.append(kwargs)
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 5 * 60 * 1000)
    bot._calc_unstuck_allowance_for_logging = lambda pside: (
        {
            "status": "ok",
            "allowance": -41.01,
            "peak": 16400.0,
            "pct_from_peak": -0.3,
        }
        if pside == "long"
        else {"status": "disabled"}
    )

    with caplog.at_level(logging.INFO):
        bot._maybe_log_unstuck_status()

    assert captured_events
    assert not any("[unstuck]" in record.message for record in caplog.records)


def test_unstuck_allowance_materiality_uses_five_percent_boundary():
    baseline = -41.0

    assert not Passivbot._unstuck_allowance_materially_changed(0.0, 0.0)
    assert not Passivbot._unstuck_allowance_materially_changed(baseline, -43.049)
    assert Passivbot._unstuck_allowance_materially_changed(baseline, -43.05)

    bot = Passivbot.__new__(Passivbot)
    bot._unstuck_unchanged_info_log_interval_ms = 60 * 60 * 1000
    bot._unstuck_operator_visible_baseline_by_pside = {}
    bot._unstuck_operator_visible_ms_by_pside = {}
    first = bot._unstuck_status_operator_decision(
        {"long": {"status": "ok", "allowance": -100.0}}, now_ms=1
    )
    gradual = bot._unstuck_status_operator_decision(
        {"long": {"status": "ok", "allowance": -103.0}}, now_ms=2
    )
    accumulated = bot._unstuck_status_operator_decision(
        {"long": {"status": "ok", "allowance": -105.0}}, now_ms=3
    )

    assert first == (True, True)
    assert gradual == (False, False)
    assert accumulated == (True, True)


def test_unstuck_status_reports_coin_override_loss_allowance_pct():
    bot = Passivbot.__new__(Passivbot)
    bot._pnls_manager = object()
    bot.coin_overrides = {
        "HYPE/USDT:USDT": {"bot": {"long": {"unstuck_loss_allowance_pct": 0.005}}},
        "BTC/USDT:USDT": {"bot": {"long": {}}},
    }
    bot.get_raw_balance = lambda: 2_000.0
    bot._get_effective_pnl_events = lambda: [SimpleNamespace(pnl=0.0, fee_paid=0.0)]
    bot._assert_no_pending_pnl_events = lambda events, context: None

    def bot_value(pside, key):
        if key == "total_wallet_exposure_limit":
            return 1.5 if pside == "long" else 0.0
        if key == "unstuck_loss_allowance_pct":
            return 0.02 if pside == "long" else 0.0
        return 0.0

    def bp(pside, key, symbol=None):
        if (
            pside == "long"
            and key == "unstuck_loss_allowance_pct"
            and symbol == "HYPE/USDT:USDT"
        ):
            return 0.005
        return bot_value(pside, key)

    bot.bot_value = bot_value
    bot.bp = bp

    info = bot._calc_unstuck_allowance_for_logging("long")
    parts, signature = bot._get_unstuck_status_parts_and_signature()

    assert info["allowance"] == pytest.approx(60.0)
    assert info["override_loss_allowance_pcts"] == {"HYPE/USDT:USDT": pytest.approx(0.005)}
    assert info["override_allowances"]["HYPE/USDT:USDT"] == pytest.approx(15.0)
    assert "HYPE/USDT:USDT=0.0050" in parts[0]
    assert signature[0][3] == (("HYPE/USDT:USDT", pytest.approx(0.005)),)


def test_unstuck_selection_logs_on_change_then_hourly(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._unstuck_unchanged_info_log_interval_ms = 60 * 60 * 1000
    bot._unstuck_last_selection_signature = None
    bot._unstuck_last_selection_info_ms = 0
    captured_events = []
    bot._emit_unstuck_selection_event = lambda **kwargs: captured_events.append(kwargs)
    now = [0]
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now[0])

    with caplog.at_level(logging.INFO):
        bot._maybe_log_unstuck_selection(
            symbol="SUI/USDT:USDT",
            pside="long",
            entry_price=1.0,
            current_price=1.1,
            allowance=-41.0,
        )
        now[0] += 5 * 60 * 1000
        bot._maybe_log_unstuck_selection(
            symbol="SUI/USDT:USDT",
            pside="long",
            entry_price=1.0,
            current_price=1.1,
            allowance=-41.0,
        )
        now[0] += 60 * 60 * 1000
        bot._maybe_log_unstuck_selection(
            symbol="SUI/USDT:USDT",
            pside="long",
            entry_price=1.0,
            current_price=1.1,
            allowance=-41.0,
        )
        now[0] += 1
        bot._maybe_log_unstuck_selection(
            symbol="ADA/USDT:USDT",
            pside="long",
            entry_price=1.0,
            current_price=1.05,
            allowance=-41.0,
        )

    unstuck_logs = [
        record.message
        for record in caplog.records
        if "[unstuck] selecting" in record.message
    ]
    assert len(unstuck_logs) == 3
    assert len(captured_events) == 3
    assert [event["changed"] for event in captured_events] == [True, False, True]
    assert captured_events[0]["symbol"] == "SUI/USDT:USDT"
    assert captured_events[2]["symbol"] == "ADA/USDT:USDT"


def test_unstuck_selection_suppresses_legacy_log_when_event_console_active(
    monkeypatch, caplog
):
    bot = Passivbot.__new__(Passivbot)
    bot.live_event_console_enabled = True
    bot._live_event_pipeline = object()
    bot._unstuck_unchanged_info_log_interval_ms = 60 * 60 * 1000
    bot._unstuck_last_selection_signature = None
    bot._unstuck_last_selection_info_ms = 0
    captured_events = []
    bot._emit_unstuck_selection_event = lambda **kwargs: captured_events.append(kwargs)
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 0)

    with caplog.at_level(logging.INFO):
        bot._maybe_log_unstuck_selection(
            symbol="SUI/USDT:USDT",
            pside="long",
            entry_price=1.0,
            current_price=1.1,
            allowance=-41.0,
        )

    assert captured_events
    assert not any("[unstuck] selecting" in record.message for record in caplog.records)


def test_trailing_status_emits_on_change_then_hourly(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot._trailing_last_status_check_ms = 0
    bot._trailing_status_check_interval_ms = 5 * 60 * 1000
    bot._trailing_unchanged_info_log_interval_ms = 60 * 60 * 1000
    bot._trailing_operator_visible_baseline_by_key = {}
    bot._trailing_operator_visible_ms_by_key = {}
    captured_events = []
    bot._emit_trailing_status_event = lambda **kwargs: captured_events.append(kwargs)
    now = [5 * 60 * 1000]
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now[0])

    item = {
        "symbol": "BTC/USDT:USDT",
        "pside": "long",
        "kind": "entry",
        "payload": {
            "kind": "entry",
            "status": "waiting_threshold",
            "threshold_met": False,
            "retracement_met": False,
            "threshold_pct": 0.01,
            "retracement_pct": 0.004,
            "threshold_price": 99_000.0,
            "retracement_price": 99_396.0,
        },
    }
    items = [deepcopy(item)]
    bot._build_trailing_status_items = lambda: deepcopy(items)

    bot._maybe_log_trailing_status()
    now[0] += 5 * 60 * 1000
    bot._maybe_log_trailing_status()
    items[0]["payload"]["status"] = "waiting_retracement"
    items[0]["payload"]["threshold_met"] = True
    now[0] += 5 * 60 * 1000
    bot._maybe_log_trailing_status()
    now[0] += 5 * 60 * 1000
    bot._maybe_log_trailing_status()
    now[0] += 60 * 60 * 1000
    bot._maybe_log_trailing_status()

    assert len(captured_events) == 5
    assert [event["changed"] for event in captured_events] == [True, False, True, False, False]
    assert [event["operator_visible"] for event in captured_events] == [
        True,
        False,
        True,
        False,
        True,
    ]
    assert captured_events[0]["symbol"] == "BTC/USDT:USDT"
    assert captured_events[0]["pside"] == "long"
    assert captured_events[0]["kind"] == "entry"
    assert captured_events[2]["payload"]["status"] == "waiting_retracement"


def test_trailing_status_materiality_boundaries_and_per_item_visibility(monkeypatch):
    assert not Passivbot._trailing_status_number_materially_changed(
        0.0, 0.0, relative=True
    )
    assert not Passivbot._trailing_status_number_materially_changed(
        0.01, 0.010499, relative=False
    )
    assert Passivbot._trailing_status_number_materially_changed(
        0.01, 0.0105, relative=False
    )
    assert not Passivbot._trailing_status_number_materially_changed(
        100.0, 100.499, relative=True
    )
    assert Passivbot._trailing_status_number_materially_changed(
        100.0, 100.5, relative=True
    )

    bot = Passivbot.__new__(Passivbot)
    bot._trailing_last_status_check_ms = 0
    bot._trailing_status_check_interval_ms = 5 * 60 * 1000
    bot._trailing_unchanged_info_log_interval_ms = 60 * 60 * 1000
    bot._trailing_operator_visible_baseline_by_key = {}
    bot._trailing_operator_visible_ms_by_key = {}
    captured_events = []
    bot._emit_trailing_status_event = lambda **kwargs: captured_events.append(kwargs)
    now = [5 * 60 * 1000]
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now[0])
    items = [
        {
            "symbol": "BTC/USDT:USDT",
            "pside": "long",
            "kind": "entry",
            "payload": {"status": "waiting_threshold", "threshold_pct": 0.01},
        },
        {
            "symbol": "ETH/USDT:USDT",
            "pside": "long",
            "kind": "entry",
            "payload": {"status": "waiting_threshold", "threshold_pct": 0.01},
        },
    ]
    bot._build_trailing_status_items = lambda: deepcopy(items)

    bot._maybe_log_trailing_status()
    now[0] += 5 * 60 * 1000
    items[0]["payload"]["threshold_pct"] = 0.0105
    bot._maybe_log_trailing_status()

    assert [event["operator_visible"] for event in captured_events] == [
        True,
        True,
        True,
        False,
    ]
    assert [event["changed"] for event in captured_events] == [True, True, True, False]


def test_trailing_status_reappearing_item_is_operator_visible(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot._trailing_last_status_check_ms = 0
    bot._trailing_status_check_interval_ms = 5 * 60 * 1000
    bot._trailing_unchanged_info_log_interval_ms = 60 * 60 * 1000
    bot._trailing_operator_visible_baseline_by_key = {}
    bot._trailing_operator_visible_ms_by_key = {}
    captured_events = []
    bot._emit_trailing_status_event = lambda **kwargs: captured_events.append(kwargs)
    now = [5 * 60 * 1000]
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now[0])
    item = {
        "symbol": "BTC/USDT:USDT",
        "pside": "long",
        "kind": "entry",
        "payload": {"status": "waiting_threshold", "threshold_pct": 0.01},
    }
    items = [deepcopy(item)]
    bot._build_trailing_status_items = lambda: deepcopy(items)

    bot._maybe_log_trailing_status()
    items.clear()
    now[0] += 5 * 60 * 1000
    bot._maybe_log_trailing_status()
    items.append(deepcopy(item))
    now[0] += 5 * 60 * 1000
    bot._maybe_log_trailing_status()

    assert [event["operator_visible"] for event in captured_events] == [True, True]
    assert [event["changed"] for event in captured_events] == [True, True]


@pytest.mark.asyncio
async def test_update_pnls_all_lookback_backfills_when_cache_scope_is_narrower():
    bot = Passivbot.__new__(Passivbot)
    cached_events = [
        SimpleNamespace(timestamp=1_700_000_000_000, id="fill-1", source_ids=["fill-1"])
    ]

    class _Manager:
        def __init__(self, events, *, history_scope="unknown"):
            self._events = list(events)
            self.refresh = AsyncMock()
            self.refresh_latest = AsyncMock()
            self.history_scope = history_scope

        def get_events(self):
            return list(self._events)

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    bot.stop_signal_received = False
    bot._pnls_manager = _Manager(cached_events, history_scope="window")
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    bot.get_exchange_time = lambda: 1_700_000_060_000
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: None
    bot.logging_level = 0
    bot._health_rate_limits = 0

    result = await bot.update_pnls()

    assert result is True
    bot._pnls_manager.refresh.assert_awaited_once_with(start_ms=None, end_ms=None)
    bot._pnls_manager.refresh_latest.assert_not_awaited()
    assert bot._pnls_manager.history_scope == "all"


@pytest.mark.asyncio
async def test_update_pnls_all_lookback_uses_incremental_refresh_when_cache_is_full_history():
    bot = Passivbot.__new__(Passivbot)
    cached_events = [
        SimpleNamespace(timestamp=1_700_000_000_000, id="fill-1", source_ids=["fill-1"])
    ]

    class _Manager:
        def __init__(self, events, *, history_scope="unknown"):
            self._events = list(events)
            self.refresh = AsyncMock()
            self.refresh_latest = AsyncMock()
            self.history_scope = history_scope

        def get_events(self):
            return list(self._events)

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    bot.stop_signal_received = False
    bot.config = {
        "live": {
            "fills_recent_overlap_minutes": 10.0,
            "pnls_max_lookback_days": "all",
        }
    }
    bot._pnls_manager = _Manager(cached_events, history_scope="all")
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    bot.get_exchange_time = lambda: 1_700_000_060_000
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: None
    bot.logging_level = 0
    bot._health_rate_limits = 0
    bot._trailing_fill_refresh_started_generation = 4
    bot._trailing_fill_refresh_generation = 4

    result = await bot.update_pnls()

    assert result is True
    bot._pnls_manager.refresh.assert_not_awaited()
    bot._pnls_manager.refresh_latest.assert_awaited_once_with(
        overlap=20,
        last_refresh_overlap_ms=10 * 60 * 1000,
    )
    assert bot._pnls_manager.history_scope == "all"
    assert bot._trailing_fill_refresh_generation == 5


@pytest.mark.asyncio
async def test_update_pnls_window_lookback_bootstraps_when_coverage_unproven():
    bot = Passivbot.__new__(Passivbot)
    now_ms = 1_800_000_000_000
    lookback_days = 30.0
    start_ms = now_ms - int(lookback_days * 86_400_000)
    cached_events = [
        SimpleNamespace(
            timestamp=start_ms - 60_000,
            id="old-fill",
            source_ids=["old-fill"],
            symbol="BTC/USDT:USDT",
            position_side="long",
            side="sell",
            qty=0.1,
            price=10.0,
            pnl=1.0,
            fee_paid=0.0,
            pnl_status="complete",
        )
    ]

    class _Cache:
        def __init__(self):
            self.covered_start_ms = 0

        def load_metadata(self):
            return {
                "covered_start_ms": self.covered_start_ms,
                "oldest_event_ts": cached_events[0].timestamp,
                "history_scope": "window",
                "known_gaps": [],
            }

        def get_known_gaps(self):
            return []

        def get_covered_start_ms(self):
            return self.covered_start_ms

        def mark_covered_start(self, value):
            self.covered_start_ms = int(value)

    class _Manager:
        def __init__(self, events):
            self._events = list(events)
            self.cache = _Cache()
            self.refresh = AsyncMock()
            self.refresh_latest = AsyncMock()
            self.refresh_for_lookback = AsyncMock(side_effect=self._refresh_for_lookback)
            self.history_scope = "window"

        async def _refresh_for_lookback(self, *, start_ms):
            self.cache.mark_covered_start(start_ms)

        def get_events(self, start_ms=None):
            if start_ms is None:
                return list(self._events)
            return [event for event in self._events if event.timestamp >= int(start_ms)]

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    bot.stop_signal_received = False
    bot.config = {
        "live": {
            "fills_recent_overlap_minutes": 10.0,
            "pnls_max_lookback_days": lookback_days,
        }
    }
    bot._authoritative_pending_confirmations = {}
    bot._pnls_manager = _Manager(cached_events)
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: bot.config["live"][key]
    bot.get_exchange_time = lambda: now_ms
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: None
    bot.logging_level = 0
    bot._health_rate_limits = 0

    result = await bot.update_pnls(source="staged_blocking")

    assert result is True
    bot._pnls_manager.refresh_for_lookback.assert_awaited_once_with(start_ms=start_ms)
    bot._pnls_manager.refresh.assert_not_awaited()
    bot._pnls_manager.refresh_latest.assert_not_awaited()
    assert bot._pnls_manager.cache.get_covered_start_ms() == start_ms
    assert bot._pnls_manager.history_scope == "window"


@pytest.mark.asyncio
async def test_update_pnls_window_lookback_stays_blocked_when_known_gap_persists(
    monkeypatch,
):
    bot = Passivbot.__new__(Passivbot)
    now_ms = 1_800_000_000_000
    wall_ms = {"value": now_ms}
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: wall_ms["value"])
    lookback_days = 30.0
    start_ms = now_ms - int(lookback_days * 86_400_000)
    cached_events = [
        SimpleNamespace(
            timestamp=start_ms + 180_000,
            id="recent-fill",
            source_ids=["recent-fill"],
            symbol="BTC/USDT:USDT",
            position_side="long",
            side="sell",
            qty=0.1,
            price=10.0,
            pnl=1.0,
            fee_paid=0.0,
            pnl_status="complete",
        )
    ]
    known_gap = {
        "start_ts": start_ms + 60_000,
        "end_ts": start_ms + 120_000,
        "retry_count": 0,
        "reason": "fetch_failed",
        "confidence": 0.0,
    }

    class _Cache:
        def __init__(self):
            self.covered_start_ms = start_ms

        def load_metadata(self):
            return {
                "covered_start_ms": self.covered_start_ms,
                "oldest_event_ts": cached_events[0].timestamp,
                "history_scope": "window",
                "known_gaps": [dict(known_gap)],
            }

        def get_known_gaps(self):
            return [dict(known_gap)]

        def get_covered_start_ms(self):
            return self.covered_start_ms

        def mark_covered_start(self, value):
            self.covered_start_ms = int(value)

    class _Manager:
        def __init__(self, events):
            self._events = list(events)
            self.cache = _Cache()
            self.refresh = AsyncMock()
            self.refresh_latest = AsyncMock()
            self.refresh_for_lookback = AsyncMock(side_effect=self._refresh_for_lookback)
            self.history_scope = "window"

        async def _refresh_for_lookback(self, *, start_ms):
            self.cache.mark_covered_start(start_ms)

        def get_events(self, start_ms=None):
            if start_ms is None:
                return list(self._events)
            return [event for event in self._events if event.timestamp >= int(start_ms)]

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    bot.stop_signal_received = False
    bot.config = {
        "live": {
            "fills_recent_overlap_minutes": 10.0,
            "pnls_max_lookback_days": lookback_days,
        }
    }
    bot._authoritative_pending_confirmations = {}
    bot._fill_coverage_retry_state = {
        "key": ("stale",),
        "retry_count": 3,
        "next_retry_ms": now_ms + 60_000,
    }
    bot._pnls_manager = _Manager(cached_events)
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: bot.config["live"][key]
    bot.get_exchange_time = lambda: now_ms
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: None
    bot.logging_level = 0
    bot._health_rate_limits = 0

    result = await bot.update_pnls(source="staged_blocking")

    assert result is False
    bot._pnls_manager.refresh_for_lookback.assert_awaited_once_with(start_ms=start_ms)
    bot._pnls_manager.refresh.assert_not_awaited()
    bot._pnls_manager.refresh_latest.assert_not_awaited()
    assert bot._last_fill_refresh_pending_pnl_count == 0
    assert "fills" not in getattr(bot, "_authoritative_surface_signatures", {})
    retry_state = getattr(bot, "_fill_coverage_retry_state", {})
    assert retry_state["next_retry_ms"] > wall_ms["value"]

    result = await bot.update_pnls(source="staged_blocking")

    assert result is False
    bot._pnls_manager.refresh_for_lookback.assert_awaited_once_with(start_ms=start_ms)

    wall_ms["value"] = int(retry_state["next_retry_ms"]) + 1
    result = await bot.update_pnls(source="staged_blocking")

    assert result is False
    assert bot._pnls_manager.refresh_for_lookback.await_count == 2


@pytest.mark.asyncio
async def test_update_pnls_window_lookback_records_fills_after_known_gap_repair():
    bot = Passivbot.__new__(Passivbot)
    now_ms = 1_800_000_000_000
    lookback_days = 30.0
    start_ms = now_ms - int(lookback_days * 86_400_000)
    cached_events = [
        SimpleNamespace(
            timestamp=start_ms + 180_000,
            id="recent-fill",
            source_ids=["recent-fill"],
            symbol="BTC/USDT:USDT",
            position_side="long",
            side="sell",
            qty=0.1,
            price=10.0,
            pnl=1.0,
            fee_paid=0.0,
            pnl_status="complete",
        )
    ]
    known_gap = {
        "start_ts": start_ms + 60_000,
        "end_ts": start_ms + 120_000,
        "retry_count": 0,
        "reason": "fetch_failed",
        "confidence": 0.0,
    }

    class _Cache:
        def __init__(self):
            self.covered_start_ms = start_ms
            self.known_gaps = [dict(known_gap)]

        def load_metadata(self):
            return {
                "covered_start_ms": self.covered_start_ms,
                "oldest_event_ts": cached_events[0].timestamp,
                "history_scope": "window",
                "known_gaps": [dict(gap) for gap in self.known_gaps],
            }

        def get_known_gaps(self):
            return [dict(gap) for gap in self.known_gaps]

        def get_covered_start_ms(self):
            return self.covered_start_ms

        def mark_covered_start(self, value):
            self.covered_start_ms = int(value)

    class _Manager:
        def __init__(self, events):
            self._events = list(events)
            self.cache = _Cache()
            self.refresh = AsyncMock()
            self.refresh_latest = AsyncMock()
            self.refresh_for_lookback = AsyncMock(side_effect=self._refresh_for_lookback)
            self.history_scope = "window"

        async def _refresh_for_lookback(self, *, start_ms):
            self.cache.known_gaps = []
            self.cache.mark_covered_start(start_ms)

        def get_events(self, start_ms=None):
            if start_ms is None:
                return list(self._events)
            return [event for event in self._events if event.timestamp >= int(start_ms)]

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    bot.stop_signal_received = False
    bot.config = {
        "live": {
            "fills_recent_overlap_minutes": 10.0,
            "pnls_max_lookback_days": lookback_days,
        }
    }
    bot._authoritative_pending_confirmations = {}
    bot._pnls_manager = _Manager(cached_events)
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: bot.config["live"][key]
    bot.get_exchange_time = lambda: now_ms
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: None
    bot.logging_level = 0
    bot._health_rate_limits = 0

    result = await bot.update_pnls(source="staged_blocking")

    assert result is True
    bot._pnls_manager.refresh_for_lookback.assert_awaited_once_with(start_ms=start_ms)
    bot._pnls_manager.refresh.assert_not_awaited()
    bot._pnls_manager.refresh_latest.assert_not_awaited()
    assert "fills" in getattr(bot, "_authoritative_surface_signatures", {})
    assert getattr(bot, "_fill_coverage_retry_state", {}) == {}


@pytest.mark.asyncio
async def test_update_pnls_window_lookback_uses_incremental_when_coverage_proven():
    bot = Passivbot.__new__(Passivbot)
    now_ms = 1_800_000_000_000
    lookback_days = 30.0
    start_ms = now_ms - int(lookback_days * 86_400_000)
    cached_events = [
        SimpleNamespace(
            timestamp=start_ms + 60_000,
            id="recent-fill",
            source_ids=["recent-fill"],
            symbol="BTC/USDT:USDT",
            position_side="long",
            side="sell",
            qty=0.1,
            price=10.0,
            pnl=1.0,
            fee_paid=0.0,
            pnl_status="complete",
        )
    ]

    class _Cache:
        def load_metadata(self):
            return {
                "covered_start_ms": start_ms,
                "oldest_event_ts": cached_events[0].timestamp,
                "history_scope": "window",
                "known_gaps": [],
            }

        def get_known_gaps(self):
            return []

        def get_covered_start_ms(self):
            return start_ms

    class _Manager:
        def __init__(self, events):
            self._events = list(events)
            self.cache = _Cache()
            self.refresh = AsyncMock()
            self.refresh_latest = AsyncMock()
            self.refresh_for_lookback = AsyncMock()
            self.history_scope = "window"

        def get_events(self, start_ms=None):
            if start_ms is None:
                return list(self._events)
            return [event for event in self._events if event.timestamp >= int(start_ms)]

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    bot.stop_signal_received = False
    bot.config = {
        "live": {
            "fills_recent_overlap_minutes": 10.0,
            "pnls_max_lookback_days": lookback_days,
        }
    }
    bot._authoritative_pending_confirmations = {}
    sink = ListEventSink()
    bot.exchange = "bybit"
    bot.user = "bybit_01"
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_fills"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._pnls_manager = _Manager(cached_events)
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: bot.config["live"][key]
    bot.get_exchange_time = lambda: now_ms
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: None
    bot.logging_level = 0
    bot._health_rate_limits = 0

    result = await bot.update_pnls()

    assert result is True
    bot._pnls_manager.refresh_for_lookback.assert_not_awaited()
    bot._pnls_manager.refresh.assert_not_awaited()
    bot._pnls_manager.refresh_latest.assert_awaited_once_with(
        overlap=20,
        last_refresh_overlap_ms=10 * 60 * 1000,
    )
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.FILLS_REFRESH_SUMMARY
    ]
    assert len(events) == 1
    event = events[0]
    assert event.cycle_id == "cy_fills"
    assert event.status == "succeeded"
    assert event.reason_code == "fills_refresh_succeeded"
    assert event.data["refresh_mode"] == "incremental_recent"
    assert event.data["event_count_before"] == 1
    assert event.data["event_count_after"] == 1
    assert event.data["coverage_ready_before"] is True
    assert event.data["coverage_ready_after"] is True
    assert event.data["coverage_reason_after"] == "window_covered"
    assert event.data["pending_pnl_count"] == 0
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_update_pnls_uses_confirmation_overlap_when_fills_pending():
    bot = Passivbot.__new__(Passivbot)
    cached_events = [
        SimpleNamespace(timestamp=1_700_000_000_000, id="fill-1", source_ids=["fill-1"])
    ]

    class _Manager:
        def __init__(self, events, *, history_scope="unknown"):
            self._events = list(events)
            self.refresh = AsyncMock()
            self.refresh_latest = AsyncMock()
            self.history_scope = history_scope

        def get_events(self):
            return list(self._events)

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    bot.stop_signal_received = False
    bot.config = {
        "live": {
            "fills_confirmation_overlap_minutes": 60.0,
            "fills_recent_overlap_minutes": 10.0,
            "pnls_max_lookback_days": "all",
        }
    }
    bot._authoritative_pending_confirmations = {"fills": 2}
    bot._pnls_manager = _Manager(cached_events, history_scope="all")
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    bot.get_exchange_time = lambda: 1_700_000_060_000
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: None
    bot.logging_level = 0
    bot._health_rate_limits = 0

    result = await bot.update_pnls()

    assert result is True
    bot._pnls_manager.refresh.assert_not_awaited()
    bot._pnls_manager.refresh_latest.assert_awaited_once_with(
        overlap=20,
        last_refresh_overlap_ms=60 * 60 * 1000,
    )


@pytest.mark.asyncio
async def test_update_pnls_propagates_unexpected_refresh_errors():
    bot = Passivbot.__new__(Passivbot)
    cached_events = [
        SimpleNamespace(timestamp=1_700_000_000_000, id="fill-1", source_ids=["fill-1"])
    ]

    class _Manager:
        def __init__(self, events, *, history_scope="unknown"):
            self._events = list(events)
            self.refresh = AsyncMock()
            self.refresh_latest = AsyncMock(
                side_effect=RuntimeError("fill refresh failed")
            )
            self.history_scope = history_scope

        def get_events(self):
            return list(self._events)

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    bot.stop_signal_received = False
    bot.config = {
        "live": {
            "fills_recent_overlap_minutes": 10.0,
            "pnls_max_lookback_days": "all",
        }
    }
    sink = ListEventSink()
    bot.exchange = "bybit"
    bot.user = "bybit_01"
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_fills_error"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._pnls_manager = _Manager(cached_events, history_scope="all")
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    bot.get_exchange_time = lambda: 1_700_000_060_000
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: None
    bot.logging_level = 0
    bot._health_rate_limits = 0

    with pytest.raises(RuntimeError, match="fill refresh failed"):
        await bot.update_pnls()
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.FILLS_REFRESH_SUMMARY
    ]
    assert len(events) == 1
    event = events[0]
    assert event.status == "failed"
    assert event.reason_code == "fill_refresh_failed"
    assert event.data["error_type"] == "RuntimeError"
    assert event.data["error"] == "fill refresh failed"
    assert event.data["coverage_ready_before"] is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_update_pnls_emits_summary_when_exchange_time_resync_handles_error():
    bot = Passivbot.__new__(Passivbot)
    cached_events = [
        SimpleNamespace(timestamp=1_700_000_000_000, id="fill-1", source_ids=["fill-1"])
    ]

    class _Manager:
        def __init__(self, events, *, history_scope="unknown"):
            self._events = list(events)
            self.refresh = AsyncMock()
            self.refresh_latest = AsyncMock(
                side_effect=RuntimeError("timestamp outside recvWindow")
            )
            self.history_scope = history_scope

        def get_events(self):
            return list(self._events)

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    bot.stop_signal_received = False
    bot.config = {
        "live": {
            "fills_recent_overlap_minutes": 10.0,
            "pnls_max_lookback_days": "all",
        }
    }
    sink = ListEventSink()
    bot.exchange = "binance"
    bot.user = "binance_01"
    bot.bot_id = "bot_1"
    bot._live_event_current_cycle_id = "cy_fills_resync"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot._pnls_manager = _Manager(cached_events, history_scope="all")
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    bot.get_exchange_time = lambda: 1_700_000_060_000
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: None
    bot._maybe_recover_exchange_time_sync = AsyncMock(return_value=True)
    bot.logging_level = 0
    bot._health_rate_limits = 0

    result = await bot.update_pnls()

    assert result is False
    bot._maybe_recover_exchange_time_sync.assert_awaited_once()
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.FILLS_REFRESH_SUMMARY
    ]
    assert len(events) == 1
    event = events[0]
    assert event.status == "deferred"
    assert event.reason_code == "exchange_time_resync"
    assert event.data["refresh_mode"] == "incremental_recent"
    assert event.data["error_type"] == "RuntimeError"
    assert event.data["coverage_ready_before"] is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_update_pnls_suppresses_inflight_shutdown_refresh_error(caplog):
    bot = Passivbot.__new__(Passivbot)
    cached_events = [
        SimpleNamespace(timestamp=1_700_000_000_000, id="fill-1", source_ids=["fill-1"])
    ]
    monitor_errors = []

    class _Manager:
        def __init__(self, events):
            self._events = list(events)
            self.refresh = AsyncMock()
            self.history_scope = "all"

        async def refresh_latest(self, overlap=20, last_refresh_overlap_ms=None):
            bot.stop_signal_received = True
            raise RuntimeError("connector is closed")

        def get_events(self):
            return list(self._events)

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    bot.stop_signal_received = False
    bot._shutdown_in_progress = False
    bot.config = {
        "live": {
            "fills_recent_overlap_minutes": 10.0,
            "pnls_max_lookback_days": "all",
        }
    }
    bot._pnls_manager = _Manager(cached_events)
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    bot.get_exchange_time = lambda: 1_700_000_060_000
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: monitor_errors.append(
        (args, kwargs)
    )
    bot.logging_level = 2
    bot._health_rate_limits = 0

    with caplog.at_level(logging.DEBUG):
        result = await bot.update_pnls()

    assert result is False
    assert monitor_errors == []
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert any(
        "fill refresh stopped during in-flight request" in r.message
        for r in caplog.records
    )


def _disable_entry_cooldown_delta_guard_for_staged_refresh_test(bot) -> None:
    bot.coin_overrides = {}
    bot.config.setdefault("bot", {})
    bot.config["bot"].setdefault("long", {})["risk_entry_cooldown_minutes"] = 0.0
    bot.config["bot"].setdefault("short", {})["risk_entry_cooldown_minutes"] = 0.0
    bot.get_exchange_time = lambda: 1_700_000_000_000


@pytest.mark.asyncio
async def test_refresh_authoritative_state_staged_applies_fake_snapshots():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    _disable_entry_cooldown_delta_guard_for_staged_refresh_test(bot)
    bot.exchange = "fake"
    bot.stop_signal_received = False
    bot.balance_override = None
    bot._balance_override_logged = False
    bot.previous_hysteresis_balance = None
    bot.balance_hysteresis_snap_pct = 0.02
    bot.balance_raw = 0.0
    bot.balance = 0.0
    bot._exchange_reported_balance_raw = 0.0
    bot.fetched_positions = []
    bot.positions = {}
    bot.open_orders = {}
    bot.fetched_open_orders = []
    bot.active_symbols = []
    bot.state_change_detected_by_symbol = set()
    bot.execution_scheduled = False
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot._authoritative_refresh_epoch = 0
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()
    bot.recent_order_cancellations = []
    bot.fetch_balance = AsyncMock(return_value=123.45)
    bot.fetch_positions = AsyncMock(
        return_value=[
            {
                "symbol": "BTC/USDT:USDT",
                "position_side": "long",
                "size": 0.01,
                "price": 100000.0,
            }
        ]
    )
    bot.fetch_open_orders = AsyncMock(
        return_value=[
            {
                "id": "1",
                "symbol": "BTC/USDT:USDT",
                "side": "buy",
                "position_side": "long",
                "qty": 0.01,
                "amount": 0.01,
                "price": 99000.0,
                "timestamp": 1,
                "reduce_only": False,
            }
        ]
    )
    bot.update_pnls = AsyncMock(return_value=True)
    seen = {}

    async def _log_position_changes(*args, **kwargs):
        del args, kwargs
        seen["balance_raw_when_logged"] = bot.balance_raw

    bot.log_position_changes = AsyncMock(side_effect=_log_position_changes)
    bot.handle_balance_update = AsyncMock()
    bot.order_matches_bot_cancellation = lambda order: False
    bot.order_was_recently_cancelled = lambda order: 0.0
    bot.log_order_action = lambda *args, **kwargs: None
    bot._reconcile_balance_after_positions_and_balance_refresh = lambda: False
    bot._reconcile_balance_after_open_orders_refresh = lambda: False

    result = await bot.refresh_authoritative_state()

    assert result is True
    bot.fetch_balance.assert_awaited_once()
    bot.fetch_positions.assert_awaited_once()
    bot.fetch_open_orders.assert_awaited_once()
    bot.update_pnls.assert_awaited_once()
    assert bot.balance_raw == pytest.approx(123.45)
    assert seen["balance_raw_when_logged"] == pytest.approx(123.45)
    assert bot.positions["BTC/USDT:USDT"]["long"]["size"] == pytest.approx(0.01)
    assert bot.open_orders["BTC/USDT:USDT"][0]["id"] == "1"
    bot.handle_balance_update.assert_awaited_once_with(source="REST")


@pytest.mark.asyncio
async def test_refresh_authoritative_state_staged_applies_bybit_snapshots():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    _disable_entry_cooldown_delta_guard_for_staged_refresh_test(bot)
    bot.exchange = "bybit"
    bot.stop_signal_received = False
    bot.balance_override = None
    bot._balance_override_logged = False
    bot.previous_hysteresis_balance = None
    bot.balance_hysteresis_snap_pct = 0.02
    bot.balance_raw = 0.0
    bot.balance = 0.0
    bot._exchange_reported_balance_raw = 0.0
    bot.fetched_positions = []
    bot.positions = {}
    bot.open_orders = {}
    bot.fetched_open_orders = []
    bot.active_symbols = []
    bot.state_change_detected_by_symbol = set()
    bot.execution_scheduled = False
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot._authoritative_refresh_epoch = 0
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()
    bot.recent_order_cancellations = []
    seen = {}

    async def fake_capture_balance_snapshot():
        return {"raw": "balance"}, 456.78

    async def fake_capture_positions_snapshot():
        return (
            {"raw": "positions"},
            [
                {
                    "symbol": "ETH/USDT:USDT",
                    "position_side": "long",
                    "size": 0.25,
                    "price": 2500.0,
                }
            ],
        )

    async def fake_fetch_open_orders():
        return [
            {
                "id": "2",
                "symbol": "ETH/USDT:USDT",
                "side": "buy",
                "position_side": "long",
                "qty": 0.25,
                "amount": 0.25,
                "price": 2400.0,
                "timestamp": 2,
                "reduce_only": False,
            }
        ]

    async def _log_position_changes(*args, **kwargs):
        del args, kwargs
        seen["balance_raw_when_logged"] = bot.balance_raw

    bot.capture_balance_snapshot = fake_capture_balance_snapshot
    bot.capture_positions_snapshot = fake_capture_positions_snapshot
    bot.fetch_open_orders = fake_fetch_open_orders
    bot.update_pnls = AsyncMock(return_value=True)
    bot.log_position_changes = AsyncMock(side_effect=_log_position_changes)
    bot.handle_balance_update = AsyncMock()
    bot.order_matches_bot_cancellation = lambda order: False
    bot.order_was_recently_cancelled = lambda order: 0.0
    bot.log_order_action = lambda *args, **kwargs: None
    bot._reconcile_balance_after_positions_and_balance_refresh = lambda: False
    bot._reconcile_balance_after_open_orders_refresh = lambda: False

    result = await bot.refresh_authoritative_state()

    assert result is True
    assert bot.balance_raw == pytest.approx(456.78)
    assert seen["balance_raw_when_logged"] == pytest.approx(456.78)
    assert bot.positions["ETH/USDT:USDT"]["long"]["size"] == pytest.approx(0.25)
    assert bot.open_orders["ETH/USDT:USDT"][0]["id"] == "2"
    bot.handle_balance_update.assert_awaited_once_with(source="REST")


@pytest.mark.asyncio
async def test_fetch_authoritative_state_staged_snapshot_uses_exchange_cohort_hook():
    bot = Passivbot.__new__(Passivbot)
    seen = {}

    async def fake_capture(plan, timings_ms):
        seen["plan"] = set(plan)
        timings_ms["positions_balance"] = 321
        return {"plan": set(plan), "balance": 12.34, "positions": [], "pnls_ok": True}

    bot.capture_authoritative_state_staged_snapshot = fake_capture

    snapshot = await bot._fetch_authoritative_state_staged_snapshot(
        {"balance", "positions", "open_orders", "fills"}
    )

    assert seen["plan"] == {"balance", "positions", "open_orders", "fills"}
    assert snapshot["balance"] == pytest.approx(12.34)
    assert snapshot["positions"] == []
    assert snapshot["pnls_ok"] is True


@pytest.mark.asyncio
async def test_refresh_authoritative_state_staged_uses_generic_staged_fetch_for_any_exchange():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    _disable_entry_cooldown_delta_guard_for_staged_refresh_test(bot)
    bot.exchange = "binance"
    bot.stop_signal_received = False
    bot._authoritative_refresh_epoch = 0
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot.fetch_balance = AsyncMock(return_value=100.0)
    bot.fetch_positions = AsyncMock(return_value=[])
    bot.fetch_open_orders = AsyncMock(return_value=[])
    bot.positions = {}
    bot.update_pnls = AsyncMock(return_value=True)
    bot._apply_positions_snapshot = lambda positions: ({}, {})
    bot._apply_balance_snapshot = lambda balance: True
    bot._record_authoritative_surface = lambda surface, signature: None
    bot._staged_defer_balance_publication = lambda: False
    bot._reconcile_balance_after_positions_and_balance_refresh = lambda: True
    bot.get_hysteresis_snapped_balance = lambda: 100.0
    bot.log_position_changes = AsyncMock()
    bot.handle_balance_update = AsyncMock()
    bot._apply_open_orders_snapshot = AsyncMock(return_value=True)
    finalized = []
    bot._finalize_authoritative_refresh_consistency = lambda plan: finalized.append(
        set(plan)
    )

    result = await bot.refresh_authoritative_state()

    assert result is True
    bot.fetch_balance.assert_awaited_once()
    bot.fetch_positions.assert_awaited_once()
    bot.fetch_open_orders.assert_awaited_once()
    bot.update_pnls.assert_awaited_once()
    assert finalized == [{"balance", "positions", "open_orders", "fills"}]


@pytest.mark.asyncio
async def test_refresh_protective_authoritative_state_uses_account_critical_surfaces():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    _disable_entry_cooldown_delta_guard_for_staged_refresh_test(bot)
    bot.exchange = "binance"
    bot.stop_signal_received = False
    bot.positions = {}
    bot.open_orders = {}
    bot._authoritative_refresh_epoch = 0
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot.balance_override = None
    bot._balance_override_logged = False
    bot.previous_hysteresis_balance = None
    bot.balance_hysteresis_snap_pct = 0.02
    bot.balance_raw = 0.0
    bot.balance = 0.0
    bot._exchange_reported_balance_raw = 0.0
    fetched_positions = [
        {
            "symbol": "BTC/USDT:USDT",
            "position_side": "long",
            "size": 0.1,
            "price": 100.0,
        }
    ]
    fetched_orders = [
        {
            "id": "1",
            "symbol": "BTC/USDT:USDT",
            "side": "sell",
            "position_side": "long",
            "qty": 0.1,
            "price": 101.0,
        }
    ]
    bot._fetch_authoritative_state_staged_snapshot = AsyncMock(
        return_value={
            "plan": {"balance", "positions", "open_orders"},
            "balance": 123.45,
            "positions": fetched_positions,
            "open_orders": fetched_orders,
        }
    )
    bot._apply_open_orders_snapshot = AsyncMock(return_value=True)

    def apply_positions(positions):
        bot.positions = {
            "BTC/USDT:USDT": {
                "long": {"size": 0.1, "price": 100.0},
                "short": {"size": 0.0, "price": 0.0},
            }
        }
        return [], positions

    bot._apply_positions_snapshot = apply_positions
    bot._positions_signature = lambda positions: tuple(
        (p["symbol"], p["position_side"], p["size"], p["price"]) for p in positions
    )
    recorded = []
    bot._record_authoritative_surface = lambda surface, signature: recorded.append(
        (surface, signature)
    )
    cooldown_updates = []
    bot._update_entry_cooldown_position_delta_guard = (
        lambda symbols, now_ms: cooldown_updates.append((tuple(symbols), now_ms))
    )
    finalized = []
    bot._finalize_authoritative_refresh_consistency = lambda plan: finalized.append(
        set(plan)
    )
    bot.update_pnls = AsyncMock(side_effect=AssertionError("fills not required"))

    ok = await bot.refresh_protective_authoritative_state()

    assert ok is True
    bot._fetch_authoritative_state_staged_snapshot.assert_awaited_once_with(
        {"balance", "positions", "open_orders"}
    )
    bot._apply_open_orders_snapshot.assert_awaited_once_with(
        fetched_orders,
        allow_followup_positions_refresh=False,
        reconcile_balance=False,
    )
    assert recorded == [
        (
            "balance",
            123.45,
        ),
        (
            "positions",
            (("BTC/USDT:USDT", "long", 0.1, 100.0),),
        )
    ]
    assert bot.balance_raw == pytest.approx(123.45)
    assert cooldown_updates == [(("BTC/USDT:USDT",), 1_700_000_000_000)]
    assert finalized == [{"balance", "positions", "open_orders"}]


def test_protective_planning_snapshot_requires_balance_not_fills_or_candles():
    import passivbot as pb_mod

    symbol = "BTC/USDT:USDT"
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "binance"
    bot.config_get = lambda keys: None
    bot._authoritative_refresh_epoch = 1
    bot._authoritative_pending_confirmations = {
        "balance": 2,
        "positions": 2,
        "open_orders": 2,
        "fills": 2,
    }
    bot._staged_planner_surface_min_epochs = (
        pb_mod.Passivbot._staged_planner_surface_min_epochs.__get__(bot, Passivbot)
    )
    bot._ensure_freshness_ledger = (
        pb_mod.Passivbot._ensure_freshness_ledger.__get__(bot, Passivbot)
    )
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot._market_snapshot_signature_invalid = lambda symbols: []
    now_ms = pb_mod.utc_ms()
    ledger = FreshnessLedger(now_ms=now_ms)
    ledger.begin_epoch(now_ms=now_ms)
    ledger.stamp("positions", (symbol, "long", 0.1), now_ms=now_ms)
    ledger.stamp("open_orders", (), now_ms=now_ms)
    ledger.stamp("balance", 100.0, now_ms=now_ms)
    ledger.stamp("market_snapshot", (symbol, 99.5, 100.5), now_ms=now_ms)
    bot.freshness_ledger = ledger
    snapshots = {
        symbol: MarketSnapshot(
            symbol=symbol,
            bid=99.5,
            ask=100.5,
            last=100.0,
            fetched_ms=now_ms,
            source="test",
        )
    }

    snapshot = pb_mod.planning_gates.build_protective_planning_snapshot(
        bot, [symbol], snapshots
    )

    assert set(snapshot.required_surfaces) == {
        "balance",
        "positions",
        "open_orders",
        "market_snapshot",
    }
    assert "fills" not in snapshot.required_surfaces
    assert "completed_candles" not in snapshot.required_surfaces
    assert {surface.name: surface.min_epoch for surface in snapshot.surfaces} == {
        "balance": 1,
        "positions": 1,
        "open_orders": 1,
        "market_snapshot": 1,
    }


@pytest.mark.asyncio
async def test_fetch_authoritative_state_staged_snapshot_cleans_up_on_cancelled_surface():
    from live import state_refresh

    class FakeBot:
        def __init__(self):
            self.progress_cancelled = False
            self.positions_cancelled = False
            self.timing_logs = []

        async def capture_authoritative_state_staged_snapshot(self, plan, timings_ms):
            return None

        async def _timed_authoritative_fetch(self, surface, coro, timings_ms):
            return await state_refresh.timed_authoritative_fetch(
                self, surface, coro, timings_ms
            )

        async def _capture_balance_staged_snapshot(self):
            raise asyncio.CancelledError("balance fetch cancelled")

        async def _capture_positions_staged_snapshot(self):
            try:
                await asyncio.sleep(60.0)
            except asyncio.CancelledError:
                self.positions_cancelled = True
                raise

        async def _log_staged_refresh_progress_until(
            self, plan, timings_ms, tasks, wall_started_ms
        ):
            try:
                await asyncio.sleep(60.0)
            except asyncio.CancelledError:
                self.progress_cancelled = True
                raise

        def _log_staged_refresh_timings(self, plan, timings_ms, wall_ms):
            self.timing_logs.append((set(plan), dict(timings_ms), int(wall_ms)))

    bot = FakeBot()

    with pytest.raises(asyncio.CancelledError, match="balance fetch cancelled"):
        await state_refresh.fetch_authoritative_state_staged_snapshot(
            bot, {"balance", "positions"}
        )

    assert bot.positions_cancelled is True
    assert bot.progress_cancelled is True
    assert bot.timing_logs
    assert bot.timing_logs[-1][0] == {"balance", "positions"}


@pytest.mark.asyncio
async def test_refresh_authoritative_state_staged_does_not_publish_when_fills_fail():
    bot = Passivbot.__new__(Passivbot)
    plan = {"balance", "positions", "open_orders", "fills"}
    bot._authoritative_staged_refresh_plan = lambda: set(plan)
    bot._fetch_authoritative_state_staged_snapshot = AsyncMock(
        return_value={
            "balance": 100.0,
            "positions": [{"symbol": "BTC/USDT:USDT"}],
            "open_orders": [],
            "pnls_ok": False,
        }
    )
    bot._apply_positions_snapshot = MagicMock()
    bot._apply_balance_snapshot = MagicMock()
    bot._apply_open_orders_snapshot = AsyncMock()
    bot.handle_balance_update = AsyncMock()
    bot._finalize_authoritative_refresh_consistency = MagicMock()

    result = await bot._refresh_authoritative_state_staged()

    assert result is False
    bot._apply_positions_snapshot.assert_not_called()
    bot._apply_balance_snapshot.assert_not_called()
    bot._apply_open_orders_snapshot.assert_not_awaited()
    bot.handle_balance_update.assert_not_awaited()
    bot._finalize_authoritative_refresh_consistency.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_authoritative_state_staged_does_not_publish_when_open_orders_apply_fails():
    bot = Passivbot.__new__(Passivbot)
    plan = {"balance", "positions", "open_orders", "fills"}
    bot._authoritative_staged_refresh_plan = lambda: set(plan)
    bot._fetch_authoritative_state_staged_snapshot = AsyncMock(
        return_value={
            "balance": 100.0,
            "positions": [{"symbol": "BTC/USDT:USDT"}],
            "open_orders": [],
            "pnls_ok": True,
        }
    )
    bot._apply_open_orders_snapshot = AsyncMock(return_value=False)
    bot._apply_positions_snapshot = MagicMock()
    bot._apply_balance_snapshot = MagicMock()
    bot.handle_balance_update = AsyncMock()
    bot._finalize_authoritative_refresh_consistency = MagicMock()

    result = await bot._refresh_authoritative_state_staged()

    assert result is False
    bot._apply_open_orders_snapshot.assert_awaited_once_with(
        [],
        allow_followup_positions_refresh=False,
        reconcile_balance=False,
    )
    bot._apply_positions_snapshot.assert_not_called()
    bot._apply_balance_snapshot.assert_not_called()
    bot.handle_balance_update.assert_not_awaited()
    bot._finalize_authoritative_refresh_consistency.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_authoritative_state_staged_invalid_exchange_balance_raises_without_commit():
    bot = Passivbot.__new__(Passivbot)
    plan = {"balance", "positions", "open_orders", "fills"}
    bot._authoritative_staged_refresh_plan = lambda: set(plan)
    bot._fetch_authoritative_state_staged_snapshot = AsyncMock(
        return_value={
            "balance": float("nan"),
            "positions": [{"symbol": "BTC/USDT:USDT"}],
            "open_orders": [],
            "pnls_ok": True,
        }
    )
    bot.quote = "USDT"
    bot.balance = 50.0
    bot.balance_raw = 50.0
    bot.balance_override = None
    bot.previous_hysteresis_balance = 50.0
    bot.balance_hysteresis_snap_pct = 0.02
    bot._exchange_reported_balance_raw = 50.0
    bot._apply_open_orders_snapshot = AsyncMock()
    bot._apply_positions_snapshot = MagicMock()
    bot.handle_balance_update = AsyncMock()
    bot._finalize_authoritative_refresh_consistency = MagicMock()

    with pytest.raises(RuntimeError, match="invalid exchange balance fetch result"):
        await bot._refresh_authoritative_state_staged()

    bot._apply_open_orders_snapshot.assert_not_awaited()
    bot._apply_positions_snapshot.assert_not_called()
    bot.handle_balance_update.assert_not_awaited()
    bot._finalize_authoritative_refresh_consistency.assert_not_called()
    assert bot.balance == pytest.approx(50.0)
    assert bot.balance_raw == pytest.approx(50.0)
    assert bot.previous_hysteresis_balance == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_refresh_authoritative_state_staged_does_not_partially_commit_on_invalid_balance():
    bot = Passivbot.__new__(Passivbot)
    plan = {"balance", "positions", "open_orders", "fills"}
    bot._authoritative_staged_refresh_plan = lambda: set(plan)
    bot._fetch_authoritative_state_staged_snapshot = AsyncMock(
        return_value={
            "balance": 100.0,
            "positions": [
                {
                    "symbol": "BTC/USDT:USDT",
                    "position_side": "long",
                    "price": 1.0,
                    "size": 1.0,
                }
            ],
            "open_orders": [{"id": "new", "symbol": "BTC/USDT:USDT"}],
            "pnls_ok": True,
        }
    )
    bot.balance = 50.0
    bot.balance_raw = 50.0
    bot.previous_hysteresis_balance = 50.0
    bot.balance_hysteresis_snap_pct = 0.02
    bot.balance_override = "invalid"
    bot._balance_override_logged = False
    bot._exchange_reported_balance_raw = 50.0
    bot.positions = {"OLD/USDT:USDT": {"long": {"size": 1.0, "price": 1.0}}}
    bot.fetched_positions = [{"symbol": "OLD/USDT:USDT"}]
    bot.open_orders = {"OLD/USDT:USDT": [{"id": "old", "symbol": "OLD/USDT:USDT"}]}
    bot.fetched_open_orders = [{"id": "old", "symbol": "OLD/USDT:USDT"}]
    bot._apply_open_orders_snapshot = AsyncMock(return_value=True)
    bot._apply_positions_snapshot = MagicMock()
    bot.handle_balance_update = AsyncMock()
    bot._finalize_authoritative_refresh_consistency = MagicMock()

    old_positions = deepcopy(bot.positions)
    old_fetched_positions = deepcopy(bot.fetched_positions)
    old_open_orders = deepcopy(bot.open_orders)
    old_fetched_open_orders = deepcopy(bot.fetched_open_orders)

    result = await bot._refresh_authoritative_state_staged()

    assert result is False
    bot._apply_open_orders_snapshot.assert_not_awaited()
    bot._apply_positions_snapshot.assert_not_called()
    bot.handle_balance_update.assert_not_awaited()
    bot._finalize_authoritative_refresh_consistency.assert_not_called()
    assert bot.balance == 50.0
    assert bot.balance_raw == 50.0
    assert bot.previous_hysteresis_balance == 50.0
    assert bot._exchange_reported_balance_raw == 50.0
    assert bot.positions == old_positions
    assert bot.fetched_positions == old_fetched_positions
    assert bot.open_orders == old_open_orders
    assert bot.fetched_open_orders == old_fetched_open_orders


def test_get_exchange_time_uses_direct_utc_ms(monkeypatch):
    import passivbot as pb_mod

    bot = Passivbot.__new__(Passivbot)
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: 123_456.0)

    assert bot.get_exchange_time() == pytest.approx(123_456.0)


@pytest.mark.asyncio
async def test_update_positions_only_updates_positions(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.positions = {}
    bot.active_symbols = []
    bot.fetched_positions = []
    bot.quote = "USDT"

    async def fake_fetch_positions():
        return [
            {
                "symbol": "BTC/USDT:USDT",
                "position_side": "long",
                "size": 1.0,
                "price": 100.0,
            }
        ]

    async def fake_log_position_changes(old, new):
        return

    async def fail_handle_balance_update(source="REST"):
        raise AssertionError(
            "handle_balance_update should not be called by update_positions"
        )

    bot.fetch_positions = fake_fetch_positions
    bot.log_position_changes = fake_log_position_changes
    bot.handle_balance_update = fail_handle_balance_update

    ok = await bot.update_positions()
    assert ok is True
    assert bot.fetched_positions[0]["symbol"] == "BTC/USDT:USDT"
    assert bot.positions["BTC/USDT:USDT"]["long"]["size"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_update_balance_uses_fetch_balance_when_no_cache():
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    calls = {"balance": 0}

    async def fake_handle_balance_update(source="REST"):
        calls["balance"] += 1
        assert bot.balance == pytest.approx(123.0)

    async def fake_fetch_balance():
        return 123.0

    bot.fetch_balance = fake_fetch_balance

    ok = await bot.update_balance()
    assert ok is True
    # handle_balance_update should not be called automatically
    assert calls["balance"] == 0
    # manual invocation uses latest balance/positions
    bot.handle_balance_update = fake_handle_balance_update
    await bot.handle_balance_update()
    assert calls["balance"] == 1


@pytest.mark.asyncio
async def test_update_balance_failure_keeps_previous():
    """Test that when fetch_balance raises an exception, previous balance is preserved.

    Per CLAUDE.md: exchange fetch methods should raise exceptions on failure,
    not return False. The exception propagates to the caller who handles it.
    """
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    bot.balance = 50.0
    bot.previous_hysteresis_balance = 50.0

    async def fake_fetch_balance():
        raise Exception("network error")  # simulate failed fetch

    bot.fetch_balance = fake_fetch_balance

    with pytest.raises(Exception, match="network error"):
        await bot.update_balance()
    # Balance should remain unchanged because assignment was never reached
    assert bot.balance == pytest.approx(50.0)
    assert bot.previous_hysteresis_balance == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_update_balance_override_does_not_reset_hysteresis_anchor():
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    bot.balance = 100.0
    bot.balance_raw = 100.0
    bot.balance_override = 250.0
    bot._balance_override_logged = False
    bot.previous_hysteresis_balance = 133.0
    bot.balance_hysteresis_snap_pct = 0.02

    ok = await bot.update_balance()
    assert ok is True
    assert bot.balance == pytest.approx(250.0)
    assert bot.balance_raw == pytest.approx(250.0)
    # Keep hysteresis anchor unchanged while override is active.
    assert bot.previous_hysteresis_balance == pytest.approx(133.0)


def test_apply_balance_snapshot_honors_override_and_retains_exchange_raw(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    bot.balance = 100.0
    bot.balance_raw = 100.0
    bot._exchange_reported_balance_raw = 100.0
    bot.balance_override = 250.0
    bot._balance_override_logged = False
    bot.previous_hysteresis_balance = 133.0
    bot.balance_hysteresis_snap_pct = 0.02

    with caplog.at_level(logging.INFO):
        ok = bot._apply_balance_snapshot(123.45)

    assert ok is True
    assert bot.balance == pytest.approx(250.0)
    assert bot.balance_raw == pytest.approx(250.0)
    assert bot._exchange_reported_balance_raw == pytest.approx(123.45)
    assert bot.previous_hysteresis_balance == pytest.approx(133.0)
    assert "Using balance override: 250.000000" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fetched_balance", "reason"),
    [
        (None, "none"),
        ("not-a-number", "non-numeric"),
        (float("nan"), "non-finite"),
        (float("inf"), "non-finite"),
    ],
)
async def test_update_balance_invalid_exchange_payload_raises_without_mutating(
    fetched_balance, reason
):
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    bot.balance = 50.0
    bot.balance_raw = 50.0
    bot.previous_hysteresis_balance = 50.0

    async def fake_fetch_balance():
        return fetched_balance

    bot.fetch_balance = fake_fetch_balance

    with pytest.raises(RuntimeError, match=rf"invalid exchange balance fetch result \({reason}\)"):
        await bot.update_balance()
    assert bot.balance == pytest.approx(50.0)
    assert bot.balance_raw == pytest.approx(50.0)
    assert bot.previous_hysteresis_balance == pytest.approx(50.0)


def test_accessor_fallback_when_balance_raw_absent():
    """get_raw_balance() falls back to snapped balance when balance_raw is absent."""
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 500.0
    # Intentionally do NOT set balance_raw
    if hasattr(bot, "balance_raw"):
        del bot.balance_raw
    assert bot.get_raw_balance() == pytest.approx(500.0)


def test_accessor_returns_raw_when_present():
    """get_raw_balance() returns balance_raw when it exists."""
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 1000.0
    bot.balance_raw = 1010.0
    assert bot.get_raw_balance() == pytest.approx(1010.0)
    assert bot.get_hysteresis_snapped_balance() == pytest.approx(1000.0)


def test_get_wallet_exposure_limit_uses_configured_n_positions_only():
    bot = Passivbot.__new__(Passivbot)
    bot.coin_overrides = {}

    def bot_value(pside, key):
        if key == "total_wallet_exposure_limit":
            return 1.4
        if key == "n_positions":
            return 7
        raise KeyError(key)

    bot.bot_value = bot_value
    bot.get_max_n_positions = lambda pside: 1
    bot.get_current_n_positions = lambda pside: 99

    wel = bot.get_wallet_exposure_limit("long")
    assert wel == pytest.approx(0.2)


def test_balance_peak_uses_raw_not_snapped():
    """Core bug regression: balance_peak must be derived from raw balance, not snapped.

    After a profit fill, raw balance advances (e.g. 1010) but snapped balance
    may remain stale (e.g. 1000) due to hysteresis. The peak calculation must
    use raw balance so that balance_peak is correct.

    balance_peak = balance_raw + (pnls_cumsum_max - pnls_cumsum_last)

    With raw=1010, cumsum_max=100, cumsum_last=100:
        peak_from_raw = 1010 + (100 - 100) = 1010  ← correct
        peak_from_snap = 1000 + (100 - 100) = 1000  ← WRONG (stale)
    """
    import types
    from unittest.mock import MagicMock

    import numpy as np

    bot = Passivbot.__new__(Passivbot)
    bot.balance = 1000.0  # snapped (stale)
    bot.balance_raw = 1010.0  # raw (advanced after profit fill)
    bot._pnls_manager = MagicMock()
    bot._pnls_manager.get_events.return_value = [
        types.SimpleNamespace(pnl=100.0, timestamp=1.0),
    ]

    # Simulate the calc_auto_unstuck_allowance_from_scratch peak calculation
    events = bot._pnls_manager.get_events()
    pnls_cumsum = np.array([ev.pnl for ev in events]).cumsum()
    pnls_cumsum_max = float(pnls_cumsum.max())
    pnls_cumsum_last = float(pnls_cumsum[-1])

    # Using raw balance (correct)
    balance_peak_raw = bot.get_raw_balance() + (pnls_cumsum_max - pnls_cumsum_last)
    assert balance_peak_raw == pytest.approx(1010.0)

    # Using snapped balance (would be wrong)
    balance_peak_snapped = bot.get_hysteresis_snapped_balance() + (
        pnls_cumsum_max - pnls_cumsum_last
    )
    assert balance_peak_snapped == pytest.approx(1000.0)

    # The raw-based peak is correct and higher
    assert balance_peak_raw > balance_peak_snapped


def test_effective_min_cost_filter_uses_snapped_balance():
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 100.0
    bot.balance_raw = 10.0
    bot.effective_min_cost = {"BTC/USDT:USDT": 40.0}
    bot.live_value = lambda key: key == "filter_by_min_effective_cost"
    bot.get_wallet_exposure_limit = lambda pside, symbol=None: 1.0
    bot.bot_value = lambda pside, key: 1.0 if key == "total_wallet_exposure_limit" else 0.0
    bot.bp = lambda pside, key, symbol=None: (
        0.5 if key == "entry_initial_qty_pct" else 0.0
    )

    # Passes only when snapped balance is used:
    # 100 * 1.0 * 0.5 = 50 >= 40; raw path would fail (10 * 1.0 * 0.5 = 5).
    assert bot.effective_min_cost_is_low_enough("long", "BTC/USDT:USDT") is True


def test_effective_min_cost_filter_uses_active_strategy_initial_sizing(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 100.0
    bot.effective_min_cost = {"BTC/USDT:USDT": 40.0}
    bot.config = {"live": {"strategy_kind": "ema_anchor"}, "bot": {"long": {}}}
    bot.live_value = lambda key: key == "filter_by_min_effective_cost"
    bot.get_wallet_exposure_limit = lambda pside, symbol=None: 1.0
    bot.bp = lambda pside, key, symbol=None: (
        0.0
        if key == "risk_we_excess_allowance_pct"
        else "bounded"
        if key == "risk_we_excess_allowance_mode"
        else (_ for _ in ()).throw(KeyError(key))
    )
    bot.bot_value = lambda pside, key: 1.0 if key == "total_wallet_exposure_limit" else 0.0
    bot._strategy_params_to_rust_dict = lambda pside, symbol=None: {"base_qty_pct": 0.5}
    monkeypatch.setattr(passivbot_module, "normalize_strategy_kind", lambda value: "ema_anchor")

    assert bot.effective_min_cost_is_low_enough("long", "BTC/USDT:USDT") is True


@pytest.mark.asyncio
async def test_update_effective_min_cost_uses_executable_min_qty():
    symbol = "SOL/USDT:USDT"
    bot = Passivbot.__new__(Passivbot)
    bot.effective_min_cost = {}
    bot.min_qtys = {symbol: 0.0}
    bot.qty_steps = {symbol: 1.0}
    bot.min_costs = {symbol: 0.1}
    bot.c_mults = {symbol: 1.0}
    bot.get_symbols_approved_or_has_pos = lambda: [symbol]

    async def fake_get_live_last_prices(symbols, **kwargs):
        return {symbol: 88.165}

    bot._get_live_last_prices = fake_get_live_last_prices

    await bot.update_effective_min_cost()

    assert bot.effective_min_cost[symbol] == pytest.approx(88.165)


def test_open_unstuck_order_does_not_gate_live_unstuck_emission(monkeypatch):
    # Rust owns auto-unstuck emission from the realized-PnL cumsum facts. A
    # resting unstuck order must not add a Python-only suppression path; any
    # duplicate-order risk rides normal order reconciliation.
    import passivbot as pb_mod

    bot = Passivbot.__new__(Passivbot)
    bot.balance = 100.0
    bot.balance_raw = 200.0
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda: [
            types.SimpleNamespace(pnl=10.0, fee_paid=-1.0),
        ],
        cache=_SafeRiskCache(),
        get_history_scope=lambda: "all",
    )
    bot.bot_value = lambda pside, key: {
        "unstuck_loss_allowance_pct": 0.2 if pside == "long" else 0.0,
        "total_wallet_exposure_limit": 0.5,
    }.get(key, 0.0)
    monkeypatch.setattr(
        pb_mod.pbr, "calc_auto_unstuck_allowance", lambda *a: 77.0
    )
    # An unstuck order is live on the exchange.
    bot.open_orders = {
        "BTC/USDT:USDT": [
            {"custom_id": _make_unstuck_custom_id()},
        ]
    }
    assert bot.has_open_unstuck_order() is True

    out = bot._calc_unstuck_allowances()
    assert out["long"] == pytest.approx(77.0)

    # The emission input stays available while the order is open.
    assert bot._auto_unstuck_allowed_live() is True


def _make_unstuck_custom_id() -> str:
    import passivbot as pb_mod

    bot = Passivbot.__new__(Passivbot)
    bot.broker_code = ""
    bot.custom_id_max_length = 36
    return bot.format_custom_id_single(
        pb_mod.pbr.get_order_id_type_from_string("close_unstuck_long")
    )


def test_unstuck_allowance_routes_raw_balance_to_rust(monkeypatch):
    import passivbot as pb_mod

    bot = Passivbot.__new__(Passivbot)
    bot.balance = 100.0
    bot.balance_raw = 200.0
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda: [
            types.SimpleNamespace(pnl=10.0, fee_paid=-1.0),
            types.SimpleNamespace(pnl=-4.0, fee_paid=-0.5),
        ],
        cache=_SafeRiskCache(),
        get_history_scope=lambda: "all",
    )

    def bot_value(pside, key):
        if key == "unstuck_loss_allowance_pct":
            return 0.2 if pside == "long" else 0.0
        if key == "total_wallet_exposure_limit":
            return 0.5
        return 0.0

    bot.bot_value = bot_value

    calls = []

    def fake_calc_auto_unstuck_allowance(
        balance, loss_allowance_pct, pnl_cumsum_max, pnl_cumsum_last
    ):
        calls.append((balance, loss_allowance_pct, pnl_cumsum_max, pnl_cumsum_last))
        return 123.45

    monkeypatch.setattr(
        pb_mod.pbr, "calc_auto_unstuck_allowance", fake_calc_auto_unstuck_allowance
    )

    out = bot._calc_unstuck_allowances()

    assert out["long"] == pytest.approx(123.45)
    assert out["short"] == pytest.approx(0.0)
    assert len(calls) == 1
    assert calls[0][0] == pytest.approx(200.0)  # raw balance
    assert calls[0][2] == pytest.approx(9.0)
    assert calls[0][3] == pytest.approx(4.5)


def test_unstuck_allowance_uses_only_configured_pnl_lookback(monkeypatch):
    import passivbot as pb_mod

    now_ms = 10 * 86_400_000
    bot = Passivbot.__new__(Passivbot)
    bot.balance_raw = 1000.0
    bot.get_raw_balance = lambda: float(bot.balance_raw)
    _set_pnl_lookback(bot, lookback_days=1.0, now_ms=now_ms)
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda start_ms=None, end_ms=None, symbol=None: [
            ev
            for ev in [
                types.SimpleNamespace(pnl=100.0, timestamp=now_ms - 3 * 86_400_000),
                types.SimpleNamespace(pnl=-80.0, timestamp=now_ms - 3 * 86_400_000 + 1),
                types.SimpleNamespace(pnl=10.0, timestamp=now_ms - 60_000),
            ]
            if start_ms is None or ev.timestamp >= start_ms
        ],
        cache=_SafeRiskCache(),
        get_history_scope=lambda: "all",
    )

    def bot_value(pside, key):
        if key == "unstuck_loss_allowance_pct":
            return 0.01 if pside == "long" else 0.0
        if key == "total_wallet_exposure_limit":
            return 1.0
        return 0.0

    bot.bot_value = bot_value

    calls = []

    def fake_calc_auto_unstuck_allowance(
        balance, loss_allowance_pct, pnl_cumsum_max, pnl_cumsum_last
    ):
        calls.append((balance, loss_allowance_pct, pnl_cumsum_max, pnl_cumsum_last))
        return 10.0

    monkeypatch.setattr(
        pb_mod.pbr, "calc_auto_unstuck_allowance", fake_calc_auto_unstuck_allowance
    )

    out = bot._calc_unstuck_allowances()

    assert out["long"] == pytest.approx(10.0)
    assert len(calls) == 1
    assert calls[0] == pytest.approx((1000.0, 0.01, 10.0, 10.0))


def test_unstuck_allowance_blocks_degraded_synthetic_pnl():
    bot = Passivbot.__new__(Passivbot)
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda start_ms=None, end_ms=None, symbol=None: [
            types.SimpleNamespace(
                pnl=0.0,
                fee_paid=0.0,
                timestamp=1_000,
                pnl_status="complete",
                pnl_source="synthetic_fill_reconstruction_degraded",
                id="degraded-close",
                symbol="BTC/USDT:USDT",
                position_side="long",
                pb_order_type="close_grid_long",
            )
        ],
        cache=None,
    )

    def bot_value(pside, key):
        if key == "unstuck_loss_allowance_pct":
            return 0.01
        if key == "total_wallet_exposure_limit":
            return 1.0
        return 0.0

    bot.bot_value = bot_value

    with pytest.raises(RuntimeError, match="degraded realized PnL"):
        bot._calc_unstuck_allowances()


@pytest.mark.asyncio
async def test_orchestrator_snapshot_payload_routes_split_balances(monkeypatch):
    import passivbot as pb_mod

    class FakeBot:
        positions = {}
        balance = 120.0  # snapped
        balance_raw = 175.0  # raw
        PB_modes = {}
        effective_min_cost = {}
        _config_hedge_mode = False
        hedge_mode = False
        equity_hard_stop_loss = {"panic_close_order_type": "limit"}
        _monitor_record_price_ticks = pb_mod.Passivbot._monitor_record_price_ticks
        _build_monitor_runtime_market_hints = (
            pb_mod.Passivbot._build_monitor_runtime_market_hints
        )
        _build_monitor_runtime_unstuck_hints = (
            pb_mod.Passivbot._build_monitor_runtime_unstuck_hints
        )
        _update_monitor_runtime_hints = pb_mod.Passivbot._update_monitor_runtime_hints

        def config_get(self, keys):
            return None

        def _bot_params_to_rust_dict(self, pside, symbol):
            return {}

        def _strategy_params_to_rust_dict(self, pside, symbol):
            return {}

        def live_value(self, key):
            return False

        def _log_realized_loss_gate_blocks(self, out, idx_to_symbol):
            return None

        def _log_ema_gating(self, ideal_orders, m1_close_emas, last_prices, symbols):
            return None

        def _to_executable_orders(self, ideal_orders, last_prices):
            return ideal_orders, []

        def _finalize_reduce_only_orders(self, ideal_orders_f, last_prices):
            return ideal_orders_f

        def get_raw_balance(self):
            return float(self.balance_raw)

        def get_hysteresis_snapped_balance(self):
            return float(self.balance)

    snapshot = {
        "symbols": [],
        "last_prices": {},
        "m1_close_emas": {},
        "m1_volume_emas": {},
        "m1_log_range_emas": {},
        "h1_log_range_emas": {},
        "unstuck_allowances": {"long": 0.0, "short": 0.0},
        "realized_pnl_cumsum": {"max": 0.0, "last": 0.0},
    }

    captured = {}

    def fake_compute(json_str):
        captured["input"] = json.loads(json_str)
        return json.dumps({"orders": [], "diagnostics": {"loss_gate_blocks": []}})

    monkeypatch.setattr(pb_mod.pbr, "compute_ideal_orders_json", fake_compute)

    bot = FakeBot()
    method = pb_mod.Passivbot.calc_ideal_orders_orchestrator_from_snapshot
    await method(bot, snapshot, return_snapshot=False)

    assert captured["input"]["balance"] == pytest.approx(120.0)
    assert captured["input"]["balance_raw"] == pytest.approx(175.0)


@pytest.mark.asyncio
async def test_live_orchestrator_input_omits_backtest_market_slippage(monkeypatch):
    import passivbot as pb_mod

    class FakeBot:
        positions = {}
        balance = 120.0
        balance_raw = 175.0
        PB_modes = {}
        effective_min_cost = {}
        _config_hedge_mode = False
        hedge_mode = False
        equity_hard_stop_loss = {"panic_close_order_type": "limit"}
        config = {
            "live": {"strategy_kind": "trailing_martingale"},
            "backtest": {"market_order_slippage_pct": 0.25},
        }
        _monitor_record_price_ticks = pb_mod.Passivbot._monitor_record_price_ticks
        _build_monitor_runtime_market_hints = (
            pb_mod.Passivbot._build_monitor_runtime_market_hints
        )
        _build_monitor_runtime_unstuck_hints = (
            pb_mod.Passivbot._build_monitor_runtime_unstuck_hints
        )
        _update_monitor_runtime_hints = pb_mod.Passivbot._update_monitor_runtime_hints

        def config_get(self, keys):
            return None

        def _bot_params_to_rust_dict(self, pside, symbol):
            return {}

        def _strategy_params_to_rust_dict(self, pside, symbol):
            return {}

        def live_value(self, key):
            values = {
                "max_realized_loss_pct": 0.0,
                "filter_by_min_effective_cost": False,
                "market_orders_allowed": False,
                "market_order_near_touch_threshold": 0.0,
            }
            return values.get(key, False)

        def _log_realized_loss_gate_blocks(self, out, idx_to_symbol):
            return None

        def _log_ema_gating(self, ideal_orders, m1_close_emas, last_prices, symbols):
            return None

        def _to_executable_orders(self, ideal_orders, last_prices):
            return ideal_orders, []

        def _finalize_reduce_only_orders(self, ideal_orders_f, last_prices):
            return ideal_orders_f

        def get_raw_balance(self):
            return float(self.balance_raw)

        def get_hysteresis_snapped_balance(self):
            return float(self.balance)

    snapshot = {
        "symbols": [],
        "last_prices": {},
        "m1_close_emas": {},
        "m1_volume_emas": {},
        "m1_log_range_emas": {},
        "h1_log_range_emas": {},
        "unstuck_allowances": {"long": 0.0, "short": 0.0},
        "realized_pnl_cumsum": {"max": 0.0, "last": 0.0},
    }

    captured = {}

    def fake_compute(json_str):
        captured["input"] = json.loads(json_str)
        return json.dumps({"orders": [], "diagnostics": {"loss_gate_blocks": []}})

    monkeypatch.setattr(pb_mod.pbr, "compute_ideal_orders_json", fake_compute)

    method = pb_mod.Passivbot.calc_ideal_orders_orchestrator_from_snapshot
    await method(FakeBot(), snapshot, return_snapshot=False)

    assert "market_order_slippage_pct" not in captured["input"]["global"]
    assert captured["input"]["global"]["max_realized_loss_pct"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_protective_panic_orchestrator_payload_omits_ema_dependencies(monkeypatch):
    import passivbot as pb_mod

    symbol = "BTC/USDT:USDT"
    healthy_symbol = "ETH/USDT:USDT"

    class FakePlanningSnapshot:
        def last_prices(self):
            return {symbol: 100.0}

    class FakeBot:
        exchange = "binance"
        user = "tester"
        balance = 120.0
        balance_raw = 120.0
        positions = {
            symbol: {
                "long": {"size": 1.0, "price": 100.0},
                "short": {"size": -1.0, "price": 100.0},
            },
            healthy_symbol: {
                "long": {"size": 2.0, "price": 200.0},
                "short": {"size": 0.0, "price": 0.0},
            },
        }
        open_orders = {}
        active_symbols = []
        _config_hedge_mode = True
        hedge_mode = True
        config = {"live": {"strategy_kind": "trailing_martingale"}}
        markets_dict = {symbol: {"active": True, "maker": 0.0002, "taker": 0.0004}}
        qty_steps = {symbol: 0.001}
        price_steps = {symbol: 0.1}
        min_qtys = {symbol: 0.001}
        min_costs = {symbol: 5.0}
        c_mults = {symbol: 1.0}
        effective_min_cost = {symbol: 5.0}
        _get_exchange_fee_rates = pb_mod.Passivbot._get_exchange_fee_rates

        def config_get(self, keys):
            return None

        def live_value(self, key):
            values = {
                "filter_by_min_effective_cost": False,
                "market_orders_allowed": True,
                "market_order_near_touch_threshold": 0.0,
                "max_realized_loss_pct": 1.0,
            }
            return values.get(key, False)

        def get_exchange_time(self):
            return 1_700_000_000_000

        def get_forced_PB_mode(self, pside, sym=None):
            if sym == symbol and pside == "long":
                return "panic"
            return None

        def get_hysteresis_snapped_balance(self):
            return self.balance

        def get_raw_balance(self):
            return self.balance_raw

        async def _get_orchestrator_market_snapshots(self, symbols):
            return {
                sym: MarketSnapshot(
                    symbol=sym,
                    bid=99.5,
                    ask=100.5,
                    last=100.0,
                    fetched_ms=pb_mod.utc_ms(),
                    source="test",
                )
                for sym in symbols
            }

        async def _load_orchestrator_ema_bundle(self, symbols, modes):
            raise AssertionError("protective panic path must not load EMA bundles")

        def _bot_params_to_rust_dict(self, pside, sym):
            return {}

        def _strategy_params_to_rust_dict(self, pside, sym):
            return {}

        def _calc_effective_min_cost_at_price(self, sym, price):
            return 5.0

        def _to_executable_orders(self, ideal_orders, last_prices):
            return ideal_orders, set()

        def _finalize_reduce_only_orders(self, ideal_orders_f, last_prices):
            return ideal_orders_f

    captured = {}

    def fake_build_snapshot(bot, symbols, market_snapshots):
        assert symbols == [symbol]
        return FakePlanningSnapshot()

    def fake_compute(json_str):
        captured["input"] = json.loads(json_str)
        return json.dumps({"orders": [], "diagnostics": {"warnings": []}})

    monkeypatch.setattr(
        pb_mod.planning_gates,
        "build_protective_planning_snapshot",
        fake_build_snapshot,
    )
    monkeypatch.setattr(pb_mod.Passivbot, "_monitor_record_price_ticks", lambda *a, **k: None)
    monkeypatch.setattr(
        pb_mod.Passivbot, "_build_orchestrator_runtime_hints", lambda self, symbol_to_idx: {}
    )
    monkeypatch.setattr(
        pb_mod.Passivbot,
        "_equity_hard_stop_enabled",
        lambda self, pside=None: True,
    )
    monkeypatch.setattr(
        pb_mod.Passivbot,
        "_equity_hard_stop_panic_close_order_type",
        lambda self, pside: "market",
    )
    monkeypatch.setattr(pb_mod.pbr, "compute_ideal_orders_json", fake_compute)

    out = await pb_mod.Passivbot.calc_protective_panic_ideal_orders_orchestrator(
        FakeBot()
    )

    assert out == {}
    rust_symbol = captured["input"]["symbols"][0]
    assert rust_symbol["long"]["mode"] == "panic"
    assert rust_symbol["short"]["mode"] == "manual"
    assert captured["input"]["global"]["panic_close_market"] is False
    assert len(captured["input"]["symbols"]) == 1
    assert rust_symbol["emas"] == {
        "m1": {"close": [], "log_range": [], "volume": []},
        "h1": {"close": [], "log_range": [], "volume": []},
    }
    assert "unstuck_allowance_long" not in captured["input"]["global"]
    assert captured["input"]["global"]["realized_pnl_cumsum_last"] == 0.0
    assert pb_mod.Passivbot._protective_panic_target_psides_by_symbol(FakeBot()) == {
        symbol: {"long"}
    }


def test_live_max_realized_loss_pct_preserves_zero_and_defaults_none():
    import passivbot as pb_mod

    class FakeBot:
        def __init__(self, value):
            self.value = value

        def live_value(self, key):
            assert key == "max_realized_loss_pct"
            return self.value

    helper = pb_mod.Passivbot._live_max_realized_loss_pct
    assert helper(FakeBot(0.0)) == pytest.approx(0.0)
    assert helper(FakeBot(None)) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_orchestrator_snapshot_payload_does_not_require_backtest_config(monkeypatch):
    import passivbot as pb_mod

    class FakeBot:
        config = prepare_config(
            get_template_config(), verbose=False, target="live", runtime="live"
        )
        positions = {}
        balance = 120.0
        balance_raw = 175.0
        PB_modes = {}
        effective_min_cost = {}
        _config_hedge_mode = False
        hedge_mode = False
        equity_hard_stop_loss = {"panic_close_order_type": "limit"}
        _monitor_record_price_ticks = pb_mod.Passivbot._monitor_record_price_ticks
        _build_monitor_runtime_market_hints = (
            pb_mod.Passivbot._build_monitor_runtime_market_hints
        )
        _build_monitor_runtime_unstuck_hints = (
            pb_mod.Passivbot._build_monitor_runtime_unstuck_hints
        )
        _update_monitor_runtime_hints = pb_mod.Passivbot._update_monitor_runtime_hints

        def config_get(self, keys):
            return None

        def _bot_params_to_rust_dict(self, pside, symbol):
            return {}

        def _strategy_params_to_rust_dict(self, pside, symbol):
            return {}

        def live_value(self, key):
            values = {
                "max_realized_loss_pct": 1.0,
                "filter_by_min_effective_cost": False,
                "market_orders_allowed": True,
                "market_order_near_touch_threshold": 0.001,
            }
            return values.get(key, False)

        def _log_realized_loss_gate_blocks(self, out, idx_to_symbol):
            return None

        def _log_ema_gating(self, ideal_orders, m1_close_emas, last_prices, symbols):
            return None

        def _to_executable_orders(self, ideal_orders, last_prices):
            return ideal_orders, []

        def _finalize_reduce_only_orders(self, ideal_orders_f, last_prices):
            return ideal_orders_f

        def get_raw_balance(self):
            return float(self.balance_raw)

        def get_hysteresis_snapped_balance(self):
            return float(self.balance)

    snapshot = {
        "symbols": [],
        "last_prices": {},
        "m1_close_emas": {},
        "m1_volume_emas": {},
        "m1_log_range_emas": {},
        "h1_log_range_emas": {},
        "unstuck_allowances": {"long": 0.0, "short": 0.0},
        "realized_pnl_cumsum": {"max": 0.0, "last": 0.0},
    }

    captured = {}

    def fake_compute(json_str):
        captured["input"] = json.loads(json_str)
        return json.dumps({"orders": [], "diagnostics": {"loss_gate_blocks": []}})

    monkeypatch.setattr(pb_mod.pbr, "compute_ideal_orders_json", fake_compute)

    method = pb_mod.Passivbot.calc_ideal_orders_orchestrator_from_snapshot
    bot = FakeBot()
    assert "backtest" not in bot.config
    await method(bot, snapshot, return_snapshot=False)

    assert captured["input"]["global"]["market_orders_allowed"] is True
    assert captured["input"]["global"]["market_order_near_touch_threshold"] == pytest.approx(
        0.001
    )
    assert "market_order_slippage_pct" not in captured["input"]["global"]


@pytest.mark.asyncio
async def test_orchestrator_snapshot_payload_includes_exchange_fees(monkeypatch):
    import passivbot as pb_mod

    symbol = "BTC/USDT:USDT"

    class FakeBot:
        positions = {}
        balance = 120.0
        balance_raw = 175.0
        PB_modes = {"long": {symbol: "normal"}, "short": {symbol: "manual"}}
        effective_min_cost = {symbol: 1.0}
        _config_hedge_mode = False
        hedge_mode = False
        qty_steps = {symbol: 0.001}
        price_steps = {symbol: 0.1}
        min_qtys = {symbol: 0.001}
        min_costs = {symbol: 5.0}
        c_mults = {symbol: 1.0}
        markets_dict = {symbol: {"maker": 0.0001, "taker": 0.0004}}
        equity_hard_stop_loss = {"panic_close_order_type": "limit"}
        trailing_prices = {}
        _monitor_record_price_ticks = pb_mod.Passivbot._monitor_record_price_ticks
        _build_monitor_runtime_market_hints = (
            pb_mod.Passivbot._build_monitor_runtime_market_hints
        )
        _build_monitor_runtime_unstuck_hints = (
            pb_mod.Passivbot._build_monitor_runtime_unstuck_hints
        )
        _update_monitor_runtime_hints = pb_mod.Passivbot._update_monitor_runtime_hints

        def config_get(self, keys):
            return None

        def _bot_params_to_rust_dict(self, pside, symbol):
            return {}

        def _strategy_params_to_rust_dict(self, pside, symbol):
            return {}

        def live_value(self, key):
            return False

        def _log_realized_loss_gate_blocks(self, out, idx_to_symbol):
            return None

        def _log_ema_gating(self, ideal_orders, m1_close_emas, last_prices, symbols):
            return None

        def _to_executable_orders(self, ideal_orders, last_prices):
            return ideal_orders, []

        def _finalize_reduce_only_orders(self, ideal_orders_f, last_prices):
            return ideal_orders_f

        def get_raw_balance(self):
            return float(self.balance_raw)

        def get_hysteresis_snapped_balance(self):
            return float(self.balance)

        _orchestrator_exchange_params = pb_mod.Passivbot._orchestrator_exchange_params
        _get_exchange_fee_rates = pb_mod.Passivbot._get_exchange_fee_rates

        def _pb_mode_to_orchestrator_mode(self, mode):
            return pb_mod.Passivbot._pb_mode_to_orchestrator_mode(self, mode)

    snapshot = {
        "symbols": [symbol],
        "last_prices": {symbol: 100.0},
        "m1_close_emas": {symbol: {10.0: 100.0}},
        "m1_volume_emas": {symbol: {10.0: 1000.0}},
        "m1_log_range_emas": {symbol: {10.0: 0.01}},
        "h1_log_range_emas": {symbol: {10.0: 0.01}},
        "unstuck_allowances": {"long": 0.0, "short": 0.0},
        "realized_pnl_cumsum": {"max": 0.0, "last": 0.0},
    }

    captured = {}

    def fake_compute(json_str):
        captured["input"] = json.loads(json_str)
        return json.dumps({"orders": [], "diagnostics": {"loss_gate_blocks": []}})

    monkeypatch.setattr(pb_mod.pbr, "compute_ideal_orders_json", fake_compute)

    method = pb_mod.Passivbot.calc_ideal_orders_orchestrator_from_snapshot
    await method(FakeBot(), snapshot, return_snapshot=False)

    exchange = captured["input"]["symbols"][0]["exchange"]
    assert exchange["maker_fee"] == pytest.approx(0.0001)
    assert exchange["taker_fee"] == pytest.approx(0.0004)


def test_orchestrator_exchange_params_require_live_fee_metadata():
    symbol = "BTC/USDT:USDT"
    bot = Passivbot.__new__(Passivbot)
    bot.markets_dict = {symbol: {"maker": 0.0001}}
    bot.qty_steps = {symbol: 0.001}
    bot.price_steps = {symbol: 0.1}
    bot.min_qtys = {symbol: 0.001}
    bot.min_costs = {symbol: 5.0}
    bot.c_mults = {symbol: 1.0}

    with pytest.raises(ValueError, match="missing taker_fee"):
        Passivbot._orchestrator_exchange_params(bot, symbol)


def test_unstuck_logging_peak_stays_stable_when_profit_updates_both_balance_and_pnl():
    """Regression for peak drift: peak must not decay when profits increase pnl_last and balance_raw."""
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 100.0  # stale snapped value to simulate hysteresis lag

    def bot_value(pside, key):
        if key == "total_wallet_exposure_limit":
            return 1.0
        if key == "unstuck_loss_allowance_pct":
            return 0.1
        return 0.0

    bot.bot_value = bot_value

    # State A: peak = 100 + (100 - 0) = 200
    bot.balance_raw = 100.0
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda: [
            types.SimpleNamespace(pnl=100.0),
            types.SimpleNamespace(pnl=-100.0),
        ]
    )
    info_a = bot._calc_unstuck_allowance_for_logging("long")

    # State B (after net +50 realized since trough): peak should still be 200
    # If snapped balance were incorrectly used here, peak would drift down to 150.
    bot.balance_raw = 150.0
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda: [
            types.SimpleNamespace(pnl=100.0),
            types.SimpleNamespace(pnl=-50.0),
        ]
    )
    info_b = bot._calc_unstuck_allowance_for_logging("long")

    assert info_a["status"] == "ok"
    assert info_b["status"] == "ok"
    assert info_a["peak"] == pytest.approx(200.0)
    assert info_b["peak"] == pytest.approx(200.0)


def test_unstuck_logging_uses_only_configured_pnl_lookback():
    now_ms = 10 * 86_400_000
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 1000.0
    bot.balance_raw = 1000.0
    _set_pnl_lookback(bot, lookback_days=1.0, now_ms=now_ms)

    def bot_value(pside, key):
        if key == "total_wallet_exposure_limit":
            return 1.0
        if key == "unstuck_loss_allowance_pct":
            return 0.01
        return 0.0

    bot.bot_value = bot_value
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda start_ms=None, end_ms=None, symbol=None: [
            ev
            for ev in [
                types.SimpleNamespace(pnl=100.0, timestamp=now_ms - 3 * 86_400_000),
                types.SimpleNamespace(pnl=-80.0, timestamp=now_ms - 3 * 86_400_000 + 1),
                types.SimpleNamespace(pnl=10.0, timestamp=now_ms - 60_000),
            ]
            if start_ms is None or ev.timestamp >= start_ms
        ]
    )

    info = bot._calc_unstuck_allowance_for_logging("long")

    assert info["status"] == "ok"
    assert info["peak"] == pytest.approx(1000.0)
    assert info["pct_from_peak"] == pytest.approx(0.0)
    assert info["allowance"] == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_update_balance_hysteresis_divergence():
    """Calls update_balance() twice with values within snap threshold, verifies balance_raw != balance."""
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"

    call_count = [0]

    async def fake_fetch_balance():
        call_count[0] += 1
        # First call: 100.0, second call: 100.5 (within 2% snap threshold)
        return 100.0 if call_count[0] == 1 else 100.5

    bot.fetch_balance = fake_fetch_balance

    ok1 = await bot.update_balance()
    assert ok1 is True
    # After first call, both should be the same
    assert bot.balance_raw == pytest.approx(100.0)
    assert bot.balance == pytest.approx(100.0)

    ok2 = await bot.update_balance()
    assert ok2 is True
    # Raw should update to 100.5, but snapped stays at 100.0 (within 2% threshold)
    assert bot.balance_raw == pytest.approx(100.5)
    assert bot.balance == pytest.approx(100.0)
    # Verify divergence
    assert bot.balance_raw != bot.balance


@pytest.mark.asyncio
async def test_update_balance_divergence_routes_to_orchestrator_input():
    """After creating divergence, builds orchestrator input dict and asserts both values pass through."""
    import passivbot as pb_mod

    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"

    call_count = [0]

    async def fake_fetch_balance():
        call_count[0] += 1
        return 1000.0 if call_count[0] == 1 else 1005.0

    bot.fetch_balance = fake_fetch_balance

    await bot.update_balance()
    await bot.update_balance()

    # Verify divergence exists
    assert bot.balance_raw == pytest.approx(1005.0)
    assert bot.balance == pytest.approx(1000.0)

    # Now verify get_raw_balance / get_hysteresis_snapped_balance route correctly
    assert bot.get_raw_balance() == pytest.approx(1005.0)
    assert bot.get_hysteresis_snapped_balance() == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_ws_balance_update_sets_balance_raw_correctly():
    """Simulate a WS-triggered balance update and verify balance_raw is set."""
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"

    # First establish a baseline via REST update_balance
    call_count = [0]

    async def fake_fetch_balance():
        call_count[0] += 1
        return 1000.0

    bot.fetch_balance = fake_fetch_balance
    await bot.update_balance()

    assert bot.balance_raw == pytest.approx(1000.0)
    assert bot.balance == pytest.approx(1000.0)

    # Now simulate a WS balance update: exchange pushes a new raw balance
    # directly onto the bot attributes (as exchange-specific subclasses do).
    bot.balance_raw = 1002.0
    # Snapped balance stays at 1000.0 since WS doesn't re-run hysteresis

    # Verify handle_balance_update sees the divergence and schedules execution
    async def fake_calc_upnl_sum():
        return 50.0

    bot.calc_upnl_sum = fake_calc_upnl_sum
    bot.execution_scheduled = False

    await bot.handle_balance_update(source="WS")

    # After WS update, the raw accessor should return the new value
    assert bot.get_raw_balance() == pytest.approx(1002.0)
    # Snapped balance hasn't changed (no hysteresis re-snap via WS path)
    assert bot.get_hysteresis_snapped_balance() == pytest.approx(1000.0)
    # Execution should have been scheduled due to balance_raw change
    assert bot.execution_scheduled is True


def test_authoritative_barrier_allows_coherent_changed_cycle():
    bot = Passivbot.__new__(Passivbot)

    bot._begin_authoritative_refresh_epoch()
    bot._record_authoritative_surface("balance", ("b", 1))
    bot._record_authoritative_surface("positions", ("p", 1))
    bot._record_authoritative_surface("open_orders", ("o", 1))
    bot._record_authoritative_surface("fills", ("f", 1))

    blocked, details = bot._authoritative_execution_barrier_state()
    assert blocked is False
    assert details["changed"] == ["fills", "open_orders", "positions"]

    bot._begin_authoritative_refresh_epoch()
    bot._record_authoritative_surface("balance", ("b", 1))
    bot._record_authoritative_surface("positions", ("p", 1))
    bot._record_authoritative_surface("open_orders", ("o", 1))
    bot._record_authoritative_surface("fills", ("f", 1))

    blocked, details = bot._authoritative_execution_barrier_state()
    assert blocked is False
    assert details["changed"] == []


def test_authoritative_barrier_waits_for_next_epoch_confirmation():
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)

    bot._begin_authoritative_refresh_epoch()
    for surface, sig in (
        ("balance", ("b", 1)),
        ("positions", ("p", 1)),
        ("open_orders", ("o", 1)),
        ("fills", ("f", 1)),
    ):
        bot._record_authoritative_surface(surface, sig)
    bot._authoritative_execution_barrier_state()

    bot._request_authoritative_confirmation({"open_orders"})
    blocked, details = bot._authoritative_execution_barrier_state()
    assert blocked is True
    assert details["missing"] == ["open_orders"]

    bot._begin_authoritative_refresh_epoch()
    bot._record_authoritative_surface("open_orders", ("o", 1))
    blocked, details = bot._authoritative_execution_barrier_state()
    assert blocked is False
    assert details["missing"] == []
    assert getattr(bot, "_authoritative_pending_confirmations", {}) == {}
    assert bot.freshness_ledger.surface_epoch("open_orders") == 2


def test_staged_refresh_plan_defers_fills_until_next_minute(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {}
    bot._pnl_history_coverage_ready_for_risk = lambda: True
    bot.freshness_ledger.stamp("fills", ("fills", "fresh"), now_ms=120_010, epoch=1)

    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 120_500)
    plan = bot._authoritative_staged_refresh_plan()

    assert plan == {"balance", "positions", "open_orders"}
    assert bot._authoritative_refresh_plan_surfaces == plan

    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 180_000)
    plan = bot._authoritative_staged_refresh_plan()

    assert plan == {"balance", "positions", "open_orders", "fills"}


def test_staged_refresh_plan_keeps_fills_when_risk_coverage_unproven(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {}
    bot._pnl_history_coverage_ready_for_risk = lambda: False
    bot.freshness_ledger.stamp("fills", ("fills", "fresh"), now_ms=120_010, epoch=1)

    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 120_500)

    assert bot._authoritative_staged_refresh_plan() == {
        "balance",
        "positions",
        "open_orders",
        "fills",
    }


def test_staged_refresh_plan_keeps_fills_when_known_gap_overlaps_lookback(monkeypatch):
    now_ms = 10 * 86_400_000
    start_ms = now_ms - 86_400_000

    class _Cache:
        def load_metadata(self):
            return {
                "known_gaps": [
                    {
                        "start_ts": start_ms + 60_000,
                        "end_ts": start_ms + 120_000,
                        "retry_count": 0,
                        "reason": "fetch_failed",
                        "confidence": 0.0,
                    }
                ],
                "covered_start_ms": start_ms,
                "history_scope": "window",
                "oldest_event_ts": start_ms,
            }

        def get_known_gaps(self):
            return list(self.load_metadata()["known_gaps"])

        def get_covered_start_ms(self):
            return start_ms

    cache = _Cache()
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {}
    bot._pnls_manager = SimpleNamespace(
        cache=cache,
        get_history_scope=lambda: "window",
    )
    bot.config = {"live": {"pnls_max_lookback_days": 1.0}}
    bot.live_value = lambda key: bot.config["live"][key]
    bot.get_exchange_time = lambda: now_ms
    bot.freshness_ledger.stamp("fills", ("fills", "fresh"), now_ms=120_010, epoch=1)

    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 120_500)

    assert bot._pnl_history_coverage_ready_for_risk() is False
    assert bot._authoritative_staged_refresh_plan() == {
        "balance",
        "positions",
        "open_orders",
        "fills",
    }


def test_staged_refresh_plan_keeps_pending_fills_even_with_recent_fills(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {"fills": 2}
    bot.freshness_ledger.stamp("fills", ("fills", "fresh"), now_ms=120_010, epoch=1)

    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 120_500)

    assert bot._authoritative_staged_refresh_plan() == {
        "balance",
        "positions",
        "open_orders",
        "fills",
    }


def test_authoritative_barrier_uses_current_staged_plan_when_no_pending():
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {}

    bot._begin_authoritative_refresh_epoch()
    bot._authoritative_refresh_plan_surfaces = {"balance", "positions", "open_orders"}
    for surface in ("balance", "positions", "open_orders"):
        bot._record_authoritative_surface(surface, (surface, "fresh"))

    blocked, details = bot._authoritative_execution_barrier_state()

    assert blocked is False
    assert details["required"] == ["balance", "open_orders", "positions"]


def test_positions_change_without_fills_requests_full_confirmation():
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {}

    bot._begin_authoritative_refresh_epoch()
    bot._record_authoritative_surface("balance", ("b", 1))
    bot._record_authoritative_surface("positions", ("p", 1))
    bot._record_authoritative_surface("open_orders", ("o", 1))

    bot._finalize_authoritative_refresh_consistency(
        {"balance", "positions", "open_orders"}
    )

    assert bot._authoritative_pending_confirmations == {
        "balance": 2,
        "positions": 2,
        "open_orders": 2,
        "fills": 2,
    }


def test_staged_planner_preconditions_require_current_epoch_surfaces():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)

    bot._begin_authoritative_refresh_epoch()
    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=False)
    assert ok is False
    assert details["missing"] == sorted(ACCOUNT_SURFACES | {"completed_candles"})

    for surface in ACCOUNT_SURFACES:
        bot._record_authoritative_surface(surface, (surface, "fresh"))
    bot._record_authoritative_surface("completed_candles", tuple())

    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=False)
    assert ok is True
    assert details["missing"] == []

    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=True)
    assert ok is False
    assert details["missing"] == ["market_snapshot"]

    bot._record_authoritative_surface("market_snapshot", ("market", "fresh"))
    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=True)
    assert ok is True
    assert set(details["required"]) == set(LIVE_STATE_SURFACES)


def test_staged_planner_preconditions_allow_open_orders_only_confirmation_epoch():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {}

    bot._begin_authoritative_refresh_epoch()
    for surface in ACCOUNT_SURFACES:
        bot._record_authoritative_surface(surface, (surface, "baseline"))
    bot._record_authoritative_surface("completed_candles", tuple())

    bot._request_authoritative_confirmation({"open_orders"})
    blocked, details = bot._authoritative_execution_barrier_state()
    assert blocked is True
    assert details["missing"] == ["open_orders"]

    bot._begin_authoritative_refresh_epoch()
    bot._record_authoritative_surface("open_orders", ("open_orders", "confirmed"))
    bot._record_authoritative_surface("completed_candles", tuple())

    blocked, details = bot._authoritative_execution_barrier_state()
    assert blocked is False
    assert details["missing"] == []

    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=False)
    assert ok is True
    assert details["missing"] == []
    assert bot.freshness_ledger.surface_epoch("balance") == 1
    assert bot.freshness_ledger.surface_epoch("positions") == 1
    assert bot.freshness_ledger.surface_epoch("fills") == 1
    assert bot.freshness_ledger.surface_epoch("open_orders") == 2

    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=True)
    assert ok is False
    assert details["missing"] == ["market_snapshot"]

    bot._record_authoritative_surface("market_snapshot", ("market_snapshot", "current"))
    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=True)
    assert ok is True
    assert details["missing"] == []


def test_staged_planner_preconditions_raise_before_rust_planning():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)

    bot._begin_authoritative_refresh_epoch()

    with pytest.raises(RuntimeError, match="staged planner precondition failed"):
        bot._assert_staged_planner_preconditions(
            include_market_snapshot=False, context="market snapshot refresh"
        )


def test_staged_planner_preconditions_reject_stale_completed_candle_signature():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot.active_symbols = ["BTC/USDT:USDT"]
    bot.positions = {"BTC/USDT:USDT": {"long": {"size": 1.0}, "short": {"size": 0.0}}}
    bot.open_orders = {}
    bot.PB_modes = {"long": {}, "short": {}}
    bot.cm = SimpleNamespace(
        get_completed_candle_health=lambda symbol, windows=None, now_ms=None: {
            "ok": False,
            "timeframes": {
                "1m": {
                    "coverage_ok": False,
                    "latest_expected_ts": 120_000,
                    "last_cached_ts": 60_000,
                    "missing_candles": 1,
                }
            },
        }
    )
    bot._begin_authoritative_refresh_epoch()
    for surface in ACCOUNT_SURFACES:
        bot._record_authoritative_surface(surface, (surface, "fresh"))
    bot._record_authoritative_surface("completed_candles", (("BTC/USDT:USDT", 60_000),))

    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=False)

    assert ok is False
    assert "completed_candles" in details["missing"]
    assert details["invalid"]["completed_candles"][0]["symbol"] == "BTC/USDT:USDT"


def test_staged_planner_preconditions_explain_completed_candle_signature_mismatch():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot.active_symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    bot.positions = {
        "BTC/USDT:USDT": {"long": {"size": 1.0}, "short": {"size": 0.0}},
        "ETH/USDT:USDT": {"long": {"size": 1.0}, "short": {"size": 0.0}},
    }
    bot.open_orders = {}
    bot.PB_modes = {"long": {}, "short": {}}

    def completed_candle_health(symbol, windows=None, now_ms=None):
        return {
            "ok": True,
            "timeframes": {
                "1m": {
                    "coverage_ok": True,
                    "latest_expected_ts": 120_000,
                    "last_cached_ts": 120_000,
                    "missing_candles": 0,
                    "runtime_synthetic_count": 0,
                }
            },
        }

    bot.cm = SimpleNamespace(get_completed_candle_health=completed_candle_health)

    bot._begin_authoritative_refresh_epoch()
    for surface in ACCOUNT_SURFACES:
        bot._record_authoritative_surface(surface, (surface, "fresh"))
    bot._record_authoritative_surface("completed_candles", (("BTC/USDT:USDT", 60_000),))

    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=False)

    assert ok is False
    mismatch = details["invalid"]["completed_candles"][0]
    assert mismatch["reason"] == "signature_mismatch"
    assert mismatch["mismatch_type"] == "planning_universe_changed"
    assert mismatch["expected_count"] == 2
    assert mismatch["stamped_count"] == 1
    assert mismatch["missing_symbols"] == ["ETH/USDT:USDT"]
    assert mismatch["changed_symbols"] == ["BTC/USDT:USDT"]


def test_completed_candle_signature_ignores_later_cache_improvements():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    symbol = "BTC/USDT:USDT"
    stamp_ms = 120_500
    bot.active_symbols = [symbol]
    bot.positions = {symbol: {"long": {"size": 1.0}, "short": {"size": 0.0}}}
    bot.open_orders = {}
    bot.PB_modes = {"long": {}, "short": {}}

    def completed_candle_health(_symbol, windows=None, now_ms=None):
        return {
            "ok": True,
            "timeframes": {
                "1m": {
                    "coverage_ok": True,
                    "latest_expected_ts": 60_000,
                    "last_cached_ts": 60_000 if int(now_ms) == stamp_ms else 120_000,
                    "missing_candles": 0,
                    "runtime_synthetic_count": 0 if int(now_ms) == stamp_ms else 3,
                }
            },
        }

    bot.cm = SimpleNamespace(get_completed_candle_health=completed_candle_health)

    bot._begin_authoritative_refresh_epoch()
    for surface in ACCOUNT_SURFACES:
        bot._record_authoritative_surface(surface, (surface, "fresh"))
    bot.freshness_ledger.stamp(
        "completed_candles",
        ((symbol, 60_000),),
        now_ms=stamp_ms,
        epoch=bot._authoritative_refresh_epoch,
    )

    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=False)

    assert ok is True
    assert details["missing"] == []


def test_staged_planner_preconditions_validate_candles_at_surface_stamp_time():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    symbol = "BTC/USDT:USDT"
    stamp_ms = 120_500
    stamped_signature = ((symbol, 60_000),)
    bot.active_symbols = [symbol]
    bot.positions = {symbol: {"long": {"size": 1.0}, "short": {"size": 0.0}}}
    bot.open_orders = {}
    bot.PB_modes = {"long": {}, "short": {}}

    def completed_candle_health(_symbol, windows=None, now_ms=None):
        if int(now_ms) == stamp_ms:
            return {
                "ok": True,
                "timeframes": {
                    "1m": {
                        "coverage_ok": True,
                        "latest_expected_ts": 60_000,
                        "last_cached_ts": 60_000,
                        "missing_candles": 0,
                        "runtime_synthetic_count": 0,
                    }
                },
            }
        return {
            "ok": False,
            "timeframes": {
                "1m": {
                    "coverage_ok": False,
                    "latest_expected_ts": 120_000,
                    "last_cached_ts": 60_000,
                    "missing_candles": 1,
                }
            },
        }

    bot.cm = SimpleNamespace(get_completed_candle_health=completed_candle_health)

    bot._begin_authoritative_refresh_epoch()
    for surface in ACCOUNT_SURFACES:
        bot._record_authoritative_surface(surface, (surface, "fresh"))
    bot.freshness_ledger.stamp(
        "completed_candles",
        stamped_signature,
        now_ms=stamp_ms,
        epoch=bot._authoritative_refresh_epoch,
    )

    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=False)

    assert ok is True
    assert details["missing"] == []


def test_staged_planner_preconditions_allow_tail_fallback_shape_recovery():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    symbol = "BTC/USDT:USDT"
    stamp_ms = 120_500
    stamped_signature = (
        (symbol, 120_000, "tail_gap_fallback", 60_000, 60_000),
    )
    bot.active_symbols = [symbol]
    bot.positions = {symbol: {"long": {"size": 1.0}, "short": {"size": 0.0}}}
    bot.open_orders = {}
    bot.PB_modes = {"long": {}, "short": {}}
    bot.cm = SimpleNamespace(
        get_completed_candle_health=lambda _symbol, windows=None, now_ms=None: {
            "ok": True,
            "timeframes": {
                "1m": {
                    "coverage_ok": True,
                    "latest_expected_ts": 120_000,
                    "last_cached_ts": 120_000,
                    "missing_candles": 0,
                    "runtime_synthetic_count": 0,
                }
            },
        }
    )

    bot._begin_authoritative_refresh_epoch()
    for surface in ACCOUNT_SURFACES:
        bot._record_authoritative_surface(surface, (surface, "fresh"))
    bot.freshness_ledger.stamp(
        "completed_candles",
        stamped_signature,
        now_ms=stamp_ms,
        epoch=bot._authoritative_refresh_epoch,
    )

    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=False)

    assert ok is True
    assert details["missing"] == []


def test_build_staged_planning_snapshot_captures_exact_surface_contract():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"user": "tester"}}
    bot.exchange = "bybit"
    bot.coin_overrides = {}
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {}
    symbol = "BTC/USDT:USDT"
    bot.active_symbols = [symbol]
    bot.positions = {symbol: {"long": {"size": 1.0}, "short": {"size": 0.0}}}
    bot.open_orders = {}
    bot.PB_modes = {"long": {}, "short": {}}
    bot.cm = SimpleNamespace(
        get_completed_candle_health=lambda sym, windows=None, now_ms=None: {
            "ok": True,
            "timeframes": {
                "1m": {
                    "coverage_ok": True,
                    "latest_expected_ts": 120_000,
                    "last_cached_ts": 120_000,
                    "missing_candles": 0,
                    "runtime_synthetic_count": 0,
                }
            },
        }
    )
    snapshots = {
        symbol: MarketSnapshot(
            symbol=symbol,
            bid=100.0,
            ask=101.0,
            last=100.5,
            fetched_ms=passivbot_module.utc_ms(),
            source="test",
        )
    }

    bot._begin_authoritative_refresh_epoch()
    for surface in ACCOUNT_SURFACES:
        bot._record_authoritative_surface(surface, (surface, "fresh"))
    candle_signature = ((symbol, 120_000),)
    bot._record_authoritative_surface("completed_candles", candle_signature)
    bot._record_market_snapshot_surface([symbol], snapshots)

    planning_snapshot = bot._build_staged_planning_snapshot([symbol], snapshots)

    assert planning_snapshot is not None
    assert planning_snapshot.symbols == (symbol,)
    assert planning_snapshot.last_prices() == {symbol: 100.5}
    assert planning_snapshot.completed_candle_signature == candle_signature
    assert set(planning_snapshot.required_surfaces) == set(LIVE_STATE_SURFACES)
    assert planning_snapshot.invalid_details(now_ms=passivbot_module.utc_ms()) == []


@pytest.mark.asyncio
async def test_planning_snapshot_freezes_data_packet_revisions_through_cycle(monkeypatch):
    symbol = "BTC/USDT:USDT"
    positions = [
        {
            "symbol": symbol,
            "position_side": "long",
            "size": 0.1,
            "price": 100.0,
            "lastUpdateTimestamp": 120_000,
        }
    ]
    bot, _counts = _counted_staged_account_refresh_bot(
        balance=100.0,
        positions=positions,
        open_orders=[],
    )
    bot.config = {"live": {"user": "tester"}}
    bot.exchange = "fake"
    bot.coin_overrides = {}
    _disable_entry_cooldown_delta_guard_for_staged_refresh_test(bot)
    bot.active_symbols = [symbol]
    bot.PB_modes = {"long": {}, "short": {}}
    bot.cm = SimpleNamespace(
        get_completed_candle_health=lambda sym, windows=None, now_ms=None: {
            "ok": True,
            "timeframes": {
                "1m": {
                    "coverage_ok": True,
                    "latest_expected_ts": 120_000,
                    "last_cached_ts": 120_000,
                    "missing_candles": 0,
                    "runtime_synthetic_count": 0,
                }
            },
        }
    )
    events = []
    bot._monitor_record_event = (
        lambda kind, tags, payload=None, **kwargs: events.append(
            {"kind": kind, "tags": tuple(tags), "payload": dict(payload or {}), **kwargs}
        )
    )

    assert await bot.refresh_authoritative_state() is True
    candle_signature = ((symbol, 120_000),)
    bot._record_authoritative_surface("completed_candles", candle_signature)
    snapshots = {
        symbol: MarketSnapshot(
            symbol=symbol,
            bid=99.5,
            ask=100.5,
            last=100.0,
            fetched_ms=passivbot_module.utc_ms(),
            source="test",
        )
    }
    bot._record_market_snapshot_surface([symbol], snapshots)
    bot._live_event_current_cycle_id = "cy_snapshot"
    bot._current_live_event_cycle_id = Passivbot._current_live_event_cycle_id.__get__(
        bot, Passivbot
    )
    diagnostic_build_calls = []
    original_diagnostic_build = passivbot_module.DiagnosticEvent.build

    def recording_diagnostic_build(kind, tags, payload=None, **kwargs):
        event = original_diagnostic_build(kind, tags, payload, **kwargs)
        if kind == "snapshot.built":
            diagnostic_build_calls.append((payload, kwargs, event))
        return event

    monkeypatch.setattr(
        passivbot_module.DiagnosticEvent,
        "build",
        staticmethod(recording_diagnostic_build),
    )

    planning_snapshot = bot._build_staged_planning_snapshot([symbol], snapshots)
    frozen_revisions = {
        packet.kind: packet.revision for packet in planning_snapshot.data_packets
    }

    assert frozen_revisions == {
        "balance": 1,
        "positions": 1,
        "open_orders": 1,
    }
    snapshot_events = [
        event for event in events if event["kind"] == "snapshot.built"
    ]
    snapshot_payload = snapshot_events[-1]["payload"]
    assert snapshot_payload["snapshot_id"] == planning_snapshot.snapshot_id
    assert diagnostic_build_calls[-1][1]["cycle_id"] == "cy_snapshot"
    assert diagnostic_build_calls[-1][1]["snapshot_id"] == planning_snapshot.snapshot_id
    surface_age_names = {item["name"] for item in snapshot_payload["surface_ages"]}
    assert {"balance", "positions", "open_orders", "completed_candles", "market_snapshot"} <= (
        surface_age_names
    )
    market_summary = snapshot_payload["market_snapshot_summary"]
    assert market_summary["count"] == 1
    assert market_summary["symbol_count"] == 1
    assert market_summary["missing_count"] == 0
    assert market_summary["configured_max_age_ms"] == 10_000
    assert market_summary["sources"] == ["test"]
    rendered_snapshot = json.dumps(snapshot_payload, sort_keys=True)
    assert "\"last\"" not in rendered_snapshot
    assert "\"bid\"" not in rendered_snapshot
    assert "\"ask\"" not in rendered_snapshot
    availability = snapshot_payload["planning_availability"]
    assert availability["snapshot_id"] == planning_snapshot.snapshot_id
    assert availability["record_count"] == 18
    assert availability["status_counts"] == {"available": 18}
    assert "records" not in availability

    positions[0]["size"] = 0.2
    assert await bot.refresh_authoritative_state() is True
    assert bot._live_data_packets["positions"].revision == 2
    assert {
        packet.kind: packet.revision for packet in planning_snapshot.data_packets
    } == frozen_revisions
    assert planning_snapshot.snapshot_id == snapshot_events[-1]["payload"]["snapshot_id"]


def test_build_staged_planning_snapshot_rejects_stale_market_snapshot():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"user": "tester"}}
    bot.exchange = "bybit"
    bot.coin_overrides = {}
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {}
    symbol = "BTC/USDT:USDT"
    snapshots = {
        symbol: MarketSnapshot(
            symbol=symbol,
            bid=100.0,
            ask=101.0,
            last=100.5,
            fetched_ms=passivbot_module.utc_ms() - 20_000,
            source="test",
        )
    }

    bot._begin_authoritative_refresh_epoch()
    for surface in ACCOUNT_SURFACES:
        bot._record_authoritative_surface(surface, (surface, "fresh"))
    bot._record_authoritative_surface("completed_candles", tuple())
    bot._record_market_snapshot_surface([symbol], snapshots)

    with pytest.raises(RuntimeError, match="planning snapshot invalid"):
        bot._build_staged_planning_snapshot([symbol], snapshots)


def _availability_test_snapshot(
    *,
    now_ms: int,
    symbols: tuple[str, ...],
    surfaces: dict[str, tuple[int, int, object]],
    market_symbols: tuple[str, ...] | None = None,
    completed_candle_signature: tuple = tuple(),
) -> PlanningSnapshot:
    return PlanningSnapshot(
        ts_ms=now_ms,
        exchange="bybit",
        user="tester",
        epoch=7,
        symbols=symbols,
        required_surfaces=tuple(sorted(surfaces)),
        surfaces=tuple(
            PlanningSurfaceStamp(
                name=name,
                epoch=epoch,
                updated_ms=now_ms,
                signature=signature,
                min_epoch=min_epoch,
            )
            for name, (epoch, min_epoch, signature) in sorted(surfaces.items())
        ),
        market_snapshots=tuple(
            PlanningMarketSnapshot(
                symbol=symbol,
                bid=99.5,
                ask=100.5,
                last=100.0,
                fetched_ms=now_ms,
                source="test",
            )
            for symbol in (market_symbols if market_symbols is not None else symbols)
        ),
        market_snapshot_max_age_ms=10_000,
        completed_candle_signature=completed_candle_signature,
        snapshot_id="snapshot-7",
    )


def _availability_record(
    availability: dict, *, symbol: str, position_side: str, order_class: str
) -> dict:
    matches = [
        record
        for record in availability["records"]
        if record["symbol"] == symbol
        and record["position_side"] == position_side
        and record["order_class"] == order_class
    ]
    assert len(matches) == 1
    return matches[0]


def test_planning_availability_reports_snapshot_invalid_without_enforcement():
    symbol = "BTC/USDT:USDT"
    now_ms = passivbot_module.utc_ms()
    snapshot = _availability_test_snapshot(
        now_ms=now_ms,
        symbols=(symbol,),
        surfaces={
            "balance": (2, 2, ("balance", "fresh")),
            "positions": (1, 2, ("positions", "old")),
            "open_orders": (2, 2, ("open_orders", "fresh")),
            "market_snapshot": (2, 2, ("market_snapshot", "fresh")),
            "completed_candles": (2, 2, ((symbol, 120_000),)),
            "fills": (2, 2, ("fills", "fresh")),
        },
        completed_candle_signature=((symbol, 120_000),),
    )

    availability = PlanningAvailability.from_snapshot(snapshot, now_ms=now_ms).to_dict()

    assert availability["summary"]["status_counts"] == {"unavailable": 18}
    assert availability["records"][0]["status"] == "unavailable"
    assert availability["records"][0]["reason_code"] == "surface_epoch_too_old"
    assert "positions" in availability["records"][0]["unavailable_surfaces"]
    entry_cancel = _availability_record(
        availability,
        symbol=symbol,
        position_side="long",
        order_class="entry_cancel",
    )
    assert entry_cancel["status"] == "unavailable"
    assert entry_cancel["unavailable_surfaces"] == ["positions"]


def test_planning_availability_panic_close_does_not_require_candles_or_emas():
    symbol = "BTC/USDT:USDT"
    now_ms = passivbot_module.utc_ms()
    snapshot = _availability_test_snapshot(
        now_ms=now_ms,
        symbols=(symbol,),
        surfaces={
            "balance": (7, 7, ("balance", "fresh")),
            "positions": (7, 7, ((symbol, "long", 0.1, 100.0),)),
            "open_orders": (7, 7, ("open_orders", "fresh")),
            "market_snapshot": (7, 7, ("market_snapshot", "fresh")),
        },
    )

    availability = PlanningAvailability.from_snapshot(snapshot, now_ms=now_ms).to_dict()
    panic = _availability_record(
        availability,
        symbol=symbol,
        position_side="long",
        order_class="hsl_panic_close",
    )
    entry = _availability_record(
        availability,
        symbol=symbol,
        position_side="long",
        order_class="initial_entry",
    )

    assert panic["status"] == "available"
    assert "canonical_candles" not in panic["required_surfaces"]
    assert entry["status"] == "unavailable"
    assert "canonical_candles" in entry["required_surfaces"]


def test_planning_availability_initial_entries_require_canonical_candles():
    symbol = "BTC/USDT:USDT"
    now_ms = passivbot_module.utc_ms()
    snapshot = _availability_test_snapshot(
        now_ms=now_ms,
        symbols=(symbol,),
        surfaces={
            "balance": (7, 7, ("balance", "fresh")),
            "positions": (7, 7, tuple()),
            "open_orders": (7, 7, tuple()),
            "market_snapshot": (7, 7, ("market_snapshot", "fresh")),
            "completed_candles": (7, 7, tuple()),
            "fills": (7, 7, ("fills", "fresh")),
        },
        completed_candle_signature=tuple(),
    )

    availability = PlanningAvailability.from_snapshot(snapshot, now_ms=now_ms).to_dict()
    record = _availability_record(
        availability,
        symbol=symbol,
        position_side="long",
        order_class="initial_entry",
    )

    assert record["status"] == "unavailable"
    assert record["reason_code"] == "missing_canonical_candles"
    assert record["unavailable_surfaces"] == ["canonical_candles"]


def test_planning_availability_stale_flat_symbol_candles_do_not_poison_position_close():
    positioned = "BTC/USDT:USDT"
    flat = "XMR/USDT:USDT"
    now_ms = passivbot_module.utc_ms()
    snapshot = _availability_test_snapshot(
        now_ms=now_ms,
        symbols=(flat, positioned),
        surfaces={
            "balance": (7, 7, ("balance", "fresh")),
            "positions": (7, 7, ((positioned, "long", 0.1, 100.0),)),
            "open_orders": (7, 7, tuple()),
            "market_snapshot": (7, 7, ("market_snapshot", "fresh")),
            "completed_candles": (7, 7, ((positioned, 120_000),)),
            "fills": (7, 7, ("fills", "fresh")),
        },
        completed_candle_signature=((positioned, 120_000),),
    )

    availability = PlanningAvailability.from_snapshot(snapshot, now_ms=now_ms).to_dict()
    positioned_close = _availability_record(
        availability,
        symbol=positioned,
        position_side="long",
        order_class="take_profit_close",
    )
    flat_entry = _availability_record(
        availability,
        symbol=flat,
        position_side="long",
        order_class="initial_entry",
    )

    assert positioned_close["status"] == "available"
    assert "canonical_candles" not in positioned_close["required_surfaces"]
    assert flat_entry["status"] == "unavailable"
    assert flat_entry["reason_code"] == "missing_canonical_candles"


def test_planning_availability_trailing_close_requires_canonical_candles():
    symbol = "BTC/USDT:USDT"
    now_ms = passivbot_module.utc_ms()
    snapshot = _availability_test_snapshot(
        now_ms=now_ms,
        symbols=(symbol,),
        surfaces={
            "balance": (7, 7, ("balance", "fresh")),
            "positions": (7, 7, ((symbol, "long", 0.1, 100.0),)),
            "open_orders": (7, 7, tuple()),
            "market_snapshot": (7, 7, ("market_snapshot", "fresh")),
            "completed_candles": (7, 7, tuple()),
            "fills": (7, 7, ("fills", "fresh")),
        },
        completed_candle_signature=tuple(),
    )

    availability = PlanningAvailability.from_snapshot(snapshot, now_ms=now_ms).to_dict()
    trailing = _availability_record(
        availability,
        symbol=symbol,
        position_side="long",
        order_class="trailing_close",
    )
    non_trailing_tp = _availability_record(
        availability,
        symbol=symbol,
        position_side="long",
        order_class="take_profit_close",
    )

    assert trailing["status"] == "unavailable"
    assert trailing["reason_code"] == "missing_canonical_candles"
    assert "canonical_candles" in trailing["required_surfaces"]
    unstuck = _availability_record(
        availability,
        symbol=symbol,
        position_side="long",
        order_class="unstuck_close",
    )
    assert unstuck["status"] == "unavailable"
    assert unstuck["reason_code"] == "missing_canonical_candles"
    assert "canonical_candles" in unstuck["required_surfaces"]
    assert non_trailing_tp["status"] == "available"
    assert "canonical_candles" not in non_trailing_tp["required_surfaces"]


def test_planning_availability_twel_reduce_does_not_require_candles():
    symbol = "BTC/USDT:USDT"
    now_ms = passivbot_module.utc_ms()
    snapshot = _availability_test_snapshot(
        now_ms=now_ms,
        symbols=(symbol,),
        surfaces={
            "balance": (7, 7, ("balance", "fresh")),
            "positions": (7, 7, ((symbol, "long", 0.1, 100.0),)),
            "open_orders": (7, 7, tuple()),
            "market_snapshot": (7, 7, ("market_snapshot", "fresh")),
            "fills": (7, 7, ("fills", "fresh")),
        },
    )

    availability = PlanningAvailability.from_snapshot(snapshot, now_ms=now_ms).to_dict()
    record = _availability_record(
        availability,
        symbol=symbol,
        position_side="long",
        order_class="wel_twel_reduce_close",
    )

    assert record["status"] == "available"
    assert record["required_surfaces"] == [
        "balance",
        "positions",
        "open_orders",
        "market_prices",
        "fill_history",
    ]


def test_planning_availability_missing_fills_only_affects_fill_required_classes():
    symbol = "BTC/USDT:USDT"
    now_ms = passivbot_module.utc_ms()
    snapshot = _availability_test_snapshot(
        now_ms=now_ms,
        symbols=(symbol,),
        surfaces={
            "balance": (7, 7, ("balance", "fresh")),
            "positions": (7, 7, ((symbol, "long", 0.1, 100.0),)),
            "open_orders": (7, 7, tuple()),
            "market_snapshot": (7, 7, ("market_snapshot", "fresh")),
            "completed_candles": (7, 7, ((symbol, 120_000),)),
        },
        completed_candle_signature=((symbol, 120_000),),
    )

    availability = PlanningAvailability.from_snapshot(snapshot, now_ms=now_ms).to_dict()

    assert _availability_record(
        availability,
        symbol=symbol,
        position_side="long",
        order_class="hsl_panic_close",
    )["status"] == "available"
    assert _availability_record(
        availability,
        symbol=symbol,
        position_side="long",
        order_class="take_profit_close",
    )["status"] == "available"
    entry = _availability_record(
        availability,
        symbol=symbol,
        position_side="long",
        order_class="initial_entry",
    )
    assert entry["status"] == "unavailable"
    assert entry["unavailable_surfaces"] == ["fill_history"]
    assert _availability_record(
        availability,
        symbol=symbol,
        position_side="long",
        order_class="entry_cancel",
    )["status"] == "available"
    fill_required = _availability_record(
        availability,
        symbol=symbol,
        position_side="long",
        order_class="unstuck_close",
    )
    assert fill_required["status"] == "unavailable"
    assert fill_required["reason_code"] == "missing_surface"
    assert fill_required["unavailable_surfaces"] == ["fill_history"]


def test_build_staged_planning_snapshot_survives_diagnostic_failures():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"user": "tester"}}
    bot.exchange = "bybit"
    bot.coin_overrides = {}
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {}
    symbol = "BTC/USDT:USDT"
    snapshots = {
        symbol: MarketSnapshot(
            symbol=symbol,
            bid=100.0,
            ask=101.0,
            last=100.5,
            fetched_ms=passivbot_module.utc_ms(),
            source="test",
        )
    }

    bot._begin_authoritative_refresh_epoch()
    for surface in ACCOUNT_SURFACES:
        bot._record_authoritative_surface(surface, (surface, "fresh"))
    bot._record_authoritative_surface("completed_candles", tuple())
    bot._record_market_snapshot_surface([symbol], snapshots)

    def failing_packets(_required):
        raise RuntimeError("packet metadata unavailable")

    def failing_emitter(*_args, **_kwargs):
        raise RuntimeError("monitor unavailable")

    bot._planning_data_packets = failing_packets
    bot._emit_snapshot_built_diagnostic = failing_emitter

    planning_snapshot = bot._build_staged_planning_snapshot([symbol], snapshots)

    assert planning_snapshot.symbols == (symbol,)
    assert planning_snapshot.data_packets == tuple()


def test_staged_execution_defer_emits_planning_unavailable_diagnostic():
    bot = Passivbot.__new__(Passivbot)
    events = []
    bot._monitor_record_event = (
        lambda kind, tags, payload=None, **kwargs: events.append(
            {"kind": kind, "tags": tuple(tags), "payload": dict(payload or {}), **kwargs}
        )
    )
    details = {
        "missing": ["positions"],
        "required": ["balance", "positions", "open_orders"],
        "epoch": 3,
        "min_epochs": {"positions": 3},
        "invalid": {},
        "context": "rust order calculation",
        "defer_reason": "staged_planner_inputs_not_fresh",
    }

    bot._log_staged_execution_defer(details)

    assert len(events) == 1
    assert events[0]["kind"] == "planning_unavailable"
    assert events[0]["tags"] == ("diagnostic", "planning", "unavailable")
    assert events[0]["payload"] == {
        "cycle_id": 3,
        "context": "rust order calculation",
        "missing": ["positions"],
        "required": ["balance", "positions", "open_orders"],
        "min_epochs": {"positions": 3},
        "invalid": {},
        "defer_reason": "staged_planner_inputs_not_fresh",
    }
    assert isinstance(events[0]["ts"], int)


def test_staged_execution_defer_survives_planning_unavailable_emit_failure():
    bot = Passivbot.__new__(Passivbot)

    def failing_emitter(_details):
        raise RuntimeError("monitor unavailable")

    bot._emit_planning_unavailable_diagnostic = failing_emitter
    details = {
        "missing": ["positions"],
        "required": ["balance", "positions", "open_orders"],
        "epoch": 3,
        "min_epochs": {"positions": 3},
        "invalid": {},
        "context": "rust order calculation",
        "defer_reason": "staged_planner_inputs_not_fresh",
    }

    bot._log_staged_execution_defer(details)


@pytest.mark.asyncio
async def test_pre_create_snapshot_filter_blocks_stale_market_snapshots():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_refresh_epoch = 1
    sink = ListEventSink()
    bot._live_event_current_cycle_id = "cy_stale_pre_create_market_snapshot"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink], monitor_sinks=[]
    )
    symbol = "BTC/USDT:USDT"

    async def fake_get_snapshots(symbols, max_age_ms=None):
        return {
            s: MarketSnapshot(
                symbol=s,
                bid=100.0,
                ask=100.0,
                last=100.0,
                fetched_ms=passivbot_module.utc_ms() - 20_000,
                source="test",
            )
            for s in symbols
        }

    bot.market_snapshot_provider = SimpleNamespace(get_snapshots=fake_get_snapshots)
    bot._current_planning_snapshot = PlanningSnapshot(
        ts_ms=passivbot_module.utc_ms(),
        exchange="bybit",
        user="tester",
        epoch=1,
        symbols=(symbol,),
        required_surfaces=tuple(),
        surfaces=tuple(),
        market_snapshots=(
            PlanningMarketSnapshot(
                symbol=symbol,
                bid=100.0,
                ask=100.0,
                last=100.0,
                fetched_ms=passivbot_module.utc_ms(),
                source="test",
            ),
        ),
        market_snapshot_max_age_ms=10_000,
        completed_candle_signature=tuple(),
    )

    orders = [
        {
            "symbol": symbol,
            "side": "buy",
            "position_side": "long",
            "qty": 1.0,
            "price": 99.0,
        }
    ]

    assert await bot._filter_fresh_market_snapshot_creations(orders) == []
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.EXECUTION_CREATE_SKIPPED
    ]
    assert len(events) == 1
    event = events[0]
    assert event.reason_code == ReasonCodes.PRE_CREATE_MARKET_SNAPSHOT_UNAVAILABLE
    assert event.status == "skipped"
    assert event.level == "warning"
    assert event.cycle_id == "cy_stale_pre_create_market_snapshot"
    assert event.data["order_count"] == 1
    assert event.data["symbols"] == [symbol]
    assert event.data["stage"] == "market_snapshot_validation"
    assert event.data["details_count"] == 1
    assert event.data["details"][0]["symbol"] == symbol
    assert event.data["details"][0]["reason"] == "stale"
    assert event.data["details"][0]["age_ms"] >= 20_000
    assert event.data["details"][0]["max_age_ms"] == 10_000
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_pre_create_snapshot_filter_emits_failed_refresh_skip_event():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_refresh_epoch = 1
    sink = ListEventSink()
    bot._live_event_current_cycle_id = "cy_failed_pre_create_market_snapshot"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink], monitor_sinks=[]
    )
    symbol = "BTC/USDT:USDT"
    now_ms = passivbot_module.utc_ms()
    bot._current_planning_snapshot = PlanningSnapshot(
        ts_ms=now_ms,
        exchange="bybit",
        user="tester",
        epoch=1,
        symbols=(symbol,),
        required_surfaces=tuple(),
        surfaces=tuple(),
        market_snapshots=(
            PlanningMarketSnapshot(
                symbol=symbol,
                bid=100.0,
                ask=100.0,
                last=100.0,
                fetched_ms=now_ms,
                source="test",
            ),
        ),
        market_snapshot_max_age_ms=10_000,
        completed_candle_signature=tuple(),
    )

    async def fail_get_snapshots(symbols, max_age_ms=None):
        raise RuntimeError("exchange unavailable with raw detail")

    bot.market_snapshot_provider = SimpleNamespace(get_snapshots=fail_get_snapshots)
    orders = [
        {
            "symbol": symbol,
            "side": "buy",
            "position_side": "long",
            "qty": 1.0,
            "price": 99.0,
        }
    ]

    assert await bot._filter_fresh_market_snapshot_creations(orders) == []
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.EXECUTION_CREATE_SKIPPED
    ]
    assert len(events) == 1
    event = events[0]
    assert event.reason_code == ReasonCodes.PRE_CREATE_MARKET_SNAPSHOT_UNAVAILABLE
    assert event.status == "skipped"
    assert event.level == "warning"
    assert event.cycle_id == "cy_failed_pre_create_market_snapshot"
    assert event.data["order_count"] == 1
    assert event.data["symbols"] == [symbol]
    assert event.data["stage"] == "market_snapshot_refresh"
    assert event.data["error_type"] == "RuntimeError"
    assert "exchange unavailable" not in json.dumps(event.data)
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_pre_create_snapshot_filter_refreshes_stale_planning_market_snapshot():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_refresh_epoch = 1
    symbol = "BTC/USDT:USDT"
    now_ms = passivbot_module.utc_ms()
    bot._current_planning_snapshot = PlanningSnapshot(
        ts_ms=now_ms - 20_000,
        exchange="bybit",
        user="tester",
        epoch=1,
        symbols=(symbol,),
        required_surfaces=tuple(),
        surfaces=tuple(),
        market_snapshots=(
            PlanningMarketSnapshot(
                symbol=symbol,
                bid=100.0,
                ask=100.0,
                last=100.0,
                fetched_ms=now_ms - 20_000,
                source="test",
            ),
        ),
        market_snapshot_max_age_ms=10_000,
        completed_candle_signature=tuple(),
    )

    async def fake_get_snapshots(symbols, max_age_ms=None):
        return {
            s: MarketSnapshot(
                symbol=s,
                bid=101.0,
                ask=101.0,
                last=101.0,
                fetched_ms=passivbot_module.utc_ms(),
                source="test",
            )
            for s in symbols
        }

    bot.market_snapshot_provider = SimpleNamespace(get_snapshots=fake_get_snapshots)
    orders = [
        {
            "symbol": symbol,
            "side": "buy",
            "position_side": "long",
            "qty": 1.0,
            "price": 99.0,
        }
    ]

    assert await bot._filter_fresh_market_snapshot_creations(orders) == orders


def _make_pre_create_distance_guard_bot(
    symbol: str,
    *,
    threshold: float = 0.8,
    market_price: float = 100.0,
):
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"limit_order_create_max_market_dist_pct": threshold}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_refresh_epoch = 1
    now_ms = passivbot_module.utc_ms()
    bot._current_planning_snapshot = PlanningSnapshot(
        ts_ms=now_ms,
        exchange="bybit",
        user="tester",
        epoch=1,
        symbols=(symbol,),
        required_surfaces=tuple(),
        surfaces=tuple(),
        market_snapshots=(
            PlanningMarketSnapshot(
                symbol=symbol,
                bid=market_price,
                ask=market_price,
                last=market_price,
                fetched_ms=now_ms,
                source="test",
            ),
        ),
        market_snapshot_max_age_ms=10_000,
        completed_candle_signature=tuple(),
    )

    async def fake_get_snapshots(symbols, max_age_ms=None):
        return {
            s: MarketSnapshot(
                symbol=s,
                bid=market_price,
                ask=market_price,
                last=market_price,
                fetched_ms=passivbot_module.utc_ms(),
                source="test",
            )
            for s in symbols
        }

    bot.market_snapshot_provider = SimpleNamespace(get_snapshots=fake_get_snapshots)
    return bot


@pytest.mark.asyncio
async def test_pre_create_snapshot_filter_skips_far_limit_order_creations(
    monkeypatch, caplog
):
    def fake_order_market_diff(side, price, market_price):
        side = str(side).lower()
        if side == "buy":
            return 1.0 - float(price) / float(market_price)
        if side == "sell":
            return float(price) / float(market_price) - 1.0
        raise ValueError("invalid side")

    monkeypatch.setattr(passivbot_module, "order_market_diff", fake_order_market_diff)

    symbol = "POPCAT/USDT:USDT"
    bot = _make_pre_create_distance_guard_bot(symbol)
    sink = ListEventSink()
    bot._live_event_current_cycle_id = "cy_distance_guard"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink], monitor_sinks=[]
    )
    kept_buy = {
        "symbol": symbol,
        "side": "buy",
        "position_side": "long",
        "qty": 1.0,
        "price": 20.0,
        "pb_order_type": "entry_grid_normal_long",
    }
    skipped_buy = {
        "symbol": symbol,
        "side": "buy",
        "position_side": "long",
        "qty": 1.0,
        "price": 19.99,
        "pb_order_type": "entry_grid_cropped_long",
    }
    kept_sell = {
        "symbol": symbol,
        "side": "sell",
        "position_side": "short",
        "qty": 1.0,
        "price": 180.0,
        "pb_order_type": "entry_grid_normal_short",
    }
    skipped_sell = {
        "symbol": symbol,
        "side": "sell",
        "position_side": "short",
        "qty": 1.0,
        "price": 180.01,
        "pb_order_type": "entry_grid_cropped_short",
    }
    skipped_close = {
        "symbol": symbol,
        "side": "sell",
        "position_side": "long",
        "qty": 1.0,
        "price": 180.02,
        "reduce_only": True,
        "pb_order_type": "close_grid_long",
    }
    market_order = {
        "symbol": symbol,
        "side": "buy",
        "position_side": "long",
        "qty": 1.0,
        "price": 1.0,
        "type": "market",
        "pb_order_type": "close_panic_long",
    }

    with caplog.at_level(logging.INFO):
        filtered = await bot._filter_fresh_market_snapshot_creations(
            [
                kept_buy,
                skipped_buy,
                kept_sell,
                skipped_sell,
                skipped_close,
                market_order,
            ]
        )

    assert filtered == [kept_buy, kept_sell, market_order]
    assert any(
        "reason=limit_order_create_market_distance" in rec.message
        and "skipped=3" in rec.message
        for rec in caplog.records
    )
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.EXECUTION_CREATE_SKIPPED
    ]
    assert len(events) == 1
    event = events[0]
    assert event.reason_code == ReasonCodes.LIMIT_ORDER_CREATE_MARKET_DISTANCE
    assert event.status == "skipped"
    assert event.level == "info"
    assert event.cycle_id == "cy_distance_guard"
    assert event.data["order_count"] == 3
    assert event.data["symbols"] == [symbol]
    assert event.data["symbols_count"] == 1
    assert event.data["threshold"] == pytest.approx(0.8)
    assert event.data["min_multiplier"] == pytest.approx(0.2)
    assert event.data["max_multiplier"] == pytest.approx(1.8)
    assert event.data["suppressed_text_log"] is False
    assert event.data["groups"] == [
        {
            "symbol": symbol,
            "pside": "long",
            "side": "buy",
            "pb_order_type": "entry_grid_cropped_long",
            "count": 1,
        },
        {
            "symbol": symbol,
            "pside": "short",
            "side": "sell",
            "pb_order_type": "entry_grid_cropped_short",
            "count": 1,
        },
        {
            "symbol": symbol,
            "pside": "long",
            "side": "sell",
            "pb_order_type": "close_grid_long",
            "count": 1,
        },
    ]
    assert len(event.data["sample"]) == 3
    assert all(item["symbol"] == symbol for item in event.data["sample"])
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_pre_create_snapshot_filter_disables_limit_order_distance_guard(
    monkeypatch,
):
    monkeypatch.setattr(
        passivbot_module,
        "order_market_diff",
        lambda side, price, market_price: 999.0,
    )

    symbol = "POPCAT/USDT:USDT"
    bot = _make_pre_create_distance_guard_bot(symbol, threshold=0.0)
    orders = [
        {
            "symbol": symbol,
            "side": "buy",
            "position_side": "long",
            "qty": 1.0,
            "price": 1.0,
            "pb_order_type": "entry_grid_normal_long",
        }
    ]

    assert await bot._filter_fresh_market_snapshot_creations(orders) == orders


@pytest.mark.asyncio
async def test_pre_create_snapshot_filter_blocks_non_market_planning_invalidation():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "bybit"
    sink = ListEventSink()
    bot._live_event_current_cycle_id = "cy_invalid_pre_create_planning_snapshot"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink], monitor_sinks=[]
    )
    symbol = "BTC/USDT:USDT"
    now_ms = passivbot_module.utc_ms()
    bot._current_planning_snapshot = PlanningSnapshot(
        ts_ms=now_ms,
        exchange="bybit",
        user="tester",
        epoch=1,
        symbols=(symbol,),
        required_surfaces=("positions",),
        surfaces=(
            PlanningSurfaceStamp(
                name="positions",
                epoch=1,
                updated_ms=now_ms,
                signature=("old",),
                min_epoch=2,
            ),
        ),
        market_snapshots=(
            PlanningMarketSnapshot(
                symbol=symbol,
                bid=100.0,
                ask=100.0,
                last=100.0,
                fetched_ms=now_ms,
                source="test",
            ),
        ),
        market_snapshot_max_age_ms=10_000,
        completed_candle_signature=tuple(),
    )

    async def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "pre-create refresh must not rescue non-market invalid state"
        )

    bot.market_snapshot_provider = SimpleNamespace(get_snapshots=fail_if_called)
    orders = [
        {
            "symbol": symbol,
            "side": "buy",
            "position_side": "long",
            "qty": 1.0,
            "price": 99.0,
        }
    ]

    assert await bot._filter_fresh_market_snapshot_creations(orders) == []
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.EXECUTION_CREATE_SKIPPED
    ]
    assert len(events) == 1
    event = events[0]
    assert event.reason_code == ReasonCodes.PRE_CREATE_PLANNING_SNAPSHOT_INVALID
    assert event.status == "skipped"
    assert event.level == "warning"
    assert event.cycle_id == "cy_invalid_pre_create_planning_snapshot"
    assert event.data["order_count"] == 1
    assert event.data["symbols"] == [symbol]
    assert event.data["stage"] == "planning_snapshot"
    assert event.data["details_count"] == 1
    assert event.data["details"][0]["surface"] == "positions"
    assert event.data["details"][0]["reason"] == "surface_epoch_too_old"
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def _make_open_order_guardrail_bot(*, epoch: int = 3):
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot._authoritative_refresh_epoch = epoch
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()
    bot._authoritative_pending_confirmations = {}
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.recent_order_cancellations = []
    bot.recent_order_executions = []
    bot.orders_emitted_to_exchange = []
    bot.log_order_action = lambda *args, **kwargs: None
    bot._reconcile_balance_after_open_orders_refresh = lambda: False
    bot._detect_foreign_passivbot_orders = AsyncMock(return_value=None)
    return bot


def _guardrail_order(**overrides):
    order = {
        "id": "created-1",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 0.01,
        "amount": 0.01,
        "price": 99_000.0,
        "reduce_only": False,
    }
    order.update(overrides)
    return order


@pytest.mark.asyncio
async def test_disappeared_self_order_blocks_creations_until_full_freshness():
    bot = _make_open_order_guardrail_bot(epoch=3)
    order = _guardrail_order()
    bot.open_orders = {"BTC/USDT:USDT": [order]}
    bot.fetched_open_orders = [order]
    bot.order_matches_recent_execution = lambda _order, max_age_ms=None: True
    bot.order_matches_bot_cancellation = lambda _order: False
    bot.order_was_recently_cancelled = lambda _order: 0.0

    ok = await Passivbot._apply_open_orders_snapshot(
        bot,
        [],
        allow_followup_positions_refresh=False,
    )

    assert ok is True
    assert bot.execution_scheduled is True
    assert bot.state_change_detected_by_symbol == {"BTC/USDT:USDT"}
    assert set(bot.freshness_ledger.blocked_symbols()) == {"BTC/USDT:USDT"}
    assert bot.freshness_ledger.blocked_symbols()["BTC/USDT:USDT"].min_epoch == 4
    assert bot._authoritative_pending_confirmations == {
        surface: 4 for surface in ACCOUNT_SURFACES
    }

    to_create, skipped = Passivbot._apply_freshness_creation_guardrails(
        bot,
        [
            {"symbol": "BTC/USDT:USDT", "side": "buy", "position_side": "long"},
            {"symbol": "ETH/USDT:USDT", "side": "buy", "position_side": "long"},
        ],
    )

    assert skipped == 1
    assert [order["symbol"] for order in to_create] == ["ETH/USDT:USDT"]

    bot._begin_authoritative_refresh_epoch()
    for surface in ACCOUNT_SURFACES:
        bot._record_authoritative_surface(surface, (surface, "fresh"))

    assert bot.freshness_ledger.blocked_symbols() == {}


@pytest.mark.asyncio
async def test_bot_cancelled_order_disappearance_does_not_arm_duplicate_guardrail():
    bot = _make_open_order_guardrail_bot(epoch=5)
    order = _guardrail_order()
    bot.open_orders = {"BTC/USDT:USDT": [order]}
    bot.fetched_open_orders = [order]
    bot.order_matches_bot_cancellation = lambda _order: True
    bot.order_was_recently_cancelled = lambda _order: 0.0
    bot.order_matches_recent_execution = lambda _order, max_age_ms=None: True

    ok = await Passivbot._apply_open_orders_snapshot(
        bot,
        [],
        allow_followup_positions_refresh=False,
    )

    assert ok is True
    assert bot.freshness_ledger.blocked_symbols() == {}
    assert bot._authoritative_pending_confirmations == {}
    assert bot.state_change_detected_by_symbol == set()


@pytest.mark.asyncio
async def test_unknown_manual_or_exchange_cancel_requests_confirmation_without_symbol_block():
    bot = _make_open_order_guardrail_bot(epoch=5)
    order = _guardrail_order(id="manual-or-exchange-cancel")
    bot.open_orders = {"BTC/USDT:USDT": [order]}
    bot.fetched_open_orders = [order]
    bot.order_matches_bot_cancellation = lambda _order: False
    bot.order_was_recently_cancelled = lambda _order: 0.0
    bot.order_matches_recent_execution = lambda _order, max_age_ms=None: False

    ok = await Passivbot._apply_open_orders_snapshot(
        bot,
        [],
        allow_followup_positions_refresh=False,
    )

    assert ok is True
    assert bot.freshness_ledger.blocked_symbols() == {}
    assert bot.state_change_detected_by_symbol == {"BTC/USDT:USDT"}
    assert bot._authoritative_pending_confirmations == {
        surface: 6 for surface in ACCOUNT_SURFACES
    }


@pytest.mark.asyncio
async def test_disappeared_emitted_order_record_arms_duplicate_guardrail_without_recent_execution():
    bot = _make_open_order_guardrail_bot(epoch=8)
    order = _guardrail_order(
        custom_id="entry_grid_normal_long",
        info={"clientOrderId": "entry_grid_normal_long"},
    )
    bot.get_exchange_time = lambda: 10_000
    bot.open_orders = {"BTC/USDT:USDT": [order]}
    bot.fetched_open_orders = [order]
    bot.order_matches_bot_cancellation = lambda _order: False
    bot.order_was_recently_cancelled = lambda _order: 0.0
    bot.order_matches_recent_execution = lambda _order, max_age_ms=None: False
    Passivbot._record_emitted_order_custom_id(bot, order, emitted_ts=9_000)

    ok = await Passivbot._apply_open_orders_snapshot(
        bot,
        [],
        allow_followup_positions_refresh=False,
    )

    assert ok is True
    assert set(bot.freshness_ledger.blocked_symbols()) == {"BTC/USDT:USDT"}
    assert bot.freshness_ledger.blocked_symbols()["BTC/USDT:USDT"].min_epoch == 9
    assert bot._authoritative_pending_confirmations == {
        surface: 9 for surface in ACCOUNT_SURFACES
    }


@pytest.mark.asyncio
async def test_restarted_inherited_order_disappearance_uses_confirmation_not_symbol_block():
    bot = _make_open_order_guardrail_bot(epoch=8)
    order = _guardrail_order(
        id="inherited-after-restart",
        custom_id="entry_grid_normal_long",
        info={"clientOrderId": "entry_grid_normal_long"},
    )
    bot.open_orders = {"BTC/USDT:USDT": [order]}
    bot.fetched_open_orders = [order]
    bot.order_matches_bot_cancellation = lambda _order: False
    bot.order_was_recently_cancelled = lambda _order: 0.0
    bot.order_matches_recent_execution = lambda _order, max_age_ms=None: False
    bot.orders_emitted_to_exchange = []

    ok = await Passivbot._apply_open_orders_snapshot(
        bot,
        [],
        allow_followup_positions_refresh=False,
    )

    assert ok is True
    assert bot.freshness_ledger.blocked_symbols() == {}
    assert bot.state_change_detected_by_symbol == {"BTC/USDT:USDT"}
    assert bot._authoritative_pending_confirmations == {
        surface: 9 for surface in ACCOUNT_SURFACES
    }


@pytest.mark.asyncio
async def test_disappeared_self_order_guardrail_blocks_real_plan_create_until_refresh(
    caplog,
):
    bot = _make_open_order_guardrail_bot(epoch=7)
    bot._last_plan_detail = {}
    bot._order_plan_summary_is_interesting = lambda **kwargs: False
    bot.PB_modes = {"long": {}, "short": {}}
    bot.active_symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]

    class _CM:
        def get_current_close(self, symbol, max_age_ms=None):
            return 100.0

    bot.cm = _CM()

    async def fake_get_live_last_prices(symbols, **kwargs):
        return {symbol: 100.0 for symbol in symbols}

    bot._get_live_last_prices = fake_get_live_last_prices
    bot.live_value = lambda key: 0.0 if key == "order_match_tolerance_pct" else 0.0

    disappeared_order = _guardrail_order()
    bot.open_orders = {"BTC/USDT:USDT": [disappeared_order], "ETH/USDT:USDT": []}
    bot.fetched_open_orders = [disappeared_order]
    bot.order_matches_recent_execution = lambda _order, max_age_ms=None: True
    bot.order_matches_bot_cancellation = lambda _order: False
    bot.order_was_recently_cancelled = lambda _order: 0.0

    await Passivbot._apply_open_orders_snapshot(
        bot,
        [],
        allow_followup_positions_refresh=False,
    )
    bot.state_change_detected_by_symbol = set()

    ideal_orders = {
        "BTC/USDT:USDT": [
            {
                "symbol": "BTC/USDT:USDT",
                "side": "buy",
                "position_side": "long",
                "qty": 0.01,
                "price": 99_000.0,
                "reduce_only": False,
            }
        ],
        "ETH/USDT:USDT": [
            {
                "symbol": "ETH/USDT:USDT",
                "side": "buy",
                "position_side": "long",
                "qty": 0.1,
                "price": 3_000.0,
                "reduce_only": False,
            }
        ],
    }

    async def fake_calc_ideal_orders():
        return ideal_orders

    bot.calc_ideal_orders = fake_calc_ideal_orders

    with caplog.at_level(logging.INFO):
        _to_cancel, to_create = await Passivbot.calc_orders_to_cancel_and_create(bot)

    assert [order["symbol"] for order in to_create] == ["ETH/USDT:USDT"]
    assert "freshness guardrail blocking order creation" in caplog.text

    bot._begin_authoritative_refresh_epoch()
    for surface in ACCOUNT_SURFACES:
        bot._record_authoritative_surface(surface, (surface, "fresh"))

    _to_cancel, to_create = await Passivbot.calc_orders_to_cancel_and_create(bot)

    assert sorted(order["symbol"] for order in to_create) == [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
    ]


@pytest.mark.asyncio
async def test_malformed_actual_open_order_blocks_symbol_plan(caplog):
    bot = _make_open_order_guardrail_bot(epoch=11)
    bot._last_plan_detail = {}
    bot._order_plan_summary_is_interesting = lambda **kwargs: False
    bot.PB_modes = {"long": {}, "short": {}}
    bot.active_symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]

    class _CM:
        def get_current_close(self, symbol, max_age_ms=None):
            return 100.0

    bot.cm = _CM()

    async def fake_get_live_last_prices(symbols, **kwargs):
        return {symbol: 100.0 for symbol in symbols}

    bot._get_live_last_prices = fake_get_live_last_prices
    bot.live_value = lambda key: 0.0 if key == "order_match_tolerance_pct" else 0.0

    malformed_order = _guardrail_order(id="malformed-entry")
    malformed_order.pop("qty")
    bot.open_orders = {"BTC/USDT:USDT": [malformed_order], "ETH/USDT:USDT": []}
    ideal_orders = {
        "BTC/USDT:USDT": [
            {
                "symbol": "BTC/USDT:USDT",
                "side": "buy",
                "position_side": "long",
                "qty": 0.01,
                "price": 99_000.0,
                "reduce_only": False,
            }
        ],
        "ETH/USDT:USDT": [
            {
                "symbol": "ETH/USDT:USDT",
                "side": "buy",
                "position_side": "long",
                "qty": 0.1,
                "price": 3_000.0,
                "reduce_only": False,
            }
        ],
    }

    async def fake_calc_ideal_orders():
        return ideal_orders

    bot.calc_ideal_orders = fake_calc_ideal_orders

    with caplog.at_level(logging.ERROR):
        to_cancel, to_create = await Passivbot.calc_orders_to_cancel_and_create(bot)

    assert to_cancel == []
    assert [order["symbol"] for order in to_create] == ["ETH/USDT:USDT"]
    assert bot._malformed_actual_order_symbols == {"BTC/USDT:USDT"}
    assert bot._malformed_actual_order_counts == {"BTC/USDT:USDT": 1}
    assert bot.state_change_detected_by_symbol == {"BTC/USDT:USDT"}
    assert bot.execution_scheduled is True
    assert bot._authoritative_pending_confirmations == {
        surface: 12 for surface in ACCOUNT_SURFACES
    }
    assert "malformed open order snapshot" in caplog.text
    assert "malformed_open_order_snapshot" in caplog.text


@pytest.mark.asyncio
async def test_ambiguous_cancel_forces_full_authoritative_confirmation():
    bot = Passivbot.__new__(Passivbot)
    bot.debug_mode = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot._health_orders_cancelled = 0
    confirmations = []

    order = {
        "id": "abc123",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 0.001,
        "price": 100_000.0,
        "reduce_only": False,
    }

    async def fake_execute_cancellations(orders):
        assert orders == [order]
        return [
            {
                "status": "success",
                "_passivbot_cancel_requires_full_authoritative_confirmation": True,
            }
        ]

    bot.live_value = lambda key: 10 if key == "max_n_cancellations_per_batch" else None
    bot.add_to_recent_order_cancellations = lambda _order: None
    bot.log_order_action = lambda *args, **kwargs: None
    bot._log_order_action_summary = lambda *args, **kwargs: None
    bot.execute_cancellations = fake_execute_cancellations
    bot.did_cancel_order = lambda executed, _order: executed.get("status") == "success"
    bot.remove_order = lambda *args, **kwargs: None
    bot._monitor_order_payload = lambda *args, **kwargs: {}
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._request_authoritative_confirmation = lambda surfaces: confirmations.append(
        set(surfaces)
    )

    res = await Passivbot.execute_cancellations_parent(bot, [order])

    assert len(res) == 1
    assert confirmations == [{"balance", "positions", "open_orders", "fills"}]
    assert bot.state_change_detected_by_symbol == {"BTC/USDT:USDT"}


@pytest.mark.asyncio
@pytest.mark.parametrize("console_sink_fails", [False, True])
async def test_ambiguous_cancel_structured_console_owns_confirmation_message(
    caplog, console_sink_fails
):
    class FailingConsoleSink:
        def write(self, _event):
            raise OSError("console unavailable")

    bot = Passivbot.__new__(Passivbot)
    bot.debug_mode = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot._health_orders_cancelled = 0
    bot.live_event_console_enabled = True
    bot._live_event_current_cycle_id = "cy_ambiguous_cancel"
    structured_sink = ListEventSink()
    console_sink = FailingConsoleSink() if console_sink_fails else ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[structured_sink],
        monitor_sinks=[],
        console_sink=console_sink,
    )
    confirmations = []
    order = {
        "id": "abc123",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 0.001,
        "price": 100_000.0,
        "reduce_only": False,
    }

    async def fake_execute_cancellations(orders):
        assert orders == [order]
        return [
            {
                "status": "success",
                "_passivbot_cancel_requires_full_authoritative_confirmation": True,
            }
        ]

    bot.live_value = lambda key: 10 if key == "max_n_cancellations_per_batch" else None
    bot.add_to_recent_order_cancellations = lambda _order: None
    bot.log_order_action = lambda *args, **kwargs: None
    bot._log_order_action_summary = lambda *args, **kwargs: None
    bot.execute_cancellations = fake_execute_cancellations
    bot.did_cancel_order = lambda executed, _order: executed.get("status") == "success"
    bot.remove_order = lambda *args, **kwargs: None
    bot._monitor_order_payload = lambda *args, **kwargs: {}
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._request_authoritative_confirmation = lambda surfaces: confirmations.append(
        set(surfaces)
    )

    with caplog.at_level(logging.INFO):
        res = await Passivbot.execute_cancellations_parent(bot, [order])

    assert len(res) == 1
    assert confirmations == [{"balance", "positions", "open_orders", "fills"}]
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    terminal_events = [
        event
        for event in structured_sink.events
        if event.event_type == EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL
    ]
    assert len(terminal_events) == 1
    assert not [
        record
        for record in caplog.records
        if "ambiguous cancel terminal state; forcing full account confirmation" in record.message
    ]
    if console_sink_fails:
        assert bot._live_event_pipeline.sink_error_counters["console"] >= 1
    else:
        assert console_sink.events == terminal_events
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_ambiguous_cancel_uses_legacy_confirmation_message_without_structured_console(
    caplog,
):
    bot = Passivbot.__new__(Passivbot)
    bot.debug_mode = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot._health_orders_cancelled = 0
    bot.live_event_console_enabled = True
    bot._live_event_current_cycle_id = "cy_ambiguous_cancel"
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
        console_sink=None,
    )
    order = {
        "id": "abc123",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 0.001,
        "price": 100_000.0,
        "reduce_only": False,
    }

    async def fake_execute_cancellations(orders):
        assert orders == [order]
        return [
            {
                "status": "success",
                "_passivbot_cancel_requires_full_authoritative_confirmation": True,
            }
        ]

    bot.live_value = lambda key: 10 if key == "max_n_cancellations_per_batch" else None
    bot.add_to_recent_order_cancellations = lambda _order: None
    bot.log_order_action = lambda *args, **kwargs: None
    bot._log_order_action_summary = lambda *args, **kwargs: None
    bot.execute_cancellations = fake_execute_cancellations
    bot.did_cancel_order = lambda executed, _order: executed.get("status") == "success"
    bot.remove_order = lambda *args, **kwargs: None
    bot._monitor_order_payload = lambda *args, **kwargs: {}
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._request_authoritative_confirmation = lambda _surfaces: None

    with caplog.at_level(logging.INFO):
        res = await Passivbot.execute_cancellations_parent(bot, [order])

    assert len(res) == 1
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert [
        event.event_type
        for event in sink.events
        if event.event_type == EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL
    ] == [EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL]
    legacy_messages = [
        record.message
        for record in caplog.records
        if "ambiguous cancel terminal state; forcing full account confirmation" in record.message
    ]
    assert legacy_messages == [
        "[order] ambiguous cancel terminal state; forcing full account confirmation "
        "before next cycle | symbols=BTC"
    ]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_positions_signature_ignores_margin_used_and_margin_mode_noise():
    bot = Passivbot.__new__(Passivbot)
    positions_a = [
        {
            "symbol": "BTC/USDT:USDT",
            "position_side": "long",
            "size": 0.1,
            "price": 100000.0,
            "margin_mode": "cross",
            "margin_used": 50.0,
        }
    ]
    positions_b = [
        {
            "symbol": "BTC/USDT:USDT",
            "position_side": "long",
            "size": 0.1,
            "price": 100000.0,
            "margin_mode": "isolated",
            "margin_used": 55.0,
        }
    ]

    assert bot._positions_signature(positions_a) == bot._positions_signature(
        positions_b
    )


def test_authoritative_barrier_does_not_block_on_balance_only_change():
    bot = Passivbot.__new__(Passivbot)

    bot._begin_authoritative_refresh_epoch()
    for surface, sig in (
        ("balance", ("b", 1)),
        ("positions", ("p", 1)),
        ("open_orders", ("o", 1)),
        ("fills", ("f", 1)),
    ):
        bot._record_authoritative_surface(surface, sig)
    bot._authoritative_execution_barrier_state()

    bot._begin_authoritative_refresh_epoch()
    bot._record_authoritative_surface("balance", ("b", 2))
    bot._record_authoritative_surface("positions", ("p", 1))
    bot._record_authoritative_surface("open_orders", ("o", 1))
    bot._record_authoritative_surface("fills", ("f", 1))
    blocked, details = bot._authoritative_execution_barrier_state()

    assert blocked is False
    assert details["changed"] == []


@pytest.mark.asyncio
async def test_run_execution_loop_waits_for_clean_authoritative_cycle_before_execute():
    bot = Passivbot.__new__(Passivbot)
    cycle = {"n": 0}
    executes = []

    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = True
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    async def fake_refresh_authoritative_state():
        cycle["n"] += 1
        bot._begin_authoritative_refresh_epoch()
        for surface, sig in (
            ("balance", ("b", 1)),
            ("positions", ("p", 1)),
            ("open_orders", ("o", 1)),
            ("fills", ("f", 1)),
        ):
            bot._record_authoritative_surface(surface, sig)
        bot._record_authoritative_surface("completed_candles", tuple())
        return True

    async def fake_refresh_market_state_if_needed():
        executes.append("market")
        return True

    async def fake_prepare_planning_universe():
        executes.append("universe")

    async def fake_execute_to_exchange(*, prepare_cycle=True):
        executes.append(("execute", prepare_cycle, cycle["n"]))
        return {"executed_cycle": cycle["n"]}

    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.refresh_market_state_if_needed = fake_refresh_market_state_if_needed
    bot.prepare_planning_universe = fake_prepare_planning_universe
    bot.execute_to_exchange = fake_execute_to_exchange

    result = await bot.run_execution_loop()

    assert result == {"executed_cycle": 1}
    assert executes == ["universe", "market", ("execute", False, 1)]


@pytest.mark.asyncio
async def test_run_execution_loop_does_not_reinitialize_coin_hsl_after_startup():
    bot = Passivbot.__new__(Passivbot)
    calls = []

    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = True
    bot._equity_hard_stop_coin_initialized = True
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: True
    bot._equity_hard_stop_signal_mode = lambda: "coin"
    bot._equity_hard_stop_check = AsyncMock(return_value=None)
    bot._equity_hard_stop_coin_red_active = lambda: False
    bot._equity_hard_stop_initialize_coin_from_history = AsyncMock()
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    async def fake_refresh_authoritative_state():
        bot._begin_authoritative_refresh_epoch()
        for surface, sig in (
            ("balance", ("b", 1)),
            ("positions", ("p", 1)),
            ("open_orders", ("o", 1)),
            ("fills", ("f", 1)),
            ("completed_candles", tuple()),
        ):
            bot._record_authoritative_surface(surface, sig)
        return True

    async def fake_prepare_planning_universe():
        calls.append("universe")

    async def fake_refresh_market_state_if_needed():
        calls.append("market")
        return True

    async def fake_execute_to_exchange(*, prepare_cycle=True):
        calls.append(("execute", prepare_cycle))
        return {"ok": True}

    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.prepare_planning_universe = fake_prepare_planning_universe
    bot.refresh_market_state_if_needed = fake_refresh_market_state_if_needed
    bot.execute_to_exchange = fake_execute_to_exchange

    result = await bot.run_execution_loop()

    assert result == {"ok": True}
    bot._equity_hard_stop_initialize_coin_from_history.assert_not_awaited()
    assert calls == ["universe", "market", ("execute", False)]


@pytest.mark.asyncio
async def test_run_execution_loop_defers_staged_precondition_without_error_count():
    bot = Passivbot.__new__(Passivbot)
    cycle = {"n": 0}
    executes = []

    bot.config = {"live": {}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = True
    bot.active_symbols = []
    bot.positions = {}
    bot.open_orders = {}
    bot.PB_modes = {"long": {}, "short": {}}
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    async def fake_refresh_authoritative_state():
        cycle["n"] += 1
        bot._begin_authoritative_refresh_epoch()
        for surface, sig in (
            ("balance", ("b", cycle["n"])),
            ("positions", ("p", cycle["n"])),
            ("open_orders", ("o", cycle["n"])),
            ("fills", ("f", cycle["n"])),
        ):
            bot._record_authoritative_surface(surface, sig)
        if cycle["n"] > 1:
            bot._record_authoritative_surface("completed_candles", tuple())
        return True

    async def fake_prepare_planning_universe():
        executes.append(("universe", cycle["n"]))

    async def fake_refresh_market_state_if_needed():
        executes.append(("market", cycle["n"]))
        return True

    async def fake_execute_to_exchange(*, prepare_cycle=True):
        executes.append(("execute", prepare_cycle, cycle["n"]))
        return {"executed_cycle": cycle["n"]}

    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.prepare_planning_universe = fake_prepare_planning_universe
    bot.refresh_market_state_if_needed = fake_refresh_market_state_if_needed
    bot.execute_to_exchange = fake_execute_to_exchange

    result = await bot.run_execution_loop()

    assert result == {"executed_cycle": 2}
    assert executes == [
        ("universe", 1),
        ("market", 1),
        ("universe", 2),
        ("market", 2),
        ("execute", False, 2),
    ]
    bot.restart_bot_on_too_many_errors.assert_not_called()


@pytest.mark.asyncio
async def test_run_execution_loop_waits_on_pending_pnl_without_restart(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    cycle = {"n": 0}
    executes = []
    sleeps = []

    async def fake_sleep_unless_shutdown(seconds, *, stage):
        sleeps.append((seconds, stage))
        return None

    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = True
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot._sleep_unless_shutdown = fake_sleep_unless_shutdown
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    async def fake_refresh_authoritative_state():
        cycle["n"] += 1
        if cycle["n"] <= 12:
            bot._last_authoritative_block_reason = "pending_pnl"
            bot._last_authoritative_pending_pnl_count = 2
            return False
        bot._begin_authoritative_refresh_epoch()
        for surface, sig in (
            ("balance", ("b", 1)),
            ("positions", ("p", 1)),
            ("open_orders", ("o", 1)),
            ("fills", ("f", 1)),
            ("completed_candles", tuple()),
        ):
            bot._record_authoritative_surface(surface, sig)
        return True

    async def fake_refresh_market_state_if_needed():
        executes.append("market")
        return True

    async def fake_prepare_planning_universe():
        executes.append("universe")

    async def fake_execute_to_exchange(*, prepare_cycle=True):
        executes.append(("execute", prepare_cycle, cycle["n"]))
        return {"executed_cycle": cycle["n"]}

    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.refresh_market_state_if_needed = fake_refresh_market_state_if_needed
    bot.prepare_planning_universe = fake_prepare_planning_universe
    bot.execute_to_exchange = fake_execute_to_exchange

    result = await bot.run_execution_loop()

    assert result == {"executed_cycle": 13}
    bot.restart_bot_on_too_many_errors.assert_not_awaited()
    assert executes == ["universe", "market", ("execute", False, 13)]
    sleep_seconds = [seconds for seconds, _stage in sleeps]
    assert sleep_seconds[:7] == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0]
    assert sleep_seconds[7:12] == [60.0] * 5
    assert {stage for _seconds, stage in sleeps} == {"pending_pnl_authoritative_retry"}


@pytest.mark.asyncio
async def test_run_execution_loop_retries_fill_history_coverage_without_restart(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    cycle = {"n": 0}
    executes = []
    sleeps = []

    async def fake_sleep_unless_shutdown(seconds, *, stage):
        sleeps.append((seconds, stage))
        return None

    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = True
    bot.active_symbols = []
    bot.positions = {}
    bot.open_orders = {}
    bot.PB_modes = {"long": {}, "short": {}}
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot._authoritative_refresh_epoch = 0
    bot._authoritative_pending_confirmations = {}
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot._emit_live_cycle_degraded = MagicMock()
    bot._sleep_unless_shutdown = fake_sleep_unless_shutdown
    bot._log_settled_order_waves = lambda *args, **kwargs: None
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    async def fake_refresh_authoritative_state():
        cycle["n"] += 1
        bot._begin_authoritative_refresh_epoch()
        for surface, sig in (
            ("balance", ("b", cycle["n"])),
            ("positions", ("p", cycle["n"])),
            ("open_orders", ("o", cycle["n"])),
            ("fills", ("f", cycle["n"])),
            ("completed_candles", tuple()),
        ):
            bot._record_authoritative_surface(surface, sig)
        return True

    async def fake_refresh_market_state_if_needed():
        executes.append(("market", cycle["n"]))
        return True

    async def fake_prepare_planning_universe():
        executes.append(("universe", cycle["n"]))

    async def fake_execute_to_exchange(*, prepare_cycle=True):
        executes.append(("execute", prepare_cycle, cycle["n"]))
        if cycle["n"] == 1:
            raise passivbot_module.FillHistoryCoverageUnavailable(
                "fill history coverage unknown for risk lookback"
            )
        return {"executed_cycle": cycle["n"]}

    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.refresh_market_state_if_needed = fake_refresh_market_state_if_needed
    bot.prepare_planning_universe = fake_prepare_planning_universe
    bot.execute_to_exchange = fake_execute_to_exchange

    result = await bot.run_execution_loop()

    assert result == {"executed_cycle": 2}
    bot.restart_bot_on_too_many_errors.assert_not_awaited()
    assert sleeps == [(1.0, "fill_history_coverage_retry")]
    assert executes == [
        ("universe", 1),
        ("market", 1),
        ("execute", False, 1),
        ("universe", 2),
        ("market", 2),
        ("execute", False, 2),
    ]
    coverage_event = next(
        call.kwargs
        for call in bot._emit_live_cycle_degraded.call_args_list
        if call.kwargs["reason_code"] == "fill_history_coverage_unavailable"
    )
    assert coverage_event["level"] == "warning"
    assert (
        coverage_event["data"]["error_type"]
        == "FillHistoryCoverageUnavailable"
    )
    assert "error" not in coverage_event["data"]


@pytest.mark.asyncio
async def test_refresh_market_state_updates_trailing_after_candles():
    bot = Passivbot.__new__(Passivbot)
    bot.stop_signal_received = False
    events = []

    async def fake_update_ohlcvs_1m_for_actives():
        events.append("candles")
        return True

    async def fake_update_trailing_data():
        events.append("trailing")

    bot.update_ohlcvs_1m_for_actives = fake_update_ohlcvs_1m_for_actives
    bot.update_trailing_data = fake_update_trailing_data

    assert await bot.refresh_market_state_if_needed() is True
    assert events == ["candles", "trailing"]


@pytest.mark.asyncio
async def test_run_execution_loop_stops_before_execute_when_signal_arrives_after_refresh():
    bot = Passivbot.__new__(Passivbot)
    executes = []

    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = False
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    async def fake_refresh_authoritative_state():
        bot._begin_authoritative_refresh_epoch()
        for surface, sig in (
            ("balance", ("b", 1)),
            ("positions", ("p", 1)),
            ("open_orders", ("o", 1)),
            ("fills", ("f", 1)),
        ):
            bot._record_authoritative_surface(surface, sig)
        bot.stop_signal_received = True
        return True

    async def fake_refresh_market_state_if_needed():
        return True

    async def fake_execute_to_exchange():
        executes.append(True)
        return {"unexpected": True}

    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.refresh_market_state_if_needed = fake_refresh_market_state_if_needed
    bot.execute_to_exchange = fake_execute_to_exchange

    result = await bot.run_execution_loop()

    assert result is None
    assert executes == []


@pytest.mark.asyncio
async def test_run_execution_loop_suppresses_inflight_shutdown_refresh_error(caplog):
    bot = Passivbot.__new__(Passivbot)

    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = False
    bot._health_errors = 0
    bot._health_rate_limits = 0
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot._monitor_record_error = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("shutdown errors should not be recorded as runtime errors")
    )
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    async def fake_refresh_authoritative_state():
        bot.stop_signal_received = True
        raise RuntimeError("connector is closed")

    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.execute_to_exchange = AsyncMock()

    with caplog.at_level(logging.DEBUG):
        result = await bot.run_execution_loop()

    assert result is None
    assert bot._health_errors == 0
    assert bot._health_rate_limits == 0
    bot.restart_bot_on_too_many_errors.assert_not_awaited()
    bot.execute_to_exchange.assert_not_awaited()
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert any(
        "execution loop stopped during in-flight refresh" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_run_execution_loop_records_nonshutdown_cancelled_error(
    caplog, monkeypatch
):
    bot = Passivbot.__new__(Passivbot)

    bot.exchange = "gateio"
    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = False
    bot._health_errors = 0
    bot._health_rate_limits = 0
    bot.error_counts = []
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    monitor_events = []
    bot._monitor_record_event = lambda *args, **kwargs: monitor_events.append(
        (args, kwargs)
    )
    bot._monitor_record_error = MagicMock()
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    async def fake_sleep(_seconds):
        return None

    async def fake_refresh_authoritative_state():
        raise asyncio.CancelledError("ccxt load_markets cancelled")

    async def fake_restart():
        bot.stop_signal_received = True

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.restart_bot_on_too_many_errors.side_effect = fake_restart
    bot.execute_to_exchange = AsyncMock()

    with caplog.at_level(logging.ERROR):
        result = await bot.run_execution_loop()

    assert result is None
    assert bot._health_errors == 1
    bot._monitor_record_error.assert_not_called()
    assert monitor_events == [
        (
            ("error.bot", ("error", "bot"), {
                "source": "run_execution_loop",
                "error_type": "CancelledError",
                "status": "-",
                "code": "-",
                "endpoint": "unknown",
                "action": "record_error_restart_backoff",
                "cycle": "abandoned",
            }),
            {},
        )
    ]
    bot.restart_bot_on_too_many_errors.assert_awaited_once()
    bot.execute_to_exchange.assert_not_awaited()
    messages = [record.message for record in caplog.records]
    assert any(
        "[error] operation=run_execution_loop" in message
        and "error_type=CancelledError" in message
        and "endpoint=unknown" in message
        and "action=record_error_restart_backoff" in message
        for message in messages
    )


@pytest.mark.asyncio
async def test_run_execution_loop_treats_shutdown_cancelled_error_as_clean_stop(caplog):
    bot = Passivbot.__new__(Passivbot)

    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = False
    bot._health_errors = 0
    bot._health_rate_limits = 0
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    bot._monitor_record_error = MagicMock(
        side_effect=AssertionError("shutdown cancellation should not be recorded")
    )
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    async def fake_refresh_authoritative_state():
        bot.stop_signal_received = True
        raise asyncio.CancelledError("shutdown")

    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.execute_to_exchange = AsyncMock()

    with caplog.at_level(logging.DEBUG):
        result = await bot.run_execution_loop()

    assert result is None
    assert bot._health_errors == 0
    bot.restart_bot_on_too_many_errors.assert_not_awaited()
    bot.execute_to_exchange.assert_not_awaited()
    assert any(
        "authoritative refresh cancelled during shutdown" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_exchange_time_sync_recovery_refreshes_ccxt_clients(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "binance"
    bot.user = "binance_01"
    bot.bot_id = "bot_1"
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot.cca = SimpleNamespace(
        options={"timeDifference": 10},
        load_time_difference=AsyncMock(
            side_effect=lambda: bot.cca.options.__setitem__("timeDifference", 25)
        ),
    )
    bot.ccp = SimpleNamespace(
        options={"timeDifference": -5},
        load_time_difference=AsyncMock(
            side_effect=lambda: bot.ccp.options.__setitem__("timeDifference", 30)
        ),
    )

    raw_error = (
        'binanceusdm {"code":-1021,"msg":"Timestamp for this request is outside of '
        'the recvWindow.","apiKey":"supersecret"}'
    )
    exc = RuntimeError(raw_error)
    with caplog.at_level(logging.WARNING):
        recovered = await bot._maybe_recover_exchange_time_sync(
            exc, source="test"
        )

    assert recovered is True
    bot.cca.load_time_difference.assert_awaited_once()
    bot.ccp.load_time_difference.assert_awaited_once()
    assert bot.cca.options["timeDifference"] == 25
    assert bot.ccp.options["timeDifference"] == 30
    warnings = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings == [
        "[time] clock offset | action=offset_refreshed source=test status=succeeded "
        "recovered=true synced=2[cca:10->25,+1] failed=0[-] error_type=RuntimeError"
    ]
    assert raw_error not in warnings[0]
    assert "supersecret" not in warnings[0]
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.event_type == EventTypes.EXCHANGE_TIME_SYNC
    assert event.component == "exchange.time_sync"
    assert event.tags == (EventTags.EXCHANGE, EventTags.TIME_SYNC)
    assert event.status == "succeeded"
    assert event.reason_code == ReasonCodes.EXCHANGE_TIME_SYNC
    assert event.data["source"] == "test"
    assert event.data["error_type"] == "RuntimeError"
    assert event.data["hook_available"] is True
    assert event.data["recovered"] is True
    assert event.data["synced_clients"] == ["cca:10->25", "ccp:-5->30"]
    assert event.data["failed_clients"] == []
    assert "error" not in event.data
    assert raw_error not in str(event.data)
    assert "supersecret" not in str(event.data)
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_exchange_time_sync_no_hook_emits_unavailable_event(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "kucoin"
    bot.user = "kucoin_01"
    bot.bot_id = "bot_1"
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot.cca = SimpleNamespace(options={})
    bot.ccp = None

    exc = RuntimeError("Invalid nonce apiKey=supersecret")
    with caplog.at_level(logging.WARNING):
        recovered = await bot._maybe_recover_exchange_time_sync(
            exc, source="fetch_balance"
        )

    assert recovered is False
    warnings = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings == [
        "[time] clock offset | action=no_hook source=fetch_balance status=skipped "
        "recovered=false synced=0[-] failed=0[-] error_type=RuntimeError"
    ]
    assert "supersecret" not in warnings[0]
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.event_type == EventTypes.EXCHANGE_TIME_SYNC
    assert event.status == "skipped"
    assert event.reason_code == ReasonCodes.EXCHANGE_TIME_SYNC_UNAVAILABLE
    assert event.data["source"] == "fetch_balance"
    assert event.data["hook_available"] is False
    assert event.data["recovered"] is False
    assert event.data["synced_count"] == 0
    assert event.data["failed_count"] == 0
    assert event.data["error_type"] == "RuntimeError"
    assert "error" not in event.data
    assert "supersecret" not in str(event.data)
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_exchange_time_sync_warning_stays_within_full_console_line_budget():
    error_type = type("TimeSync" + ("Error" * 100), (RuntimeError,), {})
    raw_error = "timestamp failure apiKey=supersecret " + ("x" * 500)
    message = Passivbot._format_exchange_time_sync_warning(
        source="refresh_" + ("source" * 100),
        error=error_type(raw_error),
        synced_clients=["cca:" + ("1" * 100) + "->" + ("2" * 100)],
        failed_clients=["ccp:" + ("RequestTimeout" * 100)],
        hook_available=True,
    )
    record = logging.LogRecord(
        "test", logging.WARNING, __file__, 0, "%s", (message,), None
    )
    record.created = 0.0
    record.log_prefix = "hyperliquid"
    formatter = logging.Formatter(DEFAULT_FORMAT_WITH_PREFIX, datefmt=DEFAULT_DATEFMT)
    formatter.converter = time.gmtime
    rendered = formatter.format(record)

    assert raw_error not in message
    assert "supersecret" not in message
    assert "\n" not in message
    assert "\r" not in message
    assert "\t" not in message
    assert len(message) <= Passivbot.EXCHANGE_TIME_SYNC_CONSOLE_MESSAGE_MAX_LEN
    assert Passivbot.EXCHANGE_TIME_SYNC_CONSOLE_MESSAGE_MAX_LEN == 240 - (
        len(rendered) - len(message)
    )
    assert len(rendered) <= 240


@pytest.mark.asyncio
async def test_run_execution_loop_recovers_timestamp_error_without_traceback(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "binance"
    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = False
    bot._health_errors = 0
    bot._health_rate_limits = 0
    bot.error_counts = []
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    bot._monitor_record_error = MagicMock()
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False
    bot.cca = SimpleNamespace(
        options={"timeDifference": 0},
        load_time_difference=AsyncMock(),
    )
    bot.ccp = None
    calls = 0

    async def fake_refresh_authoritative_state():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Timestamp for this request is outside of the recvWindow")
        bot.stop_signal_received = True
        return False

    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.execute_to_exchange = AsyncMock()

    with caplog.at_level(logging.WARNING):
        result = await bot.run_execution_loop()

    assert result is None
    assert calls == 2
    assert bot._health_errors == 1
    bot.cca.load_time_difference.assert_awaited_once()
    bot.restart_bot_on_too_many_errors.assert_awaited_once()
    bot.execute_to_exchange.assert_not_awaited()
    assert any(
        "[time] clock offset | action=offset_refreshed "
        "source=run_execution_loop status=succeeded" in r.message
        for r in caplog.records
    )
    assert not [r for r in caplog.records if "error with run_execution_loop" in r.message]


@pytest.mark.asyncio
async def test_run_execution_loop_error_log_includes_type_status_and_action(caplog, monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "kucoin"
    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = False
    bot._health_errors = 0
    bot._health_rate_limits = 0
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    monitor_events = []
    bot._monitor_record_event = lambda *args, **kwargs: monitor_events.append(
        (args, kwargs)
    )
    bot._monitor_record_error = MagicMock()
    bot._emit_live_cycle_degraded = MagicMock()
    bot._maybe_recover_exchange_time_sync = AsyncMock(return_value=False)
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    class FakeExchangeError(RuntimeError):
        pass

    raw_error = (
        "kucoinfutures GET https://example.invalid/account-overview?api_key=SECRET "
        "connector said raw-message"
    )
    exc = FakeExchangeError(raw_error)
    exc.http_status = 500
    exc.code = "500000"

    async def fake_sleep(_seconds):
        return None

    async def fake_refresh_authoritative_state():
        raise exc

    async def fake_restart():
        bot.stop_signal_received = True

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.restart_bot_on_too_many_errors.side_effect = fake_restart

    with caplog.at_level(logging.DEBUG):
        result = await bot.run_execution_loop()

    assert result is None
    assert bot._health_errors == 1
    bot.restart_bot_on_too_many_errors.assert_awaited_once()
    bot._monitor_record_error.assert_not_called()
    assert monitor_events == [
        (
            ("error.bot", ("error", "bot"), {
                "source": "run_execution_loop",
                "error_type": "FakeExchangeError",
                "status": "500",
                "code": "500000",
                "endpoint": "account-overview",
                "action": "record_error_restart_backoff",
                "cycle": "abandoned",
            }),
            {},
        )
    ]
    messages = [record.message for record in caplog.records]
    assert not any("error with run_execution_loop" in message for message in messages)
    assert any(
        "[error] operation=run_execution_loop" in message
        and "error_type=FakeExchangeError" in message
        and "status=500" in message
        and "code=500000" in message
        and "endpoint=account-overview" in message
        and "cycle=abandoned" in message
        and "action=record_error_restart_backoff" in message
        for message in messages
    )
    assert all(raw_error not in message for message in messages)
    assert all("SECRET" not in message for message in messages)
    degraded_event = bot._emit_live_cycle_degraded.call_args.kwargs
    assert degraded_event["reason_code"] == "FakeExchangeError"
    assert degraded_event["level"] == "error"
    assert degraded_event["data"]["error_type"] == "FakeExchangeError"
    assert "error" not in degraded_event["data"]
    assert raw_error not in str(degraded_event)
    assert "SECRET" not in str(degraded_event)
    debug_stack_messages = [
        record.message
        for record in caplog.records
        if record.levelno == logging.DEBUG and "stack frames" in record.message
    ]
    assert len(debug_stack_messages) == 1
    assert raw_error not in debug_stack_messages[0]
    assert "FakeExchangeError" not in debug_stack_messages[0]


@pytest.mark.asyncio
async def test_execution_loop_error_normal_info_is_bounded_and_skips_nondebug_frames(
    caplog, monkeypatch
):
    bot = Passivbot.__new__(Passivbot)
    bot.stop_signal_received = False
    bot._health_errors = 0
    bot.error_counts = []
    bot._maybe_recover_exchange_time_sync = AsyncMock(return_value=False)
    bot._monitor_record_event = MagicMock()
    bot.restart_bot = AsyncMock()

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    format_tb = MagicMock()
    monkeypatch.setattr(passivbot_module.traceback, "format_tb", format_tb)
    raw_error = "raw-message https://example.invalid/private?api_key=SECRET"

    try:
        raise RuntimeError(raw_error)
    except RuntimeError as exc:
        with caplog.at_level(logging.INFO):
            assert await bot._handle_execution_loop_failure(
                exc, allow_time_sync_recovery=False
            ) is True

    assert bot._health_errors == 1
    assert len(bot.error_counts) == 1
    bot.restart_bot.assert_not_awaited()
    format_tb.assert_not_called()
    normal_messages = [
        record.message for record in caplog.records if record.levelno >= logging.INFO
    ]
    assert "[health] error_budget count=1 limit=10 window=1h action=continue" in normal_messages
    assert all(raw_error not in message for message in normal_messages)
    assert all("SECRET" not in message for message in normal_messages)


def test_execution_loop_error_fields_are_bounded_and_classify_unknown_endpoint():
    bot = Passivbot.__new__(Passivbot)

    class FakeExchangeError(RuntimeError):
        pass

    exc = FakeExchangeError(
        "GET https://example.invalid/" + "x" * 80 + "?api_key=SECRET"
    )
    exc.status = "500?api_key=SECRET"
    exc.code = "500000?signature=SIG"
    exc.info = {"status": "429", "retCode": "RATE_LIMIT"}

    fields = bot._execution_loop_error_fields(exc)

    assert fields == {
        "error_type": "FakeExchangeError",
        "status": "429",
        "code": "RATE_LIMIT",
        "endpoint": "unknown",
    }
    assert "SECRET" not in str(fields)
    assert "SIG" not in str(fields)
    assert "example.invalid" not in str(fields)


def test_execution_loop_error_burst_summarizes_repeated_endpoints(caplog, monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    now = {"value": 1_000_000}
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now["value"])

    fields = {
        "error_type": "RequestTimeout",
        "status": "-",
        "code": "-",
        "endpoint": "account-overview",
    }

    with caplog.at_level(logging.WARNING):
        bot._log_execution_loop_error_burst(fields)
        bot._log_execution_loop_error_burst(fields)
        bot._log_execution_loop_error_burst(fields)

    messages = [record.message for record in caplog.records]
    assert len(messages) == 1
    assert "[health] execution loop error burst" in messages[0]
    assert "count=3" in messages[0]
    assert "top=account-overview:3" in messages[0]
    assert "action=restart_backoff_continues" in messages[0]


def test_execution_loop_error_burst_structured_console_owns_warning(caplog, monkeypatch):
    sink = ListEventSink()
    console_sink = ListEventSink()
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "kucoin"
    bot.user = "kucoin_01"
    bot.bot_id = "bot_1"
    bot.live_event_console_enabled = True
    bot._live_event_current_cycle_id = "cy_error"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
        console_sink=console_sink,
    )
    now = {"value": 1_000_000}
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now["value"])

    fields = {
        "error_type": "RequestTimeout",
        "status": "-",
        "code": "-",
        "endpoint": "account-overview",
        "error": (
            "kucoinfutures GET https://api-futures.kucoin.com/api/v1/"
            "account-overview?api_key=SECRET&signature=SIG"
        ),
    }

    with caplog.at_level(logging.WARNING):
        bot._log_execution_loop_error_burst(fields)
        bot._log_execution_loop_error_burst(fields)
        bot._log_execution_loop_error_burst(fields)

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert not [
        record
        for record in caplog.records
        if "[health] execution loop error burst" in record.message
    ]
    events = sink.events
    assert len(events) == 1
    event = events[0]
    assert event.event_type == EventTypes.HEALTH_SUMMARY
    assert event.level == "warning"
    assert event.component == "execution_loop"
    assert event.tags == (
        EventTags.HEALTH,
        EventTags.EXECUTION,
        EventTags.SUMMARY,
    )
    assert event.cycle_id == "cy_error"
    assert event.status == "degraded"
    assert event.reason_code == ReasonCodes.EXECUTION_LOOP_ERROR_BURST
    assert event.data["count"] == 3
    assert event.data["window_s"] == 1
    assert event.data["top_endpoints"] == [
        {"endpoint": "account-overview", "count": 3}
    ]
    assert event.data["latest_error_type"] == "RequestTimeout"
    assert event.data["latest_status"] == "-"
    assert event.data["latest_code"] == "-"
    assert event.data["latest_endpoint"] == "account-overview"
    assert "latest_error" not in event.data
    assert "SECRET" not in str(event.data)
    assert "SIG" not in str(event.data)
    assert "api-futures.kucoin.com" not in str(event.data)
    assert console_sink.events == [event]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.parametrize(
    ("console_enabled", "has_pipeline"),
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_execution_loop_error_burst_uses_legacy_fallback_without_structured_console(
    caplog, monkeypatch, console_enabled, has_pipeline
):
    bot = Passivbot.__new__(Passivbot)
    bot.live_event_console_enabled = console_enabled
    structured_sink = ListEventSink()
    if has_pipeline:
        bot._live_event_pipeline = LiveEventPipeline(
            structured_sinks=[structured_sink],
            monitor_sinks=[],
            console_sink=None,
        )
    now = {"value": 1_000_000}
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now["value"])
    fields = {
        "error_type": "RequestTimeout",
        "status": "-",
        "code": "-",
        "endpoint": "account-overview",
    }

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            bot._log_execution_loop_error_burst(fields)

    messages = [record.message for record in caplog.records]
    assert len(messages) == 1
    assert "[health] execution loop error burst" in messages[0]
    if has_pipeline:
        assert bot._live_event_pipeline.flush(timeout=2.0) is True
        assert [event.reason_code for event in structured_sink.events] == [
            ReasonCodes.EXECUTION_LOOP_ERROR_BURST
        ]
        assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.parametrize("emitter_state", ["missing", "noncallable"])
def test_execution_loop_error_burst_uses_legacy_fallback_when_emitter_unavailable(
    caplog, monkeypatch, emitter_state
):
    bot = Passivbot.__new__(Passivbot)
    bot.live_event_console_enabled = True
    if emitter_state == "missing":
        monkeypatch.delattr(Passivbot, "_emit_execution_loop_error_burst_event")
    else:
        bot._emit_execution_loop_error_burst_event = object()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[],
        monitor_sinks=[],
        console_sink=ListEventSink(),
    )
    now = {"value": 1_000_000}
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now["value"])
    fields = {
        "error_type": "RequestTimeout",
        "status": "-",
        "code": "-",
        "endpoint": "account-overview",
    }

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            bot._log_execution_loop_error_burst(fields)

    messages = [record.message for record in caplog.records]
    assert len(messages) == 1
    assert "[health] execution loop error burst" in messages[0]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_execution_loop_error_burst_keeps_structured_console_owner_when_queue_is_full(
    caplog, monkeypatch
):
    console_sink = ListEventSink()
    pipeline = LiveEventPipeline(
        structured_sinks=[ListEventSink()],
        monitor_sinks=[],
        console_sink=console_sink,
        queue_maxsize=1,
        start=False,
    )
    pipeline.emit(LiveEvent(EventTypes.CYCLE_STARTED))
    bot = Passivbot.__new__(Passivbot)
    bot.live_event_console_enabled = True
    bot._live_event_pipeline = pipeline
    now = {"value": 1_000_000}
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now["value"])
    fields = {
        "error_type": "RequestTimeout",
        "status": "-",
        "code": "-",
        "endpoint": "account-overview",
    }

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            bot._log_execution_loop_error_burst(fields)

    assert not [
        record
        for record in caplog.records
        if "[health] execution loop error burst" in record.message
    ]
    assert [
        event.reason_code
        for event in console_sink.events
        if event.reason_code == ReasonCodes.EXECUTION_LOOP_ERROR_BURST
    ] == [ReasonCodes.EXECUTION_LOOP_ERROR_BURST]
    assert pipeline.health_snapshot()["event_dropped_total"] == 1
    pipeline._queue.get_nowait()
    pipeline._queue.task_done()
    assert pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_execution_loop_error_budget_keeps_count_and_restart_threshold(caplog, monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    now = 1_000_000
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now)
    bot.error_counts = [now - 60 * 60 * 1000] + [now - 1] * 8
    bot.restart_bot = AsyncMock()

    with caplog.at_level(logging.INFO):
        await bot.restart_bot_on_too_many_errors()
        with pytest.raises(Exception, match="too many errors"):
            await bot.restart_bot_on_too_many_errors()

    assert len(bot.error_counts) == 10
    bot.restart_bot.assert_awaited_once()
    messages = [record.message for record in caplog.records]
    assert messages == [
        "[health] error_budget count=9 limit=10 window=1h action=continue",
        "[health] error_budget count=10 limit=10 window=1h action=restart_at_limit",
    ]


def test_staged_refresh_timing_summary_aggregates_routine_fast_refreshes(
    caplog, monkeypatch
):
    bot = Passivbot.__new__(Passivbot)
    now = {"value": 1_000_000}
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now["value"])
    bot._authoritative_pending_confirmations = {}
    bot._authoritative_refresh_epoch_changed = set()

    with caplog.at_level(logging.INFO):
        for i in range(60):
            bot._log_staged_refresh_timings(
                {"open_orders"},
                {"open_orders": 100 + i},
                100 + i,
            )

    messages = [record.message for record in caplog.records]
    assert len(messages) == 1
    assert "[state] staged refresh timing summary" in messages[0]
    assert "plan=open_orders" in messages[0]
    assert "count=60" in messages[0]
    assert "open_orders=100/130/159ms" in messages[0]


def test_staged_refresh_timing_summary_emits_structured_event(monkeypatch):
    sink = ListEventSink()
    bot = Passivbot.__new__(Passivbot)
    now = {"value": 1_000_000}
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now["value"])
    bot._live_event_current_cycle_id = "cy_refresh_summary"
    bot._live_event_pipeline = LiveEventPipeline(structured_sinks=[sink], monitor_sinks=[])
    bot._authoritative_pending_confirmations = {}
    bot._authoritative_refresh_epoch_changed = set()

    for i in range(60):
        bot._log_staged_refresh_timings(
            {"open_orders"},
            {"open_orders": 100 + i},
            100 + i,
        )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[0]
    assert event.event_type == EventTypes.STATE_REFRESH_TIMING
    assert event.level == "info"
    assert event.component == "state.refresh"
    assert event.tags == ("state", "refresh", "account", "summary")
    assert event.cycle_id == "cy_refresh_summary"
    assert event.status == "succeeded"
    assert event.reason_code == ReasonCodes.STAGED_REFRESH_TIMING
    assert event.data["summary"] is True
    assert event.data["plan"] == ["open_orders"]
    assert event.data["count"] == 60
    assert event.data["wall_ms"] == {"count": 60, "min": 100, "mean": 130, "max": 159}
    assert event.data["surface_sum_ms"] == {
        "count": 60,
        "min": 100,
        "mean": 130,
        "max": 159,
    }
    assert event.data["surface_max_ms"] == {
        "count": 60,
        "min": 100,
        "mean": 130,
        "max": 159,
    }
    assert event.data["residual_ms"] == {"count": 60, "min": 0, "mean": 0, "max": 0}
    assert event.data["surfaces_ms"] == {
        "open_orders": {"count": 60, "min": 100, "mean": 130, "max": 159}
    }
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_staged_refresh_timing_threshold_uses_structured_console_owner(caplog, monkeypatch):
    structured_sink = ListEventSink()
    console_sink = ListEventSink()
    text_sink = ListEventSink()
    bot = Passivbot.__new__(Passivbot)
    bot.live_event_console_enabled = True
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[structured_sink],
        monitor_sinks=[],
        console_sink=console_sink,
        text_sink=text_sink,
    )
    bot._authoritative_pending_confirmations = {}
    bot._authoritative_refresh_epoch_changed = set()
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 1_000_000)

    with caplog.at_level(logging.DEBUG):
        bot._log_staged_refresh_timings(
            {"balance", "fills", "open_orders", "positions"},
            {"balance": 2500, "fills": 4000, "open_orders": 2000, "positions": 3000},
            9999,
        )
        bot._log_staged_refresh_timings(
            {"balance", "fills", "open_orders", "positions"},
            {"balance": 2500, "fills": 4000, "open_orders": 2000, "positions": 3000},
            10_000,
        )

    legacy_records = [
        record
        for record in caplog.records
        if "[state] staged refresh timings" in record.message
    ]
    assert [(record.levelno, record.message) for record in legacy_records] == [
        (
            logging.DEBUG,
            "[state] staged refresh timings | plan=balance,fills,open_orders,positions | "
            "wall=9999ms | surface_sum=11500ms | surface_max=4000ms | parallel=yes | "
            "balance=2500ms fills=4000ms open_orders=2000ms positions=3000ms "
            "residual=5999ms residual_hint=scheduler_or_lock_wait",
        )
    ]
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert len(structured_sink.events) == 1
    event = structured_sink.events[0]
    assert event.data["wall_ms"] == 10_000
    assert event.data["timings_ms"] == {
        "balance": 2500,
        "fills": 4000,
        "open_orders": 2000,
        "positions": 3000,
    }
    assert console_sink.events == [event]
    assert text_sink.events == [event]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_staged_refresh_timing_uses_exact_legacy_fallback_without_console(caplog, monkeypatch):
    structured_sink = ListEventSink()
    bot = Passivbot.__new__(Passivbot)
    bot.live_event_console_enabled = True
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[structured_sink],
        monitor_sinks=[],
        console_sink=None,
    )
    bot._authoritative_pending_confirmations = {}
    bot._authoritative_refresh_epoch_changed = set()
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 1_000_000)

    with caplog.at_level(logging.INFO):
        bot._log_staged_refresh_timings(
            {"balance", "fills", "open_orders", "positions"},
            {"balance": 2500, "fills": 4000, "open_orders": 2000, "positions": 3000},
            10_000,
        )

    assert [record.message for record in caplog.records] == [
        "[state] staged refresh timings | plan=balance,fills,open_orders,positions | "
        "wall=10000ms | surface_sum=11500ms | surface_max=4000ms | parallel=yes | "
        "balance=2500ms fills=4000ms open_orders=2000ms positions=3000ms "
        "residual=6000ms residual_hint=scheduler_or_lock_wait"
    ]
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert structured_sink.events[0].data["wall_ms"] == 10_000
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_staged_refresh_timing_summary_uses_structured_console_owner(caplog, monkeypatch):
    structured_sink = ListEventSink()
    console_sink = ListEventSink()
    bot = Passivbot.__new__(Passivbot)
    bot.live_event_console_enabled = True
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[structured_sink],
        monitor_sinks=[],
        console_sink=console_sink,
    )
    bot._authoritative_pending_confirmations = {}
    bot._authoritative_refresh_epoch_changed = set()
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 1_000_000)

    with caplog.at_level(logging.INFO):
        for i in range(60):
            bot._log_staged_refresh_timings(
                {"open_orders"},
                {"open_orders": 100 + i},
                100 + i,
            )

    assert not [
        record
        for record in caplog.records
        if "[state] staged refresh timing summary" in record.message
    ]
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert len(structured_sink.events) == 1
    event = structured_sink.events[0]
    assert event.data["summary"] is True
    assert console_sink.events == [event]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_staged_refresh_timing_summary_includes_debug_moderate_refreshes(
    caplog, monkeypatch
):
    bot = Passivbot.__new__(Passivbot)
    now = {"value": 2_000_000}
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now["value"])
    bot._authoritative_pending_confirmations = {}
    bot._authoritative_refresh_epoch_changed = set()

    with caplog.at_level(logging.INFO):
        for _ in range(60):
            bot._log_staged_refresh_timings(
                {"balance", "fills", "open_orders", "positions"},
                {
                    "balance": 300,
                    "fills": 400,
                    "open_orders": 500,
                    "positions": 600,
                },
                1_500,
            )

    messages = [record.message for record in caplog.records]
    assert len(messages) == 1
    assert "[state] staged refresh timing summary" in messages[0]
    assert "plan=balance,fills,open_orders,positions" in messages[0]
    assert "count=60" in messages[0]
    assert "wall=1500/1500/1500ms" in messages[0]


@pytest.mark.asyncio
async def test_refresh_authoritative_state_staged_uses_open_orders_only_confirmation_plan():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.exchange = "fake"
    bot.stop_signal_received = False
    bot.balance_override = None
    bot._balance_override_logged = False
    bot.previous_hysteresis_balance = None
    bot.balance_hysteresis_snap_pct = 0.02
    bot.balance_raw = 0.0
    bot.balance = 0.0
    bot._exchange_reported_balance_raw = 0.0
    bot.fetched_positions = []
    bot.positions = {}
    bot.open_orders = {}
    bot.fetched_open_orders = []
    bot.active_symbols = []
    bot.state_change_detected_by_symbol = set()
    bot.execution_scheduled = False
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot._authoritative_refresh_epoch = 0
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()
    bot._authoritative_pending_confirmations = {"open_orders": 1}
    bot.recent_order_cancellations = []
    bot.recent_order_executions = [
        {
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "position_side": "long",
            "qty": 0.01,
            "price": 99000.0,
            "execution_timestamp": 10**15,
        }
    ]
    bot.fetch_balance = AsyncMock(
        side_effect=AssertionError("balance fetch should not run")
    )
    bot.fetch_positions = AsyncMock(
        side_effect=AssertionError("positions fetch should not run")
    )
    bot.update_pnls = AsyncMock(
        side_effect=AssertionError("fills refresh should not run")
    )
    bot.fetch_open_orders = AsyncMock(
        return_value=[
            {
                "id": "1",
                "symbol": "BTC/USDT:USDT",
                "side": "buy",
                "position_side": "long",
                "qty": 0.01,
                "amount": 0.01,
                "price": 99000.0,
                "timestamp": 1,
                "reduce_only": False,
            }
        ]
    )
    bot.handle_balance_update = AsyncMock()
    bot.order_was_recently_cancelled = lambda order: 0.0
    bot.log_order_action = lambda *args, **kwargs: None
    bot._reconcile_balance_after_open_orders_refresh = lambda: False

    result = await bot.refresh_authoritative_state()
    blocked, details = bot._authoritative_execution_barrier_state()

    assert result is True
    bot.fetch_open_orders.assert_awaited_once()
    assert bot.fetch_balance.await_count == 0
    assert bot.fetch_positions.await_count == 0
    assert bot.update_pnls.await_count == 0
    assert blocked is False
    assert details["missing"] == []


@pytest.mark.asyncio
async def test_staged_account_refresh_request_counts_steady_state_defers_recent_fills(
    monkeypatch,
):
    bot, counts = _counted_staged_account_refresh_bot()
    bot.freshness_ledger.stamp("fills", (), now_ms=120_010, epoch=0)
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 120_500)

    result = await bot.refresh_authoritative_state()

    assert result is True
    assert counts == {
        "fetch_balance": 1,
        "fetch_positions": 1,
        "fetch_open_orders": 1,
        "update_pnls": 0,
    }
    assert bot._authoritative_refresh_plan_surfaces == {
        "balance",
        "positions",
        "open_orders",
    }
    assert bot._authoritative_pending_confirmations == {}


@pytest.mark.asyncio
async def test_staged_account_refresh_prefetches_due_routine_fills_without_blocking(
    monkeypatch,
):
    bot, counts = _counted_staged_account_refresh_bot()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_update_pnls(**_kwargs):
        counts["update_pnls"] += 1
        started.set()
        await release.wait()
        bot._record_authoritative_surface("fills", ())
        return True

    bot.update_pnls = slow_update_pnls
    bot.freshness_ledger.stamp("fills", (), now_ms=1_000, epoch=1)
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 61_500)

    result = await bot.refresh_authoritative_state()
    await asyncio.wait_for(started.wait(), timeout=1.0)

    assert result is True
    assert bot._authoritative_refresh_plan_surfaces == {
        "balance",
        "positions",
        "open_orders",
    }
    assert counts["fetch_balance"] == 1
    assert counts["fetch_positions"] == 1
    assert counts["fetch_open_orders"] == 1
    assert counts["update_pnls"] == 1
    fill_task = bot.maintainers["routine_fill_refresh"]
    assert fill_task.done() is False

    release.set()
    await asyncio.wait_for(fill_task, timeout=1.0)


@pytest.mark.asyncio
async def test_staged_account_refresh_blocks_on_due_fills_when_prefetch_is_too_stale(
    monkeypatch,
):
    bot, counts = _counted_staged_account_refresh_bot()
    bot.freshness_ledger.stamp("fills", (), now_ms=1_000, epoch=1)
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 4 * 60_000 + 1_000)

    result = await bot.refresh_authoritative_state()

    assert result is True
    assert bot._authoritative_refresh_plan_surfaces == ACCOUNT_SURFACES
    assert counts == {
        "fetch_balance": 1,
        "fetch_positions": 1,
        "fetch_open_orders": 1,
        "update_pnls": 1,
    }
    assert "routine_fill_refresh" not in getattr(bot, "maintainers", {})


@pytest.mark.asyncio
async def test_staged_account_refresh_request_counts_open_orders_only_confirmation():
    bot, counts = _counted_staged_account_refresh_bot(
        pending_confirmations={"open_orders": 1}
    )

    result = await bot.refresh_authoritative_state()
    blocked, details = bot._authoritative_execution_barrier_state()

    assert result is True
    assert counts == {
        "fetch_balance": 0,
        "fetch_positions": 0,
        "fetch_open_orders": 1,
        "update_pnls": 0,
    }
    assert bot._authoritative_refresh_plan_surfaces == {"open_orders"}
    assert blocked is False
    assert details["missing"] == []


@pytest.mark.asyncio
async def test_staged_account_refresh_request_counts_missing_self_order_escalates_next_cycle():
    stale_order = {
        "id": "stale-entry",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 0.01,
        "amount": 0.01,
        "price": 90_000.0,
        "timestamp": 1,
        "reduce_only": False,
    }
    bot, counts = _counted_staged_account_refresh_bot(
        open_orders=[],
        pending_confirmations={"open_orders": 1},
    )
    bot.open_orders = {"BTC/USDT:USDT": [dict(stale_order)]}
    bot._authoritative_surface_signatures["open_orders"] = (
        Passivbot._open_orders_signature(bot, [stale_order])
    )

    result = await bot.refresh_authoritative_state()

    assert result is True
    assert counts == {
        "fetch_balance": 0,
        "fetch_positions": 0,
        "fetch_open_orders": 1,
        "update_pnls": 0,
    }
    assert set(bot._authoritative_pending_confirmations) == ACCOUNT_SURFACES

    result = await bot.refresh_authoritative_state()

    assert result is True
    assert counts == {
        "fetch_balance": 1,
        "fetch_positions": 1,
        "fetch_open_orders": 2,
        "update_pnls": 1,
    }
    assert bot._authoritative_refresh_plan_surfaces == ACCOUNT_SURFACES


@pytest.mark.asyncio
async def test_run_execution_loop_propagates_fatal_bot_exception():
    bot = Passivbot.__new__(Passivbot)

    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = False
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    async def fake_refresh_authoritative_state():
        raise FatalBotException("fatal")

    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.execute_to_exchange = AsyncMock()

    with pytest.raises(FatalBotException, match="fatal"):
        await bot.run_execution_loop()


def _balance_composition_fixture(amount: str) -> dict:
    return normalize_okx_balance_composition(
        {"info": {"data": [{"details": [{"ccy": "USDT", "cashBal": amount}]}]}}
    )


@pytest.mark.asyncio
async def test_composition_only_balance_change_is_durable_but_does_not_schedule_execution():
    bot = Passivbot.__new__(Passivbot)
    bot.balance_raw = 100.0
    bot.balance = 100.0
    bot._previous_balance_raw = 100.0
    bot._previous_balance_snapped = 100.0
    bot._balance_composition = _balance_composition_fixture("2")
    bot._previous_balance_composition_signature = balance_composition_signature(
        _balance_composition_fixture("1")
    )
    bot.execution_scheduled = False
    bot.live_event_console_enabled = False
    bot._live_event_pipeline = None
    bot._monitor_record_event = MagicMock()
    bot._emit_balance_changed_event = MagicMock()
    bot._log_noncritical_market_snapshot_error = lambda *_args: False

    async def calc_upnl_sum():
        return 0.0

    bot.calc_upnl_sum = calc_upnl_sum

    await bot.handle_balance_update(source="REST")

    bot._emit_balance_changed_event.assert_called_once()
    assert bot._emit_balance_changed_event.call_args.kwargs["balance_composition"][
        "asset_balances"
    ][0]["amount"] == pytest.approx(2.0)
    assert bot.execution_scheduled is False

    await bot.handle_balance_update(source="REST")

    bot._emit_balance_changed_event.assert_called_once()


def test_legacy_balance_apply_preserves_existing_composition_when_none_is_supplied():
    bot = Passivbot.__new__(Passivbot)
    composition = _balance_composition_fixture("1")
    bot.quote = "USDT"
    bot.balance = 100.0
    bot.balance_raw = 100.0
    bot._exchange_reported_balance_raw = 100.0
    bot.balance_override = None
    bot._balance_override_logged = False
    bot.previous_hysteresis_balance = 100.0
    bot.balance_hysteresis_snap_pct = 0.02
    bot._balance_composition = composition

    assert bot._apply_balance_snapshot(100.0) is True
    assert bot.balance == pytest.approx(100.0)
    assert bot.balance_raw == pytest.approx(100.0)
    assert bot._balance_composition is composition
