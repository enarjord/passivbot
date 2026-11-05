"""Passive input and permission projections; no exchange/network activity."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from config.strategy_spec import get_strategy_defaults
from live import ordinary_diagnostics as ordinary
from live.console_health import readiness_payload
from live.event_bus import EventTypes, ReasonCodes, format_periodic_health_summary
from live.freshness import FreshnessLedger
from live.hsl_live import Wave
from live.hsl_runtime import Scope, Unavailable
import passivbot_monitor as monitor


def strategy_bot(kind="trailing_martingale"):
    from passivbot import Passivbot
    config = {"bot": {"portfolio": {"strategy_kind": kind}}}
    for side in ("long", "short"):
        config["bot"][side] = {"strategy": {kind: get_strategy_defaults(kind)[side]}}
    bot = SimpleNamespace(config=config, coin_overrides={},
        _monitor_runtime_indicator_captured_at_ms=1000)
    bot._strategy_params_to_rust_dict = Passivbot._strategy_params_to_rust_dict.__get__(bot)
    return bot


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_canonical_spans_missing_inputs_zero_and_recovery(kind):
    bot = strategy_bot(kind)
    params = bot._strategy_params_to_rust_dict("long", "BTC")
    h1_key = "log_range_ema_span_1h_candles" if kind == "trailing_martingale" else "offset_log_range_ema_span_1h_candles"
    m1_key = "log_range_ema_span_minutes" if kind == "trailing_martingale" else "offset_log_range_ema_span_minutes"
    # Fractional horizons and symbol overrides must survive projection unchanged.
    bot.coin_overrides = {"BTC": {"bot": {"long": {"strategy": {kind: {h1_key: 24.5}}}}}}
    missing = monitor._monitor_logrange_inputs(bot, "long", "BTC", now_ms=1250)
    assert missing["h1_log_range"] == dict(value=None, status="unavailable",
        reason="prepared_value_unavailable", span=24.5, span_unit="hour_candles",
        captured_at_ms=1000, age_ms=250)
    bot._monitor_runtime_h1_log_range_emas = {"BTC": {24.5: 0.0}}
    bot._monitor_runtime_m1_log_range_emas = {"BTC": {params[m1_key]: 0.02}}
    recovered = monitor._monitor_logrange_inputs(bot, "long", "BTC", now_ms=1500)
    assert recovered["h1_log_range"]["value"] == 0.0
    assert recovered["h1_log_range"]["status"] == "available"
    assert recovered["m1_log_range"]["value"] == 0.02
    bot._monitor_runtime_h1_log_range_emas["BTC"][24.5] = float("nan")
    assert monitor._monitor_logrange_inputs(bot, "long", "BTC")["h1_log_range"]["value"] is None
    json.dumps(missing, allow_nan=False)


def test_missing_strategy_is_explicit_and_does_not_invent_span():
    result = monitor._monitor_logrange_inputs(SimpleNamespace(), "long", "BTC", now_ms=1000)
    assert all(row["value"] is None and row["span"] is None
               and row["reason"] == "strategy_span_unavailable" for row in result.values())


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_price_bands_use_canonical_entry_horizons(kind):
    bot = strategy_bot(kind)
    bot.bp = lambda *args: (_ for _ in ()).throw(KeyError("not a legacy strategy input"))
    params = bot._strategy_params_to_rust_dict("long", "BTC")
    spans = params["entry"] if kind == "trailing_martingale" else params
    s0, s1 = (spans[f"ema_span_{i}_minutes"] for i in (0, 1))
    cache = {"BTC": {s0: 101., s1: 103., (s0 * s1) ** .5: 102.}}
    hints = monitor._build_monitor_runtime_market_hints(bot, ["BTC"], {"BTC": 104.}, cache)
    assert hints["BTC"]["ema_bands"]["long"]["lower"] == 101.
    assert hints["BTC"]["ema_bands"]["long"]["upper"] == 103.
    if kind == "ema_anchor":
        assert "entry_trigger_price" not in hints["BTC"]["ema_bands"]["long"]


def plan_bot(count=1):
    ledger = FreshnessLedger(now_ms=1000)
    for name in ordinary.SURFACES:
        ledger.stamp(name, now_ms=1000)
    scope = Scope("portfolio", None, None)
    bot = SimpleNamespace(positions={"BTC": {"long": {"size": 1}}},
        freshness_ledger=ledger, _account_invalidation_generation=2,
        _live_market_snapshot_max_age_ms=lambda: 1000,
        _hsl_planning_wave=Wave(captured_ms=1000, position_observed_ms=900,
            mark_observed_ms=(950,), decisions=(), unavailable=(Unavailable(scope, "current_balance_unavailable"),),
            positions="unused", open_orders=(), balance=100, generation=2))
    rows = []
    for idx in range(count):
        rows.append(dict(symbol_idx=idx, long=dict(input_mode="manual", effective_mode="manual", active=True, allow_initial=False),
            short=dict(input_mode=None, effective_mode="normal", active=False, allow_initial=False)))
    return bot, dict(symbol_states=rows), {idx: "BTC" if idx == 0 else f"S{idx}" for idx in range(count)}


def test_last_ordinary_manual_is_separate_from_fast_green_and_market_tradable():
    bot, diagnostics, mapping = plan_bot()
    bot._hsl_diagnostic_observation = {"tier": "green"}
    bot.active_symbols = ["BTC"]
    bot.has_position = lambda **_: True
    # A point market/input readiness flag cannot grant ordinary permission.
    market = monitor._build_monitor_market_section(bot)
    assert market["BTC"]["tradable"] is True
    assert market["BTC"]["tradable_basis"] == "market_and_prepared_inputs_only"
    ordinary.observe_plan(bot, diagnostics, mapping, now_ms=1100)
    section = ordinary.snapshot(bot, now_ms=1200)
    row = section["scopes"][0]
    assert section["authority"] == "observation_only"
    assert section["stage"] == "computed_native_plan"
    assert section["observation_status"] == "current"
    assert row["effective_mode"] == "manual"
    assert row["hsl_permission"] == "unavailable"
    assert row["hsl_unavailable_reason"] == "current_balance_unavailable"
    assert section["restricted_held_count"] == 1
    health = readiness_payload(bot, 1200)
    console = format_periodic_health_summary(health)
    assert "ordinary_computed=current/0.2s" in console
    assert "ordinary_held_restricted=1" in console
    assert "ordinary_scope=BTC:long:manual" in console
    assert "ordinary_hsl_cause=current_balance_unavailable" in console
    assert bot._hsl_diagnostic_observation == {"tier": "green"}
    assert diagnostics["symbol_states"][0]["long"]["effective_mode"] == "manual"


def test_ordinary_capture_age_generation_pending_and_recovery():
    bot, diagnostics, mapping = plan_bot()
    ordinary.observe_plan(bot, diagnostics, mapping, now_ms=1100)
    assert ordinary.snapshot(bot, now_ms=1900)["observation_status"] == "current"
    assert ordinary.snapshot(bot, now_ms=1901)["stale_reasons"] == ["input_expired"]
    bot._account_invalidation_generation += 1
    assert "account_generation_changed" in ordinary.snapshot(bot, now_ms=1200)["stale_reasons"]
    bot._account_invalidation_generation -= 1
    epoch = bot.freshness_ledger.surfaces["positions"].epoch
    bot._authoritative_pending_confirmations = {"positions": epoch + 1, "password": 999}
    section = ordinary.snapshot(bot, now_ms=1200)
    assert section["pending_confirmations"] == [dict(surface="positions", requested_epoch=epoch + 1, current_epoch=epoch)]
    assert "account_confirmation_pending" in section["stale_reasons"]
    bot._authoritative_pending_confirmations.clear()
    assert ordinary.snapshot(bot, now_ms=1200)["observation_status"] == "current"
    assert ordinary.snapshot(bot, now_ms=999)["observation_status"] == "stale"


def test_barrier_is_last_outcome_not_active_gate_and_sink_failure_isolated(monkeypatch):
    import live.event_emitters as emitters
    bot, diagnostics, mapping = plan_bot()
    bot._current_live_event_cycle_id = lambda: None
    def fail(*args, **kwargs):
        raise OSError("private payload must not escape")
    bot._emit_live_event = fail
    monkeypatch.setattr(emitters, "utc_ms", lambda: 1100)
    assert emitters.emit_execution_create_filter_event(bot,
        event_type=EventTypes.EXECUTION_CANCEL_FIRST_BARRIER, status="deferred",
        reason_code=ReasonCodes.ACCOUNT_CANCEL_FIRST_BARRIER, order_count=2,
        symbols=["BTC"], wave={"id": 7},
        data={"required_surfaces": ["positions", "password", "open_orders"], "secret": "private"}) is False
    result = ordinary.snapshot(bot, now_ms=5000)
    barrier = result["last_cancel_first_barrier"]
    assert barrier == dict(observed_at_ms=1100, reason="account_cancel_first_barrier", order_count=2,
        wave_id=7, required_surfaces=["positions", "open_orders"], age_ms=3900, status="last_observed_outcome")
    assert result["pending_confirmations"] == []
    assert "private" not in json.dumps(result)


def test_bounds_redaction_and_projection_failure_isolation(monkeypatch):
    bot, diagnostics, mapping = plan_bot(200)
    ordinary.observe_plan(bot, diagnostics, mapping, now_ms=1100)
    section = ordinary.snapshot(bot, now_ms=1200)
    assert section["scope_count"] == 400
    assert len(section["scopes"]) == ordinary.ROW_LIMIT
    assert section["omitted_scopes"] == 400 - ordinary.ROW_LIMIT
    assert section["mode_counts"]["manual"] == 200
    assert section["scopes"][0]["symbol"] == "BTC"  # Held scopes precede flat sampling.
    bad = deepcopy(diagnostics)
    bad["symbol_states"][0]["long"]["effective_mode"] = "https://private.invalid/?secret=credential"
    monkeypatch.setattr(ordinary.logging, "debug", lambda *a, **k: (_ for _ in ()).throw(OSError("sink")))
    ordinary.observe_plan(bot, bad, mapping, now_ms=1100)
    assert ordinary.snapshot(bot, now_ms=1200)["observation_status"] == "diagnostic_unavailable"
    assert "credential" not in json.dumps(ordinary.snapshot(bot, now_ms=1200))
    assert ordinary._reason("https://private.invalid/?secret=credential") == "unavailable"


def test_missing_input_never_runs_native_trailing_diagnostic(monkeypatch):
    bot = strategy_bot()
    def fail(*args, **kwargs):
        pytest.fail("missing inputs must not be replaced with native diagnostic zeros")
    monkeypatch.setattr(monitor, "build_trailing_entry_diagnostic", fail)
    monkeypatch.setattr(monitor, "build_trailing_martingale_close_diagnostic", fail)
    common = dict(symbol="BTC", pside="long", balance_raw=100., current_price=1.,
        position_size=1., position_price=1., trailing_bundle={})
    assert monitor._build_monitor_trailing_entry_payload(bot, **common,
        market_entry={"ema_bands": {"long": {"lower": 1., "upper": 1.}}}) is None
    assert monitor._build_monitor_trailing_close_payload(bot, **common, balance_strategy=100.) is None


def test_native_diagnostic_failure_keeps_other_snapshot_observations(monkeypatch):
    bot = strategy_bot()
    bot.positions = {"BTC": {"long": {"size": 1., "price": 1.}}}
    def fail(*args, **kwargs):
        raise RuntimeError("private malformed output")
    monkeypatch.setattr(monitor, "_build_monitor_trailing_entry_payload", fail)
    monkeypatch.setattr(monitor, "_build_monitor_trailing_close_payload", fail)
    section = monitor._build_monitor_trailing_section(bot,
        balance_raw=100., market={"BTC": {"last_price": 1.}})
    assert section["BTC"]["long"]["entry_diagnostic_status"] == "unavailable"
    assert section["BTC"]["long"]["close_diagnostic_status"] == "unavailable"
    assert "strategy_inputs" in section["BTC"]["long"]
    assert "private" not in json.dumps(section)


def test_nullable_diagnostics_remain_renderable_in_tui():
    from monitor_tui import _format_tick_line, _render_trailing_panel
    bot = strategy_bot()
    inputs = monitor._monitor_logrange_inputs(bot, "long", "BTC", now_ms=2000)
    market = {"entry_log_range_ema_1h": {"long": None}, "strategy_inputs": {"long": inputs},
        "ema_bands": {"long": {"lower": None, "upper": None}}}
    assert "lo=-" in _format_tick_line("BTC", {"payload": {"last": 1.}}, market)
    assert isinstance(_render_trailing_panel({"trailing": {"BTC": {"long": {"strategy_inputs": inputs}}}}), list)
