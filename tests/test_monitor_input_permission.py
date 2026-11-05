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
from live.planning_snapshot import PlanningSnapshot
from live.market_snapshot import MarketSnapshot
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
    if kind == "ema_anchor":
        for timeframe in ("1m", "1h"):
            bot.config["bot"]["long"]["strategy"][kind][f"offset_log_range_{timeframe}_coefficient"] = .1
    params = bot._strategy_params_to_rust_dict("long", "BTC")
    h1_key = "log_range_ema_span_1h_candles" if kind == "trailing_martingale" else "offset_log_range_ema_span_1h_candles"
    m1_key = "log_range_ema_span_minutes" if kind == "trailing_martingale" else "offset_log_range_ema_span_minutes"
    # Fractional horizons and symbol overrides must survive projection unchanged.
    bot.coin_overrides = {"BTC": {"bot": {"long": {"strategy": {kind: {h1_key: 24.5}}}}}}
    missing = monitor._monitor_logrange_inputs(bot, "long", "BTC", now_ms=1250)
    assert missing["h1_log_range"] == dict(value=None, status="unavailable", required=True,
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
    ledger.begin_epoch(now_ms=1000)
    for name in (*ordinary.SURFACES, "market_snapshot", "completed_candles"):
        ledger.stamp(name, now_ms=1000)
    scope = Scope("portfolio", None, None)
    bot = SimpleNamespace(positions={"BTC": {"long": {"size": 1}}},
        freshness_ledger=ledger, _account_invalidation_generation=2,
        _live_market_snapshot_max_age_ms=lambda: 1000,
        _hsl_planning_wave=Wave(captured_ms=1000, position_observed_ms=900,
            mark_observed_ms=(950,), decisions=(), unavailable=(Unavailable(scope, "current_balance_unavailable"),),
            positions="unused", open_orders=(), balance=100, generation=2, balance_observed_ms=1000))
    rows = []
    for idx in range(count):
        rows.append(dict(symbol_idx=idx, long=dict(input_mode="manual", effective_mode="manual", active=True, allow_initial=False),
            short=dict(input_mode=None, effective_mode="normal", active=False, allow_initial=False)))
    mapping = {idx: "BTC" if idx == 0 else f"S{idx}" for idx in range(count)}
    bot._current_planning_snapshot = PlanningSnapshot.capture(ts_ms=1000, exchange="fake", user="example",
        ledger=ledger, required_surfaces=(*ordinary.SURFACES, "market_snapshot", "completed_candles"),
        min_epochs={name:ledger.surfaces[name].epoch for name in ledger.surfaces},
        symbols=mapping.values(), market_snapshots={s:MarketSnapshot(s, 1., 1., 1., 1000, "fixture") for s in mapping.values()},
        market_snapshot_max_age_ms=1000, account_invalidation_generation=2)
    return bot, dict(symbol_states=rows), mapping


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


def test_dex_qualified_symbols_keep_ordinary_diagnostic_coverage():
    from dataclasses import replace
    bot, diagnostics, mapping = plan_bot()
    symbol = "xyz:TSLA/USDC:USDC"
    bot.positions = {symbol: {"long": {"size": 1}}}
    mapping[0] = symbol
    snapshot = bot._current_planning_snapshot
    bot._current_planning_snapshot = replace(snapshot, symbols=(symbol,),
        market_snapshots=(replace(snapshot.market_snapshots[0], symbol=symbol),))
    ordinary.observe_plan(bot, diagnostics, mapping, now_ms=1100)
    section = ordinary.snapshot(bot, now_ms=1200)
    assert section["observation_status"] == "current"
    assert section["scopes"][0]["symbol"] == symbol


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


def native_diagnostic_bot(unused="1m"):
    from pathlib import Path
    from config.load import load_prepared_config
    from passivbot import Passivbot
    bot = strategy_bot()
    bot.config = load_prepared_config(str(Path(__file__).parents[1] / "configs/examples/suite_example.json"),
        live_only=True, target="canonical", verbose=False, log_info=False)
    # Keep the example's unused metrics, while selecting its optional entry diagnostic.
    bot.config["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["retracement_base_fraction"] = .01
    if unused == "1h":
        for branch in ("entry", "close"):
            for consumer in ("threshold", "retracement"):
                bot.config["bot"]["long"]["strategy"]["trailing_martingale"][branch][f"{consumer}_log_range_1h_coefficient"] = 0.
    symbol = "BTC/USDT:USDT"
    bot._strategy_params_to_rust_dict = Passivbot._strategy_params_to_rust_dict.__get__(bot)
    bot._monitor_runtime_m1_log_range_emas = {symbol: {999.: .99}}  # Different forager span.
    span = bot._strategy_params_to_rust_dict("long", symbol)["log_range_ema_span_1h_candles"]
    bot._monitor_runtime_h1_log_range_emas = {symbol: {span: .01}} if unused != "1h" else {}
    bot.bp = lambda side, key, symbol: .5 if key == "wallet_exposure_limit_ratio" else .1
    bot.bot_value = lambda *args: 1.5
    exchange = dict(qty_step=.001, price_step=.1, min_qty=.001, min_cost=1.,
                    c_mult=1., maker_fee_fraction=.0002, taker_fee_fraction=.0005)
    bot._orchestrator_exchange_params = lambda symbol: exchange
    bot._bot_params_to_rust_dict = lambda *args: dict(wallet_exposure_limit_ratio=.5,
        total_wallet_exposure_limit_ratio=1.5, n_positions=3, risk_we_excess_allowance_fraction=.1,
        risk_wel_enforcer_enabled=False, risk_wel_enforcer_threshold_ratio=1.)
    for attr, value in (("qty_steps", .001), ("price_steps", .1), ("min_qtys", .001),
                        ("min_costs", 1.), ("c_mults", 1.)):
        setattr(bot, attr, {symbol: value})
    return bot, symbol


def diagnostic_payloads(bot, symbol):
    common = dict(symbol=symbol, pside="long", balance_raw=1000., current_price=102.,
        position_size=1., position_price=100., trailing_bundle=dict(
            min_since_open=99., max_since_min=102., max_since_open=103., min_since_max=101.))
    return (
        monitor._build_monitor_trailing_entry_payload(bot, **common,
            market_entry={"ema_bands": {"long": {"lower": 100., "upper": 102.}}}),
        monitor._build_monitor_trailing_close_payload(bot, **common, balance_strategy=1000.),
    )


def test_unused_public_example_metric_keeps_real_native_diagnostics():
    bot, symbol = native_diagnostic_bot()
    inputs = monitor._monitor_logrange_inputs(bot, "long", symbol, now_ms=2000)
    assert inputs["m1_log_range"]["status"] == "not_required"
    assert inputs["m1_log_range"]["value"] is None
    assert inputs["m1_log_range"]["span"] == 60.
    for payload in diagnostic_payloads(bot, symbol):
        assert payload is not None
        assert payload["strategy_inputs"]["m1_log_range"]["status"] == "not_required"
        assert payload["strategy_inputs"]["m1_log_range"]["value"] is None


@pytest.mark.parametrize("branch", ["entry", "close"])
@pytest.mark.parametrize("consumer", ["threshold", "retracement"])
@pytest.mark.parametrize("timeframe", ["1m", "1h"])
@pytest.mark.parametrize("coefficient", [-.25, .25])
def test_each_branch_requires_its_nonzero_consumers_independently(branch, consumer, timeframe, coefficient):
    bot, symbol = native_diagnostic_bot(unused=timeframe)
    bot.config["bot"]["long"]["strategy"]["trailing_martingale"][branch][f"{consumer}_log_range_{timeframe}_coefficient"] = coefficient
    payloads = diagnostic_payloads(bot, symbol)
    assert payloads[0 if branch == "entry" else 1] is None
    assert payloads[1 if branch == "entry" else 0] is not None


def test_real_observed_zero_remains_available_when_dimension_unused():
    bot, symbol = native_diagnostic_bot()
    bot._monitor_runtime_m1_log_range_emas[symbol][60.] = 0.
    row = monitor._monitor_logrange_inputs(bot, "long", symbol)["m1_log_range"]
    assert row["status"] == "available" and row["value"] == 0. and row["required"] is False


def test_balance_receipt_expiry_is_retained_not_renewed_from_live_ledger():
    from dataclasses import replace
    bot, diagnostics, mapping = plan_bot()
    bot._hsl_planning_wave = replace(bot._hsl_planning_wave, balance_observed_ms=100)
    ordinary.observe_plan(bot, diagnostics, mapping, now_ms=1000)
    # A later confirming receipt cannot rewrite the already consumed balance age.
    bot.freshness_ledger.stamp("balance", now_ms=1200)
    assert ordinary.snapshot(bot, now_ms=1100)["observation_status"] == "current"
    old = ordinary.snapshot(bot, now_ms=1101)
    assert old["input_expires_at_ms"] == 1100
    assert old["balance_observed_at_ms"] == 100
    assert old["stale_reasons"] == ["input_expired"]
    assert "ordinary_computed=stale" in format_periodic_health_summary(readiness_payload(bot, 1200))
    bot._hsl_planning_wave = replace(bot._hsl_planning_wave, balance_observed_ms=1200,
        captured_ms=1200, position_observed_ms=1200, mark_observed_ms=(1200,))
    ordinary.observe_plan(bot, diagnostics, mapping, now_ms=1200)
    assert ordinary.snapshot(bot, now_ms=1300)["observation_status"] == "current"


def test_wave_without_balance_receipt_never_claims_current_input_age():
    from dataclasses import replace
    bot, diagnostics, mapping = plan_bot()
    bot._hsl_planning_wave = replace(bot._hsl_planning_wave, balance_observed_ms=None)
    ordinary.observe_plan(bot, diagnostics, mapping, now_ms=1100)
    assert ordinary.snapshot(bot, now_ms=1200)["stale_reasons"] == ["input_age_unknown"]


def test_real_wave_retains_balance_receipt_read_before_native_evaluation(monkeypatch):
    from live import hsl_live
    from test_hsl_runtime import NOW, bot as make_bot
    import utils
    bot = make_bot("coin")
    monkeypatch.setattr(utils, "utc_ms", lambda: NOW)
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot.get_exchange_time = lambda: NOW
    bot.approved_coins_minus_ignored_coins = {"long": set(bot.positions), "short": set()}
    ledger = bot._ensure_freshness_ledger()
    original = ledger.surfaces["balance"].updated_ms
    evaluate = hsl_live.runtime.evaluate
    def slow_evaluate(requests):
        result = evaluate(requests)
        ledger.stamp("balance", now_ms=NOW + 10)
        return result
    monkeypatch.setattr(hsl_live.runtime, "evaluate", slow_evaluate)
    wave = hsl_live.owner(bot).capture()
    assert wave.balance_observed_ms == original
    assert ledger.surfaces["balance"].updated_ms == NOW + 10


@pytest.mark.parametrize("hsl_disabled", [False, True])
def test_ordinary_quote_cohort_expiry_includes_flat_unscoped_candidates(hsl_disabled):
    from dataclasses import replace
    bot, diagnostics, mapping = plan_bot(2)
    wave = bot._hsl_planning_wave
    bot._hsl_planning_wave = replace(wave, mark_observed_ms=(),
        unavailable=() if hsl_disabled else (Unavailable(Scope("coin", "long", "BTC"), "current_balance_unavailable"),),
        balance_observed_ms=None if hsl_disabled else 1000,
        position_observed_ms=0 if hsl_disabled else 1000)
    snapshot = bot._current_planning_snapshot
    bot._current_planning_snapshot = replace(snapshot, market_snapshots=tuple(
        replace(row, fetched_ms=100 if row.symbol == "S1" else 1000) for row in snapshot.market_snapshots))
    ordinary.observe_plan(bot, diagnostics, mapping, now_ms=1000)
    current = ordinary.snapshot(bot, now_ms=1100)
    assert current["input_expires_at_ms"] == 1100
    assert current["input_receipts"]["problems"] == []
    assert current["input_receipts"]["required_market_symbol_count"] == 2
    assert current["observation_status"] == "current"
    assert ordinary.snapshot(bot, now_ms=1101)["stale_reasons"] == ["input_expired"]
    assert bot._current_planning_snapshot.invalid_details(now_ms=1101)[0]["reason"] == "snapshot_too_old"


@pytest.mark.parametrize("account_ttl,quote_ttl,expected", [(1000, 200, 1100), (200, 1000, 1200)])
def test_account_and_quote_receipts_keep_independent_ttls(account_ttl, quote_ttl, expected):
    from dataclasses import replace
    bot, diagnostics, mapping = plan_bot()
    bot._hsl_planning_wave = replace(bot._hsl_planning_wave, unavailable=(), mark_observed_ms=())
    bot._live_market_snapshot_max_age_ms = lambda: account_ttl
    snapshot = bot._current_planning_snapshot
    bot._current_planning_snapshot = replace(snapshot, market_snapshot_max_age_ms=quote_ttl,
        market_snapshots=tuple(replace(row, fetched_ms=900) for row in snapshot.market_snapshots))
    ordinary.observe_plan(bot, diagnostics, mapping, now_ms=1000)
    assert ordinary.snapshot(bot, now_ms=expected)["observation_status"] == "current"
    assert ordinary.snapshot(bot, now_ms=expected + 1)["input_expires_at_ms"] == expected
    assert ordinary.snapshot(bot, now_ms=expected + 1)["stale_reasons"] == ["input_expired"]


@pytest.mark.parametrize("fault", ["missing_snapshot", "missing_quote", "future_quote", "cohort", "future_capture"])
def test_unproven_planning_receipt_never_becomes_current(fault):
    from dataclasses import replace
    bot, diagnostics, mapping = plan_bot(2)
    snapshot = bot._current_planning_snapshot
    if fault == "missing_snapshot":
        bot._current_planning_snapshot = None
    elif fault == "missing_quote":
        bot._current_planning_snapshot = replace(snapshot, market_snapshots=snapshot.market_snapshots[:1])
    elif fault == "future_quote":
        bot._current_planning_snapshot = replace(snapshot,
            market_snapshots=tuple(replace(row, fetched_ms=1001) for row in snapshot.market_snapshots))
    elif fault == "cohort":
        bot._current_planning_snapshot = replace(snapshot, symbols=("BTC",))
    else:
        bot._current_planning_snapshot = replace(snapshot, ts_ms=1001)
    ordinary.observe_plan(bot, diagnostics, mapping, now_ms=1000)
    observed = ordinary.snapshot(bot, now_ms=1500)
    assert observed["input_expires_at_ms"] is None
    assert observed["input_receipts"]["problems"]
    assert observed["observation_status"] == "stale"


@pytest.mark.parametrize("surface", ["balance", "positions", "open_orders"])
@pytest.mark.parametrize("fault", ["missing", "zero", "future", "old"])
def test_every_consumed_account_receipt_constrains_observation(surface, fault):
    from dataclasses import replace
    bot, diagnostics, mapping = plan_bot()
    snapshot = bot._current_planning_snapshot
    stamps = []
    for row in snapshot.surfaces:
        if row.name == surface:
            if fault == "missing":
                continue
            row = replace(row, updated_ms={"zero": 0, "future": 1001, "old": 100}[fault])
        stamps.append(row)
    bot._current_planning_snapshot = replace(snapshot, surfaces=tuple(stamps))
    ordinary.observe_plan(bot, diagnostics, mapping, now_ms=1000)
    result = ordinary.snapshot(bot, now_ms=1200)
    assert result["observation_status"] == "stale"
    if fault == "old":
        assert result["input_expires_at_ms"] == 1100
    else:
        assert result["input_expires_at_ms"] is None


def test_old_fill_and_candle_receipts_keep_coverage_epochs_not_quote_ttl():
    from dataclasses import replace
    bot, diagnostics, mapping = plan_bot()
    snapshot = bot._current_planning_snapshot
    bot._current_planning_snapshot = replace(snapshot, surfaces=tuple(
        replace(row, updated_ms=1) if row.name in {"fills", "completed_candles"} else row
        for row in snapshot.surfaces))
    ordinary.observe_plan(bot, diagnostics, mapping, now_ms=1100)
    result = ordinary.snapshot(bot, now_ms=1200)
    assert result["input_receipts"]["problems"] == []
    assert result["observation_status"] == "current"
    assert result["input_expires_at_ms"] == 1900  # Earlier required HSL position receipt.


def test_live_cache_and_replacement_snapshot_cannot_renew_old_computation():
    from dataclasses import replace
    bot, diagnostics, mapping = plan_bot()
    snapshot = bot._current_planning_snapshot
    bot._current_planning_snapshot = replace(snapshot,
        market_snapshots=tuple(replace(row, fetched_ms=100) for row in snapshot.market_snapshots))
    ordinary.observe_plan(bot, diagnostics, mapping, now_ms=1000)
    bot.market_snapshot_provider = SimpleNamespace(_cache={"BTC": MarketSnapshot("BTC", 1., 1., 1., 1200, "new")})
    bot._current_planning_snapshot = replace(snapshot, ts_ms=1200,
        market_snapshots=tuple(replace(row, fetched_ms=1200) for row in snapshot.market_snapshots))
    assert ordinary.snapshot(bot, now_ms=1200)["input_expires_at_ms"] == 1100
    assert ordinary.snapshot(bot, now_ms=1200)["observation_status"] == "stale"


@pytest.mark.asyncio
@pytest.mark.parametrize("hsl_enabled", [False, True])
async def test_real_fake_ordinary_compute_has_complete_immutable_receipts(tmp_path, monkeypatch, hsl_enabled):
    import argparse
    import hjson
    from config import prepare_config
    from config.hsl import generated_template
    from hsl_fixture import load_fake_hsl_config, mutable_runtime_fixture, set_risk_memory_spans
    from live import hsl_live
    from test_run_fake_live import REPO_ROOT, _cleanup_fake_user_state
    import tools.run_fake_live as runner
    import utils

    user = f"fake_monitor_receipts_{tmp_path.name}"
    _cleanup_fake_user_state(user)
    legacy = prepare_config(load_fake_hsl_config(), target="canonical", runtime=None, verbose=False)
    cfg = generated_template(mutable_runtime_fixture(legacy), "coin")
    for side in ("long", "short"):
        cfg["bot"][side]["unstuck"]["enabled"] = False
        cfg["bot"][side]["entry_cooldown"]["base_duration_minutes"] = 0.
        cfg["bot"][side]["hsl"]["enabled"] = bool(hsl_enabled and side == "long")
    cfg["bot"]["portfolio"]["loss_gate"]["max_drawdown_fraction"] = 1.
    cfg["bot"]["long"]["universe"]["approved_coins"] = ["BTC"]
    cfg["bot"]["long"]["hsl"].update(drawdown_fraction_ema_span_minutes=1., drawdown_threshold_fraction=.5)
    set_risk_memory_spans(cfg, 288.)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({k:v for k,v in cfg.items() if not k.startswith("_")}))
    scenario = hjson.loads((REPO_ROOT / "scenarios/fake_live/hsl_long_red_restart.hjson").read_text())
    scenario.pop("assertions", None)
    scenario["run_initial_cycle"] = True
    scenario_path = tmp_path / "scenario.hjson"
    scenario_path.write_text(hjson.dumps(scenario))
    completed = []

    async def exercise(bot):
        owner = hsl_live.owner(bot)
        owner.poll_inputs()
        await bot.refresh_protective_authoritative_state(require_balance=True)
        assert await owner._ordinary_plan() is not None
        result = ordinary.snapshot(bot, now_ms=int(utils.utc_ms()))
        assert result["stage"] == "computed_native_plan"
        assert result["input_receipts"]["problems"] == []
        assert result["input_receipts"]["required_market_symbol_count"] > 0
        assert result["input_receipts"]["hsl_inputs_required"] is hsl_enabled
        assert result["input_expires_at_ms"] >= int(utils.utc_ms())
        assert result["observation_status"] == "current"
        completed.append(True)
        return {"ordinary_receipts_verified": True}

    monkeypatch.setattr(runner, "_run_fake_cycle", exercise)
    try:
        args = argparse.Namespace(config=str(config_path), scenario=str(scenario_path), user=user,
            max_steps=1, output_dir=str(tmp_path / "run"), log_level=1, snapshot_each_step=False)
        assert await runner._async_main(args) == 0
        assert completed
    finally:
        _cleanup_fake_user_state(user)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["accepted_empty", "accepted_sub_ms", "rejected", "absent"])
async def test_cycle_completion_records_ordinary_path_separately_from_zero_timing(monkeypatch, case):
    import asyncio
    from dataclasses import replace
    import time
    import utils
    from live import hsl_live
    from test_hsl_runtime import NOW, bot as make_bot, quotes

    monkeypatch.setattr(utils, "utc_ms", lambda: NOW)
    monkeypatch.setattr(time, "perf_counter", lambda: 10.)
    bot = make_bot("coin")
    ledger = bot._ensure_freshness_ledger()
    ledger.stamp("open_orders", now_ms=NOW - 200)
    bot.freshness_ledger = ledger
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot.get_exchange_time = lambda: NOW
    bot.approved_coins_minus_ignored_coins = {"long": set(bot.positions), "short": set()}
    bot._begin_live_event_cycle = lambda **_: setattr(bot, "_live_event_current_cycle_id", "cy_test")
    events, executed = [], []
    bot._emit_live_event = lambda kind, **data: events.append((kind, data))
    owner = hsl_live.Owner(bot)
    wave = replace(owner.capture(quotes()), ordinary_receipts=owner.ordinary_receipts())
    assert owner._account_matches(wave, NOW)  # Use real admission/immutable facts.
    owner.poll_inputs = owner.remember_position = owner.schedule_history = lambda: None
    owner.schedule_sources = owner.schedule_account_reporting = lambda: None
    async def protect(**kwargs):
        return False
    async def refresh(**kwargs):
        return True
    async def execute(cancels, creates):
        executed.append((cancels, creates))
        return None  # A returned execution path does not imply acknowledgement/writes.
    async def pending():
        await asyncio.Event().wait()
    marker = object()
    creates = [{"symbol": next(iter(bot.positions))}] if case == "accepted_sub_ms" else []
    async def prepared():
        return [], creates, marker, wave
    owner.protect, owner._ordinary_plan = protect, pending
    bot.refresh_protective_authoritative_state = refresh
    bot.execute_order_plan_to_exchange = execute
    if case != "absent":
        owner._ordinary = asyncio.create_task(prepared())
        await owner._ordinary
    if case == "rejected":
        bot._account_invalidation_generation = 1
    try:
        result = await owner.cycle()
        data = [event["data"] for kind, event in events if kind == EventTypes.CYCLE_COMPLETED]
        assert len(data) == 1
        expected = case.startswith("accepted")
        assert data[0]["ordinary_executed"] is result["ordinary_executed"] is expected
        assert data[0]["timings_ms"]["ordinary_execute"] == 0
        assert bool(executed) is expected
        assert data[0]["execution_owner"] == "hsl"
    finally:
        owner._ordinary.cancel()
        await asyncio.gather(owner._ordinary, return_exceptions=True)


def test_other_execution_owner_omits_optional_ordinary_path_flag():
    from live.event_emitters import emit_live_cycle_completed
    events = []
    bot = SimpleNamespace(_emit_live_event=lambda kind, **data: events.append(data))
    emit_live_cycle_completed(bot, cycle_id="cy_test", loop_start_ms=1000, timings_ms={})
    assert "ordinary_executed" not in events[0]["data"]
