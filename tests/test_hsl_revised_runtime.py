"""Current live observations through the real shared native evaluator, offline."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from config import prepare_config
from config.hsl_revised import generated_template
from config.schema import get_template_config
from live.freshness import FreshnessLedger
from live.hsl_revised_candles import Sources, Failure
from live.hsl_revised_inputs import capture_candles
from live.hsl_revised_runtime import capture, evaluate, InvalidHslOutput
from live.market_snapshot import MarketSnapshot
from test_hsl_revised_inputs import event

NOW = 1_800_000_000_000
SYMBOL = "TEST/USDT:USDT"


def bot(mode="coin", *, side="long", events=()):
    config = generated_template(get_template_config(), mode)
    config["live"]["pnls_max_lookback_days"] = 1
    for pside in ("long", "short"):
        config["bot"][pside]["hsl"].update(enabled=pside == side, red_threshold=.05,
            ema_span_minutes=1_000_000., restart_after_red_policy="always")
        config["bot"][pside]["risk"]["n_positions"] = 1
    if mode == "unified":
        config["bot"]["hsl"].update(config["bot"][side]["hsl"])
        config["bot"]["hsl"]["panic_close_order_type"] = "market"
        for pside in ("long", "short"):
            config["bot"][pside]["hsl"]["enabled"] = False
            config["bot"][pside]["risk"]["n_positions"] = 0
            config["bot"][pside]["risk"]["total_wallet_exposure_limit"] = 0
    config = prepare_config(config, verbose=False, target="canonical", runtime=None)
    ledger = FreshnessLedger(now_ms=NOW)
    for surface in ("balance", "positions"):
        ledger.stamp(surface, now_ms=NOW-200)
    positions = {SYMBOL: {pside: dict(size=(10. if side == "long" else -10.) if side == pside else 0.,
                                     price=100. if side == pside else 0.)
                         for pside in ("long", "short")}}
    value = SimpleNamespace(config=config, positions=positions, open_orders={}, inverse=False, coin_overrides={},
        c_mults={SYMBOL: 1.}, qty_steps={SYMBOL: .1},
        _ensure_freshness_ledger=lambda: ledger, get_raw_balance=lambda: 1000.,
        _pnls_manager=SimpleNamespace(get_events=lambda *, start_ms: [e for e in events if e.timestamp >= start_ms]))
    value.bot_value = lambda side, key: config["bot"][side]["risk"][key]
    value.bp = lambda side, key, symbol: config["bot"][side]["hsl"][key.removeprefix("hsl_")]
    return value


def quotes(side="long"):
    price = 90. if side == "long" else 110.
    return {SYMBOL: MarketSnapshot(SYMBOL, price, price, price, NOW-100, "fake")}


def run(value, marks=None, sources=None, **changes):
    kwargs = dict(symbols={"long": [SYMBOL], "short": [SYMBOL]}, now_ms=NOW, utc_now_ms=NOW, max_current_age_ms=10_000)
    kwargs.update(changes)
    if isinstance(kwargs["symbols"], list):
        kwargs["symbols"] = {side: kwargs["symbols"] for side in ("long", "short")}
    requests, unavailable = capture(value, quotes() if marks is None else marks, sources or {}, **kwargs)
    return evaluate(requests), unavailable


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_no_history_current_loss_is_evaluated_for_every_topology(mode, side):
    value = bot(mode, side=side)
    result, unavailable = run(value, quotes(side))
    assert not unavailable
    decision, = result
    assert decision.action == "panic"
    raw = json.loads(decision.payload)
    assert raw["observations"] == 1
    assert raw["decision"]["raw"] == pytest.approx(100/1100)
    assert raw["decision"]["ema"] == raw["decision"]["raw"]
    assert "fill_capture_unknown" in decision.reasons
    assert decision.execution_type == ("market" if mode == "unified" else "limit")


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_canonical_cashflows_survive_pending_quantity_and_missing_candles(mode):
    value = bot(mode, events=[event(timestamp=NOW-60_000, side="sell", qty=None,
                                    price=90., pnl=-200., fee_paid=-1., c_mult=1.)])
    decision, = run(value)[0]
    assert decision.action == "panic"
    assert "invalid_fill_quantity" in decision.reasons
    assert json.loads(decision.payload)["decision"]["raw"] == pytest.approx(301/1301)


def test_current_quote_failure_is_scoped_before_native_evaluation():
    value = bot()
    other = "OTHER/USDT:USDT"
    value.positions[other] = deepcopy(value.positions[SYMBOL])
    value.c_mults[other], value.qty_steps[other] = 1., .1
    result, unavailable = run(value)
    assert [d.scope.symbol for d in result] == [SYMBOL]
    assert result[0].action == "panic"
    assert [u.scope.symbol for u in unavailable] == [other]
    value.config["live"]["hsl_signal_mode"] = "unified"
    value.config["bot"]["hsl"] = deepcopy(value.config["bot"]["long"]["hsl"])
    result, unavailable = run(value)
    assert not result and len(unavailable) == 1
    assert unavailable[0].scope.mode == "unified"


@pytest.mark.parametrize("surface", ["balance", "positions"])
def test_stale_essentials_do_not_become_healthy_or_saved_decisions(surface):
    value = bot()
    assert run(value)[0][0].action == "panic"
    value._ensure_freshness_ledger().surfaces[surface].updated_ms = NOW-10_001
    result, unavailable = run(value)
    assert not result
    assert unavailable[0].reason == f"current_{surface}_unavailable"
    value._ensure_freshness_ledger().stamp(surface, now_ms=NOW)
    value.positions[SYMBOL]["long"]["price"] = 80.
    assert run(value)[0][0].action == "normal"


def test_clock_offset_preserves_age_without_changing_exchange_event_times():
    value = bot(events=[event(timestamp=NOW-60_000, c_mult=1.)])
    requests, unavailable = capture(value, quotes(), {}, symbols={"long": [SYMBOL], "short": [SYMBOL]}, now_ms=NOW+5000,
        utc_now_ms=NOW, max_current_age_ms=10_000,
        fills_started_ms=NOW-150, fills_completed_ms=NOW-50)
    assert not unavailable
    snap = json.loads(requests[0].payload)["snapshot"]
    pair = snap["pairs"][0]
    assert pair["position_at"] == NOW+4800
    assert pair["mark_at"] == NOW+4900
    assert pair["fills_started_at"] == NOW+4850
    assert pair["fills_at"] == NOW+4950
    assert pair["fills"][0]["timestamp"] == NOW-60_000


@pytest.mark.parametrize("offset", [-5000, 5000])
def test_candle_capture_clock_conversion_does_not_move_candle_open(offset):
    tape = capture_candles([dict(ts=NOW-60_000, o=100., h=100., l=100., c=100.)],
                          minutes=1, observed_at=NOW-offset)
    sources = {SYMBOL: Sources((tape,), (), 0)}
    value = bot()
    for surface in ("balance", "positions"):
        value._ensure_freshness_ledger().stamp(surface, now_ms=NOW-offset-200)
    marks = {SYMBOL: replace(quotes()[SYMBOL], fetched_ms=NOW-offset-100)}
    requests, unavailable = capture(value, marks, sources, symbols={"long": [SYMBOL], "short": [SYMBOL]}, now_ms=NOW,
        utc_now_ms=NOW-offset, max_current_age_ms=10_000)
    assert not unavailable
    prices = json.loads(requests[0].payload)["snapshot"]["pairs"][0]["prices"]
    assert prices[str(NOW)] == 100.
    assert max(map(int, prices)) == NOW


def test_capture_is_immutable_and_does_not_inherit_prior_state():
    value = bot()
    requests, unavailable = capture(value, quotes(), {}, symbols={"long": [SYMBOL], "short": [SYMBOL]}, now_ms=NOW,
        utc_now_ms=NOW, max_current_age_ms=10_000)
    assert not unavailable
    before = evaluate(requests)
    value.positions[SYMBOL]["long"]["price"] = 80.
    value.config["bot"]["long"]["hsl"]["red_threshold"] = .99
    assert evaluate(requests) == before
    assert run(value)[0][0].action == "normal"


def test_zero_slots_remains_inactive_without_fake_divisor():
    value = bot()
    value.config["bot"]["long"]["risk"]["n_positions"] = 0
    result, unavailable = run(value)
    assert not unavailable and result[0].action is None
    assert json.loads(result[0].payload)["decision"] is None


def test_fractional_cooldown_uses_native_backtest_millisecond_rounding():
    value = bot()
    value.config["bot"]["long"]["hsl"]["cooldown_minutes_after_red"] = 1.5/60_000
    value.config["live"]["pnls_max_lookback_days"] = (86_400_000 + 1.5)/86_400_000
    requests, _ = capture(value, quotes(), {}, symbols={"long": [SYMBOL], "short": [SYMBOL]}, now_ms=NOW,
        utc_now_ms=NOW, max_current_age_ms=10_000)
    assert json.loads(requests[0].payload)["cooldown_ms"] == 2
    assert json.loads(requests[0].payload)["snapshot"]["start"] == NOW-86_400_002


def test_coarse_source_gaps_are_native_estimates_not_candle_ledger_writes():
    value = bot()
    tape = capture_candles([dict(ts=NOW-900_000, o=100., h=120., l=80., c=90.)],
                          minutes=15, observed_at=NOW)
    sources = {SYMBOL: Sources((tape,), (Failure("1m", "fetch", "TimeoutError"),), 0)}
    before = deepcopy(sources)
    decision, = run(value, sources=sources)[0]
    assert "coarse_candle" in decision.reasons
    assert "candle_fetch_unavailable:1m" in decision.reasons
    assert sources == before
    assert json.loads(decision.payload)["observations"] > 1


def test_flat_with_stale_factual_mark_replays_without_fresh_quote_requirement():
    value = bot()
    value.positions = {}
    old = {SYMBOL: replace(quotes()[SYMBOL], fetched_ms=NOW-60_000)}
    result, unavailable = run(value, old)
    assert not unavailable and result[0].action == "normal"
    assert "stale_flat_mark" in result[0].reasons


def test_missing_flat_quote_uses_historical_close_but_never_for_held_exposure():
    value = bot()
    value.positions = {}
    tape = capture_candles([dict(ts=NOW-120_000, o=90., h=90., l=90., c=90.)],
                          minutes=1, observed_at=NOW)
    sources = {SYMBOL: Sources((tape,), (), 0)}
    result, unavailable = run(value, {}, sources)
    assert not unavailable and result[0].action == "normal"
    assert {"flat_historical_close", "stale_flat_mark"} <= set(result[0].reasons)
    value.positions = bot().positions
    result, unavailable = run(value, {}, sources)
    assert not result and unavailable[0].reason == "current_mark_unavailable"


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_observed_fill_interval_reconstructs_flat_halt_and_window_expiry(mode):
    events = [event(id="open", timestamp=NOW-180_000, qty=10., price=100., c_mult=1., fee_paid=0.),
              event(id="close", timestamp=NOW-120_000, side="sell", qty=-10., price=80.,
                    pnl=-200., c_mult=1., fee_paid=0.)]
    value = bot(mode, events=events)
    value.positions = {}
    policy = value.config["bot"]["hsl"] if mode == "unified" else value.config["bot"]["long"]["hsl"]
    policy["restart_after_red_policy"] = "never"
    decision, = run(value, fills_started_ms=NOW-150, fills_completed_ms=NOW-50)[0]
    assert decision.action == "halted"
    assert json.loads(decision.payload)["decision"]["flat_at"] == NOW-120_000
    # Cache reads cannot pretend to be a remote post-position interval.
    estimated, = run(value)[0]
    assert "fill_capture_unknown" in estimated.reasons
    # A fresh process at a later exchange time has no out-of-window authority.
    later = NOW+86_400_000
    for surface in ("balance", "positions"):
        value._ensure_freshness_ledger().stamp(surface, now_ms=later-200)
    marks = {SYMBOL: replace(quotes()[SYMBOL], fetched_ms=later-100)}
    result, unavailable = run(value, marks, now_ms=later, utc_now_ms=later,
        fills_started_ms=later-150, fills_completed_ms=later-50)
    assert not unavailable and result[0].action == "normal"


def test_missing_quote_does_not_contaminate_disabled_side_or_other_coin_policy():
    value = bot()
    other = "OTHER/USDT:USDT"
    value.coin_overrides[other] = {"bot": {"long": {"hsl": {"enabled": False}}}}
    result, unavailable = run(value, symbols=[SYMBOL, other])
    assert not unavailable and len(result) == 1


def test_malformed_later_native_scope_does_not_publish_an_earlier_usable_subset(monkeypatch):
    import live.hsl_revised_runtime as runtime
    value = bot()
    value.config["bot"]["short"]["hsl"]["enabled"] = True
    native = runtime.pbr.hsl_revised_evaluate_grids
    calls = []
    def malformed(request, grids):
        calls.append(request)
        output = json.loads(native(request, grids))
        if len(calls) == 2:
            output["decision"] = None
        return json.dumps(output)
    monkeypatch.setattr(runtime.pbr, "hsl_revised_evaluate_grids", malformed)
    with pytest.raises(InvalidHslOutput):
        run(value)
    assert len(calls) == 2


def test_out_of_window_fill_does_not_introduce_old_flat_symbols_or_quality():
    value = bot("unified", events=[event(symbol="OLD", timestamp=NOW-86_400_001, qty=None)])
    result, unavailable = run(value, symbols=["UNUSED"])
    assert not unavailable and result[0].action == "panic"
    assert "invalid_fill_quantity" not in result[0].reasons


def test_empty_aggregate_side_does_not_duplicate_price_history():
    import passivbot_rust as pbr
    value = bot("unified")
    requests, unavailable = capture(value, quotes(), {}, symbols={"long": [SYMBOL], "short": [SYMBOL]}, now_ms=NOW,
        utc_now_ms=NOW, max_current_age_ms=10_000)
    assert not unavailable
    request, = requests
    payload = json.loads(request.payload)
    assert len(payload["snapshot"]["pairs"]) == 1
    baseline = json.loads(pbr.hsl_revised_evaluate(request.payload))
    flat = deepcopy(payload["snapshot"]["pairs"][0])
    flat["position"].update(size=0., basis=0., pside="short")
    payload["snapshot"]["pairs"].append(flat)
    assert json.loads(pbr.hsl_revised_evaluate(json.dumps(payload))) == baseline


def test_empty_aggregate_side_retains_cashflows_even_without_exposure():
    value = bot("unified", events=[event(timestamp=NOW-60_000, position_side="short",
        qty=0., price=100., pnl=-200., fee_paid=0., c_mult=1.)])
    requests, unavailable = capture(value, quotes(), {}, symbols={"long": [SYMBOL], "short": [SYMBOL]}, now_ms=NOW,
        utc_now_ms=NOW, max_current_age_ms=10_000)
    assert not unavailable and len(json.loads(requests[0].payload)["snapshot"]["pairs"]) == 2
    assert json.loads(evaluate(requests)[0].payload)["decision"]["raw"] == pytest.approx(300/1300)


def test_all_flat_aggregate_with_no_history_needs_no_market_quote():
    value = bot("unified")
    value.positions = {}
    result, unavailable = run(value, {}, symbols=["UNUSED"])
    assert not unavailable and result[0].action == "normal"


@pytest.mark.parametrize("field,value", [("action", "orange"), ("raw", float("nan")),
                                        ("timestamp", NOW-1), ("action", [])])
def test_malformed_native_decision_is_fatal(field, value, monkeypatch):
    import live.hsl_revised_runtime as runtime
    native = runtime.pbr.hsl_revised_evaluate_grids
    def malformed(request, grids):
        result = json.loads(native(request, grids))
        result["decision"][field] = value
        return json.dumps(result)
    monkeypatch.setattr(runtime.pbr, "hsl_revised_evaluate_grids", malformed)
    with pytest.raises(InvalidHslOutput):
        run(bot())


def test_native_errors_use_the_bot_fatal_contract(monkeypatch):
    import live.hsl_revised_runtime as runtime
    from passivbot_exceptions import FatalBotException
    def invalid(_request, _grids):
        raise ValueError("unexpected native schema error")
    monkeypatch.setattr(runtime.pbr, "hsl_revised_evaluate_grids", invalid)
    with pytest.raises(FatalBotException):
        run(bot())


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_future_only_symbol_does_not_demand_a_current_quote_or_metadata(mode):
    value = bot(mode, events=[event(symbol="FUTURE", timestamp=NOW+60_000, c_mult=1.)])
    result, unavailable = run(value)
    assert not unavailable and result[0].action == "panic"
    assert "future_fill_outside_evaluation" in result[0].reasons


def test_coin_membership_keeps_eligible_and_historical_sides_separate():
    value = bot(events=[event(symbol="OLD_SHORT", position_side="short", side="sell",
                            qty=-1., timestamp=NOW-60_000, c_mult=1.)])
    value.config["bot"]["short"]["hsl"]["enabled"] = True
    result, unavailable = run(value, symbols={"long": [SYMBOL], "short": ["SHORT_ONLY"]})
    assert [(d.scope.symbol, d.scope.pside) for d in result] == [(SYMBOL, "long")]
    assert {(u.scope.symbol, u.scope.pside) for u in unavailable} == {
        ("OLD_SHORT", "short"), ("SHORT_ONLY", "short")}


def test_long_only_history_never_requires_metadata_for_the_empty_short_scope():
    value = bot("pside", events=[event(symbol="LONG_ONLY", timestamp=NOW-60_000)])
    value.config["bot"]["short"]["hsl"]["enabled"] = True
    result, unavailable = run(value)
    assert [(d.scope.pside, d.action) for d in result] == [("short", "normal")]
    assert len(unavailable) == 1 and unavailable[0].scope.pside == "long"


def test_simultaneous_flat_empty_observation_needs_no_invented_mark_or_metadata():
    value = bot()
    value.positions = {}
    value.c_mults = {}
    result, unavailable = run(value, {}, fills_started_ms=NOW-300, fills_completed_ms=NOW-200)
    assert not unavailable and result[0].action == "normal"
    assert json.loads(result[0].payload)["observations"] == 1
    requests, _ = capture(value, {}, {}, symbols={"long": [SYMBOL], "short": []},
        now_ms=NOW, utc_now_ms=NOW, max_current_age_ms=10_000,
        fills_started_ms=NOW-300, fills_completed_ms=NOW-200)
    snapshot = json.loads(requests[0].payload)["snapshot"]
    assert snapshot["pairs"] == []
    assert snapshot["flat_coin"] == dict(symbol=SYMBOL, pside="long",
        position_at=NOW-200, fills_at=NOW-200, history_start=NOW-86_400_000)
    # A local cache read or different remote observation is not that proof.
    for kwargs in ({}, dict(fills_started_ms=NOW-300, fills_completed_ms=NOW-100)):
        result, unavailable = run(value, {}, **kwargs)
        assert not result and unavailable


@pytest.mark.parametrize("changes", [dict(qty=None), dict(timestamp=0)])
def test_damaged_retained_fill_cannot_supply_empty_flat_proof(changes):
    value = bot(events=[event(timestamp=NOW-60_000, c_mult=1., **changes)]
                if "timestamp" not in changes else [event(c_mult=1., **changes)])
    value.positions = {}
    result, unavailable = run(value, {}, fills_started_ms=NOW-300, fills_completed_ms=NOW-200)
    # The undated row is filtered by this fixture's canonical window query, so
    # use an explicit manager response for that corrupted observation.
    if "timestamp" in changes:
        value._pnls_manager.get_events = lambda **_: [event(c_mult=1., **changes)]
        result, unavailable = run(value, {}, fills_started_ms=NOW-300, fills_completed_ms=NOW-200)
        assert not result and unavailable
    else:
        assert not unavailable
        assert {'flat_historical_fill_price', 'invalid_fill_quantity'} <= set(result[0].reasons)
        requests, _ = capture(value, {}, {}, symbols={'long': [SYMBOL], 'short': []},
            now_ms=NOW, utc_now_ms=NOW, max_current_age_ms=10_000,
            fills_started_ms=NOW-300, fills_completed_ms=NOW-200)
        assert 'flat_coin' not in json.loads(requests[0].payload)['snapshot']


def test_negative_historical_equity_can_leave_a_valid_ema_above_one():
    value = bot()
    value.positions[SYMBOL]["long"]["size"] = 100.
    value.config["bot"]["long"]["hsl"]["ema_span_minutes"] = 1.5
    rows = [dict(ts=NOW-(6-i)*60_000, o=p, h=p, l=p, c=p)
            for i, p in enumerate([1000., .001, .001, .001, .001, .001])]
    sources = {SYMBOL: Sources((capture_candles(rows, minutes=1, observed_at=NOW),), (), 0)}
    result, unavailable = run(value, sources=sources)
    assert not unavailable and result[0].action == "panic"
    assert json.loads(result[0].payload)["decision"]["ema"] > 1.


@pytest.mark.parametrize("changes", [
    dict(action="halted"), dict(action="normal"), dict(red_at=None),
    dict(red_at=True), dict(flat_at=NOW), dict(red_at=NOW+1),
    dict(red_at=NOW-86_400_001), dict(numeric_range_approximation=1),
    dict(reason=None), dict(ema=-.01), dict(raw=True), dict(timestamp=float(NOW)),
])
def test_lifecycle_incoherent_native_response_is_fatal(changes, monkeypatch):
    import live.hsl_revised_runtime as runtime
    native = runtime.pbr.hsl_revised_evaluate_grids
    def malformed(request, grids):
        result = json.loads(native(request, grids))
        result["decision"].update(changes)
        return json.dumps(result)
    monkeypatch.setattr(runtime.pbr, "hsl_revised_evaluate_grids", malformed)
    with pytest.raises(InvalidHslOutput):
        run(bot())


def test_empty_failed_candle_acquisition_keeps_flat_proof_and_diagnostics():
    value = bot()
    value.positions = {}
    sources = {SYMBOL: Sources((), (Failure("1m", "fetch", "TimeoutError"),), 0)}
    result, unavailable = run(value, {}, sources,
        fills_started_ms=NOW-300, fills_completed_ms=NOW-200)
    assert not unavailable and result[0].action == "normal"
    assert "candle_fetch_unavailable:1m" in result[0].reasons


@pytest.mark.parametrize("symbols", [[SYMBOL], {"long": [SYMBOL]},
                                      {"long": SYMBOL, "short": []}])
def test_scope_selection_requires_explicit_valid_side_membership(symbols):
    with pytest.raises(ValueError, match="membership"):
        capture(bot(), quotes(), {}, symbols=symbols, now_ms=NOW,
                utc_now_ms=NOW, max_current_age_ms=10_000)


@pytest.mark.parametrize("mode", ["pside", "unified"])
def test_empty_aggregate_retains_unknown_fill_capture_diagnostic(mode):
    value = bot(mode)
    value.positions = {}
    result, unavailable = run(value, {})
    assert not unavailable and result[0].action == "normal"
    assert "fill_capture_unknown" in result[0].reasons
    result, unavailable = run(value, {}, fills_started_ms=NOW-150, fills_completed_ms=NOW-50)
    assert not unavailable and "fill_capture_unknown" not in result[0].reasons


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_actual_passivbot_grouped_policy_and_resolved_partial_override(mode):
    from passivbot import Passivbot
    fixture = bot(mode)
    # Do not start a bot or exchange client: exercise the real runtime accessors
    # and symbol override resolver against canonical prepared configuration.
    value = Passivbot.__new__(Passivbot)
    value.__dict__.update(fixture.__dict__)
    del value.__dict__["bp"], value.__dict__["bot_value"]
    value.config["coin_overrides"] = {"TEST": {"bot": {"long": {"hsl": {
        "red_threshold": .99, "panic_close_order_type": "market"}}}}}
    value.coin_to_symbol = lambda coin, verbose=False: SYMBOL
    value.markets_dict = {SYMBOL: {}}
    value.init_coin_overrides()
    requests, unavailable = capture(value, quotes(), {},
        symbols={"long": [SYMBOL], "short": []}, now_ms=NOW, utc_now_ms=NOW,
        max_current_age_ms=10_000)
    assert not unavailable
    request, = requests
    assert json.loads(request.payload)["threshold"] == (.99 if mode == "coin" else .05)
    assert json.loads(request.payload)["span"] == 1_000_000.
    assert request.execution_type == ("limit" if mode == "pside" else "market")
    assert evaluate(requests)[0].action == ("normal" if mode == "coin" else "panic")


@pytest.mark.parametrize("mode", ["pside", "unified"])
def test_undated_flat_fill_quality_survives_without_market_requirements(mode):
    value = bot(mode)
    value.positions = {}
    value.config["bot"]["short"]["hsl"]["enabled"] = True
    value._pnls_manager.get_events = lambda **_: [event(symbol="UNKNOWN_MARKET", timestamp=0)]
    result, unavailable = run(value, {}, fills_started_ms=NOW-150, fills_completed_ms=NOW-50)
    assert not unavailable and all(d.action == "normal" for d in result)
    for decision in result:
        assert ("unidentified_or_undated_fill" in decision.reasons) == (decision.scope.pside != "short")


def test_undated_other_coin_diagnostic_does_not_cross_coin_scope():
    value = bot()
    value._pnls_manager.get_events = lambda **_: [event(symbol="OTHER", timestamp=0)]
    result, unavailable = run(value)
    assert not unavailable and len(result) == 1
    assert "unidentified_or_undated_fill" not in result[0].reasons


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("history", ["empty", "damaged", "closed"])
def test_compact_price_transport_preserves_complete_decisions(mode, side, history, monkeypatch):
    import live.hsl_revised_runtime as runtime
    sign = 1 if side == "long" else -1
    events = [] if history == "empty" else [
        event(id="open", timestamp=NOW-180_000, position_side=side,
              side="buy" if sign > 0 else "sell", qty=sign*10., price=100., c_mult=1., fee_paid=0.),
        event(id="close", timestamp=NOW-120_000, position_side=side,
              side="sell" if sign > 0 else "buy", qty=-sign*10. if history == "closed" else None,
              price=100.-sign*20., pnl=-200., c_mult=1., fee_paid=-1.)]
    value = bot(mode, side=side, events=events)
    if history == "closed":
        value.positions = {}
    coarse = capture_candles([dict(ts=NOW-900_000, o=100., h=120., l=80., c=100.)],
                            minutes=15, observed_at=NOW)
    fine = capture_candles([dict(ts=NOW-60_000, o=100., h=100., l=100., c=100.)],
                          minutes=1, observed_at=NOW)
    sources = {SYMBOL: Sources((coarse, fine), (), 0)}
    options = dict(fills_started_ms=NOW-150, fills_completed_ms=NOW-50)
    actual = run(value, quotes(side), sources, **options)
    def json_projection(start, end, rows):
        keys = ("start", "minutes", "open", "high", "low", "close", "available_at")
        out = json.loads(runtime.pbr.hsl_revised_prices(json.dumps(dict(
            start=start, end=end, candles=[dict(zip(keys, row)) for row in rows]))))
        last = out["rows"][-1] if out["rows"] else None
        return ({str(row["timestamp"]): row["close"] for row in out["rows"]},
                (last["close"], last["source_end"]) if last else None, out["reasons"])
    monkeypatch.setattr(runtime.pbr, "hsl_revised_price_grid", json_projection)
    assert run(value, quotes(side), sources, **options) == actual


@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
def test_actual_unchanged_position_read_can_precede_fill_tail_without_forged_time(mode):
    from live.hsl_revised_runtime import observe_positions
    events = [event(id='open', timestamp=NOW-180_000, qty=10., price=100., c_mult=1., fee_paid=0.),
              event(id='close', timestamp=NOW-120_000, side='sell', qty=-10., price=80.,
                    pnl=-200., c_mult=1., fee_paid=0.)]
    value = bot(mode, events=events)
    value.positions = {}
    policy = value.config['bot']['hsl'] if mode == 'unified' else value.config['bot']['long']['hsl']
    policy['restart_after_red_policy'] = 'never'
    actual_read = observe_positions(value)
    assert actual_read.observed_ms == NOW-200
    value._ensure_freshness_ledger().stamp('positions', now_ms=NOW-10)
    kwargs = dict(fills_started_ms=NOW-150, fills_completed_ms=NOW-50)
    without_read, = run(value, **kwargs)[0]
    assert 'fills_before_position' in without_read.reasons
    decision, = run(value, position_observation=actual_read, **kwargs)[0]
    assert decision.action == 'halted'
    assert 'fills_before_position' not in decision.reasons
    # Fresh acquisition reconstructs the same halt without the earlier read.
    fresh, = run(value, fills_started_ms=NOW-5, fills_completed_ms=NOW-1)[0]
    assert fresh.action == decision.action


@pytest.mark.parametrize('change', ['position', 'generation', 'timestamp'])
def test_position_read_cannot_be_reused_for_different_current_facts(change):
    from live.hsl_revised_runtime import observe_positions
    value = bot()
    observed = observe_positions(value)
    if change == 'position':
        value.positions[SYMBOL]['long']['size'] = 9.
    elif change == 'generation':
        value._account_invalidation_generation = 1
    else:
        observed = replace(observed, observed_ms=NOW)
    with pytest.raises(ValueError, match='does not match current account facts'):
        run(value, position_observation=observed)


@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
def test_flat_retained_fills_do_not_require_a_delisted_market_quote(mode):
    events = [event(id='open', timestamp=NOW-180_000, qty=10., price=100., c_mult=1., fee_paid=0.),
              event(id='close', timestamp=NOW-120_000, side='sell', qty=-10., price=80.,
                    pnl=-200., c_mult=1., fee_paid=0.)]
    value = bot(mode, events=events)
    value.positions = {}
    policy = value.config['bot']['hsl'] if mode == 'unified' else value.config['bot']['long']['hsl']
    policy['restart_after_red_policy'] = 'never'
    decisions, unavailable = run(value, {}, fills_started_ms=NOW-150, fills_completed_ms=NOW-50)
    assert not unavailable
    assert decisions[0].action == 'halted'
    assert 'flat_historical_fill_price' in decisions[0].reasons
    value.positions = {SYMBOL: {'long': dict(size=1., price=100.), 'short': dict(size=0., price=0.)}}
    decisions, unavailable = run(value, {}, fills_started_ms=NOW-150, fills_completed_ms=NOW-50)
    assert not decisions and unavailable[0].reason == 'current_mark_unavailable'


@pytest.mark.parametrize('surface', ['balance', 'positions'])
def test_recent_but_invalidated_account_observation_is_not_current(surface):
    value = bot()
    value._authoritative_pending_confirmations = {surface: 1}
    decisions, unavailable = run(value)
    assert not decisions
    assert unavailable[0].reason == f'current_{surface}_unavailable'
    value._ensure_freshness_ledger().stamp(surface, now_ms=NOW-10, epoch=1)
    decisions, unavailable = run(value)
    assert not unavailable and decisions[0].action == 'panic'


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("history", ["empty", "partial", "coarse", "complete"])
def test_native_grid_transport_matches_complete_json_and_is_immutable(mode, side, history):
    import passivbot_rust as pbr
    import numpy as np
    value = bot(mode, side=side, events=[event(timestamp=NOW-120_000, qty=None,
        position_side=side, pnl=-12., fee_paid=-.3, c_mult=1.)])
    sources = {}
    if history != "empty":
        minutes = 15 if history == "coarse" else 1
        count = 1440 if history == "complete" else 3
        rows = np.array([(NOW-(count-i)*minutes*60_000, 100., 105., 85., 90.)
            for i in range(count)], dtype=[(k, 'int64' if k == 'ts' else 'float64')
                                          for k in ('ts', 'o', 'h', 'l', 'c')])
        sources[SYMBOL] = Sources((capture_candles(rows, minutes=minutes, observed_at=NOW),), (), 0)
    requests, unavailable = capture(value, quotes(side), sources,
        symbols={"long": [SYMBOL], "short": [SYMBOL]}, now_ms=NOW,
        utc_now_ms=NOW, max_current_age_ms=10_000)
    assert not unavailable
    for request, decision in zip(requests, evaluate(requests), strict=True):
        expected = json.loads(pbr.hsl_revised_evaluate(request.payload))
        assert json.loads(decision.payload) == expected
        assert all(not pair['prices'] for pair in json.loads(request.metadata)['snapshot']['pairs'])
        for grid in request.price_grids:
            original = grid.values()
            detached = grid.values()
            detached.clear()
            assert grid.values() == original
            with pytest.raises(AttributeError):
                grid.prices = {}
        # Repeated evaluation must not consume or mutate its immutable input.
        assert json.loads(evaluate((request,))[0].payload) == expected
    with pytest.raises(TypeError):
        pbr.RevisedHslPriceGrid()


@pytest.mark.parametrize("defect", ["missing", "extra", "start", "end", "prices_at", "override", "wrong_type"])
def test_native_grid_transport_rejects_mismatched_observation(defect):
    import passivbot_rust as pbr
    requests, unavailable = capture(bot(), quotes(), {}, symbols={"long": [SYMBOL], "short": []},
        now_ms=NOW, utc_now_ms=NOW, max_current_age_ms=10_000)
    assert not unavailable
    request, = requests
    value = json.loads(request.metadata)
    grids = list(request.price_grids)
    if defect == "missing":
        grids.clear()
    elif defect == "extra":
        grids += grids
    elif defect in {"start", "end"}:
        grids[0] = pbr.hsl_revised_native_price_grid(value['snapshot']['start']-(defect == 'start'),
                                                   NOW+(defect == 'end'), [])[0]
    elif defect == "prices_at":
        value['snapshot']['pairs'][0]['prices_at'] -= 1
    elif defect == "override":
        value['snapshot']['pairs'][0]['prices'] = {str(NOW): 12.}
    else:
        grids[0] = {}
    with pytest.raises((ValueError, TypeError)):
        pbr.hsl_revised_evaluate_grids(json.dumps(value), grids)


@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
def test_native_grids_stay_with_their_symbol_across_scope_ordering(mode):
    import passivbot_rust as pbr
    from live.hsl_revised_inputs import Candle, CandleTape
    value = bot(mode)
    other = 'AAA/USDT:USDT'
    value.positions[other] = {'long': {'size': 2., 'price': 130.},
                              'short': {'size': -1., 'price': 80.}}
    value.c_mults[other], value.qty_steps[other] = 1., .1
    if mode != 'unified':
        value.config['bot']['short']['hsl']['enabled'] = True
    marks = {**quotes(), other: replace(quotes()[SYMBOL], symbol=other, last=110., bid=110., ask=110.)}
    sources = {symbol: Sources((CandleTape(tuple(
        Candle(NOW-(3-i)*60_000, 1, price, price, price, price, NOW)
        for i in range(3)), ()),), (), 0) for symbol, price in [(SYMBOL, 140.), (other, 75.)]}
    requests, unavailable = capture(value, marks, sources,
        symbols={'long': [SYMBOL, other], 'short': [other, SYMBOL]},
        now_ms=NOW, utc_now_ms=NOW, max_current_age_ms=10_000)
    assert not unavailable
    for request, actual in zip(requests, evaluate(requests), strict=True):
        assert json.loads(actual.payload) == json.loads(pbr.hsl_revised_evaluate(request.payload))
        for pair in json.loads(request.payload)['snapshot']['pairs']:
            assert set(pair['prices'].values()) == ({75.} if pair['symbol'] == other else {140.})


@pytest.mark.asyncio
@pytest.mark.parametrize("offset", [-86_400_000, 86_400_000])
async def test_owner_candle_acquisition_uses_utc_with_exchange_timeline_manager(monkeypatch, offset):
    import asyncio
    import utils
    from live.hsl_revised_live import owner
    utc_now = NOW - offset
    monkeypatch.setattr(utils, 'utc_ms', lambda: utc_now)
    value = bot()
    value.get_exchange_time = lambda: NOW
    calls = []
    async def read(symbol, **kwargs):
        calls.append(kwargs)
        return [dict(ts=NOW-60_000, o=100., h=100., l=100., c=100.)]
    value.cm = SimpleNamespace(exchange=SimpleNamespace(timeframes={'1m': 1}),
                              _now_ms=lambda: NOW, get_candles=read)
    instance = owner(value)
    instance.schedule_sources()
    await asyncio.wait_for(instance._source_task, 1.)
    assert calls[0]['end_ts'] == NOW
    assert instance.sources[SYMBOL].payload()[0]['available_at'] == utc_now
    for surface in ('balance', 'positions'):
        value._ensure_freshness_ledger().stamp(surface, now_ms=utc_now-200)
    marks = {SYMBOL: replace(quotes()[SYMBOL], fetched_ms=utc_now-100)}
    requests, unavailable = capture(value, marks, instance.sources,
        symbols={'long': [SYMBOL], 'short': []}, now_ms=NOW, utc_now_ms=utc_now,
        max_current_age_ms=10_000)
    assert not unavailable
    prices = json.loads(requests[0].payload)['snapshot']['pairs'][0]['prices']
    assert prices[str(NOW)] == 100.
    assert max(map(int, prices)) == NOW


@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
@pytest.mark.parametrize('side', ['long', 'short'])
def test_targeted_permission_keeps_the_complete_relevant_scope(mode, side):
    other = 'PEER/USDT:USDT'
    value = bot(mode, side=side)
    value.positions[other] = deepcopy(value.positions[SYMBOL])
    value.c_mults[other], value.qty_steps[other] = 1., .1
    marks = {**quotes(side), other: replace(quotes(side)[SYMBOL], symbol=other, last=100., bid=100., ask=100.)}
    kwargs = dict(symbols={'long': [other, SYMBOL], 'short': [SYMBOL, other]},
                  now_ms=NOW, utc_now_ms=NOW, max_current_age_ms=10_000)
    full, absent = capture(value, marks, {}, **kwargs)
    selected, selected_absent = capture(value, marks, {}, target=(SYMBOL, side), **kwargs)
    assert not absent and not selected_absent
    assert len(selected) == 1
    assert selected[0].payload == next(r.payload for r in full if r.scope.symbol in (None, SYMBOL))
    expected = next(d for d in evaluate(full) if d.scope.symbol in (None, SYMBOL))
    assert evaluate(selected) == (expected,)
    pairs = json.loads(selected[0].metadata)['snapshot']['pairs']
    assert {p['symbol'] for p in pairs} == ({SYMBOL} if mode == 'coin' else {SYMBOL, other})
    # Only coin has an independent peer. Aggregate modes must keep its missing
    # current quote as unavailability rather than silently evaluating a subset.
    del marks[other]
    selected, selected_absent = capture(value, marks, {}, target=(SYMBOL, side), **kwargs)
    assert bool(selected) == (mode == 'coin')
    assert bool(selected_absent) == (mode != 'coin')
