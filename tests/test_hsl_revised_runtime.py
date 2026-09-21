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
    value = SimpleNamespace(config=config, positions=positions, inverse=False, coin_overrides={},
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
    native = runtime.pbr.hsl_revised_evaluate
    calls = []
    def malformed(request):
        calls.append(request)
        output = json.loads(native(request))
        if len(calls) == 2:
            output["decision"] = None
        return json.dumps(output)
    monkeypatch.setattr(runtime.pbr, "hsl_revised_evaluate", malformed)
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
    native = runtime.pbr.hsl_revised_evaluate
    def malformed(request):
        result = json.loads(native(request))
        result["decision"][field] = value
        return json.dumps(result)
    monkeypatch.setattr(runtime.pbr, "hsl_revised_evaluate", malformed)
    with pytest.raises(InvalidHslOutput):
        run(bot())


def test_native_errors_use_the_bot_fatal_contract(monkeypatch):
    import live.hsl_revised_runtime as runtime
    from passivbot_exceptions import FatalBotException
    def invalid(_request):
        raise ValueError("unexpected native schema error")
    monkeypatch.setattr(runtime.pbr, "hsl_revised_evaluate", invalid)
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
    native = runtime.pbr.hsl_revised_evaluate
    def malformed(request):
        result = json.loads(native(request))
        result["decision"].update(changes)
        return json.dumps(result)
    monkeypatch.setattr(runtime.pbr, "hsl_revised_evaluate", malformed)
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
