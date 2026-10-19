"""Causal account clocks through Python fact transport and the real native fold."""

from copy import deepcopy
import json

import passivbot_rust as pbr
import pytest

from live.hsl_budget import BudgetObservations


def fill(identity, timestamp, delta, realized=0.0, price=100.0):
    return dict(identity=identity, timestamp=timestamp, delta=delta, price=price,
                realized=realized, fee=0.0, sequence=0, revision=0)


def facts(now, *, size=1.0, mark=100.0, position_at=None, balance=1000.0):
    return dict(
        snapshot=dict(
            now=now, start=0, balance=balance, balance_at=now, config_at=now,
            max_current_age_ms=60_000, mode="coin", pside="long", symbol="A",
            pairs=[dict(
                symbol="A", position=dict(size=size, basis=100.0 if size else 0.0,
                    mark=mark, multiplier=1.0, inverse=False, pside="long"),
                position_at=now if position_at is None else position_at,
                mark_at=now, fills_started_at=None, fills_at=None,
                prices_at=now, prices={}, fills=[], revisions=[0, 0, 0, 0],
                fills_position_anchor=None,
            )],
        ),
        slots=1, scale_budget_with_excess_allowance=False, exposure_budget=None,
        memory_ema_span_minutes=43200.0, drawdown_ratio_ema_span_minutes=1.0, threshold=0.005, cooldown_ms=60_000,
        budget_observations=[dict(timestamp=0, order=-1, balance=1000.0, slots=1,
            scale_budget_with_excess_allowance=False, exposure_budget=None, estimated=True)],
    )


def evaluate(replay, value):
    return json.loads(replay.evaluate(json.dumps(value, allow_nan=False)))


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
def test_actual_capture_fill_payload_cannot_cross_its_independent_source_cut(mode):
    """The manager may contain a fill newer than the completed fetch receipt."""
    from live.hsl_runtime import capture, evaluate as evaluate_requests
    from live.market_snapshot import MarketSnapshot
    from test_hsl_inputs import event
    from test_hsl_runtime import NOW, SYMBOL, bot

    value, executions = bot(mode), []
    value._pnls_manager.get_events = lambda **_: list(executions)

    def read(now, cut):
        ledger = value._ensure_freshness_ledger()
        for surface in ("balance", "positions"):
            ledger.stamp(surface, now_ms=now)
        requests, missing = capture(
            value, {SYMBOL: MarketSnapshot(SYMBOL, 100., 100., 100., now, "fake")}, {},
            symbols={"long": [SYMBOL], "short": []}, now_ms=now, utc_now_ms=now,
            max_current_age_ms=10_000, fills_started_ms=cut-1, fills_completed_ms=cut,
        )
        assert not missing and len(requests) == 1
        (decision,) = evaluate_requests(requests)
        return requests[0], json.loads(decision.payload)

    read(NOW-2000, NOW-2000)
    executions.append(event(id="after-cut", timestamp=NOW-500, qty=None,
                            pnl=-100.0, price=100., fee_paid=0., c_mult=1.))
    request, excluded = read(NOW, NOW-1000)
    pair = json.loads(request.payload)["snapshot"]["pairs"][0]
    assert pair["fills"][0]["timestamp"] > pair["fills_at"]
    assert "post_capture_fill" in excluded["reasons"]
    assert excluded["risk"]["drawdown"] == 0.
    request, admitted = read(NOW, NOW)
    from v9_risk_reference import coefficients, fma
    _, beta = coefficients(json.loads(request.payload)["memory_ema_span_minutes"])
    expected = 100.
    # The authored fill is in the preceding minute; the completed current
    # minute contributes its ordinary zero-loss slow decay after admission.
    for _ in range(NOW // 60_000 - (NOW-500) // 60_000):
        expected = fma(beta, expected, 0.)
    assert admitted["risk"]["drawdown"] == expected
    assert admitted["decision"]["action"] == "panic"
    request.replay.clear_derived_cache()
    assert read(NOW, NOW)[1] == admitted


@pytest.mark.parametrize("kind", ["fill", "price"])
def test_native_forward_capture_cut_matches_only_the_supported_reference(kind):
    replay, reference = pbr.HslV9Replay(), pbr.HslV9Replay()
    for owner in (replay, reference):
        evaluate(owner, facts(1))
    value = facts(120_001)
    pair = value["snapshot"]["pairs"][0]
    if kind == "fill":
        pair.update(fills_at=100, fills=[fill("uncaptured", 110, 0., -100.)])
        expected_d, diagnostic = 100., "post_capture_fill"
        from v9_risk_reference import coefficients, fma
        _, beta = coefficients(value["memory_ema_span_minutes"])
        # After the minute-zero cashflow, the current preview crosses two
        # completed integer minutes with zero additional economic loss.
        for _ in range(value["snapshot"]["now"] // 60_000 - 110 // 60_000):
            expected_d = fma(beta, expected_d, 0.)
    else:
        pair.update(prices_at=59_999, prices={"60000": 150.})
        expected_d, diagnostic = 50., "post_capture_price"
    supported = deepcopy(value)
    supported["snapshot"]["pairs"][0].update(fills=[], prices={})
    result = evaluate(replay, value)
    assert result["risk"] == evaluate(reference, supported)["risk"]
    assert diagnostic in result["reasons"]
    pair.update(fills_at=120_001, prices_at=120_001)
    admitted = evaluate(replay, value)
    assert admitted["risk"]["drawdown"] == expected_d
    assert evaluate(replay, value) == admitted
    replay.clear_derived_cache()
    assert evaluate(replay, value) == admitted


def test_native_quarantines_a_retained_revision_before_clipping_or_latest_selection():
    replay = pbr.HslV9Replay()
    evaluate(replay, facts(1))
    earlier = facts(100)
    earlier["snapshot"]["pairs"][0]["fills"] = [fill("revision", 50, 0., -10.)]
    assert evaluate(replay, earlier)["risk"]["drawdown"] == 10.
    later = facts(200)
    later["snapshot"]["start"] = 60
    later["snapshot"]["pairs"][0].update(
        fills_at=100, fills=[fill("revision", 150, 0., -100.)],
    )
    excluded = evaluate(replay, later)
    assert excluded["risk"]["drawdown"] == 0.
    replay.clear_derived_cache()
    assert evaluate(replay, later) == excluded


def test_native_deferred_equal_anchor_fill_advances_after_the_immutable_initializer():
    replay = pbr.HslV9Replay()
    first = facts(100)
    first["snapshot"]["pairs"][0].update(
        fills_at=90, fills=[fill("after-cut-at-anchor", 100, 0., -100.)],
    )
    assert evaluate(replay, first)["risk"]["drawdown"] == 0.
    later = facts(200)
    later["snapshot"]["pairs"][0].update(
        fills_at=200, fills=first["snapshot"]["pairs"][0]["fills"],
    )
    admitted = evaluate(replay, later)
    assert admitted["risk"]["drawdown"] == 100.
    assert "ambiguous_bootstrap_fill_phase" in admitted["reasons"]
    replay.clear_derived_cache()
    assert evaluate(replay, later) == admitted


def test_native_later_receipt_repairs_prefix_without_retiming_original_capture():
    replay = pbr.HslV9Replay()
    first = facts(100)
    first["snapshot"]["pairs"][0].update(fills_at=90, prices_at=90)
    evaluate(replay, first)
    later = facts(200)
    later["snapshot"]["pairs"][0].update(
        fills_at=200, fills=[fill("later-historical-receipt", 95, 0., -100.)],
    )
    repaired = evaluate(replay, later)
    assert repaired["risk"]["drawdown"] == 100.
    replay.clear_derived_cache()
    assert evaluate(replay, later) == repaired
    assert first["snapshot"]["pairs"][0]["fills_at"] == 90


def test_native_revision_correction_advances_only_after_its_source_cut():
    replay = pbr.HslV9Replay()
    evaluate(replay, facts(1))
    value = facts(200)
    revised = fill("revision", 150, 0., -20.)
    revised["revision"] = 1
    value["snapshot"]["pairs"][0].update(
        fills_at=100, fills=[fill("revision", 50, 0., -10.), revised],
    )
    assert evaluate(replay, value)["risk"]["drawdown"] == 10.
    value["snapshot"]["pairs"][0]["fills_at"] = 200
    repaired = evaluate(replay, value)
    assert repaired["risk"]["drawdown"] == 20.
    replay.clear_derived_cache()
    assert evaluate(replay, value) == repaired


def test_native_regressed_cut_keeps_all_previously_admitted_conflicting_variants():
    replay = pbr.HslV9Replay()
    evaluate(replay, facts(1))
    variants = [fill("conflict", 95, 0., -10.), fill("conflict", 150, 0., -20.)]
    supported = facts(200)
    supported["snapshot"]["pairs"][0].update(fills_at=200, fills=variants)
    conflict = evaluate(replay, supported)
    assert "conflicting_identity" in conflict["reasons"]
    assert conflict["risk"]["drawdown"] == 0.
    older = facts(300)
    older["snapshot"]["pairs"][0].update(fills_at=100, fills=variants)
    repeated = evaluate(replay, older)
    assert "conflicting_identity" in repeated["reasons"]
    assert repeated["risk"]["drawdown"] == 0.
    replay.clear_derived_cache()
    assert evaluate(replay, older) == repeated


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("known_origin", [False, True])
def test_native_fill_between_anchor_source_and_capture_keeps_actual_economic_time(
    mode, partial, known_origin
):
    initial = facts(100, position_at=90)
    initial["snapshot"].update(mode=mode, symbol="A" if mode == "coin" else None,
                               pside=None if mode == "portfolio" else "long")
    initial["snapshot"]["pairs"][0].update(
        fills_at=90, prices={"0": 100.},
        fills=[fill("entry", 0, 1.)] if known_origin else [],
    )
    replay = pbr.HslV9Replay()
    evaluate(replay, initial)
    after = facts(200, size=.5 if partial else 0., mark=90.,
                  position_at=90 if partial else 200)
    after["snapshot"].update(mode=mode, symbol="A" if mode == "coin" else None,
                             pside=None if mode == "portfolio" else "long")
    after["snapshot"]["pairs"][0].update(
        fills_at=200, prices={"0": 100.},
        fills=([fill("entry", 0, 1.)] if known_origin else []) + [
            fill("exit", 95, -.5 if partial else -1., -5. if partial else -10., 90.)
        ],
    )
    if partial:
        # A stale actual source still reports the old pre-fill quantity. The
        # current quote may not use it to debit the same loss a second time.
        after["snapshot"]["pairs"][0]["position"]["size"] = 1.
        blocked = evaluate(replay, after)
        assert blocked["unavailable_reason"] == "causal_position_snapshot_precedes_fill"
        assert blocked["risk"] is None
        after["snapshot"]["pairs"][0].update(position_at=200)
        after["snapshot"]["pairs"][0]["position"]["size"] = .5
    result = evaluate(replay, after)
    assert result["risk"]["drawdown"] == 10.
    if known_origin:
        assert result["risk"] == evaluate(pbr.HslV9Replay(), after)["risk"]
    assert result["risk"]["cooldown_until"] == (None if partial else 60_095)
    if not partial:
        assert result["decision"]["flat_at"] == 95
    replay.clear_derived_cache()
    assert evaluate(replay, after) == result


@pytest.mark.parametrize("partial", [False, True])
def test_native_unknown_source_inventory_waits_for_the_execution_mark(partial):
    initial = facts(100, position_at=90)
    initial["snapshot"]["pairs"][0]["fills_at"] = 90
    replay = pbr.HslV9Replay()
    evaluate(replay, initial)
    after = facts(200, size=.5 if partial else 0., mark=90.)
    after["snapshot"]["pairs"][0].update(
        fills_at=200, fills=[fill("exit", 95, -.5 if partial else -1.,
                                  -5. if partial else -10., 90.)],
    )
    result = evaluate(replay, after)
    assert result["risk"]["drawdown"] == 10.
    assert "unknown_prefix_causal_mark_seed" in result["reasons"]
    assert result["risk"]["cooldown_until"] == (None if partial else 60_095)
    replay.clear_derived_cache()
    assert evaluate(replay, after) == result


def test_native_bootstrap_source_tail_requires_marks_for_the_entire_held_scope():
    initial = facts(100, position_at=90)
    initial["snapshot"].update(mode="portfolio", pside=None, symbol=None)
    initial["snapshot"]["pairs"][0]["fills_at"] = 90
    other = deepcopy(initial["snapshot"]["pairs"][0])
    other["symbol"] = "B"
    initial["snapshot"]["pairs"].append(other)
    initial["snapshot"]["pairs"][0]["prices"] = {"0": 100.}
    replay = pbr.HslV9Replay()
    evaluate(replay, initial)
    after = facts(200, size=.5, mark=90.)
    after["snapshot"].update(mode="portfolio", pside=None, symbol=None)
    after["snapshot"]["pairs"][0].update(
        fills=[fill("partial", 95, -.5, -5., 90.)], prices={"0": 100.},
    )
    other = facts(200, mark=80.)["snapshot"]["pairs"][0]
    other["symbol"] = "B"
    after["snapshot"]["pairs"].append(other)
    blocked = evaluate(replay, after)
    assert blocked["unavailable_reason"] == "causal_position_mark_unavailable"
    assert blocked["risk"] is None
    after["snapshot"]["pairs"][1]["prices"] = {"0": 100.}
    recovered = evaluate(replay, after)
    assert recovered["risk"]["drawdown"] == 30.
    replay.clear_derived_cache()
    assert evaluate(replay, after) == recovered


@pytest.mark.parametrize("partial", [False, True])
def test_native_terminal_unknown_bootstrap_uses_its_original_quote_at_its_own_clock(partial):
    initial = facts(100, position_at=90)
    initial["snapshot"].update(mode="portfolio", pside=None, symbol=None)
    initial["snapshot"]["pairs"][0]["fills_at"] = 90
    other = deepcopy(initial["snapshot"]["pairs"][0])
    other.update(symbol="B", position_at=100)
    initial["snapshot"]["pairs"].append(other)
    initial["snapshot"]["pairs"][0]["prices"] = {"0": 100.}
    replay = pbr.HslV9Replay()
    evaluate(replay, initial)
    after = facts(200, size=.5 if partial else 0., mark=90.)
    after["snapshot"].update(mode="portfolio", pside=None, symbol=None)
    after["snapshot"]["pairs"][0].update(
        fills=[fill("exit", 95, -.5 if partial else -1., -5. if partial else -10., 90.)],
        prices={"0": 100.},
    )
    other = facts(200, mark=80.)["snapshot"]["pairs"][0]
    other["symbol"] = "B"
    after["snapshot"]["pairs"].append(other)
    result = evaluate(replay, after)
    if partial:
        assert result["unavailable_reason"] == "causal_position_mark_unavailable"
        assert result["risk"] is None
    else:
        assert result["unavailable_reason"] is None
        assert result["risk"]["drawdown"] == 20.
        assert "unknown_prefix_causal_mark_seed" in result["reasons"]
    replay.clear_derived_cache()
    assert evaluate(replay, after) == result


@pytest.mark.parametrize("known_origin", [False, True])
def test_changed_prewrite_inventory_does_not_reopen_a_late_exit(known_origin):
    replay = pbr.HslV9Replay()
    initial = facts(1)
    entry = fill("entry", 0, 1.0)
    addition = fill("addition", 59_999, 1.0)
    if known_origin:
        initial["snapshot"]["pairs"][0]["fills"] = [entry]
    evaluate(replay, initial)
    before = facts(60_000, size=2.0, mark=90.0, position_at=59_999)
    if known_origin:
        before["snapshot"]["pairs"][0]["fills"] = [entry, addition]
    evaluate(replay, before)
    after = facts(90_000, size=0.0, mark=90.0)
    after["snapshot"]["pairs"][0]["fills"] = (
        [entry, addition] if known_origin else []
    ) + [fill("exit", 60_000, -2.0, -20.0, 90.0)]
    stopped = evaluate(replay, after)
    assert stopped["risk"]["cooldown_until"] == 120_000
    assert stopped["decision"]["flat_at"] == 60_000
    committed = replay.cache_info()["committed_events"]
    for now in (90_001, 90_002):
        after["snapshot"].update(now=now, balance_at=now, config_at=now)
        after["snapshot"]["pairs"][0].update(position_at=now, mark_at=now, prices_at=now)
        assert evaluate(replay, after)["risk"]["cooldown_until"] == 120_000
        assert replay.cache_info()["committed_events"] == committed
    stable = evaluate(replay, after)
    replay.clear_derived_cache()
    assert evaluate(replay, after) == stable


def test_real_budget_owner_keeps_a_prewrite_balance_before_the_late_fill():
    replay, budgets = pbr.HslV9Replay(), BudgetObservations()

    def account(now, balance, *, size=1.0, position_at=None, balance_at=None, executions=()):
        value = facts(now, size=size, mark=90.0 if now > 1 else 100.0,
                      position_at=position_at, balance=balance)
        value["snapshot"]["pairs"][0]["fills"] = list(executions)
        value["snapshot"]["balance_at"] = now if balance_at is None else balance_at
        value["budget_observations"] = budgets.capture(
            "A-long", start=0, now=now, balance=balance, slots=1, scale=False,
            exposure=None, balance_at=value["snapshot"]["balance_at"],
        )
        return value

    entry = fill("entry", 0, 1.0)
    evaluate(replay, account(1, 1000.0, executions=[entry]))
    before = account(60_000, 10_000.0, position_at=59_999,
                     balance_at=59_999, executions=[entry])
    assert before["budget_observations"][-1]["timestamp"] == 59_999
    assert evaluate(replay, before)["decision"]["action"] == "normal"
    after = account(90_000, 10_000.0, size=0.0,
                    executions=[entry, fill("exit", 60_000, -1.0, -10.0, 90.0)])
    stopped = evaluate(replay, after)
    assert stopped["decision"]["action"] == "normal"
    assert stopped["risk"]["cooldown_until"] is None
    replay.clear_derived_cache()
    assert evaluate(replay, after) == stopped


def test_an_unpriceable_new_aggregate_contributor_has_no_permission_until_corrected():
    replay = pbr.HslV9Replay()
    initial = facts(60_000)
    initial["snapshot"].update(mode="portfolio", pside=None, symbol=None)
    evaluate(replay, initial)
    current = facts(120_001)
    current["snapshot"].update(mode="portfolio", pside=None, symbol=None)
    other = deepcopy(current["snapshot"]["pairs"][0])
    other.update(symbol="B", position_at=119_999)
    other["position"]["mark"] = 90.0
    current["snapshot"]["pairs"].append(other)
    unavailable = evaluate(replay, current)
    assert unavailable["unavailable_reason"] == "causal_position_mark_unavailable"
    assert unavailable["decision"] is unavailable["risk"] is None
    replay.clear_derived_cache()
    assert evaluate(replay, current) == unavailable
    other["prices"] = {"60000": 95.0}
    repaired = evaluate(replay, current)
    assert repaired["unavailable_reason"] is None
    assert repaired["risk"]["exposed"]


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("mode", ["coin", "portfolio"])
def test_partial_fill_stale_quantity_is_unavailable_then_recovers_without_double_loss(side, mode):
    replay = pbr.HslV9Replay()
    direction = 1 if side == "long" else -1

    def account(now, size, mark):
        value = facts(now, size=direction * size, mark=mark)
        value["threshold"] = 0.012
        value["snapshot"]["pside"] = side
        value["snapshot"]["pairs"][0]["position"]["pside"] = side
        if mode == "portfolio":
            value["snapshot"].update(mode=mode, pside=None, symbol=None)
            other = deepcopy(value["snapshot"]["pairs"][0])
            other["symbol"] = "B"
            other["position"].update(size=direction, mark=100.0)
            value["snapshot"]["pairs"].append(other)
        return value

    evaluate(replay, account(1, 1.0, 100.0))
    committed = replay.cache_info()["committed_events"]
    advanced = replay.cache_info()["advanced_events"]
    mark = 90.0 if side == "long" else 110.0
    execution = fill("partial", 60_000, -0.5 * direction, -5.0, mark)
    stale = account(60_001, 1.0, mark)
    stale["snapshot"]["pairs"][0].update(position_at=59_999, fills=[execution])
    unavailable = evaluate(replay, stale)
    assert unavailable["unavailable_reason"] == "causal_position_snapshot_precedes_fill"
    assert unavailable["decision"] is unavailable["risk"] is None
    assert replay.cache_info()["committed_events"] == committed
    assert replay.cache_info()["advanced_events"] == advanced
    replay.clear_derived_cache()
    assert evaluate(replay, stale) == unavailable
    fresh = account(60_002, 0.5, mark)
    fresh["snapshot"]["pairs"][0]["fills"] = [execution]
    recovered = evaluate(replay, fresh)
    assert recovered["unavailable_reason"] is None
    assert recovered["decision"]["action"] == "normal"
    assert recovered["risk"]["drawdown"] == 10.0
    replay.clear_derived_cache()
    assert evaluate(replay, fresh) == recovered


def test_live_owner_routes_native_causal_unavailability_without_publishing_old_permission(monkeypatch):
    from live.hsl_candles import Sources
    from live.hsl_inputs import capture_candles
    from live.hsl_live import Owner
    from live.market_snapshot import MarketSnapshot
    from test_hsl_runtime import NOW, SYMBOL, bot
    import utils

    value = bot("portfolio")
    now = [NOW]
    monkeypatch.setattr(utils, "utc_ms", lambda: now[0])
    value.get_exchange_time = lambda: now[0]
    value._live_market_snapshot_max_age_ms = lambda: 10_000
    value.approved_coins_minus_ignored_coins = {"long": {SYMBOL}, "short": set()}
    owner = Owner(value)
    first = owner.capture({SYMBOL: MarketSnapshot(SYMBOL, 100, 100, 100, NOW - 1, "fake")})
    assert first.permission(SYMBOL, "long")[0] == "normal"
    now[0] += 1000
    other = "OTHER/USDT:USDT"
    value.positions[other] = {"long": dict(size=10.0, price=100.0),
                              "short": dict(size=0.0, price=0.0)}
    value.c_mults[other] = 1.0
    value.qty_steps[other] = 0.1
    ledger = value._ensure_freshness_ledger()
    ledger.stamp("positions", now_ms=NOW + 900)
    ledger.stamp("balance", now_ms=NOW + 900)
    quotes = {
        SYMBOL: MarketSnapshot(SYMBOL, 100, 100, 100, NOW + 999, "fake"),
        other: MarketSnapshot(other, 90, 90, 90, NOW + 999, "fake"),
    }
    unavailable = owner.capture(quotes)
    assert unavailable.decisions == ()
    assert unavailable.permission(SYMBOL, "long") == ("unavailable", None)
    assert unavailable.permission(other, "long") == ("unavailable", None)
    assert unavailable.unavailable[0].reason == "causal_position_mark_unavailable"
    owner.sources[other] = Sources((capture_candles(
        [dict(ts=NOW - 60_000, o=95.0, h=95.0, l=95.0, c=95.0)],
        minutes=1, observed_at=now[0]),), (), 0)
    repaired = owner.capture(quotes)
    assert not repaired.unavailable
    assert repaired.permission(other, "long")[0] == "panic"


@pytest.mark.parametrize("change", [
    dict(unavailable_reason="unknown_reason"),
    dict(observations=True),
    dict(episodes=False),
    dict(reasons="reason"),
    dict(events=[{}]),
    dict(risk={}),
    dict(decision={}),
])
def test_malformed_native_unavailability_remains_a_fatal_producer_error(change):
    from types import SimpleNamespace
    from live.hsl_runtime import InvalidHslOutput, Request, Scope, evaluate as evaluate_requests

    output = dict(unavailable_reason="causal_position_mark_unavailable", decision=None,
                  risk=None, events=[], observations=0, episodes=0, reasons=[],
                  continuity=dict(initialized=True, bootstrap_origin_ms=0,
                      bootstrap_observed_ms=1, retained_from_ms=0, boundary_at_ms=None,
                      retention_horizon_ms=60_000, reconstruction="bounded_bootstrap"))
    output.update(change)
    request = Request(Scope("coin", "long", "A"), json.dumps(facts(1)), "limit", (), (), (),
                      SimpleNamespace(evaluate_grids=lambda *_: json.dumps(output)))
    with pytest.raises(InvalidHslOutput):
        evaluate_requests((request,))


@pytest.mark.parametrize("sources", [(59_999, 60_001), (60_001, 59_999)])
def test_same_receipt_actual_source_endpoints_preserve_reconciliation_and_pre_red(sources):
    replay = pbr.HslV9Replay()
    initial = facts(1)
    initial["threshold"] = 0.012
    evaluate(replay, initial)
    partial = fill("partial", 60_000, -0.5, -5.0, 90.0)
    account = facts(60_001, mark=90.0)
    account["threshold"] = 0.012
    account["snapshot"]["pairs"][0]["fills"] = [partial]
    for index, source in enumerate(sources):
        account["snapshot"]["pairs"][0]["position_at"] = source
        current = evaluate(replay, account)
        if index == 0 and source == 59_999:
            assert current["unavailable_reason"] == "causal_position_snapshot_precedes_fill"
        else:
            assert current["unavailable_reason"] is None
            assert current["risk"]["drawdown"] == 15.0
    committed = replay.cache_info()["committed_events"]
    advanced = replay.cache_info()["advanced_events"]
    stable = evaluate(replay, account)
    assert replay.cache_info()["committed_events"] == committed
    assert replay.cache_info()["advanced_events"] == advanced
    replay.clear_derived_cache()
    assert evaluate(replay, account) == stable
    flat = facts(60_002, size=0.0, mark=90.0)
    flat["threshold"] = 0.012
    flat["snapshot"]["pairs"][0]["fills"] = [
        partial, fill("rebated_exit", 60_002, -1.0, 0.0, 90.0)
    ]
    terminal = evaluate(replay, flat)
    assert terminal["risk"]["drawdown"] == 5.0
    assert terminal["risk"]["cooldown_until"] == 120_002
    assert terminal["decision"]["flat_at"] == 60_002
    replay.clear_derived_cache()
    assert evaluate(replay, flat) == terminal


@pytest.mark.parametrize("processing_delay", [1, 2])
def test_real_utc_capture_latency_cannot_retime_monotonic_account_facts(processing_delay):
    from live.hsl_runtime import capture, evaluate as evaluate_requests
    from live.market_snapshot import MarketSnapshot
    from test_hsl_inputs import event
    from test_hsl_runtime import NOW, SYMBOL, bot

    value, executions = bot(), []
    value.config["bot"]["long"]["hsl"]["red_threshold"] = 0.012
    value._pnls_manager.get_events = lambda **_: list(executions)

    def read(utc, now, source, size, mark):
        value.positions[SYMBOL]["long"].update(size=size, price=100.0 if size else 0.0)
        ledger = value._ensure_freshness_ledger()
        ledger.stamp("positions", now_ms=source)
        ledger.stamp("balance", now_ms=utc)
        requests, missing = capture(
            value, {SYMBOL: MarketSnapshot(SYMBOL, mark, mark, mark, utc, "fake")}, {},
            symbols={"long": [SYMBOL], "short": []}, now_ms=now, utc_now_ms=utc,
            max_current_age_ms=10_000,
        )
        assert not missing
        snapshot = json.loads(requests[0].payload)["snapshot"]
        assert snapshot["pairs"][0]["position_at"] == source
        assert snapshot["balance_at"] == utc
        (decision,) = evaluate_requests(requests)
        return json.loads(decision.payload), requests[0].replay

    read(NOW - 60_000, NOW - 60_000, NOW - 60_001, 1.0, 100.0)
    read(NOW, NOW + processing_delay, NOW - 1, 2.0, 90.0)
    read(NOW + 3, NOW + 3, NOW, 1.0, 90.0)
    executions.append(event(id="exit", timestamp=NOW + 4, side="sell", qty=-1.0,
                            price=90.0, pnl=-10.0, fee_paid=0.0, c_mult=1.0))
    final, replay = read(NOW + 5, NOW + 5, NOW + 5, 0.0, 90.0)
    assert final["decision"]["action"] == "normal"
    assert final["risk"]["cooldown_until"] is None
    replay.clear_derived_cache()
    assert read(NOW + 5, NOW + 5, NOW + 5, 0.0, 90.0)[0] == final


@pytest.mark.parametrize("synthetic", [False, True])
def test_owner_uses_utc_unless_the_harness_explicitly_supplies_a_clock_projection(monkeypatch, synthetic):
    import utils
    from live import hsl_runtime
    from live.hsl_live import Owner
    from live.market_snapshot import MarketSnapshot
    from test_hsl_runtime import NOW, SYMBOL, bot

    value = bot()
    utc = NOW - 86_400_000 if synthetic else NOW
    now = NOW if synthetic else NOW + 2
    monkeypatch.setattr(utils, "utc_ms", lambda: utc)
    value.get_exchange_time = lambda: now
    value._live_market_snapshot_max_age_ms = lambda: 10_000
    value.approved_coins_minus_ignored_coins = {"long": {SYMBOL}, "short": set()}
    for surface in ("positions", "balance"):
        value._ensure_freshness_ledger().stamp(surface, now_ms=utc - 200)
    owner, projected = Owner(value), []
    captured, capture = [], hsl_runtime.capture

    def observe_capture(*args, **kwargs):
        result = capture(*args, **kwargs)
        captured.extend(result[0])
        return result

    monkeypatch.setattr(hsl_runtime, "capture", observe_capture)
    if synthetic:
        def project(observed):
            projected.append(observed)
            return now - observed
        owner._observation_clock_offset = project
    wave = owner.capture({SYMBOL: MarketSnapshot(SYMBOL, 100, 100, 100, utc - 100, "fake")})
    assert not wave.unavailable
    assert wave.permission(SYMBOL, "long")[0] == "normal"
    replay = next(iter(value._hsl_factual_replays._scopes.values()))
    snapshot = json.loads(captured[0].metadata)["snapshot"]
    assert replay._pending is None
    expected = now if synthetic else utc
    assert snapshot["pairs"][0]["position_at"] == expected - 200
    assert snapshot["pairs"][0]["mark_at"] == expected - 100
    assert projected == ([utc] if synthetic else [])


@pytest.mark.parametrize("offset", [True, 1.0, None, "0", float("nan")])
def test_observation_clock_projection_requires_an_explicit_integer(offset):
    from live.hsl_runtime import capture
    from test_hsl_runtime import NOW, SYMBOL, bot, quotes

    with pytest.raises(ValueError, match="capture clock"):
        capture(bot(), quotes(), {}, symbols={"long": [SYMBOL], "short": []},
                now_ms=NOW, utc_now_ms=NOW, max_current_age_ms=10_000,
                observation_clock_offset_ms=offset)


@pytest.mark.parametrize("surface", ["positions", "balance"])
def test_impossible_future_projected_account_clock_is_rejected_without_clamping(surface):
    from live.hsl_runtime import capture
    from test_hsl_runtime import NOW, SYMBOL, bot, quotes

    value = bot()
    for name in ("positions", "balance"):
        value._ensure_freshness_ledger().stamp(name, now_ms=NOW - 10_000)
    value._ensure_freshness_ledger().stamp(surface, now_ms=NOW - 100)
    with pytest.raises(ValueError, match="projected HSL account clock"):
        capture(value, quotes(), {}, symbols={"long": [SYMBOL], "short": []},
                now_ms=NOW, utc_now_ms=NOW, max_current_age_ms=10_000,
                observation_clock_offset_ms=1000)


def test_corrected_native_position_sources_are_ordered_before_no_op_detection():
    results = []
    for readings in ([(40_000, 1.0), (50_000, 2.0), (60_000, 1.0)],
                     [(50_000, 2.0), (40_000, 1.0), (60_000, 1.0)]):
        replay = pbr.HslV9Replay()
        initial = facts(1)
        initial["threshold"] = 0.012
        evaluate(replay, initial)
        for index, (source, size) in enumerate(readings):
            account = facts(60_001 + index * 10_000, position_at=source, size=size, mark=90.0)
            account["threshold"] = 0.012
            evaluate(replay, account)
        flat = facts(90_001, size=0.0, mark=90.0)
        flat["threshold"] = 0.012
        flat["snapshot"]["pairs"][0]["fills"] = [fill("exit", 90_000, -1.0, -10.0, 90.0)]
        terminal = evaluate(replay, flat)
        assert terminal["risk"]["cooldown_until"] is None
        assert terminal["decision"]["action"] == "normal"
        replay.clear_derived_cache()
        assert evaluate(replay, flat) == terminal
        results.append(terminal["risk"])
    assert results[0] == results[1]


def test_explicit_harness_clock_configuration_survives_disposable_owner_replacement(monkeypatch):
    import utils
    from live.hsl_live import Owner
    from live.market_snapshot import MarketSnapshot
    from test_hsl_runtime import NOW, SYMBOL, bot

    value = bot()
    utc = NOW - 86_400_000
    monkeypatch.setattr(utils, "utc_ms", lambda: utc)
    value.get_exchange_time = lambda: NOW
    value._live_market_snapshot_max_age_ms = lambda: 10_000
    value.approved_coins_minus_ignored_coins = {"long": {SYMBOL}, "short": set()}
    for surface in ("positions", "balance"):
        value._ensure_freshness_ledger().stamp(surface, now_ms=utc - 200)
    project = value._hsl_observation_clock_offset = lambda observed: NOW - observed
    quote = {SYMBOL: MarketSnapshot(SYMBOL, 100, 100, 100, utc - 100, "fake")}
    first = Owner(value)
    before = first.capture(quote)
    replacement = Owner(value)
    assert replacement._observation_clock_offset is project
    after = replacement.capture(quote)
    assert not before.unavailable and not after.unavailable
    assert before.decisions[0].payload == after.decisions[0].payload
    assert before.permission(SYMBOL, "long") == after.permission(SYMBOL, "long")


def fake_account_clock_bot(monkeypatch, wall):
    import hjson
    import utils
    from exchanges.fake import FakeCCXTClient
    from live.hsl_runtime import observe_fills
    from live.market_snapshot import MarketSnapshotProvider
    from test_hsl_runtime import NOW, bot
    from test_run_fake_live import REPO_ROOT
    from tools.run_fake_live import _install_runtime_overrides

    value = bot()
    scenario = hjson.loads(
        (REPO_ROOT / "scenarios/fake_live/hsl_long_red_restart.hjson").read_text()
    )
    value.cca = FakeCCXTClient(scenario)
    value.cca.now_ms = NOW
    value._live_market_snapshot_max_age_ms = lambda: 10_000
    value.approved_coins_minus_ignored_coins = {"long": set(value.positions), "short": set()}
    monkeypatch.setattr(utils, "utc_ms", lambda: wall[0])
    value._record_authoritative_surface = lambda surface, signature: (
        value._ensure_freshness_ledger().stamp(surface, signature, now_ms=wall[0])
    )
    value.market_snapshot_provider = MarketSnapshotProvider(
        exchange_name="fake", fetch_tickers=None,
    )

    async def acquire_fill_receipt(**kwargs):
        value._hsl_fill_observation = observe_fills(value, (wall[0], wall[0]))
        return True

    value.update_pnls = acquire_fill_receipt
    _install_runtime_overrides(value, scenario)
    return value


def fake_account_clock_quotes(value, wall, price=100.):
    import asyncio
    from test_hsl_runtime import SYMBOL

    # These synchronous capture fixtures simulate successful acquisitions through
    # the installed wrappers; they must not create unregistered raw receipts.
    assert asyncio.run(value.update_pnls(source="hsl"))
    quote = value.market_snapshot_provider._snapshot_from_ticker(
        SYMBOL, dict(last=price, bid=price, ask=price), fetched_ms=wall[0],
    )
    assert quote is not None
    assert value._hsl_fact_clock_projector("quote", quote, quote.fetched_ms) == value.cca.now_ms
    return {SYMBOL: quote}


def test_fake_account_receipt_phase_is_independent_of_wall_jitter_and_owner_replacement(monkeypatch):
    from live.hsl_live import Owner
    from test_hsl_inputs import event
    from test_hsl_runtime import NOW, SYMBOL

    results = []
    for jitter in (1, 2):
        wall = [NOW + 86_400_000]
        value = fake_account_clock_bot(monkeypatch, wall)
        value.cca.now_ms = NOW - 60_000
        balance, executions = [200.0], []
        value.get_raw_balance = lambda: balance[0]
        value._pnls_manager.get_events = lambda **_: list(executions)
        value.positions[SYMBOL]["long"].update(size=5.0, price=100.0)
        for surface in ("positions", "balance"):
            value._record_authoritative_surface(surface, "before")
        owner = Owner(value)
        owner.capture(fake_account_clock_quotes(value, wall))
        value.cca.now_ms = NOW
        wall[0] += 10
        value.positions[SYMBOL]["long"].update(size=0.0, price=0.0)
        balance[0] = 185.0
        executions.append(event(id="exit", timestamp=NOW, side="sell", qty=-5.0,
                                price=97.0, pnl=-15.0, fee_paid=0.0, c_mult=1.0))
        for surface in ("positions", "balance"):
            value._record_authoritative_surface(surface, "after")
        wall[0] += jitter
        quote = fake_account_clock_quotes(value, wall, 97.)
        wave = owner.capture(quote)
        assert not wave.unavailable
        output = json.loads(wave.decisions[0].payload)
        facts_ = next(iter(value._hsl_budget_observations._scopes.values()))
        after = [fact for fact in facts_ if fact["balance"] == 185.0]
        assert after and all(fact["timestamp"] == NOW for fact in after)
        replacement = Owner(value)
        assert replacement._account_clock_projector is value._hsl_account_clock_projector
        assert replacement.capture(quote).decisions[0].payload == wave.decisions[0].payload
        value._hsl_factual_replays.clear_derived_caches()
        assert replacement.capture(quote).decisions[0].payload == wave.decisions[0].payload
        results.append(output)
    # Preserve all native trading/quality fields, including raw, EMA, budgets'
    # phase diagnostics and cooldown. Wall-clock timing cannot change these.
    assert results[0] == results[1]


def test_fake_account_clock_refuses_unknown_or_mismatched_receipt_identity(monkeypatch):
    from test_hsl_runtime import NOW

    wall = [NOW + 86_400_000]
    value = fake_account_clock_bot(monkeypatch, wall)
    value._record_authoritative_surface("positions", "actual")
    state = value._ensure_freshness_ledger().surfaces["positions"]
    project = value._hsl_account_clock_projector
    assert project("positions", state.revision, state.updated_ms) == NOW
    with pytest.raises(ValueError, match="unknown fake account observation"):
        project("positions", state.revision + 1, state.updated_ms)
    with pytest.raises(ValueError, match="unknown fake account observation"):
        project("positions", state.revision, state.updated_ms + 1)


def test_cached_position_read_keeps_its_receipt_clock_and_selects_only_an_actual_fresh_read(monkeypatch):
    from live.hsl_live import Owner
    from test_hsl_runtime import NOW

    wall = [NOW + 86_400_000]
    value = fake_account_clock_bot(monkeypatch, wall)
    for surface in ("positions", "balance"):
        value._record_authoritative_surface(surface, "first")
    owner = Owner(value)
    owner.capture(fake_account_clock_quotes(value, wall))
    first = owner._position_observation
    value.cca.now_ms += 60_000
    wall[0] += 1
    quote = fake_account_clock_quotes(value, wall)
    stale = owner.capture(quote)
    assert not stale.decisions and stale.unavailable
    assert value._hsl_account_clock_projector("positions", first.revision, first.observed_ms) == NOW
    for surface in ("positions", "balance"):
        value._record_authoritative_surface(surface, "fresh")
    fresh = owner.capture(quote)
    assert not fresh.unavailable
    current = owner._position_observation
    assert current.revision > first.revision
    assert value._hsl_account_clock_projector("positions", current.revision, current.observed_ms) == NOW + 60_000
    assert value._hsl_account_clock_projector("positions", first.revision, first.observed_ms) == NOW
