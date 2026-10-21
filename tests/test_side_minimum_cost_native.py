"""Real Rust admission covers independent sides, forager selection and held positions."""
from copy import deepcopy
from itertools import product

import pytest

from test_orchestrator_json_api import adaptive_strategy_params, bot_params_pair, compute, make_input, make_symbol


def snapshot(kind, flags, *, hedge=True, held=None, forager=False):
    bp = {"n_positions": 1, "total_wallet_exposure_limit": 1.0, "wallet_exposure_limit": 1.0,
          "entry_initial_qty_pct": 0.001, "risk_we_excess_allowance_pct": 0.0}
    strategy = (adaptive_strategy_params(entry={"initial_qty_pct": 0.001, "initial_ema_dist": 0.0})
                if kind == "trailing_martingale" else
                {"base_qty_pct": 0.001, "ema_span_0": 10.0, "ema_span_1": 20.0,
                 "offset": 0.0, "offset_psize_weight": 0.0})
    symbols = []
    for i in range(2 if forager else 1):
        kwargs = {f"{held}_pos_size": 0.1 if held == "long" else -0.1,
                  f"{held}_pos_price": 100.0} if held and i == 0 else {}
        symbol = make_symbol(i, bid=100.0, ask=100.0, effective_min_cost=50.0,
                             long_bp=bp, short_bp=bp, long_strategy=deepcopy(strategy),
                             short_strategy=deepcopy(strategy), **kwargs)
        symbol["exchange"]["min_cost"] = 50.0
        symbols.append(symbol)
    value = make_input(balance=1000.0, strategy_kind=kind,
                       global_bp=bot_params_pair(long_overrides=bp, short_overrides=bp), symbols=symbols)
    value["global"].update(hedge_mode=hedge, filter_by_min_effective_cost=dict(flags))
    return value


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("long,short", product((False, True), repeat=2))
@pytest.mark.parametrize("forager", [False, True])
def test_native_flat_initial_admission_is_side_specific(kind, long, short, forager):
    import passivbot_rust as pbr
    value = snapshot(kind, {"long": long, "short": short}, forager=forager)
    result = compute(pbr, value)
    for side, enabled in (("long", long), ("short", short)):
        entries = [o for o in result["orders"] if o["pside"] == side and o["order_type"].startswith("entry_")]
        assert bool(entries) is (not enabled)
        # Disabling only the eligibility filter never bypasses exchange minima.
        assert all(abs(o["qty"] * o["price"]) >= 50.0 for o in entries)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("passing_side", ["long", "short"])
def test_one_way_arbitration_uses_both_policies_before_choosing_side(kind, passing_side):
    import passivbot_rust as pbr
    flags = {side: side != passing_side for side in ("long", "short")}
    result = compute(pbr, snapshot(kind, flags, hedge=False))
    entries = [o for o in result["orders"] if o["order_type"].startswith("entry_")]
    assert entries and {o["pside"] for o in entries} == {passing_side}


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("held", ["long", "short"])
def test_minimum_cost_filter_does_not_change_held_position_management(kind, held):
    import passivbot_rust as pbr
    off = compute(pbr, snapshot(kind, {"long": False, "short": False}, held=held))
    on = compute(pbr, snapshot(kind, {"long": True, "short": True}, held=held))
    orders = lambda result: [o for o in result["orders"] if o["pside"] == held]
    assert orders(on) == orders(off)
    assert any(o["order_type"].startswith("close_") for o in orders(on))


@pytest.mark.parametrize("invalid", [None, True, {}, {"long": True}, {"long": True, "short": 1},
                                         {"long": True, "short": False, "extra": True}])
def test_orchestrator_requires_strict_complete_pair(invalid):
    import passivbot_rust as pbr
    value = snapshot("ema_anchor", {"long": True, "short": False})
    value["global"]["filter_by_min_effective_cost"] = invalid
    with pytest.raises(ValueError):
        compute(pbr, value)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("long,short,override", product((False, True), (False, True), (None, False, True)))
def test_actual_cpu_replay_respects_each_side_and_nullable_override(kind, long, short, override):
    from backtest import build_backtest_payload, execute_backtest
    from tools.gpu_parity import build_parser, fixture_inputs

    args = build_parser().parse_args(["--fixture", kind, "--hsl", "disabled", "--sides", "both",
                                      "--coins", "2", "--bars", "240", "--seed", "2026"])
    source, candles, markets, btc, timestamps = fixture_inputs(args)
    source["backtest"]["filter_by_min_effective_cost"] = override
    for side, flag in (("long", long), ("short", short)):
        bot = source["bot"][side]
        bot["universe"]["filter_by_min_effective_cost"] = flag
        bot["risk"]["max_positions"] = 1
        strategy = bot["strategy"][kind]
        strategy["ema_span_0_minutes"], strategy["ema_span_1_minutes"] = 2.0, 3.0
        if kind == "ema_anchor":
            strategy.update(base_qty_fraction=0.001, offset_fraction=0.0)
        else:
            strategy["entry"].update(initial_qty_fraction=0.001, initial_ema_distance_fraction=0.0)
    for coin, market in markets.items():
        if coin != "__meta__":
            market["min_cost"] = 100.0
    payload = build_backtest_payload(candles, markets, source, args.exchange, btc, timestamps)
    expected = {"long": long, "short": short} if override is None else dict.fromkeys(("long", "short"), override)
    assert payload.backtest_params["filter_by_min_effective_cost"] == expected
    fills, _, _ = execute_backtest(payload, source)
    entry_types = {str(row[13]) for row in fills if str(row[13]).startswith("entry_")}
    for side in ("long", "short"):
        assert any(value.endswith("_" + side) for value in entry_types) is (not expected[side])


@pytest.mark.parametrize("invalid", [None, True, {}, {"long": True}, {"long": True, "short": 1},
                                         {"long": True, "short": False, "extra": True}])
def test_backtest_native_packet_rejects_missing_or_malformed_pair(invalid):
    import passivbot_rust as pbr
    from backtest import build_backtest_payload
    from tools.gpu_parity import build_parser, fixture_inputs
    args = build_parser().parse_args(["--fixture", "ema_anchor", "--hsl", "disabled", "--sides", "both",
                                      "--coins", "2", "--bars", "65", "--seed", "2026"])
    source, candles, markets, btc, timestamps = fixture_inputs(args)
    payload = build_backtest_payload(candles, markets, source, args.exchange, btc, timestamps)
    params = deepcopy(payload.backtest_params)
    params["filter_by_min_effective_cost"] = invalid
    with pytest.raises(ValueError, match="filter_by_min_effective_cost"):
        pbr.run_backtest_bundle(payload.bundle, payload.bot_params_list, payload.strategy_params_list,
                               payload.exchange_params, params)
