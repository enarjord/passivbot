import copy
import json
import math

import pytest


@pytest.fixture(scope="module", autouse=True)
def require_real_passivbot_rust_module():
    import passivbot_rust as pbr

    if getattr(pbr, "__is_stub__", False):
        pytest.fail(
            "tests/test_orchestrator_json_api.py requires the real passivbot_rust extension; stub detected"
        )


def bot_params(**overrides):
    base = {
        "close_grid_markup_end": 0.01,
        "close_grid_markup_start": 0.01,
        "close_grid_qty_pct": 1.0,
        "close_trailing_retracement_pct": 0.0,
        "close_trailing_grid_ratio": 0.0,
        "close_trailing_qty_pct": 0.0,
        "close_trailing_threshold_pct": 0.0,
        "entry_grid_double_down_factor": 1.0,
        "entry_grid_spacing_volatility_weight": 0.0,
        "entry_grid_spacing_we_weight": 0.0,
        "entry_grid_spacing_pct": 0.02,
        "entry_volatility_ema_span_hours": 0.0,
        "entry_initial_ema_dist": 0.0,
        "entry_initial_qty_pct": 0.1,
        "entry_trailing_double_down_factor": 0.0,
        "entry_trailing_retracement_pct": 0.0,
        "entry_trailing_retracement_we_weight": 0.0,
        "entry_trailing_retracement_volatility_weight": 0.0,
        "entry_trailing_grid_ratio": 0.0,
        "entry_trailing_threshold_pct": 0.0,
        "entry_trailing_threshold_we_weight": 0.0,
        "entry_trailing_threshold_volatility_weight": 0.0,
        "filter_volatility_ema_span": 10.0,
        "filter_volatility_drop_pct": 0.0,
        "filter_volume_ema_span": 10.0,
        "filter_volume_drop_pct": 0.0,
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "n_positions": 1,
        "total_wallet_exposure_limit": 1.0,
        "wallet_exposure_limit": 1.0,
        "risk_wel_enforcer_threshold": 0.0,
        "risk_twel_enforcer_threshold": 0.0,
        "risk_we_excess_allowance_pct": 0.0,
        "unstuck_close_pct": 0.0,
        "unstuck_ema_dist": 0.0,
        "unstuck_loss_allowance_pct": 0.0,
        "unstuck_threshold": 0.0,
    }
    base.update(overrides)
    return base


def bot_params_pair(long_overrides=None, short_overrides=None):
    return {
        "long": bot_params(**(long_overrides or {})),
        "short": bot_params(
            **(
                {
                    "n_positions": 0,
                    "total_wallet_exposure_limit": 0.0,
                }
                | (short_overrides or {})
            )
        ),
    }


def trailing_bundle():
    return {
        "min_since_open": 0.0,
        "max_since_min": 0.0,
        "max_since_open": 0.0,
        "min_since_max": 0.0,
    }


def exchange_params(**overrides):
    base = {
        "qty_step": 0.01,
        "price_step": 0.01,
        "min_qty": 0.0,
        "min_cost": 0.0,
        "c_mult": 1.0,
    }
    base.update(overrides)
    return base


def ema_bundle(
    *,
    m1_close=None,
    m1_volume=None,
    m1_log_range=None,
    h1_close=None,
    h1_volume=None,
    h1_log_range=None,
):
    return {
        "m1": {
            "close": m1_close or [],
            "volume": m1_volume or [],
            "log_range": m1_log_range or [],
        },
        "h1": {
            "close": h1_close or [],
            "volume": h1_volume or [],
            "log_range": h1_log_range or [],
        },
    }


def make_symbol(
    symbol_idx: int,
    *,
    bid: float,
    ask: float,
    tradable=True,
    effective_min_cost=1.0,
    long_mode=None,
    short_mode=None,
    long_pos_size=0.0,
    long_pos_price=0.0,
    short_pos_size=0.0,
    short_pos_price=0.0,
    long_bp=None,
    short_bp=None,
    emas=None,
):
    return {
        "symbol_idx": symbol_idx,
        "order_book": {"bid": bid, "ask": ask},
        "exchange": exchange_params(),
        "tradable": tradable,
        "next_candle": None,
        "effective_min_cost": effective_min_cost,
        "emas": emas
        or ema_bundle(
            m1_close=[
                [10.0, bid],
                [20.0, bid],
                [math.sqrt(10.0 * 20.0), bid],
            ]
        ),
        "long": {
            "mode": long_mode,
            "position": {"size": long_pos_size, "price": long_pos_price},
            "trailing": trailing_bundle(),
            "bot_params": bot_params(**(long_bp or {})),
        },
        "short": {
            "mode": short_mode,
            "position": {"size": short_pos_size, "price": short_pos_price},
            "trailing": trailing_bundle(),
            "bot_params": bot_params(
                **(
                    {
                        "n_positions": 0,
                        "total_wallet_exposure_limit": 0.0,
                    }
                    | (short_bp or {})
                )
            ),
        },
    }


def make_input(*, balance: float, global_bp=None, symbols):
    return {
        "balance": balance,
        "balance_raw": balance,
        "global": {
            "filter_by_min_effective_cost": False,
            "unstuck_allowance_long": 0.0,
            "unstuck_allowance_short": 0.0,
            "sort_global": True,
            "global_bot_params": global_bp or bot_params_pair(),
        },
        "symbols": symbols,
        "peek_hints": None,
    }


def compute(pbr, inp: dict) -> dict:
    out_json = pbr.compute_ideal_orders_json(json.dumps(inp))
    return json.loads(out_json)


def test_json_rejects_invalid_order_book():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[make_symbol(0, bid=0.0, ask=1.0)],
    )
    with pytest.raises(ValueError, match="InvalidOrderBook|invalid order"):
        compute(pbr, inp)


def test_json_rejects_non_contiguous_symbol_idx():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[make_symbol(1, bid=100.0, ask=100.0)],
    )
    with pytest.raises(ValueError, match="NonContiguousSymbolIdx"):
        compute(pbr, inp)


def test_json_rejects_missing_ema():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[make_symbol(0, bid=100.0, ask=100.0, emas=ema_bundle(m1_close=[]))],
    )
    with pytest.raises(ValueError, match="MissingEma"):
        compute(pbr, inp)


def test_panic_mode_emits_close_panic_long():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[
            make_symbol(
                0,
                bid=95.0,
                ask=95.0,
                long_mode="panic",
                long_pos_size=1.5,
                long_pos_price=100.0,
            )
        ],
    )
    out = compute(pbr, inp)
    assert len(out["orders"]) == 1
    o = out["orders"][0]
    assert o["symbol_idx"] == 0
    assert o["pside"] == "long"
    assert o["order_type"] == "close_panic_long"
    assert o["qty"] < 0.0


def test_graceful_stop_blocks_initial_entries_only():
    import passivbot_rust as pbr

    # No position => no entries.
    inp_no_pos = make_input(
        balance=1_000.0,
        symbols=[make_symbol(0, bid=100.0, ask=100.0, long_mode="graceful_stop")],
    )
    out_no_pos = compute(pbr, inp_no_pos)
    assert out_no_pos["orders"] == []

    # With a position, GracefulStop behaves like Normal.
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
    )
    inp_normal = make_input(balance=1_000.0, symbols=[{**sym, "long": {**sym["long"], "mode": None}}])
    inp_gs = make_input(
        balance=1_000.0, symbols=[{**sym, "long": {**sym["long"], "mode": "graceful_stop"}}]
    )
    out_normal = compute(pbr, inp_normal)
    out_gs = compute(pbr, inp_gs)
    assert out_normal == out_gs
    assert any(o["order_type"].startswith("close_") for o in out_gs["orders"])


def test_forager_respects_n_positions_selects_one_coin():
    import passivbot_rust as pbr

    global_bp = bot_params_pair(
        long_overrides={
            "n_positions": 1,
            "total_wallet_exposure_limit": 1.0,
            "filter_volume_drop_pct": 0.5,
            "filter_volatility_drop_pct": 0.0,
            "filter_volume_ema_span": 10.0,
            "filter_volatility_ema_span": 10.0,
        }
    )

    sym0 = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        emas=ema_bundle(
            m1_close=[
                [10.0, 100.0],
                [20.0, 100.0],
                [math.sqrt(10.0 * 20.0), 100.0],
            ],
            m1_volume=[[10.0, 10.0]],
            m1_log_range=[[10.0, 0.1]],
        ),
        long_bp={
            "filter_volume_drop_pct": 0.5,
            "filter_volatility_drop_pct": 0.0,
            "filter_volume_ema_span": 10.0,
            "filter_volatility_ema_span": 10.0,
        },
    )
    sym1 = make_symbol(
        1,
        bid=100.0,
        ask=100.0,
        emas=ema_bundle(
            m1_close=[
                [10.0, 100.0],
                [20.0, 100.0],
                [math.sqrt(10.0 * 20.0), 100.0],
            ],
            m1_volume=[[10.0, 11.0]],
            m1_log_range=[[10.0, 0.2]],
        ),
        long_bp={
            "filter_volume_drop_pct": 0.5,
            "filter_volatility_drop_pct": 0.0,
            "filter_volume_ema_span": 10.0,
            "filter_volatility_ema_span": 10.0,
        },
    )

    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym0, sym1])
    out = compute(pbr, inp)
    assert out["orders"], "expected at least one order"
    assert {o["symbol_idx"] for o in out["orders"]} == {1}


def test_json_output_is_deterministic():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[make_symbol(0, bid=100.0, ask=100.0, long_pos_size=1.0, long_pos_price=100.0)],
    )
    out1 = compute(pbr, inp)
    out2 = compute(pbr, inp)
    assert out1 == out2


def test_unstuck_is_added_in_addition_to_close_grid_and_capped():
    import passivbot_rust as pbr

    balance = 1_000.0
    long_bp = {
        "unstuck_close_pct": 0.5,
        "unstuck_threshold": 0.001,
        "unstuck_ema_dist": 0.0,
        "unstuck_loss_allowance_pct": 0.01,
        "close_grid_qty_pct": 1.0,
        "close_grid_markup_start": 0.01,
        "close_grid_markup_end": 0.01,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)

    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=10.0,
        long_pos_price=100.0,
        long_bp=long_bp,
        emas=ema_bundle(
            m1_close=[
                [10.0, 1.0],
                [20.0, 1.0],
                [math.sqrt(10.0 * 20.0), 1.0],
            ]
        ),
    )
    inp = make_input(balance=balance, global_bp=global_bp, symbols=[sym])
    inp["global"]["unstuck_allowance_long"] = 1e9

    out = compute(pbr, inp)
    order_types = [o["order_type"] for o in out["orders"]]
    assert "close_unstuck_long" in order_types
    assert "close_grid_long" in order_types

    closes = [
        o for o in out["orders"] if o["order_type"].startswith("close_") and o["pside"] == "long"
    ]
    total_close_qty = -sum(o["qty"] for o in closes if o["qty"] < 0.0)
    assert total_close_qty <= 10.0 + 1e-9


def test_orders_include_entries_and_closes():
    import passivbot_rust as pbr

    long_bp = {
        "entry_initial_qty_pct": 0.1,
        "entry_grid_spacing_pct": 0.01,
        "close_grid_qty_pct": 1.0,
        "close_grid_markup_start": 0.01,
        "close_grid_markup_end": 0.01,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    out = compute(pbr, inp)
    order_types = {o["order_type"] for o in out["orders"]}
    assert any(t.startswith("entry_") for t in order_types)
    assert any(t.startswith("close_") for t in order_types)


def test_twel_entry_gating_blocks_new_entries():
    import passivbot_rust as pbr

    long_bp = {
        "entry_initial_qty_pct": 0.1,
        "entry_grid_spacing_pct": 0.01,
        "total_wallet_exposure_limit": 0.1,
        "wallet_exposure_limit": 0.1,
        "n_positions": 1,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    out = compute(pbr, inp)
    assert not any(o["order_type"].startswith("entry_") for o in out["orders"])


def test_twel_enforcer_emits_auto_reduce():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.4,
        "total_wallet_exposure_limit": 0.9,
        "risk_twel_enforcer_threshold": 1.0,
        "n_positions": 2,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym0 = make_symbol(
        0,
        bid=50.0,
        ask=50.0,
        long_pos_size=8.0,
        long_pos_price=50.0,
        long_bp=long_bp,
    )
    sym1 = make_symbol(
        1,
        bid=50.0,
        ask=50.0,
        long_pos_size=12.0,
        long_pos_price=50.0,
        long_bp=long_bp,
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym0, sym1])
    out = compute(pbr, inp)
    assert any(o["order_type"] == "close_auto_reduce_twel_long" for o in out["orders"])


# ---------------------------------------------------------------------------
# balance_raw semantics tests
# ---------------------------------------------------------------------------


def test_realized_loss_gate_uses_balance_raw():
    """Realized-loss gate peak/floor uses balance_raw, not snapped balance."""
    import passivbot_rust as pbr

    # Position underwater: entry 100, bid 80 → close would realize loss.
    long_bp = {
        "wallet_exposure_limit": 0.5,
        "risk_wel_enforcer_threshold": 1.0,
        "total_wallet_exposure_limit": 1.0,
        "n_positions": 1,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)

    sym = make_symbol(
        0,
        bid=80.0,
        ask=80.0,
        long_pos_size=10.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )

    # With snapped balance=1000 and raw balance=980, a tight gate (0.001),
    # the gate computes peak from balance_raw: peak = 980 + (50-(-20)) = 1050
    # floor = 1050 * (1 - 0.001) ≈ 1048.95.  Projected balance after
    # realizing the loss would be well below floor → gate blocks.
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp["balance_raw"] = 980.0
    inp["global"]["max_realized_loss_pct"] = 0.001  # very tight gate
    inp["global"]["realized_pnl_cumsum_max"] = 50.0
    inp["global"]["realized_pnl_cumsum_last"] = -20.0

    out = compute(pbr, inp)
    loss_gate_blocks = out.get("diagnostics", {}).get("loss_gate_blocks", [])
    assert len(loss_gate_blocks) > 0, (
        "expected tight loss gate (0.001) to block close orders on underwater position, "
        f"but got no loss_gate_blocks. orders: {[o['order_type'] for o in out['orders']]}"
    )


def test_balance_raw_absent_falls_back_to_balance():
    """When balance_raw is absent, Rust falls back to balance (NaN default)."""
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[make_symbol(0, bid=100.0, ask=100.0)],
    )
    # Remove balance_raw entirely - Rust default is NaN which falls back to balance
    inp.pop("balance_raw")
    out = compute(pbr, inp)
    # Should work without error
    assert isinstance(out, dict)


def test_balance_raw_zero_gate_returns_early():
    """When balance_raw is 0.0, the loss gate returns early (non-positive guard)."""
    import passivbot_rust as pbr

    long_bp = {
        "close_grid_qty_pct": 1.0,
        "close_grid_markup_start": 0.01,
        "close_grid_markup_end": 0.01,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )

    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp["balance_raw"] = 0.0
    inp["global"]["max_realized_loss_pct"] = 0.05
    inp["global"]["realized_pnl_cumsum_max"] = 10.0
    inp["global"]["realized_pnl_cumsum_last"] = 5.0

    out = compute(pbr, inp)
    # Gate skips with non-positive balance_raw, close orders should still appear
    close_orders = [o for o in out["orders"] if o["order_type"].startswith("close_")]
    assert len(close_orders) > 0


def test_balance_raw_negative_gate_returns_early():
    """When balance_raw is -1.0, the loss gate returns early (non-positive guard)."""
    import passivbot_rust as pbr

    long_bp = {
        "close_grid_qty_pct": 1.0,
        "close_grid_markup_start": 0.01,
        "close_grid_markup_end": 0.01,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )

    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp["balance_raw"] = -1.0
    inp["global"]["max_realized_loss_pct"] = 0.05
    inp["global"]["realized_pnl_cumsum_max"] = 10.0
    inp["global"]["realized_pnl_cumsum_last"] = 5.0

    out = compute(pbr, inp)
    # Gate skips with negative balance_raw, close orders should still appear
    close_orders = [o for o in out["orders"] if o["order_type"].startswith("close_")]
    assert len(close_orders) > 0


def test_balance_raw_inf_rejected():
    """When balance_raw is inf, JSON serialization rejects it (not valid JSON)."""
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[make_symbol(0, bid=100.0, ask=100.0)],
    )
    inp["balance_raw"] = float("inf")
    # json.dumps with allow_nan=False raises ValueError; with allow_nan=True
    # the output is not valid JSON per spec. Either way, the Rust parser rejects it.
    with pytest.raises(ValueError):
        compute(pbr, inp)


def test_balance_raw_nan_rejected_by_json():
    """When balance_raw is NaN, JSON serialization produces invalid JSON that Rust rejects."""
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[make_symbol(0, bid=100.0, ask=100.0)],
    )
    inp["balance_raw"] = float("nan")
    # Python json.dumps encodes NaN as 'NaN' which is not valid JSON;
    # serde rejects it at parse time.
    with pytest.raises(ValueError):
        compute(pbr, inp)


def test_loss_gate_uses_balance_raw_when_snapped_and_raw_diverge():
    import passivbot_rust as pbr

    global_bp = bot_params_pair(
        long_overrides={
            "n_positions": 1,
            "total_wallet_exposure_limit": 1.0,
        }
    )
    sym = make_symbol(
        0,
        bid=80.0,
        ask=80.0,
        long_pos_size=10.0,
        long_pos_price=100.0,
        long_bp={
            "wallet_exposure_limit": 0.5,
            "risk_wel_enforcer_threshold": 1.0,
            "total_wallet_exposure_limit": 1.0,
            "n_positions": 1,
        },
    )
    inp_blocked = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp_blocked["global"]["max_realized_loss_pct"] = 0.01

    out_blocked = compute(pbr, inp_blocked)
    blocked_types = [o["order_type"] for o in out_blocked["orders"]]
    assert "close_auto_reduce_wel_long" not in blocked_types
    assert any(
        b.get("order_type") == "close_auto_reduce_wel_long"
        for b in out_blocked.get("diagnostics", {}).get("loss_gate_blocks", [])
    )

    inp_allowed = copy.deepcopy(inp_blocked)
    inp_allowed["balance_raw"] = 1_000_000.0
    out_allowed = compute(pbr, inp_allowed)
    allowed_types = [o["order_type"] for o in out_allowed["orders"]]
    assert "close_auto_reduce_wel_long" in allowed_types
    assert not out_allowed.get("diagnostics", {}).get("loss_gate_blocks")


def test_twel_enforcer_uses_balance_raw_not_snapped():
    """TWEL enforcer should use balance_raw for wallet exposure, not snapped balance."""
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.4,
        "total_wallet_exposure_limit": 0.9,
        "risk_twel_enforcer_threshold": 1.0,
        "n_positions": 2,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    # Two positions: cost = 8*50 + 12*50 = 400 + 600 = 1000.
    # With snapped balance 2000: total WE = 1000/2000 = 0.5 (under 0.9, no trigger).
    # With raw balance 1100: total WE = 1000/1100 ≈ 0.909 (over 0.9, triggers).
    sym0 = make_symbol(0, bid=50.0, ask=50.0, long_pos_size=8.0, long_pos_price=50.0, long_bp=long_bp)
    sym1 = make_symbol(
        1, bid=50.0, ask=50.0, long_pos_size=12.0, long_pos_price=50.0, long_bp=long_bp
    )
    inp = make_input(balance=2_000.0, global_bp=global_bp, symbols=[sym0, sym1])
    inp["balance_raw"] = 1_100.0

    out = compute(pbr, inp)
    order_types = [o["order_type"] for o in out["orders"]]
    assert "close_auto_reduce_twel_long" in order_types, (
        "TWEL enforcer should trigger with raw balance (WE=0.909>0.9), "
        f"not snapped (WE=0.5<0.9). Got: {order_types}"
    )


def test_loss_gate_returns_early_when_raw_is_non_positive():
    """Non-positive balance_raw causes the loss gate to early-return (gate disabled)."""
    import passivbot_rust as pbr

    global_bp = bot_params_pair(
        long_overrides={
            "n_positions": 1,
            "total_wallet_exposure_limit": 1.0,
        }
    )
    sym = make_symbol(
        0,
        bid=80.0,
        ask=80.0,
        long_pos_size=10.0,
        long_pos_price=100.0,
        long_bp={
            "wallet_exposure_limit": 0.5,
            "risk_wel_enforcer_threshold": 1.0,
            "total_wallet_exposure_limit": 1.0,
            "n_positions": 1,
        },
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp["global"]["max_realized_loss_pct"] = 0.01

    for raw_balance in [0.0, -1.0]:
        inp_case = copy.deepcopy(inp)
        inp_case["balance_raw"] = raw_balance
        out = compute(pbr, inp_case)
        order_types = [o["order_type"] for o in out["orders"]]
        assert "close_auto_reduce_wel_long" in order_types
        blocks = out.get("diagnostics", {}).get("loss_gate_blocks", [])
        assert not blocks
