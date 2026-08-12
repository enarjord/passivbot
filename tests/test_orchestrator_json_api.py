import copy
import json
import math

import pytest

from live import reconciler
from passivbot_exceptions import FatalBotException


def test_limit_price_step_failure_reports_exact_numeric_and_order_context():
    with pytest.raises(FatalBotException) as exc_info:
        reconciler._validate_rust_limit_price_exchange_constraints(
            12.34,
            (0.001, 0.1, 0.001, 5.0, 1.0),
            (
                "Rust orchestrator order 2 symbol_idx=4 "
                "order_type=entry_initial_normal_long"
            ),
        )

    message = str(exc_info.value)
    for expected in (
        "Rust orchestrator order 2",
        "symbol_idx=4",
        "order_type=entry_initial_normal_long",
        "price=12.34",
        "price_step=0.1",
        "nearest_price=12.3",
        "delta=",
        "tolerance=",
    ):
        assert expected in message


@pytest.fixture(scope="module", autouse=True)
def require_real_passivbot_rust_module():
    import passivbot_rust as pbr

    if getattr(pbr, "__is_stub__", False):
        pytest.fail(
            "tests/test_orchestrator_json_api.py requires the real passivbot_rust extension; stub detected"
        )


ADAPTIVE_STRATEGY_KEYS = {
    "close_grid_qty_pct",
    "close_trailing_retracement_pct",
    "close_trailing_qty_pct",
    "close_trailing_threshold_pct",
    "close_weight_volatility_1h",
    "close_weight_volatility_1m",
    "entry_grid_double_down_factor",
    "entry_grid_spacing_pct",
    "entry_volatility_ema_span_1h",
    "entry_volatility_ema_span_1m",
    "entry_weight_volatility_1h",
    "entry_weight_volatility_1m",
    "entry_we_weight",
    "entry_initial_ema_dist",
    "entry_initial_qty_pct",
    "entry_trailing_double_down_factor",
    "entry_trailing_retracement_pct",
    "entry_trailing_threshold_pct",
    "ema_span_0",
    "ema_span_1",
}


def _set_nested(mapping, path, value):
    current = mapping
    for part in path[:-1]:
        current = current.setdefault(part, {})
    current[path[-1]] = value


LEGACY_STRATEGY_KEY_MAP = {
    "close_grid_qty_pct": ("close", "qty_pct"),
    "close_trailing_retracement_pct": ("close", "retracement_base_pct"),
    "close_trailing_qty_pct": ("close", "qty_pct"),
    "close_trailing_threshold_pct": ("close", "threshold_base_pct"),
    "close_weight_volatility_1h": ("close", "threshold_volatility_1h_weight"),
    "close_weight_volatility_1m": ("close", "threshold_volatility_1m_weight"),
    "entry_grid_double_down_factor": ("entry", "double_down_factor"),
    "entry_grid_spacing_pct": ("entry", "threshold_base_pct"),
    "entry_volatility_ema_span_1h": ("volatility_ema_span_1h",),
    "entry_volatility_ema_span_1m": ("volatility_ema_span_1m",),
    "entry_weight_volatility_1h": ("entry", "threshold_volatility_1h_weight"),
    "entry_weight_volatility_1m": ("entry", "threshold_volatility_1m_weight"),
    "entry_we_weight": ("entry", "threshold_we_weight"),
    "entry_initial_ema_dist": ("entry", "initial_ema_dist"),
    "entry_initial_qty_pct": ("entry", "initial_qty_pct"),
    "entry_trailing_double_down_factor": ("entry", "double_down_factor"),
    "entry_trailing_retracement_pct": ("entry", "retracement_base_pct"),
    "entry_trailing_threshold_pct": ("entry", "threshold_base_pct"),
}


def adaptive_strategy_params(**overrides):
    base = {
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "volatility_ema_span_1h": 0.0,
        "volatility_ema_span_1m": 60.0,
        "entry": {
            "double_down_factor": 1.0,
            "ema_gate_mode": "initial",
            "initial_ema_dist": 0.0,
            "initial_qty_pct": 0.1,
            "threshold_base_pct": 0.02,
            "threshold_we_weight": 0.0,
            "threshold_volatility_1h_weight": 0.0,
            "threshold_volatility_1m_weight": 0.0,
            "retracement_base_pct": 0.0,
            "retracement_we_weight": 0.0,
            "retracement_volatility_1h_weight": 0.0,
            "retracement_volatility_1m_weight": 0.0,
        },
        "close": {
            "qty_pct": 1.0,
            "threshold_base_pct": 0.01,
            "threshold_we_weight": 0.0,
            "threshold_volatility_1h_weight": 0.0,
            "threshold_volatility_1m_weight": 0.0,
            "retracement_base_pct": 0.0,
            "retracement_volatility_1h_weight": 0.0,
            "retracement_volatility_1m_weight": 0.0,
        },
    }
    for key, value in overrides.items():
        if key in LEGACY_STRATEGY_KEY_MAP:
            _set_nested(base, LEGACY_STRATEGY_KEY_MAP[key], value)
        elif "." in key:
            _set_nested(base, tuple(part for part in key.split(".") if part), value)
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


def trailing_grid_v7_strategy_params(*, close_overrides=None, **entry_overrides):
    entry = {
        "grid_double_down_factor": 1.0,
        "grid_spacing_pct": 0.02,
        "grid_spacing_we_weight": 0.0,
        "grid_spacing_volatility_weight": 0.0,
        "initial_ema_dist": 0.0,
        "initial_qty_pct": 0.1,
        "trailing_double_down_factor": 1.0,
        "trailing_grid_ratio": -0.072114,
        "trailing_retracement_pct": 0.037427,
        "trailing_retracement_we_weight": 0.0,
        "trailing_retracement_volatility_weight": 0.0,
        "trailing_threshold_pct": 0.01,
        "trailing_threshold_we_weight": 0.0,
        "trailing_threshold_volatility_weight": 0.0,
        "volatility_ema_span_hours": 1.0,
    }
    entry.update(entry_overrides)
    close = {
        "grid_markup_start": 0.01,
        "grid_markup_end": 0.01,
        "grid_qty_pct": 1.0,
        "trailing_grid_ratio": 0.0,
        "trailing_qty_pct": 1.0,
        "trailing_retracement_pct": 0.0,
        "trailing_threshold_pct": 0.0,
    }
    close.update(close_overrides or {})
    return {
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "entry": entry,
        "close": close,
    }


def _split_bot_and_adaptive_strategy_overrides(overrides):
    raw = dict(overrides or {})
    bot_overrides = {k: v for k, v in raw.items() if k not in ADAPTIVE_STRATEGY_KEYS}
    strategy_overrides = {k: v for k, v in raw.items() if k in ADAPTIVE_STRATEGY_KEYS}
    return bot_overrides, strategy_overrides


def bot_params(**overrides):
    base = {
        "filter_volatility_ema_span_1m": 10.0,
        "filter_volatility_drop_pct": 0.0,
        "filter_volume_ema_span_1m": 10.0,
        "filter_volume_drop_pct": 0.0,
        "n_positions": 1,
        "total_wallet_exposure_limit": 1.0,
        "wallet_exposure_limit": 1.0,
        "risk_wel_enforcer_threshold": 0.0,
        "risk_twel_entry_gate_enabled": True,
        "risk_twel_enforcer_policy": "reduce_overweight",
        "risk_twel_enforcer_threshold": 0.0,
        "risk_we_excess_allowance_pct": 0.0,
        "risk_entry_cooldown_minutes": 0.0,
        "unstuck_ema_gating_enabled": True,
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
        "maker_fee": 0.0002,
        "taker_fee": 0.00055,
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
            "close": (
                m1_close
                if m1_close is not None
                else [[10.0, 100.0], [20.0, 100.0], [math.sqrt(10.0 * 20.0), 100.0]]
            ),
            "volume": m1_volume if m1_volume is not None else [[10.0, 1_000.0]],
            "log_range": m1_log_range if m1_log_range is not None else [[10.0, 0.01]],
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
    long_strategy=None,
    short_strategy=None,
    emas=None,
):
    long_bot_overrides, long_strategy_overrides = _split_bot_and_adaptive_strategy_overrides(long_bp)
    short_bot_overrides, short_strategy_overrides = _split_bot_and_adaptive_strategy_overrides(short_bp)
    if long_strategy is None and long_strategy_overrides:
        long_strategy = adaptive_strategy_params(**long_strategy_overrides)
    if short_strategy is None and short_strategy_overrides:
        short_strategy = adaptive_strategy_params(**short_strategy_overrides)
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
            ],
            m1_volume=[[10.0, 1_000.0]],
            m1_log_range=[[10.0, 0.01]],
        ),
        "long": {
            "mode": long_mode,
            "position": {"size": long_pos_size, "price": long_pos_price},
            "trailing": trailing_bundle(),
            "bot_params": bot_params(**long_bot_overrides),
            "strategy_params": long_strategy,
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
                    | short_bot_overrides
                )
            ),
            "strategy_params": short_strategy,
        },
    }


def make_input(*, balance: float, global_bp=None, strategy_kind="trailing_martingale", symbols):
    if strategy_kind == "trailing_martingale":
        for symbol in symbols:
            for pside in ("long", "short"):
                current = symbol[pside].get("strategy_params")
                if current is None:
                    symbol[pside]["strategy_params"] = adaptive_strategy_params()
                else:
                    symbol[pside]["strategy_params"] = adaptive_strategy_params(**current)
    return {
        "balance": balance,
        "balance_raw": balance,
        "global": {
            "filter_by_min_effective_cost": False,
            "auto_unstuck_allowed": True,
            "unstuck_allowance_long": 0.0,
            "unstuck_allowance_short": 0.0,
            "max_realized_loss_pct": 1.0,
            "sort_global": True,
            "global_bot_params": global_bp or bot_params_pair(),
            "strategy_kind": strategy_kind,
        },
        "symbols": symbols,
        "peek_hints": None,
    }


def compute(pbr, inp: dict) -> dict:
    out_json = pbr.compute_ideal_orders_json(json.dumps(inp))
    return json.loads(out_json)


def test_live_validator_accepts_complete_rust_output():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[make_symbol(0, bid=100.0, ask=101.0)],
    )
    out, orders = reconciler.parse_and_validate_rust_orchestrator_output(
        pbr.compute_ideal_orders_json(json.dumps(inp)),
        {0: "BTC/USDT:USDT"},
        inp,
    )

    assert orders == out["orders"]
    assert orders
    conversion_identities = {
        reconciler.rust_order_conversion_identity(
            "BTC/USDT:USDT", order["qty"], order["price"], order["order_type"]
        )
        for order in orders
    }
    assert len(conversion_identities) == len(orders)
    assert out["diagnostics"]["symbol_states"][0]["symbol_idx"] == 0
    assert "loss_gate_blocks" in out["diagnostics"]


def test_real_rust_preserves_tiny_aligned_exchange_minimum():
    import passivbot_rust as pbr

    assert pbr.calc_min_entry_qty_py(100.0, 1.0, 1e-12, 1e-12, 0.0) == 1e-12


def test_real_rust_ceils_minimum_genuinely_above_quantity_step():
    import passivbot_rust as pbr

    assert pbr.calc_min_entry_qty_py(4999.999975, 1.0, 0.001, 0.0, 5.0) == 0.002


def test_execution_validation_matches_rust_for_representation_noisy_aligned_book():
    import passivbot_rust as pbr

    book_price = 1e-6 * 100
    symbol = make_symbol(
        0,
        bid=book_price,
        ask=book_price,
        long_pos_size=1.0,
        long_pos_price=9.9e-05,
        long_strategy=adaptive_strategy_params(entry={"initial_qty_pct": 0.0}),
    )
    symbol["exchange"].update(
        price_step=0.0001,
        qty_step=0.1,
        min_qty=0.0,
        min_cost=0.0,
    )
    inp = make_input(balance=1_000.0, symbols=[symbol])
    inp["global"].update(
        market_orders_allowed=True,
        market_order_near_touch_threshold=0.0,
    )

    raw = pbr.compute_ideal_orders_json(json.dumps(inp))
    _, orders = reconciler.parse_and_validate_rust_orchestrator_output(
        raw,
        {0: "TEST/USDT:USDT"},
        inp,
    )

    close = next(order for order in orders if order["order_type"] == "close_grid_long")
    assert close["price"] == 0.0001
    assert close["execution_type"] == "market"


@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "trailing_grid_v7"])
def test_short_close_rounds_to_positive_minimum_tick(strategy_kind):
    import passivbot_rust as pbr

    short_strategy = (
        adaptive_strategy_params(
            entry={"initial_qty_pct": 0.0},
            close={"threshold_base_pct": 0.6, "qty_pct": 1.0},
        )
        if strategy_kind == "trailing_martingale"
        else trailing_grid_v7_strategy_params(
            initial_qty_pct=0.0,
            close_overrides={
                "grid_markup_start": 0.6,
                "grid_markup_end": 0.6,
                "grid_qty_pct": 1.0,
            },
        )
    )
    symbol_kwargs = {"short_strategy": short_strategy}
    if strategy_kind == "trailing_grid_v7":
        symbol_kwargs["long_strategy"] = trailing_grid_v7_strategy_params(
            initial_qty_pct=0.0
        )
    symbol = make_symbol(
        0,
        bid=0.1,
        ask=0.1,
        short_pos_size=-1.0,
        short_pos_price=0.1,
        **symbol_kwargs,
    )
    symbol["exchange"].update(
        price_step=0.1,
        qty_step=0.1,
        min_qty=0.0,
        min_cost=0.0,
    )
    global_bp = bot_params_pair(
        long_overrides={"n_positions": 0, "total_wallet_exposure_limit": 0.0},
        short_overrides={"n_positions": 1, "total_wallet_exposure_limit": 1.0},
    )
    inp = make_input(
        balance=1_000.0,
        global_bp=global_bp,
        strategy_kind=strategy_kind,
        symbols=[symbol],
    )

    raw = pbr.compute_ideal_orders_json(json.dumps(inp))
    _, orders = reconciler.parse_and_validate_rust_orchestrator_output(
        raw,
        {0: "TEST/USDT:USDT"},
        inp,
    )

    close = next(order for order in orders if order["order_type"] == "close_grid_short")
    assert close["price"] == 0.1
    assert close["qty"] == 1.0


@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "trailing_grid_v7"])
def test_limit_close_minimum_uses_emitted_price_and_exact_remaining_position(
    strategy_kind,
):
    import passivbot_rust as pbr

    short_strategy = (
        adaptive_strategy_params(entry={"initial_qty_pct": 0.0})
        if strategy_kind == "trailing_martingale"
        else trailing_grid_v7_strategy_params(initial_qty_pct=0.0)
    )
    symbol_kwargs = {"short_strategy": short_strategy}
    if strategy_kind == "trailing_grid_v7":
        symbol_kwargs["long_strategy"] = trailing_grid_v7_strategy_params(
            initial_qty_pct=0.0
        )
    symbol = make_symbol(
        0,
        bid=1.50,
        ask=1.51,
        short_pos_size=-1.0,
        short_pos_price=1.0,
        **symbol_kwargs,
    )
    symbol["exchange"].update(
        price_step=0.01,
        qty_step=0.03,
        min_qty=0.0,
        min_cost=1.0,
    )
    global_bp = bot_params_pair(
        long_overrides={"n_positions": 0, "total_wallet_exposure_limit": 0.0},
        short_overrides={"n_positions": 1, "total_wallet_exposure_limit": 1.0},
    )
    inp = make_input(
        balance=1_000.0,
        global_bp=global_bp,
        strategy_kind=strategy_kind,
        symbols=[symbol],
    )

    raw = pbr.compute_ideal_orders_json(json.dumps(inp))
    _, orders = reconciler.parse_and_validate_rust_orchestrator_output(
        raw,
        {0: "TEST/USDT:USDT"},
        inp,
    )

    close = next(order for order in orders if order["order_type"] == "close_grid_short")
    assert close["price"] == 0.99
    assert close["qty"] == 1.0


def test_live_validator_accepts_rust_market_execution_policy():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[make_symbol(0, bid=100.0, ask=100.0)],
    )
    inp["global"]["market_orders_allowed"] = True
    inp["global"]["market_order_near_touch_threshold"] = 0.001
    out = compute(pbr, inp)

    assert any(order["execution_type"] == "market" for order in out["orders"])
    assert reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    ) == out["orders"]


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


def test_trailing_unavailable_does_not_hide_malformed_bundle():
    import passivbot_rust as pbr

    symbol = make_symbol(0, bid=100.0, ask=100.0)
    symbol["long"]["trailing_available"] = False
    del symbol["long"]["trailing"]["max_since_open"]
    inp = make_input(balance=1_000.0, symbols=[symbol])

    with pytest.raises(ValueError, match="missing field `max_since_open`"):
        compute(pbr, inp)


def test_live_missing_entry_trailing_preserves_close_and_other_side():
    import passivbot_rust as pbr

    enabled_short = {"n_positions": 1, "total_wallet_exposure_limit": 1.0}
    strategy = adaptive_strategy_params(
        entry={"retracement_base_pct": 0.01},
        close={"retracement_base_pct": 0.0},
    )
    symbol = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
        short_pos_size=-1.0,
        short_pos_price=100.0,
        short_bp=enabled_short,
        long_strategy=strategy,
        short_strategy=strategy,
    )
    symbol["long"]["trailing_available"] = False
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(short_overrides=enabled_short),
        symbols=[symbol],
    )
    inp["global"]["hedge_mode"] = True

    out = compute(pbr, inp)

    assert not any(
        order["pside"] == "long" and order["order_type"].startswith("entry_")
        for order in out["orders"]
    )
    assert any(
        order["pside"] == "long" and order["order_type"].startswith("close_")
        for order in out["orders"]
    )
    assert any(order["pside"] == "short" for order in out["orders"])
    assert {
        "strategy_input_unavailable": {
            "symbol_idx": 0,
            "pside": "long",
            "scope": "strategy_orders",
        }
    } in out["diagnostics"]["warnings"]


def test_live_missing_close_trailing_still_emits_wel_reducer():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.4,
        "risk_wel_enforcer_threshold": 1.0,
        "risk_twel_enforcer_enabled": False,
        "total_wallet_exposure_limit": 1.0,
        "n_positions": 1,
    }
    strategy = adaptive_strategy_params(
        entry={"retracement_base_pct": 0.0},
        close={"retracement_base_pct": 0.01},
    )
    symbol = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=6.0,
        long_pos_price=100.0,
        long_bp=long_bp,
        long_strategy=strategy,
    )
    symbol["long"]["trailing_available"] = False
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(long_overrides=long_bp),
        symbols=[symbol],
    )
    inp["global"]["hedge_mode"] = True

    out = compute(pbr, inp)

    assert any(
        order["order_type"] == "close_auto_reduce_wel_long"
        for order in out["orders"]
    )
    assert not any(
        order["pside"] == "long"
        and order["order_type"] in {"close_grid_long", "close_trailing_long"}
        for order in out["orders"]
    )


def test_trailing_unavailable_is_inert_when_strategy_does_not_consume_it():
    import passivbot_rust as pbr

    symbol = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
    )
    symbol["long"]["trailing_available"] = False
    inp = make_input(balance=1_000.0, symbols=[symbol])
    available_inp = copy.deepcopy(inp)
    available_inp["symbols"][0]["long"]["trailing_available"] = True

    out = compute(pbr, inp)
    available_out = compute(pbr, available_inp)

    assert out["orders"] == available_out["orders"]
    assert not any(
        "strategy_input_unavailable" in warning
        for warning in out["diagnostics"]["warnings"]
    )


def test_live_authorized_missing_entry_ema_preserves_independent_closes():
    import passivbot_rust as pbr

    symbol = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
        emas=ema_bundle(m1_close=[]),
    )
    symbol["allow_missing_strategy_inputs"] = True
    inp = make_input(balance=1_000.0, symbols=[symbol])
    inp["global"]["hedge_mode"] = True

    out = compute(pbr, inp)

    assert not any(
        order["pside"] == "long"
        and order["order_type"].startswith("entry_")
        for order in out["orders"]
    )
    assert any(
        order["pside"] == "long" and order["order_type"].startswith("close_")
        for order in out["orders"]
    )
    assert {
        "strategy_input_unavailable": {
            "symbol_idx": 0,
            "pside": "long",
            "scope": "strategy_orders",
        }
    } in out["diagnostics"]["warnings"]
    assert reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    ) == out["orders"]


def test_live_authorized_missing_entry_volatility_preserves_independent_closes():
    import passivbot_rust as pbr

    strategy = adaptive_strategy_params(
        entry_volatility_ema_span_1h=4.0,
        entry_weight_volatility_1h=1.0,
    )
    symbol = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
        long_strategy=strategy,
        emas=ema_bundle(h1_log_range=[]),
    )
    symbol["allow_missing_strategy_inputs"] = True
    inp = make_input(balance=1_000.0, symbols=[symbol])
    inp["global"]["hedge_mode"] = True

    out = compute(pbr, inp)
    complete_inp = copy.deepcopy(inp)
    complete_inp["symbols"][0]["emas"] = ema_bundle(h1_log_range=[[4.0, 0.01]])
    complete_inp["symbols"][0]["allow_missing_strategy_inputs"] = False
    complete_out = compute(pbr, complete_inp)

    assert not any(
        order["pside"] == "long" and order["order_type"].startswith("entry_")
        for order in out["orders"]
    )
    scoped_closes = [
        order
        for order in out["orders"]
        if order["pside"] == "long" and order["order_type"].startswith("close_")
    ]
    complete_closes = [
        order
        for order in complete_out["orders"]
        if order["pside"] == "long" and order["order_type"].startswith("close_")
    ]
    assert scoped_closes
    assert scoped_closes == complete_closes
    assert out["diagnostics"]["warnings"].count(
        {
            "strategy_input_unavailable": {
                "symbol_idx": 0,
                "pside": "long",
                "scope": "strategy_orders",
            }
        }
    ) == 1


def test_live_authorized_missing_ema_still_emits_twel_reducer():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.4,
        "total_wallet_exposure_limit": 0.9,
        "risk_twel_enforcer_threshold": 1.0,
        "n_positions": 2,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    symbols = [
        make_symbol(
            0,
            bid=50.0,
            ask=50.0,
            long_pos_size=8.0,
            long_pos_price=50.0,
            long_bp=long_bp,
            emas=ema_bundle(m1_close=[]),
        ),
        make_symbol(
            1,
            bid=50.0,
            ask=50.0,
            long_pos_size=12.0,
            long_pos_price=50.0,
            long_bp=long_bp,
            emas=ema_bundle(m1_close=[]),
        ),
    ]
    for symbol in symbols:
        symbol["allow_missing_strategy_inputs"] = True
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=symbols)
    inp["global"]["hedge_mode"] = True

    out = compute(pbr, inp)

    twel_orders = [
        order
        for order in out["orders"]
        if order["order_type"] == "close_auto_reduce_twel_long"
    ]
    assert twel_orders
    assert {order["symbol_idx"] for order in twel_orders} == {1}


def test_live_authorized_missing_ema_still_emits_wel_reducer():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.4,
        "risk_wel_enforcer_threshold": 1.0,
        "risk_twel_enforcer_enabled": False,
        "total_wallet_exposure_limit": 1.0,
        "n_positions": 1,
    }
    symbol = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=6.0,
        long_pos_price=100.0,
        long_bp=long_bp,
        emas=ema_bundle(m1_close=[]),
    )
    symbol["allow_missing_strategy_inputs"] = True
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(long_overrides=long_bp),
        symbols=[symbol],
    )
    inp["global"]["hedge_mode"] = True

    out = compute(pbr, inp)
    complete_inp = copy.deepcopy(inp)
    complete_inp["symbols"][0]["emas"] = ema_bundle()
    complete_inp["symbols"][0]["allow_missing_strategy_inputs"] = False
    complete_out = compute(pbr, complete_inp)

    wel_orders = [
        order
        for order in out["orders"]
        if order["order_type"] == "close_auto_reduce_wel_long"
    ]
    complete_wel_orders = [
        order
        for order in complete_out["orders"]
        if order["order_type"] == "close_auto_reduce_wel_long"
    ]
    assert len(wel_orders) == 1
    assert wel_orders == complete_wel_orders
    assert wel_orders[0]["symbol_idx"] == 0
    assert not any(
        order["pside"] == "long"
        and order["order_type"].startswith("entry_")
        for order in out["orders"]
    )
    assert any(
        order["pside"] == "long"
        and (
            order["order_type"].startswith("close_grid")
            or order["order_type"].startswith("close_trailing")
        )
        for order in out["orders"]
    )


def test_live_authorized_missing_ema_does_not_scope_unaffected_pside():
    import passivbot_rust as pbr

    enabled_short = {"n_positions": 1, "total_wallet_exposure_limit": 1.0}
    symbol = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
        short_pos_size=-1.0,
        short_pos_price=100.0,
        short_bp=enabled_short,
        long_strategy=adaptive_strategy_params(
            entry_volatility_ema_span_1h=4.0,
            entry_weight_volatility_1h=1.0,
        ),
        short_strategy=adaptive_strategy_params(),
        emas=ema_bundle(h1_log_range=[]),
    )
    symbol["allow_missing_strategy_inputs"] = True
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(short_overrides=enabled_short),
        symbols=[symbol],
    )
    inp["global"]["hedge_mode"] = True

    out = compute(pbr, inp)

    assert any(order["pside"] == "short" for order in out["orders"])
    assert not any(
        order["pside"] == "long" and order["order_type"].startswith("entry_")
        for order in out["orders"]
    )
    assert any(
        order["pside"] == "long"
        and (
            order["order_type"].startswith("close_grid")
            or order["order_type"].startswith("close_trailing")
        )
        for order in out["orders"]
    )
    scoped_warnings = [
        warning["strategy_input_unavailable"]
        for warning in out["diagnostics"]["warnings"]
        if "strategy_input_unavailable" in warning
    ]
    assert scoped_warnings == [
        {"symbol_idx": 0, "pside": "long", "scope": "strategy_orders"}
    ]


def test_trailing_grid_v7_close_only_ignores_entry_only_emas():
    import passivbot_rust as pbr

    strategy = trailing_grid_v7_strategy_params(
        grid_spacing_volatility_weight=1.0,
        volatility_ema_span_hours=4.0,
    )
    symbol = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_mode="tp_only",
        long_pos_size=1.0,
        long_pos_price=100.0,
        long_strategy=strategy,
        short_strategy=strategy,
        emas=ema_bundle(m1_close=[], h1_log_range=[]),
    )
    inp = make_input(
        balance=1_000.0,
        strategy_kind="trailing_grid_v7",
        symbols=[symbol],
    )

    out = compute(pbr, inp)

    assert any(
        order["pside"] == "long" and order["order_type"] == "close_grid_long"
        for order in out["orders"]
    )


def test_live_authorization_does_not_tolerate_malformed_ema_values():
    import passivbot_rust as pbr

    symbol = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
        emas=ema_bundle(
            m1_close=[
                [10.0, 0.0],
                [20.0, 0.0],
                [math.sqrt(10.0 * 20.0), 0.0],
            ]
        ),
    )
    symbol["allow_missing_strategy_inputs"] = True
    inp = make_input(balance=1_000.0, symbols=[symbol])
    inp["global"]["hedge_mode"] = True

    with pytest.raises(ValueError, match="NonFiniteInput"):
        compute(pbr, inp)


def test_ema_gate_mode_disabled_initial_long_uses_best_bid_without_ema():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=101.0,
                long_strategy=adaptive_strategy_params(
                    entry={"ema_gate_mode": "disabled", "initial_ema_dist": -0.25}
                ),
                emas=ema_bundle(m1_close=[]),
            )
        ],
    )
    inp["global"]["hedge_mode"] = True

    out = compute(pbr, inp)

    initial = next(o for o in out["orders"] if o["order_type"] == "entry_initial_normal_long")
    assert initial["price"] == pytest.approx(100.0)


def test_ema_gate_mode_reentry_leaves_flat_initial_at_best_bid_without_ema():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=101.0,
                long_strategy=adaptive_strategy_params(
                    entry={"ema_gate_mode": "reentry", "initial_ema_dist": -0.25}
                ),
                emas=ema_bundle(m1_close=[]),
            )
        ],
    )
    inp["global"]["hedge_mode"] = True

    out = compute(pbr, inp)

    initial = next(o for o in out["orders"] if o["order_type"] == "entry_initial_normal_long")
    assert initial["price"] == pytest.approx(100.0)


def test_ema_gate_mode_reentry_leaves_partial_initial_at_best_bid_without_ema():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=101.0,
                long_pos_size=0.5,
                long_pos_price=100.0,
                long_strategy=adaptive_strategy_params(
                    entry={"ema_gate_mode": "reentry", "initial_ema_dist": -0.25}
                ),
                emas=ema_bundle(m1_close=[]),
            )
        ],
    )
    inp["global"]["hedge_mode"] = True

    out = compute(pbr, inp)

    partial = next(o for o in out["orders"] if o["order_type"] == "entry_initial_partial_long")
    assert partial["price"] == pytest.approx(100.0)


def test_ema_gate_mode_all_gates_long_reentry_price():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_pos_size=1.0,
                long_pos_price=100.0,
                long_strategy=adaptive_strategy_params(
                    entry={"ema_gate_mode": "all", "threshold_base_pct": 0.02}
                ),
                emas=ema_bundle(
                    m1_close=[
                        [10.0, 95.0],
                        [20.0, 95.0],
                        [math.sqrt(10.0 * 20.0), 95.0],
                    ]
                ),
            )
        ],
    )
    inp["global"]["hedge_mode"] = True

    out = compute(pbr, inp)

    reentry = next(o for o in out["orders"] if o["order_type"] == "entry_grid_normal_long")
    assert reentry["price"] == pytest.approx(95.0)


def test_ema_gate_mode_reentry_requires_ema_for_true_reentry():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_pos_size=1.0,
                long_pos_price=100.0,
                long_strategy=adaptive_strategy_params(entry={"ema_gate_mode": "reentry"}),
                emas=ema_bundle(m1_close=[]),
            )
        ],
    )
    inp["global"]["hedge_mode"] = True

    with pytest.raises(ValueError, match="MissingEma"):
        compute(pbr, inp)


def test_one_way_flat_tie_break_requires_ema_even_when_entry_gate_disabled():
    import passivbot_rust as pbr

    side_enabled = {"n_positions": 1, "total_wallet_exposure_limit": 1.0}
    disabled_strategy = adaptive_strategy_params(entry={"ema_gate_mode": "disabled"})
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(short_overrides=side_enabled),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=101.0,
                short_bp=side_enabled,
                long_strategy=disabled_strategy,
                short_strategy=disabled_strategy,
                emas=ema_bundle(m1_close=[]),
            )
        ],
    )
    inp["global"]["hedge_mode"] = False

    with pytest.raises(ValueError, match="MissingEma"):
        compute(pbr, inp)


def test_live_validator_accepts_one_way_forager_selection_before_tie_break():
    import passivbot_rust as pbr

    side_enabled = {"n_positions": 1, "total_wallet_exposure_limit": 1.0}
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(short_overrides=side_enabled),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                short_bp=side_enabled,
            )
        ],
    )
    inp["global"]["hedge_mode"] = False

    out = compute(pbr, inp)
    states = out["diagnostics"]["symbol_states"][0]
    selections = {
        item["pside"]: item["selected_symbol_indices"]
        for item in out["diagnostics"]["forager_selections"]
    }

    assert selections == {"long": [0], "short": [0]}
    assert states["long"]["active"] is True
    assert states["long"]["allow_initial"] is True
    assert states["short"]["active"] is True
    assert states["short"]["allow_initial"] is False
    reconciler.validate_rust_orchestrator_output(
        out,
        {0: "BTC/USDT:USDT"},
        inp,
    )


def test_live_authorized_missing_ema_blocks_both_one_way_initial_sides():
    import passivbot_rust as pbr

    side_enabled = {"n_positions": 1, "total_wallet_exposure_limit": 1.0}
    symbol = make_symbol(
        0,
        bid=100.0,
        ask=101.0,
        short_bp=side_enabled,
        emas=ema_bundle(m1_close=[]),
    )
    symbol["allow_missing_strategy_inputs"] = True
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(short_overrides=side_enabled),
        symbols=[symbol],
    )
    inp["global"]["hedge_mode"] = False

    out = compute(pbr, inp)

    assert not any(order["order_type"].startswith("entry_") for order in out["orders"])
    assert {
        "strategy_input_unavailable": {
            "symbol_idx": 0,
            "pside": "long",
            "scope": "one_way_arbitration",
        }
    } in out["diagnostics"]["warnings"]


@pytest.mark.parametrize(
    "field",
    ["qty_step", "price_step", "min_qty", "min_cost", "c_mult", "maker_fee", "taker_fee"],
)
def test_json_rejects_missing_exchange_param(field):
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[make_symbol(0, bid=100.0, ask=100.0)],
    )
    del inp["symbols"][0]["exchange"][field]
    with pytest.raises(ValueError, match=rf"missing field `{field}`"):
        compute(pbr, inp)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qty_step", 0.0),
        ("price_step", 0.0),
        ("min_qty", -0.01),
        ("min_cost", -1.0),
        ("c_mult", 0.0),
    ],
)
def test_json_rejects_invalid_exchange_param(field, value):
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[make_symbol(0, bid=100.0, ask=100.0)],
    )
    inp["symbols"][0]["exchange"][field] = value
    with pytest.raises(ValueError, match=rf"InvalidExchangeParams.*{field}"):
        compute(pbr, inp)


def test_json_rejects_missing_realized_loss_gate_param():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[make_symbol(0, bid=100.0, ask=100.0)],
    )
    del inp["global"]["max_realized_loss_pct"]
    with pytest.raises(ValueError, match=r"missing field `max_realized_loss_pct`"):
        compute(pbr, inp)


def test_adaptive_grid_long_entry_output_regression():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(
            short_overrides={
                "n_positions": 0,
                "total_wallet_exposure_limit": 0.0,
            }
        ),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_bp={"entry_initial_ema_dist": -0.01},
                short_bp={
                    "n_positions": 0,
                    "total_wallet_exposure_limit": 0.0,
                },
            )
        ],
    )

    out = compute(pbr, inp)

    assert out["orders"] == [
        {
            "symbol_idx": 0,
            "pside": "long",
            "qty": 1.0,
            "price": 100.0,
            "order_type": "entry_initial_normal_long",
            "execution_type": "limit",
            "execution_priority": "ordinary",
        },
        {
            "symbol_idx": 0,
            "pside": "long",
            "qty": 1.02,
            "price": 98.0,
            "order_type": "entry_grid_normal_long",
            "execution_type": "limit",
            "execution_priority": "ordinary",
        },
        {
            "symbol_idx": 0,
            "pside": "long",
            "qty": 2.02,
            "price": 97.01,
            "order_type": "entry_grid_normal_long",
            "execution_type": "limit",
            "execution_priority": "ordinary",
        },
        {
            "symbol_idx": 0,
            "pside": "long",
            "qty": 4.04,
            "price": 96.04,
            "order_type": "entry_grid_normal_long",
            "execution_type": "limit",
            "execution_priority": "ordinary",
        },
        {
            "symbol_idx": 0,
            "pside": "long",
            "qty": 2.27,
            "price": 95.07,
            "order_type": "entry_grid_cropped_long",
            "execution_type": "limit",
            "execution_priority": "ordinary",
        },
    ]


def test_adaptive_grid_short_entry_output_regression():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(
            long_overrides={
                "n_positions": 0,
                "total_wallet_exposure_limit": 0.0,
            },
            short_overrides={
                "n_positions": 1,
                "total_wallet_exposure_limit": 1.0,
            },
        ),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_bp={
                    "n_positions": 0,
                    "total_wallet_exposure_limit": 0.0,
                },
                short_bp={
                    "n_positions": 1,
                    "total_wallet_exposure_limit": 1.0,
                    "entry_initial_ema_dist": 0.01,
                },
            )
        ],
    )

    out = compute(pbr, inp)

    assert out["orders"] == [
        {
            "symbol_idx": 0,
            "pside": "short",
            "qty": -0.99,
            "price": 101.0,
            "order_type": "entry_initial_normal_short",
            "execution_type": "limit",
            "execution_priority": "ordinary",
        },
        {
            "symbol_idx": 0,
            "pside": "short",
            "qty": -0.99,
            "price": 103.02,
            "order_type": "entry_grid_normal_short",
            "execution_type": "limit",
            "execution_priority": "ordinary",
        },
        {
            "symbol_idx": 0,
            "pside": "short",
            "qty": -1.98,
            "price": 104.06,
            "order_type": "entry_grid_normal_short",
            "execution_type": "limit",
            "execution_priority": "ordinary",
        },
        {
            "symbol_idx": 0,
            "pside": "short",
            "qty": -3.96,
            "price": 105.1,
            "order_type": "entry_grid_normal_short",
            "execution_type": "limit",
            "execution_priority": "ordinary",
        },
        {
            "symbol_idx": 0,
            "pside": "short",
            "qty": -1.65,
            "price": 106.15,
            "order_type": "entry_grid_cropped_short",
            "execution_type": "limit",
            "execution_priority": "ordinary",
        },
    ]


def test_entry_ladder_can_stage_simultaneously_only_with_zero_cooldown():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(long_overrides={"risk_entry_cooldown_minutes": 0.0}),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_bp={"risk_entry_cooldown_minutes": 0.0},
            )
        ],
    )
    inp["timestamp_ms"] = 120_000

    out = compute(pbr, inp)
    long_add_orders = [
        o
        for o in out["orders"]
        if o["pside"] == "long" and o["qty"] > 0.0 and o["order_type"].startswith("entry_")
    ]

    assert len(long_add_orders) > 1


def test_positive_fractional_entry_cooldown_throttles_ladder_to_one_order():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(long_overrides={"risk_entry_cooldown_minutes": 0.05}),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_bp={"risk_entry_cooldown_minutes": 0.05},
            )
        ],
    )
    inp["timestamp_ms"] = 120_000

    out = compute(pbr, inp)
    long_add_orders = [
        o
        for o in out["orders"]
        if o["pside"] == "long" and o["qty"] > 0.0 and o["order_type"].startswith("entry_")
    ]

    assert len(long_add_orders) == 1
    assert long_add_orders[0]["order_type"] == "entry_initial_normal_long"


def test_entry_retracement_throttles_ladder_even_with_zero_cooldown():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_strategy=adaptive_strategy_params(entry={"retracement_base_pct": 0.001}),
            )
        ],
    )
    inp["timestamp_ms"] = 120_000

    out = compute(pbr, inp)
    long_add_orders = [
        o
        for o in out["orders"]
        if o["pside"] == "long" and o["qty"] > 0.0 and o["order_type"].startswith("entry_")
    ]

    assert len(long_add_orders) == 1
    assert long_add_orders[0]["order_type"] == "entry_initial_normal_long"


@pytest.mark.parametrize(
    ("cooldown_minutes", "expected_order_count"),
    [(0.0, None), (0.05, 1)],
)
def test_trailing_grid_v7_preserves_zero_cooldown_grid_ladder_but_not_positive_cooldown(
    cooldown_minutes, expected_order_count
):
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        strategy_kind="trailing_grid_v7",
        global_bp=bot_params_pair(
            long_overrides={"risk_entry_cooldown_minutes": cooldown_minutes}
        ),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_pos_size=1.0,
                long_pos_price=100.0,
                long_bp={"risk_entry_cooldown_minutes": cooldown_minutes},
                long_strategy=trailing_grid_v7_strategy_params(),
                short_strategy=trailing_grid_v7_strategy_params(),
            )
        ],
    )
    inp["timestamp_ms"] = 120_000
    inp["peek_hints"] = {
        "expand_grid_long": [0],
        "expand_grid_short": [],
        "expand_close_long": [],
        "expand_close_short": [],
    }

    out = compute(pbr, inp)
    long_add_orders = [
        order
        for order in out["orders"]
        if order["pside"] == "long" and order["qty"] > 0.0
    ]

    if expected_order_count is None:
        assert len(long_add_orders) >= 2
        assert all(order["order_type"].startswith("entry_grid_") for order in long_add_orders)
    else:
        assert len(long_add_orders) == expected_order_count


def test_entry_cooldown_blocks_position_adding_orders_until_exact_window_expires():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(long_overrides={"risk_entry_cooldown_minutes": 1.0}),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_bp={"risk_entry_cooldown_minutes": 1.0},
            )
        ],
    )
    inp["timestamp_ms"] = 119_999
    inp["symbols"][0]["long"]["last_increase_fill_timestamp_ms"] = 60_000

    out = compute(pbr, inp)
    long_add_orders = [
        o
        for o in out["orders"]
        if o["pside"] == "long" and o["qty"] > 0.0 and o["order_type"].startswith("entry_")
    ]

    assert long_add_orders == []


def test_entry_cooldown_keeps_one_add_order_after_window_expires():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(long_overrides={"risk_entry_cooldown_minutes": 2.0}),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_bp={"risk_entry_cooldown_minutes": 2.0},
            )
        ],
    )
    inp["timestamp_ms"] = 240_000
    inp["symbols"][0]["long"]["last_increase_fill_timestamp_ms"] = 60_000

    out = compute(pbr, inp)
    long_add_orders = [
        o
        for o in out["orders"]
        if o["pside"] == "long" and o["qty"] > 0.0 and o["order_type"].startswith("entry_")
    ]

    assert len(long_add_orders) == 1
    assert long_add_orders[0]["order_type"] == "entry_initial_normal_long"


def test_entry_cooldown_keeps_close_orders_while_blocking_adds():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(long_overrides={"risk_entry_cooldown_minutes": 1.0}),
        symbols=[
            make_symbol(
                0,
                bid=102.0,
                ask=102.0,
                long_pos_size=1.0,
                long_pos_price=100.0,
                long_bp={"risk_entry_cooldown_minutes": 1.0},
            )
        ],
    )
    inp["timestamp_ms"] = 119_999
    inp["symbols"][0]["long"]["last_increase_fill_timestamp_ms"] = 60_000

    out = compute(pbr, inp)
    long_add_orders = [
        o
        for o in out["orders"]
        if o["pside"] == "long" and o["qty"] > 0.0 and o["order_type"].startswith("entry_")
    ]
    long_close_orders = [
        o
        for o in out["orders"]
        if o["pside"] == "long" and o["qty"] < 0.0 and o["order_type"].startswith("close_")
    ]

    assert long_add_orders == []
    assert long_close_orders


def test_fractional_entry_cooldown_blocks_until_seconds_elapsed_then_keeps_one_add():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(long_overrides={"risk_entry_cooldown_minutes": 0.05}),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_bp={"risk_entry_cooldown_minutes": 0.05},
            )
        ],
    )
    inp["timestamp_ms"] = 63_999
    inp["symbols"][0]["long"]["last_increase_fill_timestamp_ms"] = 61_000

    out = compute(pbr, inp)
    long_add_orders = [
        o
        for o in out["orders"]
        if o["pside"] == "long" and o["qty"] > 0.0 and o["order_type"].startswith("entry_")
    ]

    assert long_add_orders == []

    inp["timestamp_ms"] = 64_000
    out = compute(pbr, inp)
    long_add_orders = [
        o
        for o in out["orders"]
        if o["pside"] == "long" and o["qty"] > 0.0 and o["order_type"].startswith("entry_")
    ]

    assert len(long_add_orders) == 1
    assert long_add_orders[0]["order_type"] == "entry_initial_normal_long"


def test_entry_cooldown_is_separated_by_pside_in_hedge_mode():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(
            long_overrides={"risk_entry_cooldown_minutes": 1.0},
            short_overrides={
                "n_positions": 1,
                "total_wallet_exposure_limit": 1.0,
                "risk_entry_cooldown_minutes": 1.0,
            },
        ),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_bp={"risk_entry_cooldown_minutes": 1.0},
                short_bp={
                    "n_positions": 1,
                    "total_wallet_exposure_limit": 1.0,
                    "risk_entry_cooldown_minutes": 1.0,
                },
            )
        ],
    )
    inp["global"]["hedge_mode"] = True
    inp["timestamp_ms"] = 119_999
    inp["symbols"][0]["long"]["last_increase_fill_timestamp_ms"] = 60_000

    out = compute(pbr, inp)
    long_add_orders = [
        o
        for o in out["orders"]
        if o["pside"] == "long" and o["qty"] > 0.0 and o["order_type"].startswith("entry_")
    ]
    short_add_orders = [
        o
        for o in out["orders"]
        if o["pside"] == "short" and o["qty"] < 0.0 and o["order_type"].startswith("entry_")
    ]

    assert long_add_orders == []
    assert len(short_add_orders) == 1
    assert short_add_orders[0]["order_type"] == "entry_initial_normal_short"


def test_ema_anchor_long_position_emits_single_entry_and_close():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        strategy_kind="ema_anchor",
        global_bp=bot_params_pair(
            short_overrides={
                "n_positions": 0,
                "total_wallet_exposure_limit": 0.0,
            }
        ),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_pos_size=1.0,
                long_pos_price=100.0,
                long_strategy={
                    "base_qty_pct": 0.1,
                    "ema_span_0": 10.0,
                    "ema_span_1": 20.0,
                    "offset": 0.01,
                    "offset_psize_weight": 0.0,
                },
                short_strategy={
                    "base_qty_pct": 0.1,
                    "ema_span_0": 10.0,
                    "ema_span_1": 20.0,
                    "offset": 0.01,
                    "offset_psize_weight": 0.0,
                },
            )
        ],
    )

    out = compute(pbr, inp)

    assert out["orders"] == [
        {
            "symbol_idx": 0,
            "pside": "long",
            "qty": -0.99,
            "price": 101.0,
            "order_type": "close_ema_anchor_long",
            "execution_type": "limit",
            "execution_priority": "ordinary",
        },
        {
            "symbol_idx": 0,
            "pside": "long",
            "qty": 1.01,
            "price": 99.0,
            "order_type": "entry_ema_anchor_long",
            "execution_type": "limit",
            "execution_priority": "ordinary",
        },
    ]


def test_ema_anchor_market_close_uses_executable_touch_minimum():
    import passivbot_rust as pbr

    strategy = {
        "base_qty_pct": 0.1,
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "offset": 0.0,
        "offset_psize_weight": 0.0,
    }
    symbol = make_symbol(
        0,
        bid=99.95,
        ask=100.05,
        long_pos_size=10.0,
        long_pos_price=100.0,
        long_strategy=strategy,
        short_strategy=strategy,
        emas=ema_bundle(
            m1_close=[
                [10.0, 100.0],
                [20.0, 100.0],
                [math.sqrt(10.0 * 20.0), 100.0],
            ]
        ),
    )
    symbol["exchange"].update(
        {"qty_step": 0.001, "price_step": 0.01, "min_qty": 0.0, "min_cost": 100.0}
    )
    inp = make_input(
        balance=1_000.0,
        strategy_kind="ema_anchor",
        symbols=[symbol],
    )
    inp["global"]["market_orders_allowed"] = True
    inp["global"]["market_order_near_touch_threshold"] = 0.001

    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )

    close = next(
        order
        for order in out["orders"]
        if order["order_type"] == "close_ema_anchor_long"
    )
    assert close["execution_type"] == "market"
    assert close["price"] == 100.05
    assert close["qty"] == pytest.approx(-1.001)


@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "trailing_grid_v7"])
def test_grid_market_close_uses_executable_touch_minimum(strategy_kind):
    import passivbot_rust as pbr

    strategy = (
        adaptive_strategy_params(
            close={
                "qty_pct": 0.1,
                "threshold_base_pct": 0.005,
                "threshold_we_weight": 0.001,
            }
        )
        if strategy_kind == "trailing_martingale"
        else trailing_grid_v7_strategy_params(
            close_overrides={
                "grid_qty_pct": 0.1,
                "grid_markup_start": 0.005,
                "grid_markup_end": 0.01,
            }
        )
    )
    symbol = make_symbol(
        0,
        bid=99.95,
        ask=100.05,
        long_pos_size=10.0,
        long_pos_price=100.0,
        long_strategy=strategy,
        short_strategy=strategy,
    )
    symbol["exchange"].update(
        {"qty_step": 0.001, "price_step": 0.01, "min_qty": 0.0, "min_cost": 100.0}
    )
    inp = make_input(
        balance=1_000.0,
        strategy_kind=strategy_kind,
        symbols=[symbol],
    )
    inp["global"]["market_orders_allowed"] = True
    inp["global"]["market_order_near_touch_threshold"] = 0.02

    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )
    close = next(
        order
        for order in out["orders"]
        if order["order_type"] == "close_grid_long"
    )

    assert close["execution_type"] == "market"
    assert close["qty"] == pytest.approx(-1.001)
    assert abs(close["qty"]) * symbol["order_book"]["bid"] >= 100.0


def test_off_tick_ema_anchor_touch_prices_pass_live_validation():
    import passivbot_rust as pbr

    strategy = {
        "base_qty_pct": 0.1,
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "offset": 0.0,
        "offset_psize_weight": 0.0,
    }
    inp = make_input(
        balance=1_000.0,
        strategy_kind="ema_anchor",
        symbols=[
            make_symbol(
                0,
                bid=98.003,
                ask=102.007,
                long_pos_size=1.0,
                long_pos_price=100.0,
                long_strategy=strategy,
                short_strategy=strategy,
                emas=ema_bundle(
                    m1_close=[
                        [10.0, 100.0],
                        [20.0, 100.0],
                        [math.sqrt(10.0 * 20.0), 100.0],
                    ],
                    m1_volume=[[10.0, 1_000.0]],
                    m1_log_range=[[10.0, 0.01]],
                ),
            )
        ],
    )

    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )

    prices = {order["order_type"]: order["price"] for order in out["orders"]}
    assert prices["entry_ema_anchor_long"] == 98.0
    assert prices["close_ema_anchor_long"] == 102.01


@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "trailing_grid_v7"])
def test_off_tick_strategy_entry_recomputes_minimum_after_price_quantization(
    strategy_kind,
):
    import passivbot_rust as pbr

    strategy = (
        adaptive_strategy_params(entry={"initial_qty_pct": 0.0})
        if strategy_kind == "trailing_martingale"
        else trailing_grid_v7_strategy_params(initial_qty_pct=0.0)
    )
    symbol = make_symbol(
        0,
        bid=3.003,
        ask=3.01,
        long_strategy=strategy,
        short_strategy=strategy,
    )
    symbol["exchange"].update(
        {"qty_step": 0.001, "price_step": 0.01, "min_qty": 0.0, "min_cost": 5.0}
    )
    inp = make_input(
        balance=100.0,
        strategy_kind=strategy_kind,
        symbols=[symbol],
    )

    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )
    entry = next(
        order for order in out["orders"] if order["order_type"].startswith("entry_")
    )

    assert entry["price"] == 3.0
    assert abs(entry["qty"]) == pytest.approx(1.667)
    assert abs(entry["qty"]) * entry["price"] >= symbol["exchange"]["min_cost"]


@pytest.mark.parametrize(
    ("strategy_kind", "pside", "bid", "ask", "expected_price"),
    [
        ("trailing_martingale", "long", 100.006, 100.014, 100.0),
        ("trailing_martingale", "short", 100.006, 100.014, 100.02),
        ("trailing_grid_v7", "long", 100.006, 100.014, 100.0),
        ("trailing_grid_v7", "short", 100.006, 100.014, 100.02),
    ],
)
@pytest.mark.parametrize("next_only", [False, True])
def test_off_tick_strategy_entries_quantize_away_from_the_spread(
    strategy_kind, pside, bid, ask, expected_price, next_only
):
    import passivbot_rust as pbr

    strategy = (
        adaptive_strategy_params()
        if strategy_kind == "trailing_martingale"
        else trailing_grid_v7_strategy_params()
    )
    disabled = {"n_positions": 0, "total_wallet_exposure_limit": 0.0}
    enabled = {"n_positions": 1, "total_wallet_exposure_limit": 1.0}
    long_bp = enabled if pside == "long" else disabled
    short_bp = enabled if pside == "short" else disabled
    symbol = make_symbol(
        0,
        bid=bid,
        ask=ask,
        long_bp=long_bp,
        short_bp=short_bp,
        long_strategy=strategy,
        short_strategy=strategy,
    )
    symbol["exchange"]["price_step"] = 0.01
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(
            long_overrides=long_bp, short_overrides=short_bp
        ),
        strategy_kind=strategy_kind,
        symbols=[symbol],
    )
    if next_only:
        inp["peek_hints"] = {
            "expand_grid_long": [],
            "expand_grid_short": [],
            "expand_close_long": [],
            "expand_close_short": [],
        }

    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )
    entry = next(order for order in out["orders"] if order["pside"] == pside)

    assert entry["execution_type"] == "limit"
    assert entry["price"] == expected_price
    if pside == "long":
        assert entry["price"] <= bid
    else:
        assert entry["price"] >= ask


@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "trailing_grid_v7"])
def test_next_only_short_entry_recrops_quantity_after_price_quantization(strategy_kind):
    import passivbot_rust as pbr

    strategy = (
        adaptive_strategy_params(entry={"initial_qty_pct": 1.0})
        if strategy_kind == "trailing_martingale"
        else trailing_grid_v7_strategy_params(initial_qty_pct=1.0)
    )
    disabled = {"n_positions": 0, "total_wallet_exposure_limit": 0.0}
    enabled = {"n_positions": 1, "total_wallet_exposure_limit": 1.0}
    symbol = make_symbol(
        0,
        bid=100.0,
        ask=100.01,
        long_bp=disabled,
        short_bp=enabled,
        long_strategy=strategy,
        short_strategy=strategy,
    )
    symbol["exchange"].update(
        {"qty_step": 0.001, "price_step": 2.0, "min_qty": 0.0, "min_cost": 0.0}
    )
    inp = make_input(
        balance=100.0,
        global_bp=bot_params_pair(long_overrides=disabled, short_overrides=enabled),
        strategy_kind=strategy_kind,
        symbols=[symbol],
    )
    inp["peek_hints"] = {
        "expand_grid_long": [],
        "expand_grid_short": [],
        "expand_close_long": [],
        "expand_close_short": [],
    }

    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )
    entry = next(order for order in out["orders"] if order["pside"] == "short")

    assert entry["price"] == 102.0
    assert abs(entry["qty"]) == pytest.approx(0.98)
    assert entry["order_type"] == "entry_initial_normal_short"
    assert abs(entry["qty"]) * entry["price"] / inp["balance"] <= 1.0


@pytest.mark.parametrize("strategy_kind", ["trailing_martingale", "trailing_grid_v7"])
def test_short_market_entry_uses_executable_bid_minimum(strategy_kind):
    import passivbot_rust as pbr

    strategy = (
        adaptive_strategy_params(entry={"initial_qty_pct": 0.0})
        if strategy_kind == "trailing_martingale"
        else trailing_grid_v7_strategy_params(initial_qty_pct=0.0)
    )
    disabled = {"n_positions": 0, "total_wallet_exposure_limit": 0.0}
    enabled = {"n_positions": 1, "total_wallet_exposure_limit": 1.0}
    symbol = make_symbol(
        0,
        bid=99.95,
        ask=100.05,
        long_bp=disabled,
        short_bp=enabled,
        long_strategy=strategy,
        short_strategy=strategy,
    )
    symbol["exchange"].update(
        {"qty_step": 0.001, "price_step": 0.01, "min_qty": 0.0, "min_cost": 100.0}
    )
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(long_overrides=disabled, short_overrides=enabled),
        strategy_kind=strategy_kind,
        symbols=[symbol],
    )
    inp["global"]["market_orders_allowed"] = True
    inp["global"]["market_order_near_touch_threshold"] = 0.001

    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )
    entry = next(
        order
        for order in out["orders"]
        if order["order_type"].startswith("entry_")
    )

    assert entry["pside"] == "short"
    assert entry["execution_type"] == "market"
    assert entry["price"] == 100.05
    assert entry["qty"] == pytest.approx(-1.001)
    assert abs(entry["qty"]) * symbol["order_book"]["bid"] >= 100.0


def test_trailing_martingale_partial_entry_preserves_sub_ten_decimal_price_step():
    import passivbot_rust as pbr

    aligned_bid = 0.999999999999
    symbol = make_symbol(
        0,
        bid=aligned_bid,
        ask=1.000000000002,
        long_pos_size=0.5,
        long_pos_price=1.0,
        long_strategy=adaptive_strategy_params(entry={"ema_gate_mode": "reentry"}),
    )
    symbol["exchange"].update(
        {"qty_step": 0.001, "price_step": 3e-12, "min_qty": 0.0, "min_cost": 0.0}
    )
    inp = make_input(balance=1_000.0, symbols=[symbol])

    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )

    partial = next(
        order
        for order in out["orders"]
        if order["order_type"] == "entry_initial_partial_long"
    )
    assert partial["price"] == aligned_bid


def test_sub_tick_ema_anchor_bid_keeps_short_close_at_lowest_positive_tick():
    import passivbot_rust as pbr

    strategy = {
        "base_qty_pct": 0.1,
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "offset": 0.0,
        "offset_psize_weight": 0.0,
    }
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(
            short_overrides={"n_positions": 1, "total_wallet_exposure_limit": 1.0}
        ),
        strategy_kind="ema_anchor",
        symbols=[
            make_symbol(
                0,
                bid=0.005,
                ask=0.015,
                short_pos_size=-1.0,
                short_pos_price=0.02,
                short_bp={"n_positions": 1, "total_wallet_exposure_limit": 1.0},
                long_strategy=strategy,
                short_strategy=strategy,
                emas=ema_bundle(
                    m1_close=[
                        [10.0, 0.005],
                        [20.0, 0.005],
                        [math.sqrt(10.0 * 20.0), 0.005],
                    ],
                    m1_volume=[[10.0, 1_000.0]],
                    m1_log_range=[[10.0, 0.01]],
                ),
            )
        ],
    )

    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )

    short_close = next(
        order
        for order in out["orders"]
        if order["order_type"] == "close_ema_anchor_short"
    )
    assert short_close["price"] == 0.01


def test_sub_tick_ema_anchor_bid_suppresses_long_entry_above_the_book():
    import passivbot_rust as pbr

    strategy = {
        "base_qty_pct": 0.1,
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "offset": 0.0,
        "offset_psize_weight": 0.0,
    }
    symbol = make_symbol(
        0,
        bid=0.005,
        ask=0.015,
        long_strategy=strategy,
        short_strategy=strategy,
        emas=ema_bundle(
            m1_close=[
                [10.0, 0.005],
                [20.0, 0.005],
                [math.sqrt(10.0 * 20.0), 0.005],
            ],
            m1_volume=[[10.0, 1_000.0]],
            m1_log_range=[[10.0, 0.01]],
        ),
    )
    inp = make_input(balance=1_000.0, strategy_kind="ema_anchor", symbols=[symbol])
    inp["global"]["market_orders_allowed"] = True

    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )

    assert not any(
        order["order_type"] == "entry_ema_anchor_long" for order in out["orders"]
    )


def test_genuinely_above_tick_ema_anchor_ask_rounds_up():
    import passivbot_rust as pbr

    strategy = {
        "base_qty_pct": 0.1,
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "offset": 0.0,
        "offset_psize_weight": 0.0,
    }
    symbol = make_symbol(
        0,
        bid=0.09,
        ask=0.1000000005,
        long_mode="manual",
        short_bp={"n_positions": 1, "total_wallet_exposure_limit": 1.0},
        long_strategy=strategy,
        short_strategy=strategy,
        emas=ema_bundle(
            m1_close=[
                [10.0, 0.1000000005],
                [20.0, 0.1000000005],
                [math.sqrt(10.0 * 20.0), 0.1000000005],
            ],
            m1_volume=[[10.0, 1_000.0]],
            m1_log_range=[[10.0, 0.01]],
        ),
    )
    symbol["exchange"] = exchange_params(price_step=0.1)
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(
            short_overrides={"n_positions": 1, "total_wallet_exposure_limit": 1.0}
        ),
        strategy_kind="ema_anchor",
        symbols=[symbol],
    )

    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )

    short_entry = next(
        order
        for order in out["orders"]
        if order["order_type"] == "entry_ema_anchor_short"
    )
    assert short_entry["price"] == 0.2


def test_ema_anchor_entry_double_down_factor_scales_same_side_qty_only():
    import passivbot_rust as pbr

    base_strategy = {
        "base_qty_pct": 0.1,
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "offset": 0.0,
        "offset_psize_weight": 0.0,
        "entry_double_down_factor": 2.0,
    }
    long_inp = make_input(
        balance=1_000.0,
        strategy_kind="ema_anchor",
        global_bp=bot_params_pair(
            short_overrides={"n_positions": 0, "total_wallet_exposure_limit": 0.0}
        ),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_pos_size=1.0,
                long_pos_price=100.0,
                long_strategy=base_strategy,
                short_strategy=base_strategy,
            )
        ],
    )
    base_long = copy.deepcopy(long_inp)
    base_long["symbols"][0]["long"]["position"] = {"size": 0.0, "price": 0.0}

    scaled_long = next(
        o for o in compute(pbr, long_inp)["orders"] if o["order_type"] == "entry_ema_anchor_long"
    )
    neutral_long = next(
        o
        for o in compute(pbr, base_long)["orders"]
        if o["order_type"] == "entry_ema_anchor_long"
    )
    assert scaled_long["qty"] > neutral_long["qty"]
    assert scaled_long["qty"] == pytest.approx(1.2)
    assert neutral_long["qty"] == pytest.approx(1.0)

    short_inp = make_input(
        balance=1_000.0,
        strategy_kind="ema_anchor",
        global_bp=bot_params_pair(
            short_overrides={"n_positions": 1, "total_wallet_exposure_limit": 1.0}
        ),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                short_pos_size=-1.0,
                short_pos_price=100.0,
                long_strategy=base_strategy,
                short_strategy=base_strategy,
            )
        ],
    )
    base_short = copy.deepcopy(short_inp)
    base_short["symbols"][0]["short"]["position"] = {"size": 0.0, "price": 0.0}

    scaled_short = next(
        o for o in compute(pbr, short_inp)["orders"] if o["order_type"] == "entry_ema_anchor_short"
    )
    neutral_short = next(
        o
        for o in compute(pbr, base_short)["orders"]
        if o["order_type"] == "entry_ema_anchor_short"
    )
    assert abs(scaled_short["qty"]) > abs(neutral_short["qty"])
    assert scaled_short["qty"] == pytest.approx(-1.2)
    assert neutral_short["qty"] == pytest.approx(-1.0)


def test_ema_anchor_respects_runtime_budget_for_base_clip_size():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        strategy_kind="ema_anchor",
        global_bp=bot_params_pair(),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_bp={
                    "wallet_exposure_limit": 1.0,
                    "risk_we_excess_allowance_pct": 0.0,
                },
                long_strategy={
                    "base_qty_pct": 0.1,
                    "ema_span_0": 10.0,
                    "ema_span_1": 20.0,
                    "offset": 0.0,
                    "offset_psize_weight": 0.0,
                },
                short_strategy={
                    "base_qty_pct": 0.1,
                    "ema_span_0": 10.0,
                    "ema_span_1": 20.0,
                    "offset": 0.0,
                    "offset_psize_weight": 0.0,
                },
            )
        ],
    )
    inp["symbols"][0]["long"]["runtime_budget"] = {
        "configured_wallet_exposure_limit": 1.0,
        "effective_wallet_exposure_limit": 0.3,
        "configured_n_positions": 1,
        "effective_n_positions": 1,
    }

    out = compute(pbr, inp)
    assert out["orders"][0]["qty"] == pytest.approx(0.3)


def test_twel_reduce_overweight_uses_effective_tradable_slots():
    import passivbot_rust as pbr

    global_bp = bot_params_pair(
        long_overrides={
            "n_positions": 4,
            "total_wallet_exposure_limit": 0.5,
            "risk_twel_enforcer_threshold": 1.0,
            "risk_twel_enforcer_policy": "reduce_overweight",
        }
    )
    inp = make_input(
        balance=1_000.0,
        global_bp=global_bp,
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_pos_size=2.5,
                long_pos_price=100.0,
                long_bp={
                    "wallet_exposure_limit": 0.4,
                    "risk_wel_enforcer_threshold": 2.0,
                },
            ),
            make_symbol(
                1,
                bid=95.0,
                ask=95.0,
                long_pos_size=2.6,
                long_pos_price=100.0,
                long_bp={
                    "wallet_exposure_limit": 0.4,
                    "risk_wel_enforcer_threshold": 2.0,
                },
            ),
        ],
    )

    out = compute(pbr, inp)
    twel_closes = [
        order for order in out["orders"] if order["order_type"] == "close_auto_reduce_twel_long"
    ]
    assert twel_closes
    assert {order["symbol_idx"] for order in twel_closes} == {1}


def test_twel_reduce_overweight_relaxes_floor_when_tradable_slots_expand():
    import passivbot_rust as pbr

    global_bp = bot_params_pair(
        long_overrides={
            "n_positions": 4,
            "total_wallet_exposure_limit": 0.5,
            "risk_twel_enforcer_threshold": 1.0,
            "risk_twel_enforcer_policy": "reduce_overweight",
        }
    )
    common_bp = {
        "wallet_exposure_limit": 0.4,
        "risk_wel_enforcer_threshold": 2.0,
    }
    inp = make_input(
        balance=1_000.0,
        global_bp=global_bp,
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_pos_size=2.5,
                long_pos_price=100.0,
                long_bp=common_bp,
            ),
            make_symbol(
                1,
                bid=95.0,
                ask=95.0,
                long_pos_size=2.6,
                long_pos_price=100.0,
                long_bp=common_bp,
            ),
            make_symbol(2, bid=100.0, ask=100.0, long_bp=common_bp),
            make_symbol(3, bid=100.0, ask=100.0, long_bp=common_bp),
        ],
    )

    out = compute(pbr, inp)
    twel_closes = [
        order for order in out["orders"] if order["order_type"] == "close_auto_reduce_twel_long"
    ]
    assert twel_closes
    assert {order["symbol_idx"] for order in twel_closes} == {0}


def test_twel_reduce_overweight_repairs_when_no_symbols_eligible():
    import passivbot_rust as pbr

    global_bp = bot_params_pair(
        long_overrides={
            "n_positions": 4,
            "total_wallet_exposure_limit": 0.5,
            "risk_twel_enforcer_threshold": 1.0,
            "risk_twel_enforcer_policy": "reduce_overweight",
        }
    )
    inp = make_input(
        balance=1_000.0,
        global_bp=global_bp,
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                tradable=False,
                long_pos_size=2.6,
                long_pos_price=100.0,
                long_bp={
                    "wallet_exposure_limit": 0.4,
                    "risk_wel_enforcer_threshold": 2.0,
                },
            ),
            make_symbol(
                1,
                bid=95.0,
                ask=95.0,
                tradable=False,
                long_pos_size=2.6,
                long_pos_price=100.0,
                long_bp={
                    "wallet_exposure_limit": 0.4,
                    "risk_wel_enforcer_threshold": 2.0,
                },
            ),
        ],
    )

    out = compute(pbr, inp)
    twel_closes = [
        order for order in out["orders"] if order["order_type"] == "close_auto_reduce_twel_long"
    ]
    assert twel_closes


def test_ema_anchor_volatility_weights_widen_quotes():
    import passivbot_rust as pbr

    base_strategy = {
        "base_qty_pct": 0.1,
        "ema_span_0": 10.0,
        "ema_span_1": 20.0,
        "offset": 0.01,
        "offset_volatility_ema_span_1m": 15.0,
        "offset_volatility_1m_weight": 2.0,
        "offset_volatility_ema_span_1h": 8.0,
        "offset_volatility_1h_weight": 3.0,
        "offset_psize_weight": 0.0,
    }
    calm = make_input(
        balance=1_000.0,
        strategy_kind="ema_anchor",
        global_bp=bot_params_pair(
            short_overrides={
                "n_positions": 0,
                "total_wallet_exposure_limit": 0.0,
            }
        ),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_pos_size=1.0,
                long_pos_price=100.0,
                long_strategy=base_strategy,
                short_strategy=base_strategy,
                emas=ema_bundle(
                    m1_close=[[10.0, 100.0], [20.0, 100.0], [math.sqrt(10.0 * 20.0), 100.0]],
                    m1_log_range=[[10.0, 0.01], [15.0, 0.0]],
                    h1_log_range=[[8.0, 0.0]],
                ),
            )
        ],
    )
    wide = copy.deepcopy(calm)
    wide["symbols"][0]["emas"]["m1"]["log_range"] = [[10.0, 0.01], [15.0, 0.02]]
    wide["symbols"][0]["emas"]["h1"]["log_range"] = [[8.0, 0.03]]

    calm_out = compute(pbr, calm)
    wide_out = compute(pbr, wide)

    calm_entry = next(o for o in calm_out["orders"] if o["order_type"] == "entry_ema_anchor_long")
    calm_close = next(o for o in calm_out["orders"] if o["order_type"] == "close_ema_anchor_long")
    wide_entry = next(o for o in wide_out["orders"] if o["order_type"] == "entry_ema_anchor_long")
    wide_close = next(o for o in wide_out["orders"] if o["order_type"] == "close_ema_anchor_long")

    assert calm_entry["price"] == pytest.approx(99.0)
    assert calm_close["price"] == pytest.approx(101.0)
    assert wide_entry["price"] < calm_entry["price"]
    assert wide_close["price"] > calm_close["price"]


def test_ema_anchor_one_way_mode_blocks_short_entries_while_long_position_exists():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        strategy_kind="ema_anchor",
        global_bp=bot_params_pair(
            short_overrides={
                "n_positions": 1,
                "total_wallet_exposure_limit": 1.0,
            }
        ),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_pos_size=1.0,
                long_pos_price=100.0,
                long_strategy={
                    "base_qty_pct": 0.1,
                    "ema_span_0": 10.0,
                    "ema_span_1": 20.0,
                    "offset": 0.01,
                    "offset_psize_weight": 0.0,
                },
                short_strategy={
                    "base_qty_pct": 0.1,
                    "ema_span_0": 10.0,
                    "ema_span_1": 20.0,
                    "offset": 0.01,
                    "offset_psize_weight": 0.0,
                },
            )
        ],
    )
    inp["global"]["hedge_mode"] = False

    out = compute(pbr, inp)

    assert any(o["pside"] == "long" and o["order_type"] == "entry_ema_anchor_long" for o in out["orders"])
    assert any(o["pside"] == "long" and o["order_type"] == "close_ema_anchor_long" for o in out["orders"])
    assert not any(
        o["pside"] == "short" and o["order_type"].startswith("entry_") for o in out["orders"]
    )
    assert out["diagnostics"]["symbol_states"][0]["short"]["active"] is False


def test_ema_anchor_one_way_mode_blocks_long_entries_while_short_position_exists():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        strategy_kind="ema_anchor",
        global_bp=bot_params_pair(
            short_overrides={
                "n_positions": 1,
                "total_wallet_exposure_limit": 1.0,
            }
        ),
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                short_pos_size=-1.0,
                short_pos_price=100.0,
                long_strategy={
                    "base_qty_pct": 0.1,
                    "ema_span_0": 10.0,
                    "ema_span_1": 20.0,
                    "offset": 0.01,
                    "offset_psize_weight": 0.0,
                },
                short_strategy={
                    "base_qty_pct": 0.1,
                    "ema_span_0": 10.0,
                    "ema_span_1": 20.0,
                    "offset": 0.01,
                    "offset_psize_weight": 0.0,
                },
            )
        ],
    )
    inp["global"]["hedge_mode"] = False

    out = compute(pbr, inp)

    assert any(
        o["pside"] == "short" and o["order_type"] == "entry_ema_anchor_short" for o in out["orders"]
    )
    assert any(
        o["pside"] == "short" and o["order_type"] == "close_ema_anchor_short" for o in out["orders"]
    )
    assert not any(
        o["pside"] == "long" and o["order_type"].startswith("entry_") for o in out["orders"]
    )
    assert out["diagnostics"]["symbol_states"][0]["long"]["active"] is False


def test_json_non_tradable_forced_normal_flat_symbol_does_not_require_ema():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                tradable=False,
                long_mode="normal",
                emas=ema_bundle(m1_close=[], m1_volume=[], m1_log_range=[]),
            )
        ],
    )
    out = compute(pbr, inp)

    assert out["orders"] == []
    assert out["diagnostics"]["symbol_states"][0]["long"]["active"] is False
    assert out["diagnostics"]["symbol_states"][0]["long"]["allow_initial"] is False


def test_side_zero_wel_excludes_only_that_side_from_active_slots():
    import passivbot_rust as pbr

    global_bp = bot_params_pair(
        long_overrides={"n_positions": 4, "total_wallet_exposure_limit": 1000.0},
        short_overrides={
            "n_positions": 4,
            "total_wallet_exposure_limit": 1000.0,
            "wallet_exposure_limit": 1.0,
        },
    )
    symbols = []
    for idx in range(4):
        long_bp = {"wallet_exposure_limit": 0.0} if idx == 3 else None
        symbols.append(
            make_symbol(
                idx,
                bid=100.0,
                ask=100.0,
                long_bp=long_bp,
                short_bp={
                    "n_positions": 4,
                    "total_wallet_exposure_limit": 1000.0,
                    "wallet_exposure_limit": 1.0,
                },
            )
        )

    out = compute(pbr, make_input(balance=1_000_000.0, global_bp=global_bp, symbols=symbols))
    states = out["diagnostics"]["symbol_states"]

    assert sum(state["long"]["active"] for state in states) == 3
    assert sum(state["short"]["active"] for state in states) == 4
    assert states[3]["long"]["active"] is False
    assert states[3]["short"]["active"] is True


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
    assert o["qty"] == -1.5


def test_off_tick_book_panic_limit_is_quantized_and_passes_live_validation():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[
            make_symbol(
                0,
                bid=100.003,
                ask=100.007,
                long_mode="panic",
                long_pos_size=1.5,
                long_pos_price=100.0,
            )
        ],
    )
    out = compute(pbr, inp)

    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )
    assert len(out["orders"]) == 1
    assert out["orders"][0]["execution_type"] == "limit"
    assert out["orders"][0]["price"] == 99.99


@pytest.mark.parametrize(
    ("pside", "expected_price"),
    [("long", 0.2), ("short", 0.3)],
)
def test_tick_aligned_panic_limit_does_not_skip_tick_from_float_noise(
    pside, expected_price
):
    import passivbot_rust as pbr

    symbol = make_symbol(
        0,
        bid=0.2,
        ask=0.3,
        long_mode="panic" if pside == "long" else "manual",
        long_pos_size=1.0 if pside == "long" else 0.0,
        long_pos_price=0.4 if pside == "long" else 0.0,
        short_mode="panic" if pside == "short" else "manual",
        short_pos_size=-1.0 if pside == "short" else 0.0,
        short_pos_price=0.1 if pside == "short" else 0.0,
        short_bp={
            "n_positions": 1,
            "total_wallet_exposure_limit": 1.0,
            "wallet_exposure_limit": 1.0,
        },
    )
    symbol["exchange"] = exchange_params(price_step=0.1)
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(
            short_overrides={
                "n_positions": 1,
                "total_wallet_exposure_limit": 1.0,
            }
        ),
        symbols=[symbol],
    )
    out = compute(pbr, inp)

    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )
    assert len(out["orders"]) == 1
    assert out["orders"][0]["order_type"] == f"close_panic_{pside}"
    assert out["orders"][0]["price"] == expected_price


def test_genuinely_above_tick_short_panic_keeps_protective_offset():
    import passivbot_rust as pbr

    symbol = make_symbol(
        0,
        bid=0.1000000005,
        ask=0.2,
        long_mode="manual",
        short_mode="panic",
        short_pos_size=-1.0,
        short_pos_price=0.1,
        short_bp={
            "n_positions": 1,
            "total_wallet_exposure_limit": 1.0,
            "wallet_exposure_limit": 1.0,
        },
    )
    symbol["exchange"] = exchange_params(price_step=0.1)
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(
            short_overrides={"n_positions": 1, "total_wallet_exposure_limit": 1.0}
        ),
        symbols=[symbol],
    )

    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )

    assert len(out["orders"]) == 1
    assert out["orders"][0]["order_type"] == "close_panic_short"
    assert out["orders"][0]["price"] == 0.3


def test_low_off_tick_book_panic_limit_stays_positive_and_passes_live_validation():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[
            make_symbol(
                0,
                bid=0.01,
                ask=0.015,
                long_mode="panic",
                long_pos_size=1.5,
                long_pos_price=0.02,
            )
        ],
    )
    out = compute(pbr, inp)

    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )
    assert len(out["orders"]) == 1
    assert out["orders"][0]["execution_type"] == "limit"
    assert out["orders"][0]["price"] == 0.01


def test_off_step_full_panic_close_passes_live_validation():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_mode="panic",
                long_pos_size=1.005,
                long_pos_price=100.0,
            )
        ],
    )
    out = compute(pbr, inp)

    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )
    assert len(out["orders"]) == 1
    assert out["orders"][0]["qty"] == -1.005


@pytest.mark.parametrize(
    ("qty_step", "min_qty", "expected_qty"),
    [(0.01, 0.015, 0.02), (0.01, 0.07, 0.07), (0.03, 0.33, 0.33)],
)
def test_exchange_min_qty_is_quantized_without_overshooting_aligned_values(
    qty_step, min_qty, expected_qty
):
    import passivbot_rust as pbr

    symbol = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_bp={"entry_initial_qty_pct": 0.0},
    )
    symbol["exchange"].update(
        {"qty_step": qty_step, "min_qty": min_qty, "min_cost": 0.0}
    )
    inp = make_input(balance=40.0, symbols=[symbol])
    out = compute(pbr, inp)

    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )
    entry = next(
        order for order in out["orders"] if order["order_type"].startswith("entry_")
    )
    assert abs(entry["qty"]) == expected_qty


def test_positive_sub_step_effective_minimum_emits_one_contract():
    import passivbot_rust as pbr

    symbol = make_symbol(
        0,
        bid=1e9,
        ask=1e9,
        long_bp={
            "wallet_exposure_limit": 2.0,
            "total_wallet_exposure_limit": 2.0,
            "entry_initial_qty_pct": 0.1,
        },
    )
    symbol["exchange"].update(
        {"qty_step": 1.0, "min_qty": 0.0, "min_cost": 1.0}
    )
    inp = make_input(
        balance=1e9,
        global_bp=bot_params_pair(
            long_overrides={
                "wallet_exposure_limit": 2.0,
                "total_wallet_exposure_limit": 2.0,
            }
        ),
        symbols=[symbol],
    )
    out = compute(pbr, inp)

    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )
    entry = next(
        order for order in out["orders"] if order["order_type"].startswith("entry_")
    )
    assert abs(entry["qty"]) == 1.0


def test_panic_close_order_type_is_side_local():
    import passivbot_rust as pbr

    global_bp = bot_params_pair(
        long_overrides={"hsl_enabled": False, "hsl_panic_close_order_type": "limit"},
        short_overrides={
            "hsl_enabled": True,
            "hsl_panic_close_order_type": "market",
            "n_positions": 1,
            "total_wallet_exposure_limit": 1.0,
        },
    )
    inp = make_input(
        balance=1_000.0,
        global_bp=global_bp,
        symbols=[
            make_symbol(
                0,
                bid=95.0,
                ask=95.0,
                long_mode="panic",
                short_mode="panic",
                long_pos_size=1.5,
                long_pos_price=100.0,
                short_pos_size=-1.5,
                short_pos_price=100.0,
                long_bp={"hsl_enabled": True, "hsl_panic_close_order_type": "market"},
                short_bp={
                    "hsl_enabled": True,
                    "hsl_panic_close_order_type": "limit",
                },
            )
        ],
    )

    out = compute(pbr, inp)
    assert inp["global"].get("market_orders_allowed", False) is False
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )
    by_pside = {o["pside"]: o for o in out["orders"]}

    assert by_pside["long"]["order_type"] == "close_panic_long"
    assert by_pside["long"]["execution_type"] == "market"
    assert by_pside["short"]["order_type"] == "close_panic_short"
    assert by_pside["short"]["execution_type"] == "limit"


def test_panic_close_order_type_rejects_invalid_values():
    import passivbot_rust as pbr

    global_bp = bot_params_pair(
        long_overrides={"hsl_panic_close_order_type": "iceberg"},
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[])

    with pytest.raises(ValueError, match="hsl_panic_close_order_type"):
        compute(pbr, inp)


def test_panic_close_order_type_rejects_invalid_symbol_value():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=100.0,
                long_bp={"hsl_panic_close_order_type": "iceberg"},
            )
        ],
    )

    with pytest.raises(
        ValueError, match=r"symbols\[0\]\.long\.bot_params\.hsl_panic_close_order_type"
    ):
        compute(pbr, inp)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"hsl_ema_span_minutes": 0.5}, r"bot\.long\.hsl_ema_span_minutes"),
        ({"hsl_red_threshold": 0.0}, r"bot\.long\.hsl_red_threshold"),
        (
            {"hsl_red_threshold": 0.2, "hsl_no_restart_drawdown_threshold": 0.1},
            r"bot\.long\.hsl_no_restart_drawdown_threshold",
        ),
        ({"hsl_restart_after_red_policy": "sometimes"}, r"bot\.long\.hsl_restart_after_red_policy"),
        ({"risk_we_excess_allowance_pct": -0.01}, r"bot\.long\.risk_we_excess_allowance_pct"),
        ({"unstuck_ema_dist": -1.0}, r"bot\.long\.unstuck_ema_dist"),
    ],
)
def test_json_rejects_invalid_global_hsl_risk_unstuck_values(overrides, match):
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(long_overrides=overrides),
        symbols=[],
    )

    with pytest.raises(ValueError, match=match):
        compute(pbr, inp)


def test_json_rejects_invalid_symbol_hsl_risk_unstuck_values():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[
            make_symbol(
                0,
                bid=100.0,
                ask=101.0,
                long_bp={"unstuck_close_pct": 1.01},
            )
        ],
    )

    with pytest.raises(ValueError, match=r"symbols\[0\]\.long\.bot_params\.unstuck_close_pct"):
        compute(pbr, inp)


def test_graceful_stop_blocks_initial_entries_only():
    import passivbot_rust as pbr

    # No position => no entries.
    inp_no_pos = make_input(
        balance=1_000.0,
        symbols=[make_symbol(0, bid=100.0, ask=100.0, long_mode="graceful_stop")],
    )
    out_no_pos = compute(pbr, inp_no_pos)
    assert out_no_pos["orders"] == []

    # With a position, GracefulStop preserves Normal prices/quantities while
    # promoting its closes to risk-critical execution priority.
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
    )
    inp_normal = make_input(
        balance=1_000.0, symbols=[{**sym, "long": {**sym["long"], "mode": None}}]
    )
    inp_gs = make_input(
        balance=1_000.0,
        symbols=[{**sym, "long": {**sym["long"], "mode": "graceful_stop"}}],
    )
    out_normal = compute(pbr, inp_normal)
    out_gs = compute(pbr, inp_gs)
    assert [
        {key: value for key, value in order.items() if key != "execution_priority"}
        for order in out_normal["orders"]
    ] == [
        {key: value for key, value in order.items() if key != "execution_priority"}
        for order in out_gs["orders"]
    ]
    assert {order["execution_priority"] for order in out_normal["orders"]} == {
        "ordinary"
    }
    assert {
        order["execution_priority"]
        for order in out_gs["orders"]
        if order["order_type"].startswith("close_")
    } == {"risk_critical"}
    assert {
        order["execution_priority"]
        for order in out_gs["orders"]
        if order["order_type"].startswith("entry_")
    } == {"ordinary"}
    assert (
        out_normal["diagnostics"]["symbol_states"][0]["long"]["effective_mode"]
        == "normal"
    )
    assert (
        out_gs["diagnostics"]["symbol_states"][0]["long"]["effective_mode"] == "normal"
    )
    assert out_normal["diagnostics"]["symbol_states"][0]["long"]["input_mode"] is None
    assert (
        out_gs["diagnostics"]["symbol_states"][0]["long"]["input_mode"]
        == "graceful_stop"
    )
    assert any(o["order_type"].startswith("close_") for o in out_gs["orders"])

    # Rust treats every exactly nonzero position as held, including sub-epsilon
    # exchange dust, so graceful stop still uses effective normal generation.
    tiny_sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_mode="graceful_stop",
        long_pos_size=1e-13,
        long_pos_price=100.0,
    )
    tiny_inp = make_input(balance=1_000.0, symbols=[tiny_sym])
    tiny_out = compute(pbr, tiny_inp)
    reconciler.validate_rust_orchestrator_output(
        tiny_out, {0: "BTC/USDT:USDT"}, tiny_inp
    )
    assert (
        tiny_out["diagnostics"]["symbol_states"][0]["long"]["effective_mode"]
        == "normal"
    )
    assert any(order["order_type"].startswith("entry_") for order in tiny_out["orders"])


def test_forager_respects_n_positions_selects_one_coin():
    import passivbot_rust as pbr

    global_bp = bot_params_pair(
        long_overrides={
            "n_positions": 1,
            "total_wallet_exposure_limit": 1.0,
            "filter_volume_drop_pct": 0.5,
            "filter_volatility_drop_pct": 0.0,
            "filter_volume_ema_span_1m": 10.0,
            "filter_volatility_ema_span_1m": 10.0,
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
            "filter_volume_ema_span_1m": 10.0,
            "filter_volatility_ema_span_1m": 10.0,
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
            "filter_volume_ema_span_1m": 10.0,
            "filter_volatility_ema_span_1m": 10.0,
        },
    )

    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym0, sym1])
    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out,
        {0: "BTC/USDT:USDT", 1: "ETH/USDT:USDT"},
        inp,
    )
    assert out["orders"], "expected at least one order"
    assert {o["symbol_idx"] for o in out["orders"]} == {1}
    selection = out["diagnostics"]["forager_selections"][0]
    assert selection["pside"] == "long"
    assert selection["ranking_required"] is True
    assert selection["selected_symbol_indices"] == [1]
    assert selection["top_scores"][0]["symbol_idx"] == 1
    assert selection["top_scores"][0]["selected"] is True


def test_single_eligible_coin_does_not_require_forager_feature_bundle():
    import passivbot_rust as pbr

    side_params = {
        "n_positions": 1,
        "total_wallet_exposure_limit": 1.0,
        "filter_volume_drop_pct": 0.5,
        "filter_volume_ema_span_1m": 10.0,
    }
    symbol = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_bp=side_params,
    )
    symbol["forager_m1"] = {
        "close": [],
        "volume": [],
        "log_range": [],
    }
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(long_overrides=side_params),
        symbols=[symbol],
    )

    out = compute(pbr, inp)

    assert out["orders"]
    assert {order["symbol_idx"] for order in out["orders"]} == {0}
    selection = next(
        item
        for item in out["diagnostics"]["forager_selections"]
        if item["pside"] == "long"
    )
    assert selection["selected_symbol_indices"] == [0]
    assert selection["ranking_required"] is False
    assert selection["top_scores"] == []


def test_forager_ranks_remaining_candidates_after_ineligible_position_consumes_slot():
    import passivbot_rust as pbr

    side_params = {
        "n_positions": 2,
        "total_wallet_exposure_limit": 1.0,
        "filter_volume_drop_pct": 0.5,
        "filter_volatility_drop_pct": 0.0,
        "filter_volume_ema_span_1m": 10.0,
        "filter_volatility_ema_span_1m": 10.0,
    }
    held_ineligible = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        tradable=False,
        long_pos_size=1.0,
        long_pos_price=100.0,
        long_bp=side_params,
    )
    lower_ranked = make_symbol(
        1,
        bid=100.0,
        ask=100.0,
        long_bp=side_params,
        emas=ema_bundle(m1_volume=[[10.0, 10.0]]),
    )
    higher_ranked = make_symbol(
        2,
        bid=100.0,
        ask=100.0,
        long_bp=side_params,
        emas=ema_bundle(m1_volume=[[10.0, 20.0]]),
    )
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(long_overrides=side_params),
        symbols=[held_ineligible, lower_ranked, higher_ranked],
    )

    out = compute(pbr, inp)

    selection = next(
        item
        for item in out["diagnostics"]["forager_selections"]
        if item["pside"] == "long"
    )
    assert selection["slots_to_fill"] == 1
    assert selection["selected_symbol_indices"] == [2]
    assert not any(
        order["symbol_idx"] == 1 and order["order_type"].startswith("entry_")
        for order in out["orders"]
    )
    assert any(
        order["symbol_idx"] == 2 and order["order_type"].startswith("entry_")
        for order in out["orders"]
    )


def test_live_authorized_missing_forager_feature_scopes_only_candidate_side():
    import passivbot_rust as pbr

    side_params = {
        "n_positions": 1,
        "total_wallet_exposure_limit": 1.0,
        "filter_volume_drop_pct": 0.5,
        "filter_volatility_drop_pct": 0.0,
        "filter_volume_ema_span_1m": 10.0,
        "filter_volatility_ema_span_1m": 10.0,
    }
    global_bp = bot_params_pair(long_overrides=side_params)
    unavailable = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_bp=side_params,
        emas=ema_bundle(m1_volume=[]),
    )
    unavailable["allow_missing_strategy_inputs"] = True
    available = make_symbol(
        1,
        bid=100.0,
        ask=100.0,
        long_bp=side_params,
        emas=ema_bundle(m1_volume=[[10.0, 11.0]]),
    )

    inp = make_input(
        balance=1_000.0,
        global_bp=global_bp,
        symbols=[unavailable, available],
    )
    strict_inp = copy.deepcopy(inp)
    strict_inp["symbols"][0]["allow_missing_strategy_inputs"] = False

    with pytest.raises(ValueError, match="MissingEma"):
        compute(pbr, strict_inp)

    out = compute(pbr, inp)

    assert any(
        order["symbol_idx"] == 1 and order["order_type"].startswith("entry_")
        for order in out["orders"]
    )
    assert not any(
        order["symbol_idx"] == 0 and order["order_type"].startswith("entry_")
        for order in out["orders"]
    )
    assert {
        "strategy_input_unavailable": {
            "symbol_idx": 0,
            "pside": "long",
            "scope": "forager_selection",
        }
    } in out["diagnostics"]["warnings"]


def test_json_rejects_invalid_forager_hysteresis_pct():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[
            make_symbol(0, bid=100.0, ask=100.0),
            make_symbol(1, bid=100.0, ask=100.0),
        ],
    )
    inp["forager_hysteresis"] = {
        "score_hysteresis_pct": -0.1,
        "incumbent_long": [],
        "incumbent_short": [],
    }

    with pytest.raises(ValueError, match="forager_score_hysteresis_pct"):
        compute(pbr, inp)


def test_select_forager_candidates_all_zero_weights_fall_back_to_ema_readiness_only():
    import passivbot_rust as pbr

    candidates = [
        {
            "index": 0,
            "enabled": True,
            "volume_score": 1.0,
            "volatility_score": 0.1,
            "bid": 100.0,
            "ask": 100.0,
            "ema_lower": 100.0,
            "ema_upper": 100.0,
            "entry_initial_ema_dist": 0.0,
        },
        {
            "index": 1,
            "enabled": True,
            "volume_score": 1.0,
            "volatility_score": 0.9,
            "bid": 100.0,
            "ask": 100.0,
            "ema_lower": 90.0,
            "ema_upper": 100.0,
            "entry_initial_ema_dist": 0.0,
        },
    ]
    selected = pbr.select_forager_candidates_py(
        candidates,
        "long",
        1,
        0.0,
        {
            "volume": 0.0,
            "ema_readiness": 0.0,
            "volatility": 0.0,
        },
        True,
    )
    assert selected == [0]


def test_json_output_is_deterministic():
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[
            make_symbol(
                0, bid=100.0, ask=100.0, long_pos_size=1.0, long_pos_price=100.0
            )
        ],
    )
    out1 = compute(pbr, inp)
    out2 = compute(pbr, inp)
    assert out1 == out2


def test_unstuck_and_ordinary_close_coexist_and_are_capped():
    import passivbot_rust as pbr

    balance = 1_000.0
    long_bp = {
        "unstuck_close_pct": 0.5,
        "unstuck_threshold": 0.001,
        "unstuck_ema_dist": 0.0,
        "unstuck_loss_allowance_pct": 0.01,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)

    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=10.0,
        long_pos_price=100.0,
        long_bp=long_bp,
        long_strategy=adaptive_strategy_params(
            close={"qty_pct": 1.0, "threshold_base_pct": 0.01},
        ),
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
        o
        for o in out["orders"]
        if o["order_type"].startswith("close_") and o["pside"] == "long"
    ]
    total_close_qty = -sum(o["qty"] for o in closes if o["qty"] < 0.0)
    assert total_close_qty <= 10.0 + 1e-9


def test_trailing_grid_v7_emits_compatible_unstuck_and_trailing_closes_together():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 10.0,
        "total_wallet_exposure_limit": 10.0,
        "risk_wel_enforcer_enabled": False,
        "risk_twel_enforcer_enabled": False,
        "unstuck_ema_gating_enabled": False,
        "unstuck_close_pct": 0.0195,
        "unstuck_threshold": 0.9,
        "unstuck_loss_allowance_pct": 0.01,
    }
    strategy = trailing_grid_v7_strategy_params(
        close_overrides={
            "trailing_grid_ratio": 1.0,
            "trailing_qty_pct": 0.2624,
            "trailing_retracement_pct": 0.005,
            "trailing_threshold_pct": 0.01,
        }
    )
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=95.6,
        long_pos_price=100.0,
        long_bp=long_bp,
        long_strategy=strategy,
        short_strategy=strategy,
    )
    sym["long"]["trailing"] = {
        "min_since_open": 99.0,
        "max_since_min": 102.0,
        "max_since_open": 102.0,
        "min_since_max": 100.0,
    }
    inp = make_input(
        balance=1_000.0,
        strategy_kind="trailing_grid_v7",
        global_bp=bot_params_pair(long_overrides=long_bp),
        symbols=[sym],
    )
    inp["global"]["unstuck_allowance_long"] = 1e9

    out = compute(pbr, inp)
    closes = {
        order["order_type"]: order
        for order in out["orders"]
        if order["pside"] == "long" and order["qty"] < 0.0
    }

    assert closes["close_unstuck_long"]["qty"] == pytest.approx(-1.95)
    assert closes["close_trailing_long"]["qty"] == pytest.approx(-26.24)
    assert sum(abs(order["qty"]) for order in closes.values()) <= 95.6 + 1e-9


def test_unstuck_uses_symbol_loss_allowance_pct_for_loss_cap():
    import passivbot_rust as pbr

    balance = 2_000.0
    long_bp = {
        "total_wallet_exposure_limit": 1.5,
        "wallet_exposure_limit": 1.5,
        "unstuck_close_pct": 0.5,
        "unstuck_threshold": 0.001,
        "unstuck_ema_dist": 0.0,
        "unstuck_loss_allowance_pct": 0.005,
    }
    global_bp = bot_params_pair(
        long_overrides={
            "total_wallet_exposure_limit": 1.5,
            "unstuck_loss_allowance_pct": 0.02,
        }
    )
    sym = make_symbol(
        0,
        bid=120.0,
        ask=120.0,
        long_pos_size=10.0,
        long_pos_price=130.0,
        long_bp=long_bp,
    )
    inp = make_input(balance=balance, global_bp=global_bp, symbols=[sym])
    inp["global"]["unstuck_allowance_long"] = 1e9

    out = compute(pbr, inp)
    unstuck = [o for o in out["orders"] if o["order_type"] == "close_unstuck_long"]

    assert len(unstuck) == 1
    assert unstuck[0]["qty"] == pytest.approx(-1.5)


@pytest.mark.parametrize(
    ("pside", "position_size", "position_price", "bid", "ask", "expected_price"),
    [
        ("long", 10.0, 130.0, 120.0, 120.003, 120.01),
        ("short", -10.0, 110.0, 120.003, 120.01, 120.0),
    ],
)
def test_auto_unstuck_quantizes_off_tick_book_price_before_live_validation(
    pside,
    position_size,
    position_price,
    bid,
    ask,
    expected_price,
):
    import passivbot_rust as pbr

    side_bp = {
        "n_positions": 1,
        "total_wallet_exposure_limit": 1.5,
        "wallet_exposure_limit": 1.5,
        "unstuck_close_pct": 0.5,
        "unstuck_threshold": 0.001,
        "unstuck_ema_gating_enabled": False,
        "unstuck_loss_allowance_pct": 0.005,
    }
    symbol_kwargs = {
        f"{pside}_pos_size": position_size,
        f"{pside}_pos_price": position_price,
        f"{pside}_bp": side_bp,
    }
    global_kwargs = {f"{pside}_overrides": side_bp}
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(**global_kwargs),
        symbols=[make_symbol(0, bid=bid, ask=ask, **symbol_kwargs)],
    )
    inp["global"][f"unstuck_allowance_{pside}"] = 1e9

    out = compute(pbr, inp)
    unstuck = next(
        order
        for order in out["orders"]
        if order["order_type"] == f"close_unstuck_{pside}"
    )

    assert unstuck["price"] == pytest.approx(expected_price)
    reconciler.validate_rust_orchestrator_output(
        out,
        {0: "BTC/USDT:USDT"},
        inp,
    )


def test_auto_unstuck_allowed_gate_blocks_symbol_allowance():
    import passivbot_rust as pbr

    long_bp = {
        "total_wallet_exposure_limit": 1.5,
        "wallet_exposure_limit": 1.5,
        "unstuck_close_pct": 0.5,
        "unstuck_threshold": 0.001,
        "unstuck_ema_dist": 0.0,
        "unstuck_loss_allowance_pct": 0.005,
    }
    sym = make_symbol(
        0,
        bid=120.0,
        ask=120.0,
        long_pos_size=10.0,
        long_pos_price=130.0,
        long_bp=long_bp,
    )
    inp = make_input(
        balance=2_000.0,
        global_bp=bot_params_pair(long_overrides=long_bp),
        symbols=[sym],
    )
    inp["global"]["auto_unstuck_allowed"] = False

    out = compute(pbr, inp)

    assert all(o["order_type"] != "close_unstuck_long" for o in out["orders"])


def test_unstuck_ema_gating_disabled_skips_missing_ema_requirement():
    import passivbot_rust as pbr

    long_bp = {
        "total_wallet_exposure_limit": 1.5,
        "wallet_exposure_limit": 1.5,
        "unstuck_close_pct": 0.5,
        "unstuck_ema_gating_enabled": False,
        "unstuck_threshold": 0.001,
        "unstuck_ema_dist": 0.0,
        "unstuck_loss_allowance_pct": 0.005,
    }
    sym = make_symbol(
        0,
        bid=120.0,
        ask=120.0,
        long_pos_size=10.0,
        long_pos_price=130.0,
        long_bp=long_bp,
        long_strategy=adaptive_strategy_params(entry={"ema_gate_mode": "disabled"}),
        emas=ema_bundle(m1_close=[]),
    )
    inp = make_input(
        balance=2_000.0,
        global_bp=bot_params_pair(long_overrides=long_bp),
        symbols=[sym],
    )
    inp["global"]["hedge_mode"] = True
    inp["global"]["unstuck_allowance_long"] = 1e9

    out = compute(pbr, inp)

    assert any(o["order_type"] == "close_unstuck_long" for o in out["orders"])


def test_orders_include_entries_and_closes():
    import passivbot_rust as pbr

    long_bp = {
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
        long_bp=long_bp,
        long_strategy=adaptive_strategy_params(
            entry={"initial_qty_pct": 0.1, "threshold_base_pct": 0.01},
            close={"qty_pct": 1.0, "threshold_base_pct": 0.01},
        ),
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    out = compute(pbr, inp)
    order_types = {o["order_type"] for o in out["orders"]}
    assert any(t.startswith("entry_") for t in order_types)
    assert any(t.startswith("close_") for t in order_types)


def test_long_grid_close_uses_position_price_anchor():
    import passivbot_rust as pbr

    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
        long_strategy=adaptive_strategy_params(
            close={"threshold_base_pct": 0.01},
        ),
        emas=ema_bundle(
            m1_close=[
                [10.0, 110.0],
                [20.0, 110.0],
                [math.sqrt(10.0 * 20.0), 110.0],
            ]
        ),
    )

    out = compute(pbr, make_input(balance=1_000.0, symbols=[sym]))
    close = next(o for o in out["orders"] if o["order_type"] == "close_grid_long")
    assert close["price"] == pytest.approx(101.0)


def test_short_grid_close_uses_position_price_anchor():
    import passivbot_rust as pbr

    short_bp = {"n_positions": 1, "total_wallet_exposure_limit": 1.0, "wallet_exposure_limit": 1.0}
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_mode="manual",
        short_pos_size=-1.0,
        short_pos_price=100.0,
        short_mode="normal",
        short_bp=short_bp,
        short_strategy=adaptive_strategy_params(
            close={"threshold_base_pct": 0.01},
        ),
        emas=ema_bundle(
            m1_close=[
                [10.0, 90.0],
                [20.0, 90.0],
                [math.sqrt(10.0 * 20.0), 90.0],
            ]
        ),
    )

    out = compute(
        pbr,
        make_input(balance=1_000.0, global_bp=bot_params_pair(short_overrides=short_bp), symbols=[sym]),
    )
    close = next(o for o in out["orders"] if o["order_type"] == "close_grid_short")
    assert close["price"] == pytest.approx(99.0)


def test_twel_entry_gating_blocks_new_entries():
    import passivbot_rust as pbr

    long_bp = {
        "entry_initial_qty_pct": 0.1,
        "entry_initial_ema_dist": -0.01,
        "entry_grid_spacing_pct": 0.01,
        "total_wallet_exposure_limit": 0.1,
        "wallet_exposure_limit": 0.1,
        "risk_twel_enforcer_threshold": 1.0,
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


def test_min_effective_cost_uses_strategy_initial_qty_pct():
    import passivbot_rust as pbr

    long_bp = {
        "entry_initial_qty_pct": 0.0,
        "total_wallet_exposure_limit": 1.0,
        "wallet_exposure_limit": 1.0,
        "n_positions": 1,
    }
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        effective_min_cost=10.0,
        long_bp=long_bp,
        long_strategy=adaptive_strategy_params(entry={"initial_qty_pct": 0.1}),
    )
    inp = make_input(balance=1_000.0, global_bp=bot_params_pair(long_overrides=long_bp), symbols=[sym])
    inp["global"]["filter_by_min_effective_cost"] = True

    out = compute(pbr, inp)

    assert any(o["order_type"].startswith("entry_") for o in out["orders"])
    assert out["diagnostics"]["min_effective_cost_blocks"] == []


def test_live_validator_accepts_real_min_effective_cost_diagnostic():
    import passivbot_rust as pbr

    long_bp = {
        "entry_initial_qty_pct": 0.01,
        "total_wallet_exposure_limit": 1.0,
        "wallet_exposure_limit": 1.0,
        "n_positions": 1,
    }
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        effective_min_cost=100.0,
        long_bp=long_bp,
    )
    inp = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(long_overrides=long_bp),
        symbols=[sym],
    )
    inp["global"]["filter_by_min_effective_cost"] = True

    out = compute(pbr, inp)

    assert out["diagnostics"]["min_effective_cost_blocks"]
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )


def test_manual_positions_consume_twel_entry_gate_budget():
    import passivbot_rust as pbr

    long_bp = {
        "entry_initial_qty_pct": 0.01,
        "entry_initial_ema_dist": 0.0,
        "entry_grid_spacing_pct": 0.01,
        "total_wallet_exposure_limit": 0.1,
        "wallet_exposure_limit": 0.1,
        "risk_twel_enforcer_threshold": 1.0,
        "n_positions": 1,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    manual_sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_mode="manual",
        long_pos_size=2.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )
    active_sym = make_symbol(
        1,
        bid=100.0,
        ask=100.0,
        long_mode="normal",
        long_bp=long_bp,
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[manual_sym, active_sym])

    out = compute(pbr, inp)
    assert not any(
        o["symbol_idx"] == 1 and o["order_type"].startswith("entry_") for o in out["orders"]
    ), "existing manual exposure must still consume TWE before allowing bot-generated entries"


def test_twel_entry_gating_uses_snapped_balance_not_raw():
    import passivbot_rust as pbr

    long_bp = {
        "entry_initial_qty_pct": 0.1,
        "entry_grid_spacing_pct": 0.02,
        "entry_grid_double_down_factor": 10.0,
        "total_wallet_exposure_limit": 0.2,
        "wallet_exposure_limit": 1.0,
        "risk_twel_enforcer_threshold": 1.0,
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
    inp_low_raw = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp_low_raw["balance_raw"] = 500.0

    inp_high_raw = copy.deepcopy(inp_low_raw)
    inp_high_raw["balance_raw"] = 1_500.0

    out_low_raw = compute(pbr, inp_low_raw)
    out_high_raw = compute(pbr, inp_high_raw)
    entries_low_raw = [
        o for o in out_low_raw["orders"] if o["order_type"].startswith("entry_")
    ]
    entries_high_raw = [
        o for o in out_high_raw["orders"] if o["order_type"].startswith("entry_")
    ]

    assert entries_low_raw, (
        "snapped balance should permit a TWEL-gated entry even when raw balance "
        "would already put current exposure at the TWEL cap"
    )
    assert [
        (o["symbol_idx"], o["order_type"], o["pside"], o["qty"], o["price"])
        for o in entries_low_raw
    ] == [
        (o["symbol_idx"], o["order_type"], o["pside"], o["qty"], o["price"])
        for o in entries_high_raw
    ]


def test_twel_entry_gate_disabled_allows_entries_above_raw_twel():
    import passivbot_rust as pbr

    long_bp = {
        "entry_initial_qty_pct": 0.1,
        "entry_initial_ema_dist": -0.01,
        "entry_grid_spacing_pct": 0.02,
        "entry_grid_double_down_factor": 10.0,
        "total_wallet_exposure_limit": 1.0,
        "wallet_exposure_limit": 0.5,
        "risk_we_excess_allowance_pct": 0.25,
        "risk_twel_entry_gate_enabled": False,
        "risk_twel_enforcer_enabled": False,
        "risk_twel_enforcer_threshold": 0.9,
        "n_positions": 2,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    held_sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_mode="tp_only",
        long_pos_size=9.8,
        long_pos_price=100.0,
        long_bp=long_bp,
    )
    active_sym = make_symbol(1, bid=100.0, ask=100.0, long_mode="normal", long_bp=long_bp)

    out = compute(
        pbr, make_input(balance=1_000.0, global_bp=global_bp, symbols=[held_sym, active_sym])
    )

    assert any(
        o["symbol_idx"] == 1 and o["order_type"].startswith("entry_") for o in out["orders"]
    )


def test_twel_entry_gate_uses_thresholded_cap_below_one():
    import passivbot_rust as pbr

    long_bp = {
        "entry_initial_qty_pct": 0.1,
        "entry_grid_spacing_pct": 0.02,
        "entry_grid_double_down_factor": 10.0,
        "total_wallet_exposure_limit": 1.0,
        "wallet_exposure_limit": 1.0,
        "risk_twel_entry_gate_enabled": True,
        "risk_twel_enforcer_enabled": False,
        "risk_twel_enforcer_threshold": 0.9,
        "n_positions": 1,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=9.2,
        long_pos_price=100.0,
        long_bp=long_bp,
    )

    out = compute(pbr, make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym]))

    assert not any(o["order_type"].startswith("entry_") for o in out["orders"])


def test_twel_entry_gate_caps_threshold_above_one_at_raw_twel():
    import passivbot_rust as pbr

    long_bp = {
        "entry_initial_qty_pct": 0.1,
        "entry_grid_spacing_pct": 0.02,
        "entry_grid_double_down_factor": 10.0,
        "total_wallet_exposure_limit": 1.0,
        "wallet_exposure_limit": 1.0,
        "risk_twel_entry_gate_enabled": True,
        "risk_twel_enforcer_enabled": False,
        "risk_twel_enforcer_threshold": 1.2,
        "n_positions": 1,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=10.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )

    out = compute(pbr, make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym]))

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
    twel_orders = [o for o in out["orders"] if o["order_type"] == "close_auto_reduce_twel_long"]
    assert twel_orders
    assert {o["symbol_idx"] for o in twel_orders} == {1}


def test_twel_enforcer_disabled_emits_no_auto_reduce():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.4,
        "total_wallet_exposure_limit": 0.9,
        "risk_twel_enforcer_enabled": False,
        "risk_twel_enforcer_threshold": 1.0,
        "n_positions": 2,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    symbols = [
        make_symbol(0, bid=50.0, ask=50.0, long_pos_size=8.0, long_pos_price=50.0, long_bp=long_bp),
        make_symbol(1, bid=50.0, ask=50.0, long_pos_size=12.0, long_pos_price=50.0, long_bp=long_bp),
    ]

    out = compute(pbr, make_input(balance=1_000.0, global_bp=global_bp, symbols=symbols))

    assert not any(o["order_type"] == "close_auto_reduce_twel_long" for o in out["orders"])


def test_twel_reduce_portfolio_can_select_underweight_positions():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.5,
        "total_wallet_exposure_limit": 0.9,
        "risk_twel_enforcer_policy": "reduce_portfolio",
        "risk_twel_enforcer_threshold": 1.0,
        "n_positions": 2,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym0 = make_symbol(
        0,
        bid=110.0,
        ask=110.0,
        long_pos_size=4.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )
    sym1 = make_symbol(
        1,
        bid=80.0,
        ask=80.0,
        long_pos_size=10.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )

    out = compute(pbr, make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym0, sym1]))
    twel_orders = [o for o in out["orders"] if o["order_type"] == "close_auto_reduce_twel_long"]

    assert {o["symbol_idx"] for o in twel_orders} == {0, 1}


def test_larger_wel_auto_reduce_wins_over_twel_for_same_position():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.4,
        "risk_wel_enforcer_threshold": 1.0,
        "total_wallet_exposure_limit": 0.5,
        "risk_twel_enforcer_policy": "reduce_portfolio",
        "risk_twel_enforcer_threshold": 1.0,
        "n_positions": 1,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=99.995,
        ask=100.005,
        long_pos_size=6.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )

    out = compute(pbr, make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym]))
    order_types = [o["order_type"] for o in out["orders"]]

    assert "close_auto_reduce_wel_long" in order_types
    wel_order = next(
        order
        for order in out["orders"]
        if order["order_type"] == "close_auto_reduce_wel_long"
    )
    assert wel_order["price"] == 100.01
    assert wel_order["price"] >= sym["order_book"]["ask"]
    assert "close_auto_reduce_twel_long" not in order_types


def test_wel_off_tick_limit_meets_minimum_and_passes_live_validation():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 1.0,
        "risk_wel_enforcer_threshold": 1.0,
        "risk_twel_enforcer_enabled": False,
        "total_wallet_exposure_limit": 1.0,
        "n_positions": 1,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=3.0,
        ask=3.003,
        long_pos_size=10.001,
        long_pos_price=100.0,
        long_bp=long_bp,
    )
    sym["exchange"].update(
        {"qty_step": 0.001, "price_step": 0.01, "min_qty": 0.0, "min_cost": 5.0}
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])

    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )
    wel_order = next(
        order
        for order in out["orders"]
        if order["order_type"] == "close_auto_reduce_wel_long"
    )

    assert wel_order["price"] == 3.01
    assert wel_order["qty"] == pytest.approx(-1.662)
    assert abs(wel_order["qty"]) * wel_order["price"] >= 5.0


def test_larger_wel_auto_reduce_wins_over_unstuck_for_same_position():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.4,
        "risk_wel_enforcer_threshold": 1.0,
        "risk_twel_enforcer_enabled": False,
        "total_wallet_exposure_limit": 1.0,
        "n_positions": 1,
        "unstuck_close_pct": 0.5,
        "unstuck_threshold": 0.001,
        "unstuck_ema_dist": 0.0,
        "unstuck_loss_allowance_pct": 0.01,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=99.995,
        ask=100.005,
        long_pos_size=6.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp["global"]["unstuck_allowance_long"] = 1e9

    out = compute(pbr, inp)
    order_types = [o["order_type"] for o in out["orders"]]

    assert "close_auto_reduce_wel_long" in order_types
    assert "close_unstuck_long" not in order_types
    assert "close_grid_long" in order_types
    assert sum(
        abs(o["qty"])
        for o in out["orders"]
        if o["pside"] == "long" and o["qty"] < 0.0
    ) <= 6.0 + 1e-9


def test_larger_short_wel_auto_reduce_wins_over_unstuck_for_same_position():
    import passivbot_rust as pbr

    short_bp = {
        "wallet_exposure_limit": 0.4,
        "risk_wel_enforcer_threshold": 1.0,
        "risk_twel_enforcer_enabled": False,
        "total_wallet_exposure_limit": 1.0,
        "n_positions": 1,
        "unstuck_close_pct": 0.5,
        "unstuck_threshold": 0.001,
        "unstuck_ema_dist": 0.0,
        "unstuck_loss_allowance_pct": 0.01,
    }
    global_bp = bot_params_pair(short_overrides=short_bp)
    sym = make_symbol(
        0,
        bid=99.995,
        ask=100.005,
        short_pos_size=-6.0,
        short_pos_price=100.0,
        short_bp=short_bp,
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp["global"]["unstuck_allowance_short"] = 1e9

    out = compute(pbr, inp)
    order_types = [o["order_type"] for o in out["orders"]]

    assert "close_auto_reduce_wel_short" in order_types
    wel_order = next(
        order
        for order in out["orders"]
        if order["order_type"] == "close_auto_reduce_wel_short"
    )
    assert wel_order["price"] == 99.99
    assert wel_order["price"] <= sym["order_book"]["bid"]
    assert "close_unstuck_short" not in order_types
    assert "close_grid_short" in order_types
    assert sum(
        abs(o["qty"])
        for o in out["orders"]
        if o["pside"] == "short" and o["qty"] > 0.0
    ) <= 6.0 + 1e-9


def test_loss_gate_falls_back_from_larger_wel_to_smaller_unstuck():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.4,
        "risk_wel_enforcer_threshold": 1.0,
        "risk_twel_enforcer_enabled": False,
        "total_wallet_exposure_limit": 1.0,
        "n_positions": 1,
        "unstuck_close_pct": 0.4,
        "unstuck_threshold": 0.001,
        "unstuck_ema_dist": 0.0,
        "unstuck_loss_allowance_pct": 0.01,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=90.0,
        ask=90.0,
        long_pos_size=6.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp["global"]["unstuck_allowance_long"] = 1e9
    inp["global"]["max_realized_loss_pct"] = 0.019

    out = compute(pbr, inp)
    reconciler.validate_rust_orchestrator_output(
        out, {0: "BTC/USDT:USDT"}, inp
    )
    order_types = [o["order_type"] for o in out["orders"]]

    assert "close_unstuck_long" in order_types
    assert "close_auto_reduce_wel_long" not in order_types
    assert any(
        block["order_type"] == "close_auto_reduce_wel_long"
        for block in out["diagnostics"]["loss_gate_blocks"]
    )


def test_loss_gate_checks_wel_reducer_after_dust_is_absorbed():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.1,
        "risk_wel_enforcer_threshold": 1.0,
        "risk_twel_enforcer_enabled": False,
        "total_wallet_exposure_limit": 2.0,
        "n_positions": 1,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=90.0,
        ask=90.0,
        long_pos_size=11.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )
    sym["exchange"].update(
        {"qty_step": 0.01, "min_qty": 10.0, "min_cost": 0.0, "maker_fee": 0.0}
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp["global"]["max_realized_loss_pct"] = 0.105

    out = compute(pbr, inp)

    assert all(o["order_type"] != "close_auto_reduce_wel_long" for o in out["orders"])
    block = next(
        block
        for block in out["diagnostics"]["loss_gate_blocks"]
        if block["order_type"] == "close_auto_reduce_wel_long"
    )
    assert block["qty"] == pytest.approx(-11.0)
    assert block["projected_balance_after"] == pytest.approx(890.0)
    assert block["balance_floor"] == pytest.approx(895.0)


def test_loss_gate_prioritizes_larger_wel_reducer_across_symbols():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.1,
        "risk_wel_enforcer_threshold": 1.0,
        "risk_twel_enforcer_enabled": False,
        "total_wallet_exposure_limit": 2.0,
        "n_positions": 2,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym0 = make_symbol(
        0,
        bid=90.0,
        ask=90.0,
        long_pos_size=5.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )
    sym1 = make_symbol(
        1,
        bid=90.0,
        ask=90.0,
        long_pos_size=7.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )
    sym0["exchange"]["maker_fee"] = 0.0
    sym1["exchange"]["maker_fee"] = 0.0
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym0, sym1])
    inp["global"]["max_realized_loss_pct"] = 0.07

    out = compute(pbr, inp)
    wel_orders = [
        order for order in out["orders"] if order["order_type"] == "close_auto_reduce_wel_long"
    ]

    assert len(wel_orders) == 1
    assert wel_orders[0]["symbol_idx"] == 1
    assert wel_orders[0]["qty"] == pytest.approx(-6.01)
    block = next(
        block
        for block in out["diagnostics"]["loss_gate_blocks"]
        if block["order_type"] == "close_auto_reduce_wel_long"
    )
    assert block["symbol_idx"] == 0
    assert block["balance_before"] == pytest.approx(939.9)
    assert block["projected_balance_after"] == pytest.approx(899.8)
    assert block["balance_floor"] == pytest.approx(930.0)


def test_larger_unstuck_wins_over_twel_auto_reduce_for_same_position():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.4,
        "risk_wel_enforcer_enabled": False,
        "total_wallet_exposure_limit": 0.5,
        "risk_twel_enforcer_policy": "reduce_portfolio",
        "risk_twel_enforcer_threshold": 1.0,
        "n_positions": 1,
        "unstuck_close_pct": 0.5,
        "unstuck_threshold": 0.001,
        "unstuck_ema_dist": 0.0,
        "unstuck_loss_allowance_pct": 0.01,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=6.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp["global"]["unstuck_allowance_long"] = 1e9

    out = compute(pbr, inp)
    order_types = [o["order_type"] for o in out["orders"]]

    assert "close_unstuck_long" in order_types
    assert "close_auto_reduce_twel_long" not in order_types
    assert "close_grid_long" in order_types
    assert sum(
        abs(o["qty"])
        for o in out["orders"]
        if o["pside"] == "long" and o["qty"] < 0.0
    ) <= 6.0 + 1e-9


def test_larger_short_unstuck_wins_over_twel_auto_reduce_for_same_position():
    import passivbot_rust as pbr

    short_bp = {
        "wallet_exposure_limit": 0.4,
        "risk_wel_enforcer_enabled": False,
        "total_wallet_exposure_limit": 0.5,
        "risk_twel_enforcer_policy": "reduce_portfolio",
        "risk_twel_enforcer_threshold": 1.0,
        "n_positions": 1,
        "unstuck_close_pct": 0.5,
        "unstuck_threshold": 0.001,
        "unstuck_ema_dist": 0.0,
        "unstuck_loss_allowance_pct": 0.01,
    }
    global_bp = bot_params_pair(short_overrides=short_bp)
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        short_pos_size=-6.0,
        short_pos_price=100.0,
        short_bp=short_bp,
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp["global"]["unstuck_allowance_short"] = 1e9

    out = compute(pbr, inp)
    order_types = [o["order_type"] for o in out["orders"]]

    assert "close_unstuck_short" in order_types
    assert "close_auto_reduce_twel_short" not in order_types
    assert "close_grid_short" in order_types
    assert sum(
        abs(o["qty"])
        for o in out["orders"]
        if o["pside"] == "short" and o["qty"] > 0.0
    ) <= 6.0 + 1e-9


def test_twel_auto_reduce_includes_managed_modes_and_excludes_manual_panic():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.2,
        "total_wallet_exposure_limit": 0.2,
        "risk_twel_enforcer_policy": "reduce_portfolio",
        "risk_twel_enforcer_threshold": 1.0,
        "n_positions": 4,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    symbols = [
        make_symbol(
            0,
            bid=110.0,
            ask=110.0,
            long_mode="tp_only",
            long_pos_size=3.0,
            long_pos_price=100.0,
            long_bp=long_bp,
        ),
        make_symbol(
            1,
            bid=109.0,
            ask=109.0,
            long_mode="graceful_stop",
            long_pos_size=3.0,
            long_pos_price=100.0,
            long_bp=long_bp,
        ),
        make_symbol(
            2,
            bid=108.0,
            ask=108.0,
            long_mode="manual",
            long_pos_size=2.0,
            long_pos_price=100.0,
            long_bp=long_bp,
        ),
        make_symbol(
            3,
            bid=107.0,
            ask=107.0,
            long_mode="panic",
            long_pos_size=2.0,
            long_pos_price=100.0,
            long_bp=long_bp,
        ),
    ]

    out = compute(pbr, make_input(balance=1_000.0, global_bp=global_bp, symbols=symbols))
    twel_symbols = {
        o["symbol_idx"] for o in out["orders"] if o["order_type"] == "close_auto_reduce_twel_long"
    }

    assert twel_symbols == {0, 1}


def test_twel_auto_reduce_manual_panic_exposure_triggers_managed_repair():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 1.0,
        "risk_wel_enforcer_enabled": False,
        "total_wallet_exposure_limit": 0.5,
        "risk_twel_enforcer_policy": "reduce_portfolio",
        "risk_twel_enforcer_threshold": 1.0,
        "n_positions": 1,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    symbols = [
        make_symbol(
            0,
            bid=100.0,
            ask=100.0,
            long_mode="normal",
            long_pos_size=1.0,
            long_pos_price=100.0,
            long_bp=long_bp,
        ),
        make_symbol(
            1,
            bid=100.0,
            ask=100.0,
            long_mode="manual",
            long_pos_size=3.0,
            long_pos_price=100.0,
            long_bp=long_bp,
        ),
        make_symbol(
            2,
            bid=100.0,
            ask=100.0,
            long_mode="panic",
            long_pos_size=2.0,
            long_pos_price=100.0,
            long_bp=long_bp,
        ),
    ]

    out = compute(pbr, make_input(balance=1_000.0, global_bp=global_bp, symbols=symbols))
    twel_orders = [o for o in out["orders"] if o["order_type"] == "close_auto_reduce_twel_long"]

    assert {o["symbol_idx"] for o in twel_orders} == {0}


def test_twel_enforcer_can_reduce_below_per_slot_target():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.2,
        "total_wallet_exposure_limit": 1.0,
        "risk_twel_enforcer_policy": "reduce_portfolio",
        "risk_twel_enforcer_threshold": 1.0,
        "n_positions": 8,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    symbols = [
        make_symbol(
            idx,
            bid=100.0 if idx == 0 else 90.0,
            ask=100.0 if idx == 0 else 90.0,
            long_pos_size=1.2,
            long_pos_price=100.0,
            long_bp=long_bp,
        )
        for idx in range(9)
    ]

    out = compute(pbr, make_input(balance=1_000.0, global_bp=global_bp, symbols=symbols))
    twel_closes = [o for o in out["orders"] if o["order_type"] == "close_auto_reduce_twel_long"]
    assert twel_closes, "TWE above TWEL must be reduced even when every position is at/below floor"
    assert any(o["symbol_idx"] == 0 for o in twel_closes), (
        "TWEL repair should use the shallowest-loss candidate even when it is at/below target"
    )


def test_twel_enforcer_threshold_reduces_positions_at_wel():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 0.75,
        "total_wallet_exposure_limit": 1.5,
        "risk_twel_enforcer_threshold": 0.99,
        "n_positions": 2,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym0 = make_symbol(
        0,
        bid=49.0,
        ask=49.0,
        long_pos_size=15.0,
        long_pos_price=50.0,
        long_bp=long_bp,
    )
    sym1 = make_symbol(
        1,
        bid=49.5,
        ask=49.5,
        long_pos_size=15.0,
        long_pos_price=50.0,
        long_bp=long_bp,
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym0, sym1])
    out = compute(pbr, inp)
    orders = [o for o in out["orders"] if o["order_type"] == "close_auto_reduce_twel_long"]
    assert orders, (
        "TWEL enforcer should reduce positions below raw WEL when "
        "risk_twel_enforcer_threshold is below 1.0"
    )

    psize_by_symbol = {0: 15.0, 1: 15.0}
    for order in orders:
        psize_by_symbol[order["symbol_idx"]] -= abs(order["qty"])
    twe_after = sum(size * 50.0 / 1_000.0 for size in psize_by_symbol.values())
    assert twe_after <= 1.5 * 0.99 + 1e-12


def test_twel_loss_gate_block_emits_twel_specific_warning():
    import passivbot_rust as pbr

    long_bp = {
        "wallet_exposure_limit": 1.0,
        "risk_wel_enforcer_enabled": False,
        "total_wallet_exposure_limit": 0.5,
        "risk_twel_enforcer_policy": "reduce_portfolio",
        "risk_twel_enforcer_threshold": 1.0,
        "n_positions": 1,
    }
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=80.0,
        ask=80.0,
        long_pos_size=6.0,
        long_pos_price=100.0,
        long_bp=long_bp,
    )
    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp["balance_raw"] = 1_000.0
    inp["global"]["max_realized_loss_pct"] = 0.0
    inp["global"]["realized_pnl_cumsum_max"] = 0.0
    inp["global"]["realized_pnl_cumsum_last"] = 0.0

    out = compute(pbr, inp)

    assert not any(o["order_type"] == "close_auto_reduce_twel_long" for o in out["orders"])
    assert any(
        b["order_type"] == "close_auto_reduce_twel_long"
        for b in out["diagnostics"]["loss_gate_blocks"]
    )
    warning = next(
        (
            w["twel_repair_blocked_by_loss_gate"]
            for w in out["diagnostics"]["warnings"]
            if "twel_repair_blocked_by_loss_gate" in w
        ),
        None,
    )
    assert warning is not None
    assert warning["pside"] == "long"
    assert warning["policy"] == "reduce_portfolio"
    assert warning["candidate_count"] == 1
    assert warning["blocked_order_count"] == 1
    assert warning["current_twe"] == pytest.approx(0.6)
    assert warning["twel_repair_target"] == pytest.approx(0.5)
    assert warning["projected_twe_after_allowed_reductions"] == pytest.approx(0.6)


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


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda inp: inp.__setitem__("balance", 0.0), r"balance must be finite and > 0"),
        (
            lambda inp: inp["global"].__setitem__("max_realized_loss_pct", -0.01),
            r"global\.max_realized_loss_pct must be finite and >= 0",
        ),
        (
            lambda inp: inp["global"].__setitem__("unstuck_allowance_long", -1.0),
            r"global\.unstuck_allowance_long must be finite and >= 0",
        ),
        (
            lambda inp: inp["global"].__setitem__("unstuck_allowance_short", -1.0),
            r"global\.unstuck_allowance_short must be finite and >= 0",
        ),
        (
            lambda inp: (
                inp["global"].__setitem__("realized_pnl_cumsum_max", 5.0),
                inp["global"].__setitem__("realized_pnl_cumsum_last", 10.0),
            ),
            r"global\.realized_pnl_cumsum_max must be >= global\.realized_pnl_cumsum_last",
        ),
    ],
)
def test_json_rejects_invalid_account_risk_globals(mutator, match):
    import passivbot_rust as pbr

    inp = make_input(
        balance=1_000.0,
        symbols=[make_symbol(0, bid=100.0, ask=100.0)],
    )
    mutator(inp)

    with pytest.raises(ValueError, match=match):
        compute(pbr, inp)


def test_balance_raw_zero_rejected():
    """Non-positive balance_raw is rejected instead of disabling realized-loss gates."""
    import passivbot_rust as pbr

    long_bp = {}
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
        long_bp=long_bp,
        long_strategy=adaptive_strategy_params(
            close={"qty_pct": 1.0, "threshold_base_pct": 0.01},
        ),
    )

    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp["balance_raw"] = 0.0
    inp["global"]["max_realized_loss_pct"] = 0.05
    inp["global"]["realized_pnl_cumsum_max"] = 10.0
    inp["global"]["realized_pnl_cumsum_last"] = 5.0

    with pytest.raises(ValueError, match=r"balance_raw must be finite and > 0"):
        compute(pbr, inp)


def test_balance_raw_negative_rejected():
    """Negative balance_raw is rejected instead of disabling realized-loss gates."""
    import passivbot_rust as pbr

    long_bp = {}
    global_bp = bot_params_pair(long_overrides=long_bp)
    sym = make_symbol(
        0,
        bid=100.0,
        ask=100.0,
        long_pos_size=1.0,
        long_pos_price=100.0,
        long_bp=long_bp,
        long_strategy=adaptive_strategy_params(
            close={"qty_pct": 1.0, "threshold_base_pct": 0.01},
        ),
    )

    inp = make_input(balance=1_000.0, global_bp=global_bp, symbols=[sym])
    inp["balance_raw"] = -1.0
    inp["global"]["max_realized_loss_pct"] = 0.05
    inp["global"]["realized_pnl_cumsum_max"] = 10.0
    inp["global"]["realized_pnl_cumsum_last"] = 5.0

    with pytest.raises(ValueError, match=r"balance_raw must be finite and > 0"):
        compute(pbr, inp)


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
    sym0 = make_symbol(
        0, bid=50.0, ask=50.0, long_pos_size=8.0, long_pos_price=50.0, long_bp=long_bp
    )
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


def test_loss_gate_rejects_non_positive_raw_balance():
    """Non-positive balance_raw fails loudly before loss-gate planning."""
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
        with pytest.raises(ValueError, match="balance_raw must be finite and > 0"):
            compute(pbr, inp_case)
