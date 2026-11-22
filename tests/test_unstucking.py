import json

import pytest


@pytest.fixture
def require_real_passivbot_rust_module():
    native = pytest.importorskip("passivbot_rust")
    if getattr(native, "__is_stub__", False):
        pytest.skip("native extension required")
    return native


def _position(*, short=False, price=None, trigger=0.5, minimum=0.0, qty=10.0):
    price = price if price is not None else (110.0 if short else 90.0)
    return {
        "symbol": "SHORT" if short else "LONG",
        "order_book": {"bid": price, "ask": price},
        "exchange": {
            "qty_step": 0.1,
            "price_step": 0.1,
            "min_qty": minimum,
            "min_cost": 0.0,
            "c_mult": 1.0,
            "maker_fee_fraction": 0.0,
            "taker_fee_fraction": 0.0,
        },
        "execution_type": "limit",
        "post_only": True,
        "market_order_slippage_fraction": 0.0,
        "policy": {
            "idx": 0,
            "side": int(short),
            "position_size": -qty if short else qty,
            "position_price": 100.0,
            "wallet_exposure_limit_ratio": 1.0,
            "effective_we_excess_allowance_fraction": 0.0,
            "unstuck_exposure_trigger_ratio": trigger,
            "unstuck_ema_gating_enabled": False,
            "unstuck_ema_distance_fraction": 0.0,
            "unstuck_loss_allowance_fraction": 0.1,
            "total_wallet_exposure_limit_ratio": 1.0,
            "ema_band_upper": 100.0,
            "ema_band_lower": 100.0,
            "current_price": price,
            "price_step": 0.1,
            "c_mult": 1.0,
        },
    }


def _payload(*positions, enabled=False, cap=None, drawdown=0.0, balance=1000.0):
    return {
        "balance": balance,
        "realized_pnl_drawdown": drawdown,
        "loss_gate_enabled": enabled,
        "loss_gate_max_drawdown_fraction": cap,
        "positions": list(positions),
    }


@pytest.mark.parametrize("short", [False, True])
def test_standalone_unstuck_observes_exposure_floor_and_gate_flag(
    require_real_passivbot_rust_module, short
):
    native = require_real_passivbot_rust_module
    position = _position(short=short)
    off = native.calc_unstucking_close_py(json.dumps(_payload(position)))
    dormant = native.calc_unstucking_close_py(json.dumps(_payload(position, cap=0.0)))
    assert off == dormant
    expected_type = f"close_unstuck_{'short' if short else 'long'}"
    assert off == (
        0,
        int(short),
        5.2 if short else -5.2,
        position["policy"]["current_price"],
        native.order_type_snake_to_id(expected_type),
    )

    zero = _payload(position, enabled=True, cap=0.0)
    assert native.calc_unstucking_close_py(json.dumps(zero)) is None

    sized = _payload(position, enabled=True, cap=0.03)
    result = native.calc_unstucking_close_py(json.dumps(sized))
    assert result[2] == (3.0 if short else -3.0)


def test_standalone_unstuck_needs_eligible_exposure_and_legal_minimum(
    require_real_passivbot_rust_module,
):
    native = require_real_passivbot_rust_module
    for payload in (
        _payload(_position(trigger=1.0)),
        _payload(_position(minimum=6.0)),
        _payload(_position(), drawdown=500.0),
        _payload(_position(), balance=50.0),
    ):
        assert native.calc_unstucking_close_py(json.dumps(payload)) is None

    position = _position()
    position["policy"]["unstuck_ema_gating_enabled"] = True
    assert native.calc_unstucking_close_py(json.dumps(_payload(position))) is None


def test_standalone_unstuck_requires_explicit_gate_policy(require_real_passivbot_rust_module):
    native = require_real_passivbot_rust_module
    position = _position()
    for absent in ("loss_gate_enabled", "loss_gate_max_drawdown_fraction"):
        payload = _payload(position)
        del payload[absent]
        with pytest.raises(ValueError, match=absent):
            native.calc_unstucking_close_py(json.dumps(payload))
    for enabled, cap in ((True, None), (False, 1.0)):
        with pytest.raises(ValueError, match="loss_gate_max_drawdown_fraction"):
            native.calc_unstucking_close_py(json.dumps(_payload(position, enabled=enabled, cap=cap)))
