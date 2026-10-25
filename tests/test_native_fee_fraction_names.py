"""Fee fraction packet boundaries preserve external keys and signed monetary fees."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from backtest import prep_backtest_args
from live.exchange_params import _get_exchange_fee_fractions, _orchestrator_exchange_params
from test_backtest_quantity_names import _inputs
from test_hlcvs_bundle import _base_meta
from test_limit_order_fill_buffer import _native_args


FEES = ("maker", "taker")


@pytest.mark.parametrize("kind", FEES)
@pytest.mark.parametrize("boundary", ["backtest", "exchange", "metadata"])
@pytest.mark.parametrize("presence", ["old_only", "both", "null_old"])
def test_retired_native_fee_keys_are_rejected(kind, boundary, presence):
    import passivbot_rust as pbr
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    old, current = f"{kind}_fee", f"{kind}_fee_fraction"
    if boundary == "metadata":
        metadata = _base_meta()
        packet = metadata["coins"][0]
    else:
        args = _native_args()
        packet = args[-1] if boundary == "backtest" else args[-2][0]
    if presence == "old_only":
        packet.pop(current)
    packet[old] = None if presence == "null_old" else -.000125
    with pytest.raises(ValueError, match=f"unsupported retired native .* field '{old}'"):
        if boundary == "metadata":
            pbr.HlcvsBundle(np.zeros((2, 1, 4)), np.ones(2), np.array([0, 60_000]), metadata)
        else:
            pbr.run_backtest(*args)


@pytest.mark.parametrize("kind", FEES)
@pytest.mark.parametrize("boundary", ["backtest", "exchange", "metadata"])
@pytest.mark.parametrize("missing", [True, False])
def test_current_native_fee_inputs_remain_required_and_nonnull(kind, boundary, missing):
    import passivbot_rust as pbr
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    current = f"{kind}_fee_fraction"
    if boundary == "metadata":
        metadata = _base_meta()
        packet = metadata["coins"][0]
    else:
        args = _native_args()
        packet = args[-1] if boundary == "backtest" else args[-2][0]
    if missing:
        packet.pop(current)
    else:
        packet[current] = None
    with pytest.raises((ValueError, TypeError), match=current + "|NoneType"):
        if boundary == "metadata":
            pbr.HlcvsBundle(np.zeros((2, 1, 4)), np.ones(2), np.array([0, 60_000]), metadata)
        else:
            pbr.run_backtest(*args)


@pytest.mark.parametrize("source", ["ccxt", "cache", "cache_precedence"])
@pytest.mark.parametrize("override", [None, 0.0, -.000375])
def test_exchange_cache_adapter_preserves_source_keys_values_and_override_precedence(source, override):
    config, markets = _inputs()
    market = markets["BTC"]
    market.update(maker=-.000125, taker=-.00025)
    if source != "ccxt":
        market.update(maker_fee=-.000125, taker_fee=-.00025)
        if source == "cache":
            del market["maker"], market["taker"]
        else:
            market.update(maker=.001, taker=.002)
    config["backtest"].update(maker_fee_override_fraction=override, taker_fee_override_fraction=override)
    before, config_before = deepcopy(markets), deepcopy(config)
    _, _, exchange, native = prep_backtest_args(config, markets, "binance")
    for kind, expected in (("maker", -.000125), ("taker", -.00025)):
        expected = expected if override is None else override
        assert exchange[0][f"{kind}_fee_fraction"] == native[f"{kind}_fee_fraction"] == expected
        assert f"{kind}_fee" not in native and f"{kind}_fee" not in exchange[0]
    assert markets == before and config == config_before
    # Persisted market metadata remains readable in its existing wire schema.
    assert json.loads(json.dumps(markets)) == before


@pytest.mark.parametrize("source", ["ccxt", "cache"])
def test_live_adapter_preserves_external_keys_and_emits_current_fraction_fields(source):
    market = {"maker": -.000125, "taker": 0.0} if source == "ccxt" else {
        "maker_fee": -.000125, "taker_fee": 0.0, "maker": .001, "taker": .002,
    }
    before = deepcopy(market)
    bot = SimpleNamespace(markets_dict={"BTC": market}, qty_steps={"BTC": .001},
                          price_steps={"BTC": .01}, min_qtys={"BTC": .001},
                          min_costs={"BTC": 1.0}, c_mults={"BTC": 1.0})
    bot._get_exchange_fee_fractions = lambda symbol: _get_exchange_fee_fractions(bot, symbol)
    native = _orchestrator_exchange_params(bot, "BTC")
    assert native["maker_fee_fraction"] == -.000125
    assert native["taker_fee_fraction"] == 0.0
    assert "maker_fee" not in native and "taker_fee" not in native
    assert market == before


@pytest.mark.parametrize("kind", FEES)
def test_typed_exchange_rejects_old_key_even_beside_current(kind):
    import passivbot_rust as pbr
    from test_orchestrator_json_api import make_input, make_symbol

    packet = make_input(balance=1000.0, symbols=[make_symbol(0, bid=100.0, ask=100.0)])
    packet["symbols"][0]["exchange"][f"{kind}_fee"] = -.000125
    with pytest.raises(ValueError, match=f"{kind}_fee"):
        pbr.compute_ideal_orders_json(json.dumps(packet))


def test_coin_metadata_roundtrip_uses_fraction_fields_and_preserves_rebates():
    import passivbot_rust as pbr

    meta = _base_meta()
    meta["coins"][0].update(maker_fee_fraction=-.000125, taker_fee_fraction=0.0)
    bundle = pbr.HlcvsBundle(np.zeros((2, 1, 4)), np.ones(2), np.array([0, 60_000]), meta)
    assert bundle.meta == meta
    assert bundle.coin_meta("BTC")["maker_fee_fraction"] == -.000125


@pytest.mark.parametrize("market_execution", [False, True])
def test_signed_native_fee_fraction_still_produces_positive_monetary_rebate(market_execution):
    import passivbot_rust as pbr

    args = _native_args()
    args[-1].update(maker_fee_fraction=-.000125, taker_fee_fraction=-.00025,
                    market_orders_allowed=market_execution, market_order_near_touch_fraction=.5)
    args[-2][0].update(maker_fee_fraction=-.000125, taker_fee_fraction=-.00025)
    fills, *_ = pbr.run_backtest(*args)
    assert len(fills) > 0
    expected = .00025 if market_execution else .000125
    first = fills[0]
    assert first[4] == pytest.approx(abs(first[9]) * first[10] * expected)
    assert first[4] > 0.0
