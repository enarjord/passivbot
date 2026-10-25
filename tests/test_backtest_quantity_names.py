"""Public setup quantities retain their scale, sentinels and native behavior."""

from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import pytest

from backtest import build_backtest_payload, prep_backtest_args
from config import prepare_config
from config.field_registry import get_field_registry
from config.schema import get_expanded_config_prototype
from config.universe import set_side_universe_sources
from config_utils import add_config_arguments
from passivbot import compute_live_warmup_windows
from tools.clean_config import cleanup_config
from tools.migrate_config_v9 import migrate
from warmup_utils import compute_backtest_warmup_minutes, compute_per_coin_warmup_minutes


RENAMES = (
    (("backtest",), "btc_collateral_cap", "btc_collateral_target_ratio"),
    (("backtest",), "btc_collateral_ltv_cap", "max_usd_debt_to_wallet_value_ratio"),
    (("backtest",), "maker_fee_override", "maker_fee_override_fraction"),
    (("backtest",), "taker_fee_override", "taker_fee_override_fraction"),
    (("backtest",), "equity_floor_ratio", "equity_floor_fraction"),
    (("bot", "portfolio", "warmup"), "span_multiplier", "activation_span_fraction"),
)


@pytest.mark.parametrize("parts,old,new", RENAMES)
@pytest.mark.parametrize("load", [prepare_config, cleanup_config, migrate])
def test_retired_development_names_are_rejected(parts, old, new, load):
    config = get_expanded_config_prototype()
    target = config
    for part in parts:
        target = target[part]
    target[old] = target.pop(new)
    with pytest.raises(ValueError, match="unknown configuration field"):
        load(config)


def _inputs(*, target=2.0, debt_cap=0.975):
    config = get_expanded_config_prototype()
    config["backtest"].update(
        start_date="2021-01-01", coins={"binance": ["BTC"]}, starting_balance=1000.0,
        btc_collateral_target_ratio=target, max_usd_debt_to_wallet_value_ratio=debt_cap,
        maker_fee_override_fraction=0.0, taker_fee_override_fraction=0.0,
    )
    config["bot"]["portfolio"]["strategy_kind"] = "ema_anchor"
    config["bot"]["portfolio"]["warmup"].update(
        activation_span_fraction=0.2, max_minutes=0.5,
    )
    set_side_universe_sources(config, "approved_coins", {"long": ["BTC"], "short": []})
    for side in ("long", "short"):
        bot = config["bot"][side]
        bot["enabled"] = side == "long"
        bot["universe"].update(minimum_coin_age_days=0.0, filter_by_min_effective_cost=False)
        bot["hsl"]["enabled"] = False
        bot["risk"].update(max_positions=1, total_wallet_exposure_limit_ratio=1.0)
        bot["strategy"]["ema_anchor"].update(
            base_qty_fraction=0.1, ema_span_0_minutes=1.0, ema_span_1_minutes=1.0,
            entry_double_down_factor=0.0, offset_fraction=0.1,
            offset_volatility_1m_weight=0.0, offset_volatility_1h_weight=0.0,
            offset_psize_weight=0.0,
        )
    markets = {"BTC": dict(maker=-0.0001, taker=0.0005, qty_step=0.001,
                            price_step=0.01, min_qty=0.001, min_cost=1.0, c_mult=1.0)}
    return config, markets


@pytest.mark.parametrize("debt_cap", [None, 0.0, -0.125, 0.625, 1.75])
def test_native_transport_keeps_leverage_signed_fees_and_debt_sentinels(debt_cap):
    config, markets = _inputs(target=2.375, debt_cap=debt_cap)
    config["backtest"].update(maker_fee_override_fraction=-0.000125,
                              taker_fee_override_fraction=0.000375,
                              equity_floor_fraction=0.07125)
    before = deepcopy(config)
    _, _, exchange_params, native = prep_backtest_args(config, markets, "binance")
    assert native["btc_collateral_target_ratio"] == 2.375
    assert native["max_usd_debt_to_wallet_value_ratio"] == debt_cap
    assert native["maker_fee_fraction"] == exchange_params[0]["maker_fee_fraction"] == -0.000125
    assert native["taker_fee_fraction"] == exchange_params[0]["taker_fee_fraction"] == 0.000375
    assert native["equity_floor_fraction"] == 0.07125
    assert not {"btc_collateral_cap", "btc_collateral_ltv_cap",
                "maker_fee_override_fraction", "taker_fee_override_fraction"}.intersection(native)
    assert config == before


def _run(config, markets, candles, btc_prices):
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(passivbot_rust, "__is_stub__", False)
    verify_loaded_runtime_extension()
    timestamps = np.arange(len(candles), dtype=np.int64) * 60_000 + 1_609_459_200_000
    payload = build_backtest_payload(
        np.asarray(candles, dtype=np.float64), markets, config, "binance",
        np.asarray(btc_prices, dtype=np.float64), timestamps,
    )
    assert payload.backtest_params["global_warmup_bars"] == 1
    return passivbot_rust.run_backtest_bundle(
        payload.bundle, payload.bot_params_list, payload.strategy_params_list,
        payload.exchange_params, payload.backtest_params,
    )


def test_native_leveraged_collateral_target_is_not_clipped_or_rescaled():
    config, markets = _inputs(target=2.375, debt_cap=0.625)
    fills, equities, *_ = _run(
        config, markets, [[[100.0, 100.0, 100.0, 1.0]]] * 6,
        [100.0, 100.0, 100.0, 150.0, 200.0, 250.0],
    )
    assert len(fills) == 0
    # Initialization buys 23.75 BTC at 100, financed by -1375 USD.
    # The separate debt ceiling applies to top-ups, preserving initial allocation.
    assert equities[:, 1] == pytest.approx([1000.0, 2187.5, 3375.0])
    assert equities[:, 3] == pytest.approx([1000.0] * 3)


def test_native_topup_limit_divides_usd_debt_by_wallet_value():
    config, markets = _inputs(target=2.0, debt_cap=0.975)
    fills, _, *_ = _run(config, markets, [[[120.0, 80.0, 100.0, 1.0]]] * 10,
                       [100.0] * 10)
    initial, close = fills[:2]
    assert initial[6] == 20.0 and initial[7] == -1000.0
    assert close[3] > 0.0
    wallet_value_before_conversion = initial[6] * close[8] + initial[7] + close[3] + close[4]
    usd_before_conversion = initial[7] + close[3] + close[4]
    debt_limit = 0.975 * wallet_value_before_conversion
    topup = debt_limit + usd_before_conversion
    assert topup > 0.0
    assert -close[7] / wallet_value_before_conversion == pytest.approx(0.975)
    assert close[6] == pytest.approx(initial[6] + topup * 0.999 / close[8])
    assert close[5] == pytest.approx(wallet_value_before_conversion - topup * 0.001)


def test_native_negative_maker_fee_fraction_remains_a_rebate():
    config, markets = _inputs()
    config["backtest"]["maker_fee_override_fraction"] = -0.0002
    fills, _, *_ = _run(config, markets, [[[120.0, 80.0, 100.0, 1.0]]] * 10,
                       [100.0] * 10)
    assert fills[0][4] == pytest.approx(abs(fills[0][9]) * fills[0][10] * 0.0002)
    assert fills[0][4] > 0.0


def test_null_fee_overrides_retain_exchange_rebates():
    config, markets = _inputs()
    config["backtest"].update(maker_fee_override_fraction=None, taker_fee_override_fraction=None)
    _, _, exchange_params, native = prep_backtest_args(config, markets, "binance")
    assert native["maker_fee_fraction"] == exchange_params[0]["maker_fee_fraction"] == -0.0001
    assert native["taker_fee_fraction"] == exchange_params[0]["taker_fee_fraction"] == 0.0005


def test_live_acquisition_and_backtest_activation_keep_their_distinct_baselines():
    config = {"bot": {"portfolio": {"strategy_kind": "trailing_martingale", "warmup": {
        "activation_span_fraction": 0.2, "max_minutes": None,
    }}, "long": {"strategy": {"trailing_martingale": {
        "entry": {"ema_span_0_minutes": 100.0, "ema_span_1_minutes": 100.0},
        "log_range_ema_span_minutes": 0.0, "log_range_ema_span_1h_candles": 0.0,
    }}}, "short": {}}, "optimize": {"bounds": {}}}
    windows, _, _ = compute_live_warmup_windows(
        {"long": {"BTC"}}, lambda _side, key, _coin: 100.0 if key in {"ema_span_0_minutes", "ema_span_1_minutes"} else 0.0,
        activation_span_fraction=0.2,
    )
    assert windows == {"BTC": 120}
    assert compute_backtest_warmup_minutes(config, include_rms=False) == 20
    assert compute_per_coin_warmup_minutes(config, include_rms=False) == {"__default__": 20}


def test_quantity_metadata_discloses_denominators_and_signed_fee_domain():
    registry = get_field_registry()
    for path, unit, denominator in [
        (("backtest", "btc_collateral_target_ratio"), "ratio", "current BTC market value plus USD cash, excluding unrealized trading PnL"),
        (("backtest", "max_usd_debt_to_wallet_value_ratio"), "ratio", "current BTC market value plus USD cash, excluding unrealized trading PnL"),
        (("backtest", "maker_fee_override_fraction"), "fraction", "absolute fill quote notional"),
        (("backtest", "taker_fee_override_fraction"), "fraction", "absolute fill quote notional"),
        (("backtest", "equity_floor_fraction"), "fraction", "starting balance"),
        (("bot", "portfolio", "warmup", "activation_span_fraction"), "fraction", "indicator span"),
    ]:
        field = registry.find(path)
        assert (field.unit, field.denominator) == (unit, denominator)
        if path[-1].endswith("fee_override_fraction"):
            assert field.nullable and field.domain.minimum is None


@pytest.mark.parametrize("kind", ["maker", "taker"])
def test_fee_cli_uses_explicit_fraction_names_without_retired_aliases(kind):
    parser = ArgumentParser(allow_abbrev=False)
    add_config_arguments(parser, get_expanded_config_prototype(), command="backtest", help_all=True)
    value = parser.parse_args([f"--{kind}-fee-override-fraction", "-0.000125"])
    assert getattr(value, f"backtest.{kind}_fee_override_fraction") == -0.000125
    with pytest.raises(SystemExit):
        parser.parse_args([f"--{kind}-fee-override", "0.000125"])
