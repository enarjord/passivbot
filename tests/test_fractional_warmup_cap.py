"""Whole-candle acquisition keeps every positive warmup cap bounded."""

from copy import deepcopy
from argparse import ArgumentParser

import numpy as np
import pytest

from backtest import ensure_valid_index_metadata
from passivbot import compute_live_warmup_windows
from warmup_utils import (
    compute_backtest_warmup_minutes,
    compute_per_coin_warmup_minutes,
    live_warmup_cap_minutes,
)


def _config(span, cap):
    return {
        "bot": {
            "portfolio": {"strategy_kind": "trailing_martingale", "warmup": {"span_multiplier": 1.0, "max_minutes": cap}},
            "long": {"strategy": {"trailing_martingale": {
                "entry": {"ema_span_0_minutes": span, "ema_span_1_minutes": span},
                "volatility_ema_span_minutes": 0.0,
                "volatility_ema_span_1h_candles": 0.0,
            }}},
            "short": {},
        },
        "optimize": {"bounds": {}},
    }


@pytest.mark.parametrize("span,cap,expected", [
    (20.25, 0.5, 1), (20.25, 1.0, 1), (20.25, 5.5, 6),
    (20.25, None, 21), (3.25, 5.5, 4), (3.25, None, 4),
    (0.25, 0.5, 1), (0.25, 1.0, 1), (5.25, 5.5, 6), (20.25, 0.00001, 1),
])
def test_live_and_backtest_acquire_the_same_whole_minutes(span, cap, expected):
    config = _config(span, cap)
    original = deepcopy(config)
    live_cap = live_warmup_cap_minutes(config)
    windows, _, _ = compute_live_warmup_windows(
        {"long": {"BTC"}},
        lambda _side, key, _coin: span if key in {"ema_span_0", "ema_span_1"} else 0.0,
        max_warmup_minutes=live_cap,
    )
    assert windows == {"BTC": expected}
    assert compute_backtest_warmup_minutes(config, include_rms=False) == expected
    per_coin = compute_per_coin_warmup_minutes(config, include_rms=False)
    assert per_coin == {"__default__": expected}

    # Cached metadata consumes the same whole-minute window, without truncating again.
    metadata = {"BTC": {"first_valid_index": 2, "last_valid_index": 39,
                        "warmup_minutes": 30}}
    ensure_valid_index_metadata(metadata, np.ones((40, 1, 4)), ["BTC"], per_coin)
    assert metadata["BTC"]["warmup_minutes"] == expected
    assert metadata["BTC"]["trade_start_index"] == 2 + expected
    assert config == original


@pytest.mark.parametrize("cap,expected_minutes,expected_hours", [
    (0.5, 1, 1), (1.0, 1, 1), (5.5, 6, 1), (60.5, 61, 2),
    (None, 181, 4), (0, 181, 4),
])
@pytest.mark.parametrize("explicit_window", [False, True])
def test_direct_live_windows_round_cap_and_keep_native_hour_alignment(
    cap, expected_minutes, expected_hours, explicit_window,
):
    windows, hours, _ = compute_live_warmup_windows(
        {"long": {"BTC"}},
        lambda _side, key, _coin: {
            "ema_span_0": 180.25, "entry_volatility_ema_span_1h": 3.25,
        }.get(key, 0.0),
        max_warmup_minutes=cap,
        window_candles=181 if explicit_window else None,
    )
    assert windows == {"BTC": expected_minutes}
    assert hours == {"BTC": expected_hours}


def test_per_coin_cap_preserves_shorter_override_history():
    config = _config(20.25, 5.5)
    config["coin_overrides"] = {"BTC": {"bot": {"long": {"strategy": {
        "trailing_martingale": {"entry": {
            "ema_span_0_minutes": 2.25, "ema_span_1_minutes": 2.25,
        }},
    }}}}}
    original = deepcopy(config)
    assert compute_per_coin_warmup_minutes(config, include_rms=False) == {
        "__default__": 6, "BTC": 3,
    }
    assert config == original


@pytest.mark.parametrize("text,expected", [("0.5", 1), ("1.0", 1), ("5.5", 6), ("null", 0)])
def test_cli_preserves_authored_cap_and_rounds_only_for_acquisition(text, expected):
    from config import prepare_config
    from config.schema import get_template_config
    from config_utils import add_arguments_recursively, update_config_with_args

    config = get_template_config()
    parser = ArgumentParser()
    add_arguments_recursively(parser, config)
    update_config_with_args(config, parser.parse_args(["--bot.portfolio.warmup.max_minutes", text]))
    prepared = prepare_config(config, verbose=False)
    assert prepared["bot"]["portfolio"]["warmup"]["max_minutes"] == (
        None if text == "null" else float(text)
    )
    assert live_warmup_cap_minutes(prepared) == expected
    backtest = compute_backtest_warmup_minutes(prepared, include_rms=False)
    if expected:
        assert backtest == expected
    else:
        assert backtest > 6


@pytest.mark.parametrize("cap,whole_minutes", [(0.5, 1), (5.5, 6)])
def test_fractional_cap_reaches_native_backtest_as_the_same_whole_window(cap, whole_minutes):
    import passivbot_rust
    from backtest import run_backtest
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    assert not getattr(passivbot_rust, "__is_stub__", False)
    config = _ema_anchor_config(True)
    config["bot"]["portfolio"]["warmup"]["max_minutes"] = cap
    original = deepcopy(config)
    candles, markets, btc, timestamps = _synthetic_inputs()
    ensure_valid_index_metadata(
        markets, candles, ["LONGCOIN", "SHORTCOIN"],
        compute_per_coin_warmup_minutes(config, include_rms=False),
    )
    fills, equity, _, payload = run_backtest(
        candles, markets, config, "binance", btc, timestamps, return_payload=True,
    )
    assert payload.backtest_params["warmup_minutes"] == [whole_minutes, whole_minutes]
    assert payload.backtest_params["global_warmup_bars"] == whole_minutes
    assert payload.backtest_params["trade_start_indices"] == [whole_minutes, whole_minutes]
    config["bot"]["portfolio"]["warmup"]["max_minutes"] = float(whole_minutes)
    reference = run_backtest(candles, markets, config, "binance", btc, timestamps)
    assert len(fills) > 0
    np.testing.assert_array_equal(fills, reference[0])
    np.testing.assert_array_equal(equity, reference[1])
    config["bot"]["portfolio"]["warmup"]["max_minutes"] = cap
    assert config == original


def test_fractional_cap_preserves_independently_required_rms_history():
    from test_backtest_directional_eligibility import _ema_anchor_config

    config = _ema_anchor_config(True)
    config["bot"]["portfolio"]["warmup"]["max_minutes"] = 0.5
    config["optimize"]["bounds"] = {}
    config["bot"]["long"]["forager"]["unilateralness_ema_span_minutes"] = 2.25
    config["bot"]["long"]["entry_cooldown"]["max_duration_minutes"] = 60.0
    config["bot"]["long"]["entry_cooldown"]["weights_minutes"]["adverse_directionality"] = 1.0
    original = deepcopy(config)
    assert compute_backtest_warmup_minutes(config, include_rms=False) == 1
    assert compute_backtest_warmup_minutes(config) == 46
    assert compute_per_coin_warmup_minutes(config)["__default__"] == 46
    assert config == original


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ["calc_volumes_and_log_ranges", "calc_log_range"])
@pytest.mark.parametrize("span,cap,expected", [
    (5.25, 5.5, 6), (3.25, 5.5, 4), (5.25, 0.5, 1), (3.25, 60, 4), (3.25, None, 4),
])
async def test_direct_ranking_readers_keep_float_spans_until_acquisition(reader, span, cap, expected):
    from types import SimpleNamespace
    from passivbot import Passivbot

    calls = []

    class CandleReader:
        async def get_latest_ema_metrics(self, symbol, spans, **kwargs):
            calls.append((symbol, spans, kwargs))
            return {"qv": 100.0, "log_range": 0.25}

    config = {"bot": {"portfolio": {"warmup": {"span_multiplier": 0.0, "max_minutes": cap}}}}
    original = deepcopy(config)
    bot = SimpleNamespace(
        config=config,
        cm=CandleReader(),
        bot_value=lambda _side, key: 2.25 if key == "forager_volume_ema_span_1m" else span,
        _compute_fetch_budget_ttls=lambda symbols, *_: ({s: 60_000 for s in symbols}, set()),
    )
    result = await getattr(Passivbot, reader)(bot, "long", ["BTC/USDT:USDT"])
    assert result
    assert len(calls) == 1
    symbol, spans, kwargs = calls[0]
    assert symbol == "BTC/USDT:USDT"
    assert spans["log_range"] == span
    if reader == "calc_volumes_and_log_ranges":
        assert spans["qv"] == 2.25
    assert kwargs["window_candles"] == expected
    assert config == original
