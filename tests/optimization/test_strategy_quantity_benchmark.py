"""Benchmark feature flags retain their native meaning after public renaming."""

from itertools import product

import pytest

from backtest import prep_backtest_args
from tools.gpu_parity import build_parser, fixture_inputs
from tools.optimizer_benchmark.catalogue import Case
from tools.optimizer_benchmark.workloads import apply_features, sampled_population


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("volatility,exposure", list(product([False, True], repeat=2)))
def test_feature_flags_and_frozen_zero_sensitivities_match_native_packets(kind, volatility, exposure):
    config, _candles, markets, _btc, _timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", kind, "--sides", "both", "--coins", "2", "--bars", "128",
    ]))
    for side in ("long", "short"):
        params = config["bot"][side]["strategy"][kind]
        if kind == "trailing_martingale":
            params["entry"]["reentry_qty_multiplier"] = 1.375
        else:
            params["entry_qty_exposure_coefficient"] = 0.625
            params["inventory_offset_exposure_coefficient"] = 0.375
    flags = tuple(name for name, enabled in (("volatility", volatility), ("exposure_weight", exposure)) if enabled)
    case = Case(kind, sides="both", topology="fixed", features=flags)
    coins = config["backtest"]["coins"]["binance"]
    apply_features(config, case, coins, markets, 128)
    native = prep_backtest_args(config, markets, "binance")[1][0]
    for side in ("long", "short"):
        if kind == "trailing_martingale":
            assert native[side]["entry"]["double_down_factor"] == 1.375
            assert native[side]["entry"]["threshold_we_weight"] == (0.01 if exposure else 0.0)
            assert native[side]["close"]["threshold_we_weight"] == (-0.01 if exposure else 0.0)
            assert native[side]["entry"]["retracement_we_weight"] == 0.0
            assert native[side]["entry"]["threshold_volatility_1m_weight"] == (0.1 if volatility else 0.0)
        else:
            # This field used to be a double-down factor, so it must not be
            # newly classified as an optional distance weight by its suffix.
            assert native[side]["entry_double_down_factor"] == 0.625
            assert native[side]["offset_psize_weight"] == (0.1 if volatility else 0.0)
    _population, _vectors, bounds = sampled_population(config, case, 3, 7)
    for row in bounds:
        leaf = row["path"][-1]
        if kind == "trailing_martingale" and (
            leaf == "retracement_exposure_ratio_coefficient"
            or leaf == "threshold_exposure_ratio_coefficient" and not exposure
        ):
            assert row["low"] == row["high"] == 0.0
        elif kind == "ema_anchor" and leaf == "inventory_offset_exposure_coefficient" and not volatility:
            assert row["low"] == row["high"] == 0.0
        elif kind == "ema_anchor" and leaf == "entry_qty_exposure_coefficient":
            assert row["low"] < row["high"]
