from collections import Counter

import numpy as np
import pytest

from backtest import run_backtest
from config_utils import load_config


@pytest.fixture(scope="module", autouse=True)
def require_real_passivbot_rust_module():
    import passivbot_rust as pbr

    if getattr(pbr, "__is_stub__", False):
        pytest.fail("directional eligibility requires the real passivbot_rust extension")


def _ema_anchor_config(hedge_mode: bool) -> dict:
    config = load_config("configs/examples/ema_anchor.json", verbose=False)
    config["live"].update(
        {
            "approved_coins": {"long": ["LONGCOIN"], "short": ["SHORTCOIN"]},
            "ignored_coins": {"long": [], "short": []},
            "hedge_mode": hedge_mode,
            "warmup_ratio": 1.0,
            "max_warmup_minutes": 10,
        }
    )
    config["backtest"].update(
        {
            "exchanges": ["binance"],
            "coins": {"binance": ["LONGCOIN", "SHORTCOIN"]},
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "starting_balance": 1_000.0,
            "filter_by_min_effective_cost": False,
            "dynamic_wel_by_tradability": True,
            "candle_interval_minutes": 1,
        }
    )
    for pside in ("long", "short"):
        config["bot"][pside]["risk"].update(
            {"n_positions": 1, "total_wallet_exposure_limit": 1.5}
        )
        config["bot"][pside]["forager"].update(
            {
                "volume_ema_span_1m": 2,
                "volatility_ema_span_1m": 2,
                "volume_drop_pct": 0.0,
            }
        )
        config["bot"][pside]["strategy"]["ema_anchor"].update(
            {
                "ema_span_0": 2.0,
                "ema_span_1": 3.0,
                "base_qty_pct": 0.2,
                "offset": 0.001,
                "offset_psize_weight": 0.0,
                "entry_double_down_factor": 0.0,
                "offset_volatility_ema_span_1m": 2.0,
                "offset_volatility_ema_span_1h": 2.0,
                "offset_volatility_1m_weight": 0.0,
                "offset_volatility_1h_weight": 0.0,
            }
        )
    # Without side eligibility, these rankings intentionally prefer the coin
    # approved only for the opposite side and reproduce the regression.
    config["bot"]["long"]["forager"]["score_weights"] = {
        "ema_readiness": 0.0,
        "volatility": 1.0,
        "volume": 0.0,
    }
    config["bot"]["short"]["forager"]["score_weights"] = {
        "ema_readiness": 0.0,
        "volatility": 0.0,
        "volume": 1.0,
    }
    return config


def _synthetic_inputs():
    n_minutes = 60
    start_ts = 1_704_067_200_000
    timestamps = np.arange(
        start_ts,
        start_ts + n_minutes * 60_000,
        60_000,
        dtype=np.int64,
    )
    hlcvs = np.empty((n_minutes, 2, 4), dtype=np.float64)
    for index in range(n_minutes):
        hlcvs[index, 0] = [101.0, 99.0, 100.0, 10_000.0]
        hlcvs[index, 1] = [220.0, 180.0, 200.0, 100.0]
    market_settings = {
        coin: {
            "qty_step": 0.001,
            "price_step": 0.01,
            "min_qty": 0.001,
            "min_cost": 1.0,
            "c_mult": 1.0,
            "maker": 0.0,
            "taker": 0.0,
            "exchange": "binance",
            "first_valid_index": 0,
            "last_valid_index": n_minutes - 1,
            "warmup_minutes": 3,
        }
        for coin in ("LONGCOIN", "SHORTCOIN")
    }
    market_settings["__meta__"] = {
        "requested_start_ts": start_ts,
        "requested_start_date": "2024-01-01",
        "warmup_minutes_requested": 3,
    }
    return hlcvs, market_settings, np.full(n_minutes, 50_000.0), timestamps


@pytest.mark.parametrize("hedge_mode", [True, False])
def test_ema_anchor_entries_obey_side_specific_approved_coins(hedge_mode):
    config = _ema_anchor_config(hedge_mode)
    hlcvs, market_settings, btc_prices, timestamps = _synthetic_inputs()

    fills, _, _, payload = run_backtest(
        hlcvs,
        market_settings,
        config,
        "binance",
        btc_prices,
        timestamps,
        return_payload=True,
    )

    entry_counts = Counter(
        (str(row[2]), str(row[13]))
        for row in fills
        if str(row[13]).startswith("entry_")
    )
    assert set(entry_counts) == {
        ("LONGCOIN", "entry_ema_anchor_long"),
        ("SHORTCOIN", "entry_ema_anchor_short"),
    }
    params_by_coin = dict(zip(payload.backtest_params["coins"], payload.bot_params_list))
    assert params_by_coin["LONGCOIN"]["short"]["entry_eligible"] is False
    assert params_by_coin["SHORTCOIN"]["long"]["entry_eligible"] is False


def test_zero_wel_coin_override_disables_approved_side_entries():
    config = _ema_anchor_config(hedge_mode=True)
    config["coin_overrides"] = {
        "SHORTCOIN": {"bot": {"short": {"wallet_exposure_limit": 0.0}}}
    }
    hlcvs, market_settings, btc_prices, timestamps = _synthetic_inputs()

    fills, _, _, payload = run_backtest(
        hlcvs,
        market_settings,
        config,
        "binance",
        btc_prices,
        timestamps,
        return_payload=True,
    )

    entry_pairs = {
        (str(row[2]), str(row[13]))
        for row in fills
        if str(row[13]).startswith("entry_")
    }
    assert entry_pairs == {("LONGCOIN", "entry_ema_anchor_long")}
    params_by_coin = dict(zip(payload.backtest_params["coins"], payload.bot_params_list))
    assert params_by_coin["SHORTCOIN"]["short"]["entry_eligible"] is False
