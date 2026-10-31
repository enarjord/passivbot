from copy import deepcopy

import numpy as np
import pytest

from backtest import build_backtest_payload, execute_backtest
from config.universe import set_side_universe_sources
from config_utils import load_config
from suite_runner import ExchangeDataset, _prepare_dataset_subset
from utils import ts_to_date


def _inputs(interval=1, offset=0):
    coins = ["ALWAYS", "EARLY", "LATE", "MISSING"]
    n_rows = 180
    timestamps = 1_704_067_200_000 + np.arange(n_rows) * interval * 60_000
    hlcvs = np.full((n_rows, len(coins), 4), np.nan)
    ranges = [(0, 179), (0, 59), (120, 179), (180, 180)]
    mss = {}
    for idx, (coin, (first, last)) in enumerate(zip(coins, ranges)):
        if first < n_rows:
            hlcvs[first:last + 1, idx] = [101.0, 99.0, 100.0, 10.0]
        mss[coin] = {
            "first_valid_index": first * interval + offset,
            "last_valid_index": (last + 1) * interval - 1 + offset,
            "qty_step": 0.001,
            "price_step": 0.01,
            "min_qty": 0.001,
            "min_cost": 1.0,
            "c_mult": 1.0,
            "maker": 0.0,
            "taker": 0.0,
            "exchange": "binance",
        }
    mss["__meta__"] = {
        "data_interval_minutes": interval,
        "candle_interval_offset_bars": offset,
    }
    dataset = ExchangeDataset(
        exchange="binance",
        coins=coins,
        coin_index={coin: idx for idx, coin in enumerate(coins)},
        coin_exchange={coin: "binance" for coin in coins},
        available_exchanges=["binance"],
        hlcvs=hlcvs,
        mss=mss,
        btc_usd_prices=np.full(n_rows, 50_000.0),
        timestamps=timestamps,
        cache_dir="unused",
    )
    config = load_config("configs/examples/ema_anchor.json", verbose=False)
    set_side_universe_sources(config, "approved_coins", {"long": coins, "short": coins})
    set_side_universe_sources(config, "ignored_coins", {"long": [], "short": []})
    config["backtest"].update(
        exchanges=["binance"], coins={"binance": coins},
        candle_interval_minutes=interval, filter_by_min_effective_cost=False,
    )
    config["bot"]["portfolio"]["warmup"].update(activation_span_fraction=1.0, max_minutes=1)
    for side in ("long", "short"):
        config["bot"][side]["universe"]["minimum_coin_age_days"] = 0.0
        config["bot"][side]["risk"].update(
            max_positions=1, total_wallet_exposure_limit_ratio=1.5,
        )
        config["bot"][side]["forager"].update(
            volume_ema_span_minutes=2.0, log_range_ema_span_minutes=2.0,
        )
        config["bot"][side]["strategy"]["ema_anchor"].update(
            ema_span_0_minutes=2.0, ema_span_1_minutes=3.0,
            base_qty_fraction=0.2, offset_fraction=0.001,
            inventory_offset_exposure_coefficient=0.0,
            entry_qty_exposure_coefficient=0.0,
            offset_log_range_ema_span_minutes=2.0,
            offset_log_range_ema_span_1h_candles=2.0,
            offset_log_range_1m_coefficient=0.0, offset_log_range_1h_coefficient=0.0,
        )
    return dataset, config


@pytest.mark.parametrize("interval,offset", [(1, 0), (5, 0), (5, 2)])
@pytest.mark.parametrize("first,last", [(0, 59), (60, 119), (120, 179), (30, 149)])
def test_subset_intersects_valid_ranges_without_creating_edge_candles(
    monkeypatch, interval, offset, first, last
):
    dataset, config = _inputs(interval, offset)
    original_mss = deepcopy(dataset.mss)
    config["backtest"].update(
        start_date=ts_to_date(int(dataset.timestamps[first])),
        end_date=ts_to_date(int(dataset.timestamps[last])),
    )
    monkeypatch.setattr("suite_runner.compute_backtest_warmup_minutes", lambda cfg: 0)
    monkeypatch.setattr("suite_runner.compute_per_coin_warmup_minutes", lambda cfg: {"__default__": 0})

    hlcvs, _, _, mss = _prepare_dataset_subset(dataset, config, dataset.coins, "window")

    for idx, coin in enumerate(dataset.coins):
        finite = np.flatnonzero(np.isfinite(hlcvs[:, idx, 2]))
        if len(finite):
            assert mss[coin]["first_valid_index"] == finite[0] * interval
            assert mss[coin]["last_valid_index"] == (finite[-1] + 1) * interval - 1
        else:
            assert mss[coin]["first_valid_index"] == len(hlcvs) * interval
            assert mss[coin]["last_valid_index"] < mss[coin]["first_valid_index"]
            assert mss[coin]["trade_start_index"] == len(hlcvs) * interval
            assert np.isnan(hlcvs[:, idx]).all()
    assert dataset.mss == original_mss


def test_subset_infers_empty_range_when_metadata_is_missing(monkeypatch):
    dataset, config = _inputs()
    for coin in dataset.coins:
        dataset.mss[coin].pop("first_valid_index")
        dataset.mss[coin].pop("last_valid_index")
    config["backtest"].update(
        start_date=ts_to_date(int(dataset.timestamps[60])),
        end_date=ts_to_date(int(dataset.timestamps[119])),
    )
    monkeypatch.setattr("suite_runner.compute_backtest_warmup_minutes", lambda cfg: 0)
    monkeypatch.setattr("suite_runner.compute_per_coin_warmup_minutes", lambda cfg: {"__default__": 0})

    hlcvs, _, _, mss = _prepare_dataset_subset(dataset, config, dataset.coins, "inferred")

    for coin in ("EARLY", "LATE", "MISSING"):
        assert mss[coin]["first_valid_index"] == len(hlcvs)
        assert mss[coin]["last_valid_index"] < mss[coin]["first_valid_index"]


@pytest.mark.parametrize("interval", [1, 5])
@pytest.mark.parametrize("first,last", [(0, 59), (60, 119), (120, 179)])
def test_subset_empty_ranges_execute_with_real_rust(interval, first, last):
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(passivbot_rust, "__is_stub__", False)
    verify_loaded_runtime_extension()
    dataset, config = _inputs(interval)
    config["backtest"].update(
        start_date=ts_to_date(int(dataset.timestamps[first])),
        end_date=ts_to_date(int(dataset.timestamps[last])),
    )
    hlcvs, btc, timestamps, mss = _prepare_dataset_subset(dataset, config, dataset.coins, "rust")
    payload = build_backtest_payload(hlcvs, mss, config, "binance", btc, timestamps)

    fills, equities, analysis = execute_backtest(payload, config)

    assert len(fills) > 0
    assert len(equities) > 0
    assert analysis["liquidated"] is False
    empty_coins = {coin for idx, coin in enumerate(dataset.coins) if np.isnan(hlcvs[:, idx]).all()}
    assert all(str(fill[2]) not in empty_coins for fill in fills)


def test_subset_keeps_internal_missing_candles_fatal():
    import passivbot_rust

    assert not getattr(passivbot_rust, "__is_stub__", False)
    dataset, config = _inputs()
    config["backtest"].update(
        start_date=ts_to_date(int(dataset.timestamps[60])),
        end_date=ts_to_date(int(dataset.timestamps[119])),
    )
    dataset.hlcvs[80, 0, :3] = np.nan
    hlcvs, btc, timestamps, mss = _prepare_dataset_subset(dataset, config, dataset.coins, "gap")
    payload = build_backtest_payload(hlcvs, mss, config, "binance", btc, timestamps)

    with pytest.raises(ValueError, match="contiguous finite H/L/C.*coin ALWAYS"):
        execute_backtest(payload, config)
