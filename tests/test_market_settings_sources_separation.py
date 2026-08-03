"""Test that market_settings_sources doesn't affect OHLCV data selection."""

import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import OhlcvStore


def test_ohlcv_source_extraction():
    """Verify that ohlcv_source is correctly extracted from mss dict."""
    from collections import defaultdict

    # Simulate what _prepare_hlcvs_combined_impl does
    chosen_mss_per_coin = {
        "BTC": {
            "exchange": "bybit",  # market settings from bybit
            "ohlcv_source": "binance",  # OHLCV from binance
            "min_cost": 10.0,
        },
        "ETH": {
            "exchange": "binance",  # market settings from binance (same as OHLCV)
            "min_cost": 5.0,
        },
        "DOGE": {
            "exchange": "hyperliquid",  # market settings from hyperliquid
            "ohlcv_source": "binance",  # OHLCV from binance
            "min_cost": 1.0,
        },
    }

    valid_coins = ["BTC", "ETH", "DOGE"]

    # Test 1: exchanges_with_data should use OHLCV sources
    exchanges_with_data = sorted(
        set(
            [
                chosen_mss_per_coin[coin].get("ohlcv_source", chosen_mss_per_coin[coin]["exchange"])
                for coin in valid_coins
            ]
        )
    )

    # Should only have "binance" since all OHLCV comes from there
    assert exchanges_with_data == [
        "binance"
    ], f"Expected only 'binance' for OHLCV, got {exchanges_with_data}"

    # Test 2: exchanges_counts should count OHLCV sources, not market settings
    exchanges_counts = defaultdict(int)
    for coin in chosen_mss_per_coin:
        ohlcv_exchange = chosen_mss_per_coin[coin].get(
            "ohlcv_source", chosen_mss_per_coin[coin]["exchange"]
        )
        exchanges_counts[ohlcv_exchange] += 1

    # All 3 coins should count towards "binance" for OHLCV
    assert (
        exchanges_counts["binance"] == 3
    ), f"Expected 3 coins from binance, got {exchanges_counts['binance']}"
    assert "bybit" not in exchanges_counts, "bybit should not be in OHLCV counts"
    assert "hyperliquid" not in exchanges_counts, "hyperliquid should not be in OHLCV counts"

    # Test 3: reference_exchange should be determined by OHLCV sources
    reference_exchange = sorted(exchanges_counts.items(), key=lambda x: x[1])[-1][0]
    assert (
        reference_exchange == "binance"
    ), f"Expected reference_exchange='binance', got '{reference_exchange}'"

    # Test 4: verify market settings are preserved separately
    assert chosen_mss_per_coin["BTC"]["exchange"] == "bybit"
    assert chosen_mss_per_coin["DOGE"]["exchange"] == "hyperliquid"
    assert chosen_mss_per_coin["ETH"]["exchange"] == "binance"


def test_market_settings_vs_ohlcv_separation():
    """Verify that market settings exchange doesn't leak into OHLCV logic."""

    # Scenario: User wants bybit market settings but binance OHLCV data
    mss_entry = {
        "exchange": "bybit",  # market settings source
        "ohlcv_source": "binance",  # OHLCV data source
        "min_cost": 0.01,
        "min_qty": 0.1,
        "symbol": "BTC/USDT:USDT",
    }

    # Extract OHLCV source (what should be used for volume normalization)
    ohlcv_exchange = mss_entry.get("ohlcv_source", mss_entry["exchange"])

    assert ohlcv_exchange == "binance", f"OHLCV should come from binance, got {ohlcv_exchange}"

    # Extract market settings source (what should be used for min_cost, etc.)
    market_settings_exchange = mss_entry["exchange"]

    assert (
        market_settings_exchange == "bybit"
    ), f"Market settings should come from bybit, got {market_settings_exchange}"

    # Verify they are different
    assert (
        ohlcv_exchange != market_settings_exchange
    ), "OHLCV and market settings should be able to differ"


def test_coins_by_exchange_grouping():
    """Verify logging groups coins by OHLCV source, not market settings."""
    from collections import defaultdict

    chosen_mss_per_coin = {
        "BTC": {"exchange": "bybit", "ohlcv_source": "binance"},
        "ETH": {"exchange": "bybit", "ohlcv_source": "binance"},
        "DOGE": {"exchange": "hyperliquid", "ohlcv_source": "binance"},
        "SOL": {"exchange": "binance"},  # No ohlcv_source, use exchange
    }

    valid_coins = ["BTC", "ETH", "DOGE", "SOL"]

    # Simulate the grouping logic for logging
    coins_by_exchange = defaultdict(list)
    for coin in valid_coins:
        ohlcv_ex = chosen_mss_per_coin[coin].get(
            "ohlcv_source", chosen_mss_per_coin[coin]["exchange"]
        )
        coins_by_exchange[ohlcv_ex].append(coin)

    # All coins should be grouped under "binance" for OHLCV
    assert "binance" in coins_by_exchange
    assert sorted(coins_by_exchange["binance"]) == ["BTC", "DOGE", "ETH", "SOL"]

    # No coins should be grouped under bybit/hyperliquid for OHLCV
    assert "bybit" not in coins_by_exchange
    assert "hyperliquid" not in coins_by_exchange


@pytest.mark.asyncio
async def test_prepare_hlcvs_combined_impl_uses_ohlcv_source_for_normalization_provenance(
    monkeypatch, tmp_path
):
    """Verify market-settings sources do not enter normalization provenance."""
    import hlcv_preparation as hp

    start_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC
    candle_df = pd.DataFrame(
        {
            "timestamp": [start_ts, start_ts + 60_000, start_ts + 120_000],
            "open": [1.0, 1.0, 1.0],
            "high": [2.0, 2.0, 2.0],
            "low": [0.5, 0.5, 0.5],
            "close": [1.5, 1.5, 1.5],
            "volume": [10.0, 20.0, 30.0],
        }
    )

    class DummyOM:
        def __init__(self, exchange_id: str):
            self.exchange_id = exchange_id

        async def load_markets(self):
            return None

        def get_symbol(self, coin):
            return coin

        def get_market_specific_settings(self, _coin):
            return {"exchange": self.exchange_id, "min_cost": 1.0}

    om_dict = {"binanceusdm": DummyOM("binanceusdm"), "bybit": DummyOM("bybit")}

    async def fake_get_first_timestamps_unified(_coins):
        return {"BTC": start_ts}

    async def fake_fetch_data_for_coin_and_exchange(coin, ex, *_args, **_kwargs):
        if coin != "BTC":
            return None
        if ex == "binanceusdm":
            return ex, candle_df.copy(), 3, 0, 1_000.0
        if ex == "bybit":
            return ex, candle_df.copy(), 2, 0, 500.0
        return None

    monkeypatch.setattr(hp, "get_first_timestamps_unified", fake_get_first_timestamps_unified)
    monkeypatch.setattr(hp, "fetch_data_for_coin_and_exchange", fake_fetch_data_for_coin_and_exchange)

    config = {
        "backtest": {"gap_tolerance_ohlcvs_minutes": 120},
        "bot": {
            "long": {
                "n_positions": 1,
                "total_wallet_exposure_limit": 1.0,
                "wallet_exposure_limit": 1.0,
            },
            "short": {
                "n_positions": 0,
                "total_wallet_exposure_limit": 0.0,
                "wallet_exposure_limit": 0.0,
            },
        },
        "live": {
            "approved_coins": {"long": ["BTC/USDT:USDT"], "short": []},
            "minimum_coin_age_days": 0,
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
    }

    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)

    mss, _timestamps, aligned_values_by_coin = await hp._prepare_hlcvs_combined_impl(
        config=config,
        om_dict=om_dict,
        base_start_ts=start_ts,
        _requested_start_ts=start_ts,
        end_ts=start_ts + 180_000,
        forced_sources={},
        market_settings_sources={"BTC": "bybit"},
        force_refetch_gaps=False,
        catalog=catalog,
        store=store,
        legacy_root=None,
    )

    assert mss["BTC"]["exchange"] == "bybit"
    assert mss["BTC"]["ohlcv_source"] == "binance"
    normalization = mss["__preparation_meta__"]["volume_normalization"]
    assert normalization["exchange_counts"] == {"binance": 1}
    assert normalization["reference_exchange"] == "binance"
    assert aligned_values_by_coin["BTC"][:, 3].sum() == pytest.approx(candle_df["volume"].sum())


@pytest.mark.asyncio
async def test_prepare_hlcvs_combined_impl_honors_disabled_volume_normalization(
    monkeypatch, tmp_path
):
    import hlcv_preparation as hp

    start_ts = 1_704_067_200_000
    timestamps = [start_ts + i * 60_000 for i in range(3)]

    class DummyOM:
        async def load_markets(self):
            return None

        def get_symbol(self, coin):
            return coin

        def get_market_specific_settings(self, _coin):
            return {"exchange": "unused", "min_cost": 1.0}

    async def fake_get_first_timestamps_unified(_coins):
        return {"BTC": start_ts, "ETH": start_ts}

    async def fake_fetch(coin, exchange, *_args, **_kwargs):
        volume = 10.0 if exchange == "binanceusdm" else 100.0
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "high": [2.0] * 3,
                "low": [0.5] * 3,
                "close": [1.5] * 3,
                "volume": [volume] * 3,
            }
        )
        return exchange, df, 3, 0, float(df["volume"].sum())

    monkeypatch.setattr(hp, "get_first_timestamps_unified", fake_get_first_timestamps_unified)
    monkeypatch.setattr(hp, "fetch_data_for_coin_and_exchange", fake_fetch)
    monkeypatch.setattr(
        hp,
        "compute_exchange_volume_ratios_with_diagnostics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("normalization estimator must not run when disabled")
        ),
    )

    config = {
        "backtest": {
            "gap_tolerance_ohlcvs_minutes": 120,
            "volume_normalization": False,
        },
        "bot": {
            "long": {
                "n_positions": 2,
                "total_wallet_exposure_limit": 1.0,
                "wallet_exposure_limit": 0.5,
            },
            "short": {
                "n_positions": 0,
                "total_wallet_exposure_limit": 0.0,
                "wallet_exposure_limit": 0.0,
            },
        },
        "live": {
            "approved_coins": {"long": ["BTC", "ETH"], "short": []},
            "minimum_coin_age_days": 0,
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
    }
    om_dict = {"binanceusdm": DummyOM(), "bybit": DummyOM()}
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)

    mss, _timestamps, values = await hp._prepare_hlcvs_combined_impl(
        config=config,
        om_dict=om_dict,
        base_start_ts=start_ts,
        _requested_start_ts=start_ts,
        end_ts=timestamps[-1],
        forced_sources={"BTC": "binanceusdm", "ETH": "bybit"},
        force_refetch_gaps=False,
        ohlcv_exchanges=["binanceusdm", "bybit"],
        catalog=catalog,
        store=store,
        legacy_root=None,
    )

    assert values["BTC"][:, 3].sum() == pytest.approx(30.0)
    assert values["ETH"][:, 3].sum() == pytest.approx(300.0)
    normalization = mss["__preparation_meta__"]["volume_normalization"]
    assert normalization["enabled"] is False
    assert normalization["scale_factors_to_reference"] == {"binance": 1.0, "bybit": 1.0}


@pytest.mark.asyncio
async def test_prepare_hlcvs_combined_impl_normalizes_forced_override_only_exchange(
    monkeypatch, tmp_path
):
    import hlcv_preparation as hp

    start_ts = 1_704_067_200_000
    timestamps = [start_ts + i * 60_000 for i in range(1440)]
    coins = ["BTC", "ETH", "SOL"]

    class DummyOM:
        def __init__(self, exchange):
            self.exchange = exchange

        async def load_markets(self):
            return None

        def has_coin(self, coin):
            return coin in coins

        def get_symbol(self, coin):
            return coin

        def get_market_specific_settings(self, _coin):
            return {"exchange": self.exchange, "min_cost": 1.0}

    async def fake_get_first_timestamps_unified(_coins):
        return {coin: start_ts for coin in coins}

    async def fake_fetch(coin, exchange, *_args, **_kwargs):
        volume = 2.0 if exchange == "binanceusdm" else 1.0
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "high": [2.0] * len(timestamps),
                "low": [0.5] * len(timestamps),
                "close": [1.5] * len(timestamps),
                "volume": [volume] * len(timestamps),
            }
        )
        return exchange, df, len(df), 0, float(df["volume"].sum())

    monkeypatch.setattr(hp, "get_first_timestamps_unified", fake_get_first_timestamps_unified)
    monkeypatch.setattr(hp, "fetch_data_for_coin_and_exchange", fake_fetch)

    config = {
        "backtest": {
            "gap_tolerance_ohlcvs_minutes": 120,
            "volume_normalization": True,
        },
        "bot": {
            "long": {
                "n_positions": 3,
                "total_wallet_exposure_limit": 1.0,
                "wallet_exposure_limit": 1.0 / 3.0,
            },
            "short": {
                "n_positions": 0,
                "total_wallet_exposure_limit": 0.0,
                "wallet_exposure_limit": 0.0,
            },
        },
        "live": {
            "approved_coins": {"long": coins, "short": []},
            "minimum_coin_age_days": 0,
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
    }
    om_dict = {
        "binanceusdm": DummyOM("binanceusdm"),
        "bybit": DummyOM("bybit"),
    }
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)

    mss, _timestamps, values = await hp._prepare_hlcvs_combined_impl(
        config=config,
        om_dict=om_dict,
        base_start_ts=start_ts,
        _requested_start_ts=start_ts,
        end_ts=timestamps[-1],
        forced_sources={"BTC": "bybit"},
        force_refetch_gaps=False,
        ohlcv_exchanges=["binanceusdm"],
        normalization_candidate_exchanges=["binanceusdm", "bybit"],
        catalog=catalog,
        store=store,
        legacy_root=None,
    )

    assert mss["BTC"]["exchange"] == "bybit"
    assert mss["ETH"]["exchange"] == "binance"
    assert mss["SOL"]["exchange"] == "binance"
    normalization = mss["__preparation_meta__"]["volume_normalization"]
    assert normalization["scale_factors_to_reference"] == {
        "binance": pytest.approx(1.0),
        "bybit": pytest.approx(2.0),
    }
    assert values["BTC"][:, 3].sum() == pytest.approx(2880.0)
