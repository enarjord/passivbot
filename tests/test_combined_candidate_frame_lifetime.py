import asyncio
import gc
import weakref

import numpy as np
import pandas as pd
import pytest

import hlcv_preparation as hp


@pytest.mark.asyncio
async def test_combined_candidate_frames_are_released_after_normalization(monkeypatch):
    coins = ["BTC", "ETH"]
    timestamps = np.array([1_704_067_200_000, 1_704_067_260_000], dtype=np.int64)
    candidate_refs = {}
    selected_exchanges = {"BTC": "binance", "ETH": "bybit"}
    normalization_finished = False
    alignment_lifetime_checked = False

    class DummyManager:
        async def load_markets(self):
            return None

    class QuietProgress:
        def __init__(self, *_args, **_kwargs):
            pass

        def maybe_log(self, *_args, **_kwargs):
            pass

        def update(self, *_args, **_kwargs):
            pass

        def log_done(self, *_args, **_kwargs):
            pass

    async def fake_resolve_combined_coin(**kwargs):
        coin = kwargs["coin"]
        candidates = []
        selected_df = None
        for exchange, volume in (("binance", 10.0), ("bybit", 20.0)):
            df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "high": [2.0, 2.0],
                    "low": [0.5, 0.5],
                    "close": [1.5, 1.5],
                    "volume": [volume, volume],
                    "valid": [True, True],
                }
            )
            candidate_refs[(coin, exchange)] = weakref.ref(df)
            candidates.append(
                hp.CombinedExchangeCandidate(
                    exchange=exchange,
                    df=df,
                    coverage_count=2,
                    gap_count=0,
                    total_volume=volume * 2,
                )
            )
            if exchange == selected_exchanges[coin]:
                selected_df = df

        resolution_kwargs = {
            "coin": coin,
            "best_exchange": selected_exchanges[coin],
            "best_df": selected_df,
            "market_settings": {
                "exchange": selected_exchanges[coin],
                "ohlcv_source": selected_exchanges[coin],
                "warmup_minutes": 0,
            },
            "candidates": tuple(candidates),
        }
        if "selection_reason" in hp.CombinedCoinResolution.__dataclass_fields__:
            resolution_kwargs["selection_reason"] = "configured_exchange_priority"
        return hp.CombinedCoinResolution(**resolution_kwargs)

    def record_normalization_inputs(*_args, **_kwargs):
        nonlocal normalization_finished
        assert len(candidate_refs) == 4
        assert all(ref() is not None for ref in candidate_refs.values())
        normalization_finished = True
        return {("binance", "bybit"): 1.0}

    def record_normalization_inputs_with_diagnostics(*_args, **_kwargs):
        ratios = record_normalization_inputs()
        return ratios, {
            "method": "test",
            "window_start_ts": int(timestamps[0]),
            "window_end_ts_exclusive": int(timestamps[-1] + 60_000),
            "pair_estimates": {},
        }

    original_set_index = pd.DataFrame.set_index

    def assert_lifetime_at_alignment(self, keys, *args, **kwargs):
        nonlocal alignment_lifetime_checked
        if normalization_finished and not alignment_lifetime_checked and keys == "timestamp":
            gc.collect()
            for coin in coins:
                unselected = "bybit" if selected_exchanges[coin] == "binance" else "binance"
                assert candidate_refs[(coin, unselected)]() is None
                assert candidate_refs[(coin, selected_exchanges[coin])]() is not None
            alignment_lifetime_checked = True
        return original_set_index(self, keys, *args, **kwargs)

    monkeypatch.setattr(hp, "effective_backtest_data_coins", lambda _config: coins)
    monkeypatch.setattr(hp, "compute_per_coin_warmup_minutes", lambda _config: {"__default__": 0})
    monkeypatch.setattr(
        hp,
        "get_first_timestamps_unified",
        lambda _coins, **_kwargs: asyncio.sleep(0, result={}),
    )
    monkeypatch.setattr(hp, "_normalize_combined_coins", lambda values, *_args: list(values))
    monkeypatch.setattr(hp, "_resolve_combined_coin", fake_resolve_combined_coin)
    monkeypatch.setattr(hp, "ProgressTracker", QuietProgress)
    monkeypatch.setattr(
        hp,
        "compute_exchange_volume_ratios_from_candidates",
        record_normalization_inputs,
    )
    if hasattr(hp, "compute_exchange_volume_ratios_with_diagnostics"):
        monkeypatch.setattr(
            hp,
            "compute_exchange_volume_ratios_with_diagnostics",
            record_normalization_inputs_with_diagnostics,
        )
    monkeypatch.setattr(pd.DataFrame, "set_index", assert_lifetime_at_alignment)

    config = {"backtest": {}, "live": {"minimum_coin_age_days": 0.0}}
    _mss, output_timestamps, aligned = await hp._prepare_hlcvs_combined_impl(
        config,
        {"binance": DummyManager(), "bybit": DummyManager()},
        base_start_ts=int(timestamps[0]),
        _requested_start_ts=int(timestamps[0]),
        end_ts=int(timestamps[-1]),
        forced_sources={},
        force_refetch_gaps=False,
        catalog=None,
        store=None,
        legacy_root=None,
        use_v2_local=False,
    )

    assert alignment_lifetime_checked
    np.testing.assert_array_equal(output_timestamps, timestamps)
    np.testing.assert_allclose(aligned["BTC"][:, 3], [10.0, 10.0])
    np.testing.assert_allclose(aligned["ETH"][:, 3], [20.0, 20.0])
