import numpy as np
import pytest

import optimize_suite
from config.schema import get_template_config
from suite_runner import ExchangeDataset


class _NoSharedArrayManager:
    def create_from(self, _array):
        raise AssertionError("test dataset should use lazy master specs")


@pytest.mark.asyncio
async def test_prepare_suite_contexts_keeps_directional_scenarios_with_default_short_disabled(
    monkeypatch,
):
    config = get_template_config()
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-02"
    config["backtest"]["exchanges"] = ["binance", "bybit"]
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [
        {"label": "base"},
        {"label": "long_only", "overrides": {"bot.short.total_wallet_exposure_limit": 0}},
        {"label": "short_only", "overrides": {"bot.long.total_wallet_exposure_limit": 0}},
    ]
    config["live"]["approved_coins"] = {"long": ["HYPE"], "short": ["HYPE"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}
    # Schema defaults keep shorts disabled. Optimizer candidates may enable
    # shorts later, so context preparation must not dedupe base vs long_only.
    config["bot"]["short"]["total_wallet_exposure_limit"] = 0.0

    async def fake_load_markets(_exchange, verbose=False):
        return {}

    async def fake_format_approved_ignored_coins(_config, _exchanges, verbose=False):
        return None

    async def fake_prepare_master_datasets(*_args, **_kwargs):
        timestamps = np.arange(1441, dtype=np.int64) * 60_000 + 1704067200000
        hlcvs = np.ones((len(timestamps), 1, 4), dtype=np.float64)
        return {
            "combined": ExchangeDataset(
                exchange="combined",
                coins=["HYPE"],
                coin_index={"HYPE": 0},
                coin_exchange={"HYPE": "binance"},
                available_exchanges=["binance", "bybit"],
                hlcvs=hlcvs,
                mss={
                    "HYPE": {
                        "exchange": "binance",
                        "first_valid_index": 0,
                        "last_valid_index": len(timestamps) - 1,
                    },
                    "__meta__": {"data_interval_minutes": 1},
                },
                btc_usd_prices=np.ones(len(timestamps), dtype=np.float64),
                timestamps=timestamps,
                cache_dir="",
                hlcvs_spec=object(),
                btc_spec=object(),
            )
        }

    monkeypatch.setattr(optimize_suite, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        optimize_suite,
        "format_approved_ignored_coins",
        fake_format_approved_ignored_coins,
    )
    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", fake_prepare_master_datasets)

    suite_cfg = optimize_suite.extract_suite_config(config, suite_override=None)
    contexts, _aggregate_cfg = await optimize_suite.prepare_suite_contexts(
        config,
        suite_cfg,
        shared_array_manager=_NoSharedArrayManager(),
    )

    assert [ctx.label for ctx in contexts] == ["base", "long_only", "short_only"]
