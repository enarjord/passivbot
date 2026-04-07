import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import passivbot_rust as pbr
from ema_anchor_analysis import (
    calc_ema_anchor_inventory_bands,
    calc_ema_anchor_inventory_bands_from_artifact,
    calc_ema_anchor_neutral_bands,
    calc_ema_anchor_neutral_bands_from_artifact,
)


def _extension_available() -> bool:
    return not hasattr(pbr.calc_entries_long_py, "__code__")


requires_extension = pytest.mark.skipif(
    not _extension_available(), reason="passivbot_rust extension not available"
)


def _ema_anchor_config() -> dict:
    return {
        "live": {"strategy_kind": "ema_anchor"},
        "bot": {
            "long": {
                "strategy": {
                    "ema_anchor": {
                        "base_qty_pct": 0.01,
                        "ema_span_0": 2.0,
                        "ema_span_1": 6.0,
                        "offset": 0.01,
                        "offset_volatility_ema_span_minutes": 2.0,
                        "offset_volatility_1m_weight": 0.5,
                        "entry_volatility_ema_span_hours": 2.0,
                        "offset_volatility_1h_weight": 0.25,
                        "offset_psize_weight": 0.2,
                    }
                }
            },
            "short": {
                "strategy": {
                    "ema_anchor": {
                        "base_qty_pct": 0.01,
                        "ema_span_0": 2.0,
                        "ema_span_1": 6.0,
                        "offset": 0.02,
                        "offset_volatility_ema_span_minutes": 2.0,
                        "offset_volatility_1m_weight": 0.5,
                        "entry_volatility_ema_span_hours": 2.0,
                        "offset_volatility_1h_weight": 0.25,
                        "offset_psize_weight": 0.1,
                    }
                }
            },
        },
    }


def _candles_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 00:00:00", periods=5, freq="1min"),
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )


@requires_extension
def test_calc_ema_anchor_neutral_bands_returns_requested_columns():
    out = calc_ema_anchor_neutral_bands(
        _ema_anchor_config(),
        _candles_df(),
        side="long",
        price_step=0.1,
    )

    assert list(out.columns) == ["timestamp", "high", "low", "close", "bid", "ask"]
    assert len(out) == 5
    assert (out["bid"] <= out["close"]).all()
    assert (out["ask"] >= out["close"]).all()


@requires_extension
def test_calc_ema_anchor_inventory_bands_shifts_fill_state_to_next_candle():
    candles = _candles_df()
    fills = pd.DataFrame(
        {
            "timestamp": [candles.loc[1, "timestamp"], candles.loc[3, "timestamp"]],
            "coin": ["BTC", "BTC"],
            "type": ["entry_ema_anchor_long", "entry_ema_anchor_short"],
            "qty": [1.0, -2.0],
            "price": [101.0, 103.0],
            "psize": [1.0, -1.0],
            "pprice": [101.0, 103.0],
            "usd_total_balance": [999.0, 998.0],
        }
    )
    balance_eq = pd.DataFrame(
        {
            "timestamp": [candles.loc[0, "timestamp"]],
            "usd_total_balance": [1000.0],
            "usd_total_equity": [1000.0],
        }
    )

    out = calc_ema_anchor_inventory_bands(
        _ema_anchor_config(),
        candles,
        fills,
        balance_eq,
        price_step=0.1,
        coin="BTC",
        side_mode="active",
    )

    assert out.loc[1, "fill_qty_bid"] == pytest.approx(1.0)
    assert out.loc[3, "fill_qty_ask"] == pytest.approx(-2.0)
    assert out.loc[1, "psize"] == pytest.approx(0.0)
    assert out.loc[2, "psize"] == pytest.approx(1.0)
    assert out.loc[4, "psize"] == pytest.approx(-1.0)
    assert out.loc[2, "bid"] == pytest.approx(out.loc[2, "bid_long"])
    assert out.loc[4, "ask"] == pytest.approx(out.loc[4, "ask_short"])


@requires_extension
def test_ema_anchor_artifact_helpers_load_dataset_metadata(tmp_path):
    artifact_dir = tmp_path / "artifact"
    cache_dir = tmp_path / "caches" / "hlcvs_data" / "abcd1234"
    artifact_dir.mkdir()
    cache_dir.mkdir(parents=True)

    config = _ema_anchor_config()
    dataset = {
        "exchange": "combined",
        "hlcv_cache_dir": str(cache_dir.resolve()),
        "hlcvs_file": str((cache_dir / "hlcvs.npy").resolve()),
        "timestamps_file": str((cache_dir / "timestamps.npy").resolve()),
        "btc_usd_prices_file": str((cache_dir / "btc_usd_prices.npy").resolve()),
        "coins_file": str((cache_dir / "coins.json").resolve()),
        "market_specific_settings_file": str((cache_dir / "market_specific_settings.json").resolve()),
        "coins": ["BTC"],
        "coin_index": {"BTC": 0},
    }
    (artifact_dir / "config.json").write_text(json.dumps(config))
    (artifact_dir / "dataset.json").write_text(json.dumps(dataset))
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:01:00"],
            "coin": ["BTC"],
            "type": ["entry_ema_anchor_long"],
            "qty": [1.0],
            "price": [101.0],
            "psize": [1.0],
            "pprice": [101.0],
            "usd_total_balance": [999.0],
        }
    ).to_csv(artifact_dir / "fills.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00"],
            "usd_total_balance": [1000.0],
            "usd_total_equity": [1000.0],
        }
    ).to_csv(artifact_dir / "balance_and_equity.csv.gz", index=False, compression="gzip")

    timestamps = np.array(
        [int(ts.value // 1_000_000) for ts in pd.date_range("2026-01-01 00:00:00", periods=5, freq="1min")],
        dtype=np.int64,
    )
    hlcvs = np.zeros((5, 1, 4), dtype=float)
    hlcvs[:, 0, 0] = [101, 102, 103, 104, 105]
    hlcvs[:, 0, 1] = [99, 100, 101, 102, 103]
    hlcvs[:, 0, 2] = [100, 101, 102, 103, 104]
    np.save(cache_dir / "hlcvs.npy", hlcvs)
    np.save(cache_dir / "timestamps.npy", timestamps)
    np.save(cache_dir / "btc_usd_prices.npy", np.ones(5))
    (cache_dir / "coins.json").write_text(json.dumps(["BTC"]))
    (cache_dir / "market_specific_settings.json").write_text(json.dumps({"BTC": {"price_step": 0.1}}))

    neutral = calc_ema_anchor_neutral_bands_from_artifact(artifact_dir, side="long")
    inventory = calc_ema_anchor_inventory_bands_from_artifact(artifact_dir, side_mode="active")

    assert list(neutral.columns) == ["timestamp", "high", "low", "close", "bid", "ask"]
    assert "fill_qty_bid" in inventory.columns
    assert inventory.loc[2, "psize"] == pytest.approx(1.0)
