import gzip
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest

from backtest_artifacts import (
    candles_for_coin,
    load_backtest_artifact,
    load_backtest_artifact_workspace,
    plot_fills_for_coin,
)
from plotting import create_forager_coin_figures


def _save_npy_gz(path: Path, array: np.ndarray) -> None:
    with gzip.open(path, "wb") as f:
        np.save(f, array)


def _write_artifact(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "backtests" / "combined" / "run"
    cache_dir = tmp_path / "caches" / "hlcvs_data" / "combined__BTC_ETH__abc123"
    artifact_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

    timestamps = np.array(
        [
            int(ts.value // 1_000_000)
            for ts in pd.date_range("2026-01-01 00:00:00", periods=5, freq="1min")
        ],
        dtype=np.int64,
    )
    hlcvs = np.array(
        [
            [[101.0, 99.0, 100.0, 10.0], [201.0, 199.0, 200.0, 20.0]],
            [[102.0, 100.0, 101.0, 11.0], [202.0, 200.0, 201.0, 21.0]],
            [[103.0, 101.0, 102.0, 12.0], [203.0, 201.0, 202.0, 22.0]],
            [[104.0, 102.0, 103.0, 13.0], [204.0, 202.0, 203.0, 23.0]],
            [[105.0, 103.0, 104.0, 14.0], [205.0, 203.0, 204.0, 24.0]],
        ],
        dtype=np.float64,
    )
    _save_npy_gz(cache_dir / "hlcvs.npy.gz", hlcvs)
    _save_npy_gz(cache_dir / "timestamps.npy.gz", timestamps)
    np.save(cache_dir / "btc_usd_prices.npy", np.array([100.0, 101.0, 102.0, 103.0, 104.0]))
    (cache_dir / "market_specific_settings.json").write_text(
        json.dumps({"BTC": {"price_step": 0.1}, "ETH": {"price_step": 0.01}}),
        encoding="utf-8",
    )
    (cache_dir / "coins.json").write_text(json.dumps(["BTC", "ETH"]), encoding="utf-8")

    (artifact_dir / "dataset.json").write_text(
        json.dumps(
            {
                "exchange": "combined",
                "hlcv_cache_dir": str(cache_dir),
                "hlcvs_file": str(cache_dir / "hlcvs.npy.gz"),
                "timestamps_file": str(cache_dir / "timestamps.npy.gz"),
                "btc_usd_prices_file": str(cache_dir / "btc_usd_prices.npy"),
                "coins_file": str(cache_dir / "coins.json"),
                "market_specific_settings_file": str(cache_dir / "market_specific_settings.json"),
                "coins": ["BTC", "ETH"],
                "coin_index": {"BTC": 0, "ETH": 1},
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "config.json").write_text(json.dumps({"config_version": "test"}), encoding="utf-8")
    (artifact_dir / "analysis.json").write_text(json.dumps({"adg_usd": 0.01}), encoding="utf-8")
    pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:01:00",
                "2026-01-01 00:03:00",
                "2026-01-01 00:04:00",
            ],
            "coin": ["BTC", "BTC", "BTC"],
            "type": ["entry_grid_normal_long", "entry_grid_normal_long", "close_grid_long"],
            "qty": [1.0, 1.0, -2.0],
            "price": [101.0, 103.0, 104.0],
            "psize": [1.0, 2.0, 0.0],
            "pprice": [101.0, 102.0, 0.0],
            "pnl": [0.0, 0.0, 2.0],
        }
    ).to_csv(artifact_dir / "fills.csv")
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00"],
            "usd_total_balance": [1000.0],
            "usd_total_equity": [1001.0],
            "strategy_equity": [1002.0],
        }
    ).to_csv(artifact_dir / "balance_and_equity.csv.gz", compression="gzip")
    return artifact_dir


def test_load_backtest_artifact_loads_cache_arrays_and_tables(tmp_path):
    artifact_dir = _write_artifact(tmp_path)

    artifact = load_backtest_artifact(artifact_dir)

    assert artifact.config["config_version"] == "test"
    assert artifact.analysis["adg_usd"] == pytest.approx(0.01)
    assert artifact.coins == ["BTC", "ETH"]
    assert artifact.hlcvs.shape == (5, 2, 4)
    assert artifact.timestamps.tolist() == [
        int(ts.value // 1_000_000)
        for ts in pd.date_range("2026-01-01 00:00:00", periods=5, freq="1min")
    ]
    assert artifact.btc_usd_prices.tolist() == [100.0, 101.0, 102.0, 103.0, 104.0]
    assert artifact.fills.loc[0, "timestamp"] == pd.Timestamp("2026-01-01 00:01:00")
    assert "Unnamed: 0" not in artifact.fills.columns
    assert artifact.balance_and_equity.loc[0, "strategy_equity"] == pytest.approx(1002.0)


def test_load_backtest_artifact_candles_for_coin_uses_dataset_coin_order(tmp_path):
    artifact = load_backtest_artifact(_write_artifact(tmp_path))

    candles = artifact.candles_for_coin("ETH")

    assert list(candles.columns) == ["timestamp", "high", "low", "close", "volume"]
    assert candles.loc[0, "high"] == pytest.approx(201.0)
    assert candles.loc[2, "close"] == pytest.approx(202.0)
    assert candles.loc[1, "volume"] == pytest.approx(21.0)


def test_load_backtest_artifact_workspace_is_jupyter_friendly(tmp_path):
    workspace = load_backtest_artifact_workspace(_write_artifact(tmp_path))

    assert set(
        [
            "artifact",
            "config",
            "cfg",
            "analysis",
            "fills",
            "fdf",
            "balance_and_equity",
            "bdf",
            "hlcvs",
            "timestamps",
            "btc_usd_prices",
            "coins",
            "coin_index",
            "market_settings",
            "candles_for_coin",
            "plot_fills_for_coin",
        ]
    ).issubset(workspace)
    candles = candles_for_coin(workspace, "BTC")
    assert candles.loc[1, "close"] == pytest.approx(101.0)


def test_plot_fills_for_coin_returns_figure_for_loaded_artifact(tmp_path):
    artifact = load_backtest_artifact(_write_artifact(tmp_path))

    fig = plot_fills_for_coin(
        artifact,
        "BTC",
        start_date="2026-01-01T00:00:00",
        end_date="2026-01-01T00:04:00",
    )

    try:
        assert fig.axes
        ax = fig.axes[0]
        assert ax.get_title() == "Fills BTC"
        assert ax.get_xlabel() == "datetime"
        assert len(ax.lines) >= 2
        assert len(ax.collections) >= 2
        pprice_lines = [line for line in ax.lines if line.get_label() == "long pprice"]
        assert len(pprice_lines) == 1
        pprice_y = pprice_lines[0].get_ydata()
        assert np.isnan(pprice_y).any()
        assert pprice_y[np.isfinite(pprice_y)].tolist() == [101.0, 101.0]
        from plotting import plt

        assert not plt.fignum_exists(fig.number)
    finally:
        from plotting import plt

        plt.close(fig)


def test_workspace_plot_fills_for_coin_uses_loaded_artifact(tmp_path):
    workspace = load_backtest_artifact_workspace(_write_artifact(tmp_path))

    fig = workspace["plot_fills_for_coin"]("BTC", include_high_low=False)

    try:
        assert fig.axes[0].get_title() == "Fills BTC"
    finally:
        from plotting import plt

        plt.close(fig)


def test_forager_coin_fill_plot_uses_datetime_axis():
    timestamps = np.array(
        [
            int(ts.value // 1_000_000)
            for ts in pd.date_range("2026-01-01 00:00:00", periods=5, freq="1min")
        ],
        dtype=np.int64,
    )
    hlcvs = np.zeros((5, 1, 4), dtype=float)
    hlcvs[:, 0, 0] = [101.0, 102.0, 103.0, 104.0, 105.0]
    hlcvs[:, 0, 1] = [99.0, 100.0, 101.0, 102.0, 103.0]
    hlcvs[:, 0, 2] = [100.0, 101.0, 102.0, 103.0, 104.0]
    fills = pd.DataFrame(
        {
            "coin": ["BTC", "BTC"],
            "minute": [1, 3],
            "timestamp": pd.to_datetime(timestamps[[1, 3]], unit="ms"),
            "type": ["long_entry", "long_close"],
            "price": [101.0, 103.0],
            "pprice": [101.0, 0.0],
            "psize": [1.0, 0.0],
        }
    )

    fig = create_forager_coin_figures(["BTC"], fills, hlcvs, timestamps=timestamps)["BTC"]

    try:
        ax = fig.axes[0]
        assert ax.get_xlabel() == "datetime"
        assert np.issubdtype(ax.lines[0].get_xdata().dtype, np.datetime64)
        assert str(ax.lines[0].get_xdata()[0]).startswith("2026-01-01")
    finally:
        from plotting import plt

        plt.close(fig)


def test_load_backtest_artifact_fails_loudly_on_missing_cache_file(tmp_path):
    artifact_dir = _write_artifact(tmp_path)
    dataset_path = artifact_dir / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["hlcvs_file"] = str(tmp_path / "missing.npy")
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="hlcvs_file"):
        load_backtest_artifact(artifact_dir)
