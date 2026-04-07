import json
from pathlib import Path

from backtest_dataset import build_backtest_dataset_metadata, dump_backtest_dataset_metadata


def test_build_backtest_dataset_metadata_prefers_cache_coin_order_and_absolute_paths(tmp_path):
    cache_dir = tmp_path / "caches" / "hlcvs_data" / "abc123def4567890"
    cache_dir.mkdir(parents=True)
    (cache_dir / "hlcvs.npy.gz").write_bytes(b"hlcvs")
    (cache_dir / "timestamps.npy.gz").write_bytes(b"timestamps")
    (cache_dir / "btc_usd_prices.npy.gz").write_bytes(b"btc")
    (cache_dir / "cache_meta.json").write_text("{}")
    (cache_dir / "market_specific_settings.json").write_text("{}")
    (cache_dir / "coins.json").write_text(json.dumps(["ETH", "BTC", "XMR"]))

    config = {
        "backtest": {
            "cache_dir": {"combined": str(cache_dir)},
            "coins": {"combined": ["BTC", "ETH"]},
            "start_date": "2025-08-01",
            "end_date": "now",
            "candle_interval_minutes": 1,
        }
    }

    metadata = build_backtest_dataset_metadata(config, "combined")

    assert metadata["hlcv_cache_dir"] == str(cache_dir.resolve())
    assert metadata["cache_hash"] == cache_dir.name
    assert metadata["coins"] == ["ETH", "BTC", "XMR"]
    assert metadata["coin_index"] == {"ETH": 0, "BTC": 1, "XMR": 2}
    assert metadata["hlcvs_file"] == str((cache_dir / "hlcvs.npy.gz").resolve())
    assert metadata["timestamps_file"] == str((cache_dir / "timestamps.npy.gz").resolve())
    assert metadata["btc_usd_prices_file"] == str((cache_dir / "btc_usd_prices.npy.gz").resolve())
    assert metadata["coins_file"] == str((cache_dir / "coins.json").resolve())
    assert metadata["market_specific_settings_file"] == str(
        (cache_dir / "market_specific_settings.json").resolve()
    )


def test_dump_backtest_dataset_metadata_writes_dataset_json(tmp_path):
    cache_dir = tmp_path / "caches" / "hlcvs_data" / "fff111aaa222bbb3"
    cache_dir.mkdir(parents=True)
    (cache_dir / "coins.json").write_text(json.dumps(["BTC"]))
    (cache_dir / "market_specific_settings.json").write_text("{}")
    (cache_dir / "hlcvs.npy").write_bytes(b"x")
    (cache_dir / "timestamps.npy").write_bytes(b"y")
    (cache_dir / "btc_usd_prices.npy").write_bytes(b"z")

    config = {
        "backtest": {
            "cache_dir": {"binanceusdm": str(cache_dir)},
            "coins": {"binanceusdm": ["BTC"]},
            "start_date": "2025-01-01",
            "end_date": "2025-02-01",
        }
    }
    results_path = tmp_path / "results"
    results_path.mkdir()

    written = dump_backtest_dataset_metadata(config, "binanceusdm", str(results_path))

    payload = json.loads(Path(written).read_text())
    assert payload["exchange"] == "binanceusdm"
    assert payload["coins"] == ["BTC"]
    assert payload["hlcvs_file"] == str((cache_dir / "hlcvs.npy").resolve())
    assert Path(written) == results_path / "dataset.json"
