import json
from pathlib import Path
from typing import Sequence

from config.access import get_optional_config_value


def _resolve_cache_artifact_path(cache_dir: Path, filename_candidates: Sequence[str]) -> str | None:
    for filename in filename_candidates:
        candidate = cache_dir / filename
        if candidate.exists():
            return str(candidate.resolve())
    return None


def build_backtest_dataset_metadata(config: dict, exchange: str) -> dict:
    cache_dir_raw = get_optional_config_value(config, f"backtest.cache_dir.{exchange}")
    cache_dir = Path(cache_dir_raw).resolve() if cache_dir_raw else None
    coins_from_config = list(get_optional_config_value(config, f"backtest.coins.{exchange}", []) or [])

    cache_dir_str = str(cache_dir) if cache_dir else None
    coins_file = None
    market_specific_settings_file = None
    cache_meta_file = None
    hlcvs_file = None
    timestamps_file = None
    btc_usd_prices_file = None
    coins_order = list(coins_from_config)

    if cache_dir and cache_dir.exists():
        coins_file = _resolve_cache_artifact_path(cache_dir, ("coins.json",))
        market_specific_settings_file = _resolve_cache_artifact_path(
            cache_dir, ("market_specific_settings.json",)
        )
        cache_meta_file = _resolve_cache_artifact_path(cache_dir, ("cache_meta.json",))
        hlcvs_file = _resolve_cache_artifact_path(cache_dir, ("hlcvs.npy.gz", "hlcvs.npy"))
        timestamps_file = _resolve_cache_artifact_path(
            cache_dir, ("timestamps.npy.gz", "timestamps.npy")
        )
        btc_usd_prices_file = _resolve_cache_artifact_path(
            cache_dir, ("btc_usd_prices.npy.gz", "btc_usd_prices.npy")
        )
        if coins_file:
            with open(coins_file) as f:
                loaded_coins = json.load(f)
            if not isinstance(loaded_coins, list) or not all(
                isinstance(coin, str) for coin in loaded_coins
            ):
                raise TypeError(f"cache coins file must contain a list[str], got {type(loaded_coins)}")
            coins_order = loaded_coins

    return {
        "exchange": exchange,
        "hlcv_cache_dir": cache_dir_str,
        "cache_hash": cache_dir.name if cache_dir else None,
        "hlcvs_file": hlcvs_file,
        "timestamps_file": timestamps_file,
        "btc_usd_prices_file": btc_usd_prices_file,
        "coins_file": coins_file,
        "market_specific_settings_file": market_specific_settings_file,
        "cache_meta_file": cache_meta_file,
        "coins": coins_order,
        "coin_index": {coin: idx for idx, coin in enumerate(coins_order)},
        "requested_start_date": get_optional_config_value(config, "backtest.start_date"),
        "requested_end_date": get_optional_config_value(config, "backtest.end_date"),
        "candle_interval_minutes": get_optional_config_value(
            config, "backtest.candle_interval_minutes", 1
        ),
    }


def dump_backtest_dataset_metadata(config: dict, exchange: str, results_path: str) -> str:
    dataset_metadata = build_backtest_dataset_metadata(config, exchange)
    out_path = Path(results_path) / "dataset.json"
    with open(out_path, "w") as f:
        json.dump(dataset_metadata, f, indent=4, sort_keys=True)
    return str(out_path)
