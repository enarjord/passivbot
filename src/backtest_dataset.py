import json
from pathlib import Path
from typing import Sequence

from backtest_universe import effective_backtest_approved_coins_by_side
from config.access import get_optional_config_value
from config.shared_bot import get_grouped_bot_value
from hlcvs_manifest import manifest_has_required_schema

HLCVS_CACHE_DIR_SEP = "__"


def _resolve_cache_artifact_path(cache_dir: Path, filename_candidates: Sequence[str]) -> str | None:
    for filename in filename_candidates:
        candidate = cache_dir / filename
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _extract_cache_hash_from_dir(cache_dir: Path | None) -> str | None:
    if cache_dir is None:
        return None
    name = cache_dir.name
    if HLCVS_CACHE_DIR_SEP not in name:
        return name
    return name.rsplit(HLCVS_CACHE_DIR_SEP, 1)[-1] or name


def _runtime_side_membership(config: dict, coins: list[str]) -> dict[str, list[str]] | None:
    bot = config.get("bot")
    approved = config.get("live", {}).get("approved_coins")
    if not isinstance(bot, dict) or not isinstance(approved, dict):
        return None
    for pside in ("long", "short"):
        side = bot.get(pside)
        if not isinstance(side, dict):
            return None
        for field in ("n_positions", "total_wallet_exposure_limit"):
            if get_grouped_bot_value(side, field, default=None, prefer_flat=True) is None:
                return None
    coin_set = set(coins)
    return {
        pside: sorted(coin for coin in side_coins if coin in coin_set)
        for pside, side_coins in effective_backtest_approved_coins_by_side(config).items()
    }


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
    manifest_file = None
    manifest_schema_version = None
    materialization_schema_version = None
    content_hashes = {}
    preparation = {}
    cache_build_side_membership = None
    manifest_missing = None
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
        manifest_file = _resolve_cache_artifact_path(cache_dir, ("manifest.json",))
        manifest_missing = manifest_file is None
        if manifest_file:
            with open(manifest_file) as f:
                manifest = json.load(f)
            if not isinstance(manifest, dict):
                raise TypeError(f"cache manifest must contain an object, got {type(manifest)}")
            manifest_schema_version = manifest.get("schema_version")
            materialization_schema_version = manifest.get("materialization_schema_version")
            if manifest_has_required_schema(manifest):
                files = manifest.get("files", {})
                if isinstance(files, dict):
                    content_hashes = {
                        name: entry.get("sha256")
                        for name, entry in files.items()
                        if isinstance(entry, dict) and entry.get("sha256") is not None
                    }
                effective = manifest.get("effective", {})
                if isinstance(effective, dict):
                    cache_build_side_membership = effective.get("build_side_membership")
                    if not isinstance(cache_build_side_membership, dict):
                        cache_build_side_membership = effective.get("side_membership")
                manifest_preparation = manifest.get("preparation", {})
                if isinstance(manifest_preparation, dict):
                    preparation = manifest_preparation
        if coins_file:
            with open(coins_file) as f:
                loaded_coins = json.load(f)
            if not isinstance(loaded_coins, list) or not all(
                isinstance(coin, str) for coin in loaded_coins
            ):
                raise TypeError(f"cache coins file must contain a list[str], got {type(loaded_coins)}")
            coins_order = loaded_coins
    runtime_side_membership = _runtime_side_membership(config, coins_order)

    return {
        "exchange": exchange,
        "dataset_override": bool(get_optional_config_value(config, "backtest.hlcvs_data_dir")),
        "dataset_override_mode": get_optional_config_value(
            config, "backtest.hlcvs_data_override_mode"
        ),
        "dataset_override_meta": get_optional_config_value(
            config, "_hlcvs_dataset_override_meta", {}
        ),
        "hlcv_cache_dir": cache_dir_str,
        "cache_hash": _extract_cache_hash_from_dir(cache_dir),
        "cache_dir_label": cache_dir.name if cache_dir else None,
        "hlcvs_file": hlcvs_file,
        "timestamps_file": timestamps_file,
        "btc_usd_prices_file": btc_usd_prices_file,
        "coins_file": coins_file,
        "market_specific_settings_file": market_specific_settings_file,
        "cache_meta_file": cache_meta_file,
        "manifest_file": manifest_file,
        "manifest_missing": manifest_missing,
        "manifest_schema_version": manifest_schema_version,
        "materialization_schema_version": materialization_schema_version,
        "content_hashes": content_hashes,
        "preparation": preparation,
        "side_membership": (
            runtime_side_membership
            if runtime_side_membership is not None
            else cache_build_side_membership
        ),
        "cache_build_side_membership": cache_build_side_membership,
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
