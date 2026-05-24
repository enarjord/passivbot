from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np

from backtest_universe import (
    POSITION_SIDES,
    effective_backtest_approved_coins_by_side,
    effective_backtest_data_coins,
    normalize_backtest_coin,
)
from config.access import get_optional_config_value, require_config_value
from hlcvs_manifest import (
    HlcvsManifestError,
    load_hlcvs_manifest,
    load_numpy_artifact,
    manifest_has_required_schema,
    verify_hlcvs_manifest,
)
from utils import date_to_ts, format_end_date, ts_to_date
from warmup_utils import compute_backtest_warmup_minutes


def _hlcvs_cache_artifact_path(
    cache_dir: Path, manifest, name: str, candidates: tuple[str, ...]
) -> Path | None:
    if manifest_has_required_schema(manifest):
        files = manifest.get("files", {})
        entry = files.get(name) if isinstance(files, dict) else None
        if not isinstance(entry, dict):
            raise HlcvsManifestError(f"HLCV manifest missing required file entry {name!r}")
        rel_path = entry.get("path") if isinstance(entry, dict) else None
        if not rel_path:
            raise HlcvsManifestError(f"HLCV manifest file entry {name!r} is missing path")
        path = cache_dir / str(rel_path)
        if not path.exists():
            raise HlcvsManifestError(f"HLCV manifest file is missing: {path}")
        return path
    for filename in candidates:
        path = cache_dir / filename
        if path.exists():
            return path
    return None


def _load_hlcvs_cache_arrays(cache_dir: Path, manifest):
    coins_path = cache_dir / "coins.json"
    mss_path = cache_dir / "market_specific_settings.json"
    if not coins_path.exists() or not mss_path.exists():
        raise FileNotFoundError(f"HLCV dataset missing coins or market settings in {cache_dir}")
    coins = json.load(open(coins_path))
    mss = json.load(open(mss_path))
    hlcvs_path = _hlcvs_cache_artifact_path(
        cache_dir, manifest, "hlcvs", ("hlcvs.npy.gz", "hlcvs.npy")
    )
    timestamps_path = _hlcvs_cache_artifact_path(
        cache_dir, manifest, "timestamps", ("timestamps.npy.gz", "timestamps.npy")
    )
    btc_path = _hlcvs_cache_artifact_path(
        cache_dir,
        manifest,
        "btc_usd_prices",
        ("btc_usd_prices.npy.gz", "btc_usd_prices.npy"),
    )
    if hlcvs_path is None or btc_path is None:
        raise FileNotFoundError(f"HLCV dataset missing required arrays in {cache_dir}")
    if timestamps_path is None:
        raise FileNotFoundError(
            f"HLCV dataset missing timestamps.npy/timestamps.npy.gz in {cache_dir}"
        )
    hlcvs = load_numpy_artifact(hlcvs_path)
    btc_usd_prices = load_numpy_artifact(btc_path)
    timestamps = load_numpy_artifact(timestamps_path)
    return coins, hlcvs, mss, btc_usd_prices, timestamps


def _side_membership_for_override(config: dict, dataset_coins: list[str], manifest, mode: str) -> dict:
    input_sides = effective_backtest_approved_coins_by_side(config)
    dataset_coin_set = set(dataset_coins)
    if mode == "intersection":
        return {
            pside: sorted([coin for coin in side_coins if coin in dataset_coin_set])
            for pside, side_coins in input_sides.items()
        }

    manifest_sides = None
    if manifest_has_required_schema(manifest):
        effective = manifest.get("effective", {})
        if isinstance(effective, dict):
            manifest_sides = effective.get("side_membership")
    if isinstance(manifest_sides, dict):
        return {
            pside: sorted([normalize_backtest_coin(coin) for coin in manifest_sides.get(pside, [])])
            for pside in POSITION_SIDES
        }

    input_coin_set = set().union(*(set(side_coins) for side_coins in input_sides.values()))
    dataset_only = sorted(set(dataset_coins) - input_coin_set)
    if dataset_only:
        raise ValueError(
            "dataset override mode 'dataset' requires manifest side_membership for dataset-only "
            f"coins: {dataset_only}"
        )
    return {
        pside: sorted([coin for coin in side_coins if coin in dataset_coin_set])
        for pside, side_coins in input_sides.items()
    }


def load_hlcvs_data_override(config, exchange):
    override_dir = get_optional_config_value(config, "backtest.hlcvs_data_dir")
    if not override_dir:
        return None
    mode = str(
        get_optional_config_value(config, "backtest.hlcvs_data_override_mode", "intersection")
        or "intersection"
    )
    if mode not in {"intersection", "dataset"}:
        raise ValueError("backtest.hlcvs_data_override_mode must be 'intersection' or 'dataset'")
    cache_dir = Path(override_dir).expanduser().resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"HLCV dataset override directory does not exist: {cache_dir}")
    cache_permissive = bool(
        get_optional_config_value(config, "backtest.hlcvs_cache_permissive", False)
    )
    manifest = load_hlcvs_manifest(cache_dir)
    manifest_missing = manifest is None
    if manifest is None:
        if not cache_permissive:
            raise HlcvsManifestError(
                f"HLCV dataset override {cache_dir} is missing manifest.json; set "
                "backtest.hlcvs_cache_permissive=true only for legacy compatibility"
            )
        logging.warning("[hlcvs] override dataset %s missing manifest; permissive load", cache_dir)
    elif manifest_has_required_schema(manifest):
        verify_hlcvs_manifest(cache_dir, manifest)
    elif not cache_permissive:
        raise HlcvsManifestError(f"HLCV dataset override {cache_dir} has unsupported manifest schema")
    else:
        logging.warning(
            "[hlcvs] override dataset %s has unsupported manifest schema; permissive load",
            cache_dir,
        )

    dataset_coins, hlcvs, mss, btc_usd_prices, timestamps = _load_hlcvs_cache_arrays(
        cache_dir, manifest
    )
    dataset_coins = [normalize_backtest_coin(coin) for coin in dataset_coins]
    requested_coins = effective_backtest_data_coins(config)
    if mode == "intersection":
        selected_coins = [coin for coin in dataset_coins if coin in set(requested_coins)]
    else:
        selected_coins = list(dataset_coins)
    if not selected_coins:
        raise ValueError("HLCV dataset override produced an empty coin set")

    dataset_start_ts = int(timestamps[0])
    dataset_end_ts = int(timestamps[-1])
    requested_start_ts = int(date_to_ts(require_config_value(config, "backtest.start_date")))
    requested_end_ts = int(date_to_ts(format_end_date(require_config_value(config, "backtest.end_date"))))
    warmup_minutes = compute_backtest_warmup_minutes(config)
    requested_data_start_ts = max(0, requested_start_ts - int(warmup_minutes) * 60_000)
    if mode == "intersection":
        effective_start_ts = max(requested_data_start_ts, dataset_start_ts)
        effective_end_ts = min(requested_end_ts, dataset_end_ts)
        if effective_end_ts < effective_start_ts:
            raise ValueError(
                "HLCV dataset override has no date overlap with requested backtest range"
            )
        effective_config_start_ts = max(requested_start_ts, dataset_start_ts)
    else:
        effective_start_ts = dataset_start_ts
        effective_end_ts = dataset_end_ts
        effective_config_start_ts = dataset_start_ts

    row_mask = (timestamps >= effective_start_ts) & (timestamps <= effective_end_ts)
    if not row_mask.any():
        raise ValueError("HLCV dataset override selected no timestamp rows")
    row_indices = np.flatnonzero(row_mask)
    row_start = int(row_indices[0])
    row_end = int(row_indices[-1]) + 1
    coin_positions = [dataset_coins.index(coin) for coin in selected_coins]
    hlcvs = np.ascontiguousarray(hlcvs[row_start:row_end][:, coin_positions, :])
    btc_usd_prices = np.ascontiguousarray(btc_usd_prices[row_start:row_end])
    timestamps = np.ascontiguousarray(timestamps[row_start:row_end])

    selected_mss = {coin: deepcopy(mss.get(coin, {})) for coin in selected_coins}
    for meta in selected_mss.values():
        for key in ("first_valid_index", "last_valid_index", "trade_start_index"):
            meta.pop(key, None)
    side_membership = _side_membership_for_override(config, selected_coins, manifest, mode)
    side_membership = {
        pside: sorted([coin for coin in side_membership.get(pside, []) if coin in selected_coins])
        for pside in POSITION_SIDES
    }
    if not set().union(*(set(side_membership[pside]) for pside in POSITION_SIDES)):
        raise ValueError("HLCV dataset override produced no side-eligible coins")

    original_approved = deepcopy(config.get("live", {}).get("approved_coins", {}))
    config.setdefault("live", {})["approved_coins"] = side_membership
    config.setdefault("backtest", {})["start_date"] = ts_to_date(int(effective_config_start_ts))
    config["backtest"]["end_date"] = ts_to_date(int(timestamps[-1]))
    config["backtest"].setdefault("cache_dir", {})[exchange] = str(cache_dir)
    config["backtest"].setdefault("coins", {})[exchange] = selected_coins
    selected_mss["__meta__"] = {
        "dataset_override": True,
        "dataset_override_mode": mode,
        "manifest_missing": manifest_missing,
        "requested_coins": requested_coins,
        "dataset_coins": dataset_coins,
        "effective_backtested_coins": selected_coins,
        "original_requested_start_ts": requested_start_ts,
        "requested_start_ts": requested_start_ts,
        "effective_requested_start_ts": int(effective_config_start_ts),
        "requested_end_ts": requested_end_ts,
        "requested_data_start_ts": requested_data_start_ts,
        "warmup_minutes": int(warmup_minutes),
        "dataset_start_ts": dataset_start_ts,
        "dataset_end_ts": dataset_end_ts,
        "effective_start_ts": int(effective_config_start_ts),
        "effective_data_start_ts": int(timestamps[0]),
        "effective_end_ts": int(timestamps[-1]),
        "dropped_requested_coins": sorted(set(requested_coins) - set(selected_coins)),
        "added_dataset_only_coins": sorted(set(selected_coins) - set(requested_coins)),
        "input_side_membership": original_approved,
        "effective_side_membership": side_membership,
    }
    config["_hlcvs_dataset_override_meta"] = deepcopy(selected_mss["__meta__"])
    logging.info(
        "[hlcvs] override %s mode=%s coins=%s range=%s -> %s",
        cache_dir,
        mode,
        ",".join(selected_coins),
        ts_to_date(int(timestamps[0])),
        ts_to_date(int(timestamps[-1])),
    )
    results_path = str(Path(require_config_value(config, "backtest.base_dir")) / str(exchange)) + "/"
    return cache_dir, selected_coins, hlcvs, selected_mss, results_path, btc_usd_prices, timestamps
