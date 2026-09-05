"""Content identity of the arrays and market inputs actually prepared for evaluation."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from shared_arrays import attach_shared_array


PREPARED_DATASET_KEY = "_optimizer_prepared_dataset_identity"
_HASH_BUFFER_BYTES = 1024 * 1024

# These are prepared market/bundle fields consumed by backtest payload construction,
# including exchange attribution used for coin overrides. Raw exchange `info`, cache
# paths and preparation diagnostics are not simulation inputs.
_MARKET_FIELDS = (
    "exchange", "ohlcv_source", "symbol", "coin", "base", "quote", "settle", "inverse",
    "qty_step", "price_step", "min_qty", "min_cost", "c_mult",
    "maker", "maker_fee", "taker", "taker_fee",
    "first_valid_index", "last_valid_index", "warmup_minutes", "trade_start_index",
)
_BUNDLE_FIELDS = (
    "data_interval_minutes", "candle_interval_offset_bars",
    "requested_start_ts", "requested_end_ts", "effective_requested_start_ts",
    "effective_start_ts", "effective_end_ts", "warmup_minutes_requested",
    "warmup_minutes_provided", "btc_source_exchange", "effective_side_membership",
)


def _array_identity(array):
    """Hash C-order values with a bounded buffer, including noncontiguous views."""
    if array is None:
        return None
    array = np.asarray(array)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"Unsupported prepared array dtype: {array.dtype}")
    dtype = array.dtype.newbyteorder("<")
    digest = hashlib.sha256()
    with np.nditer(
        array, flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"], op_dtypes=[dtype], order="C",
        buffersize=max(1, _HASH_BUFFER_BYTES // dtype.itemsize),
    ) as chunks:
        for chunk in chunks:
            digest.update(memoryview(np.ascontiguousarray(chunk)).cast("B"))
    return {"shape": list(array.shape), "dtype": dtype.str, "sha256": digest.hexdigest()}


def _metadata_identity(mss, coins):
    payload = {
        "markets": [
            {key: mss[coin][key] for key in _MARKET_FIELDS if key in mss[coin]}
            for coin in coins
        ],
        "bundle": {
            key: mss.get("__meta__", {})[key]
            for key in _BUNDLE_FIELDS if key in mss.get("__meta__", {})
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        default=lambda value: value.item() if isinstance(value, np.generic) else _unsupported(value),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _unsupported(value):
    raise TypeError(f"Unsupported prepared metadata type: {type(value).__name__}")


def build_prepared_dataset_identity(
    *, config, hlcvs_specs, btc_usd_specs, msss, timestamps, scenario_contexts=()
):
    """Fingerprint every standalone/suite evaluation slice once before resume checks.

    Read prepared shared memory, not cache manifests: a valid manifest need not prove
    the bytes or market settings currently loaded. Shared-memory names are only local
    memoization keys and never become part of the persisted identity. Hash coin views
    separately to avoid allocating a scenario-sized fancy-indexing copy.
    """
    attachments = {}
    memo = {}

    def array_hash(spec, time_slice=None, column=None):
        if spec is None:
            return None
        key = (spec, None if time_slice is None else tuple(time_slice), column)
        if key not in memo:
            if spec not in attachments:
                attachments[spec] = attach_shared_array(spec)
            values = attachments[spec].array
            if time_slice is not None:
                start, end = time_slice
                if not 0 <= start <= end <= len(values):
                    raise ValueError("Prepared scenario time slice is outside its dataset")
                values = values[start:end]
            if column is not None:
                if values.ndim != 3 or not 0 <= column < values.shape[1]:
                    raise ValueError("Prepared coin column is outside its candle dataset")
                values = values[:, column, :]
            memo[key] = _array_identity(values)
        return memo[key]

    def exchange_identity(ex, run_config, hlcv_spec, btc_spec, mss, ts, time_slice, indices):
        if hlcv_spec is None:
            raise ValueError(f"Missing prepared candles for optimizer exchange {ex}")
        coins = list(run_config["backtest"]["coins"][ex])
        indices = list(range(len(coins))) if indices is None else list(indices)
        if len(coins) != len(indices) or not coins:
            raise ValueError(f"Prepared coin mapping is incomplete for optimizer exchange {ex}")
        return {
            "coins": coins,
            "hlcvs": [array_hash(hlcv_spec, time_slice, int(index)) for index in indices],
            "btc_usd": array_hash(btc_spec, time_slice),
            "timestamps": _array_identity(ts),
            "market_settings_sha256": _metadata_identity(mss, coins),
        }

    try:
        scenarios = []
        if scenario_contexts:
            for ctx in scenario_contexts:
                exchanges = {}
                for ex in sorted(ctx.exchanges):
                    lazy = (ctx.master_hlcvs_specs or {}).get(ex) is not None
                    exchanges[ex] = exchange_identity(
                        ex, ctx.config,
                        ctx.master_hlcvs_specs[ex] if lazy else ctx.hlcvs_specs[ex],
                        (ctx.master_btc_specs or {}).get(ex) if lazy else ctx.btc_usd_specs.get(ex),
                        ctx.msss[ex], ctx.timestamps.get(ex),
                        (ctx.time_slice or {}).get(ex) if lazy else None,
                        (ctx.coin_slice_indices or {}).get(ex) if lazy else ctx.coin_indices.get(ex),
                    )
                scenarios.append({"label": ctx.label, "exchanges": exchanges})
        else:
            exchanges = {
                ex: exchange_identity(
                    ex, config, hlcvs_specs[ex], btc_usd_specs.get(ex),
                    msss[ex], timestamps.get(ex), None, None,
                )
                for ex in sorted(hlcvs_specs)
            }
            if not exchanges:
                raise ValueError("Missing prepared optimizer datasets")
            scenarios.append({"exchanges": exchanges})
        return {"version": 1, "scenarios": scenarios}
    finally:
        for attachment in attachments.values():
            attachment.close()
