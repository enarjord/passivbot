"""CPU-prepared, immutable metadata and borrowed shared-array references.

The parent owns the shared segments and must keep them immutable and alive until
the service closes. Registration snapshots metadata, not large candle arrays.
No device runtime, simulation or search dependency is imported here.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import json
from config.universe import (
    PreparedUniverseSnapshot, universe_source_view, runtime_universe_snapshot,
    materialize_universe_snapshot, fixed_runtime_universe_view,
)
import logging

import numpy as np

from shared_arrays import SharedArraySpec, attach_shared_array


@contextmanager
def _borrow_array(spec):
    attachment = attach_shared_array(spec)
    try:
        yield attachment.array
    except BaseException:
        try:
            attachment.close()
        except BaseException:
            logging.exception("GPU dataset attachment cleanup failed after an earlier failure")
        raise
    else:
        attachment.close()


@dataclass(frozen=True, init=False)
class PreparedGpuDataset:
    hlcvs: SharedArraySpec
    btc: SharedArraySpec
    timestamps: SharedArraySpec
    exchange: str
    metrics: tuple[str, ...]
    time_range: tuple[int, int]
    timestamp_range: tuple[int, int]
    coin_indices: tuple[int, ...]
    candle_coins: tuple[str, ...]
    eligibility: SharedArraySpec | None
    eligibility_rows: SharedArraySpec | None
    config_json: str = field(repr=False)
    markets_json: str = field(repr=False)
    universe_snapshot: PreparedUniverseSnapshot = field(repr=False)

    def __init__(self, *, config, markets, hlcvs, btc, timestamps, candle_coins, exchange, metrics,
                 time_range=None, coin_indices=None, timestamp_range=None,
                 eligibility=None, eligibility_rows=None):
        specs = []
        for name, spec in (("hlcvs", hlcvs), ("btc", btc), ("timestamps", timestamps)):
            if not isinstance(spec, SharedArraySpec) or not spec.name:
                raise TypeError(f"{name} must be a named shared-array reference")
            shape = tuple(spec.shape)
            if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape):
                raise ValueError(f"invalid {name} shared-array shape")
            dtype = np.dtype(spec.dtype)
            if dtype.kind not in "iuf":
                raise ValueError(f"{name} must contain numeric values")
            specs.append(SharedArraySpec(spec.name, shape, dtype.str))
        hlcvs, btc, timestamps = specs
        if len(hlcvs.shape) != 3 or hlcvs.shape[2] != 4:
            raise ValueError("hlcvs must have shape (bars, coins, 4)")
        if (btc.shape != (hlcvs.shape[0],) or len(timestamps.shape) != 1
                or (timestamp_range is None and timestamps.shape != btc.shape)):
            raise ValueError("BTC prices and timestamps must match candle rows")
        if isinstance(candle_coins, str):
            raise ValueError("candle coins must be a collection of column identities")
        candle_coins = tuple(candle_coins)
        if (len(candle_coins) != hlcvs.shape[1]
                or any(not isinstance(coin, str) or not coin for coin in candle_coins)
                or len(set(candle_coins)) != len(candle_coins)):
            raise ValueError("candle coins must identify every source column uniquely")
        if not isinstance(exchange, str) or not exchange:
            raise ValueError("exchange must be a nonempty string")
        indices = tuple(range(hlcvs.shape[1])) if coin_indices is None else tuple(coin_indices)
        if (not 1 <= len(indices) <= 64 or len(set(indices)) != len(indices)
                or any(isinstance(i, bool) or not isinstance(i, int)
                       or not 0 <= i < hlcvs.shape[1] for i in indices)):
            raise ValueError("coin indices must identify 1..64 distinct candle columns")
        span = (0, hlcvs.shape[0]) if time_range is None else tuple(time_range)
        if (len(span) != 2 or any(isinstance(n, bool) or not isinstance(n, int) for n in span)
                or not 0 <= span[0] < span[1] <= hlcvs.shape[0] or span[1] - span[0] < 3):
            raise ValueError("time range must identify at least three available candle rows")
        timestamp_span = span if timestamp_range is None else tuple(timestamp_range)
        if (len(timestamp_span) != 2
                or any(isinstance(n, bool) or not isinstance(n, int) for n in timestamp_span)
                or not 0 <= timestamp_span[0] < timestamp_span[1] <= timestamps.shape[0]
                or timestamp_span[1] - timestamp_span[0] != span[1] - span[0]):
            raise ValueError("timestamp range must match the selected candle rows")
        coins = config["backtest"]["coins"][exchange]
        if len(coins) != len(indices) or not coins or coins != sorted(set(coins)):
            raise ValueError("config coins must be unique, sorted and match selected candle columns")
        if [candle_coins[index] for index in indices] != coins:
            raise ValueError("selected candle columns must match config coin identities and order")
        if any(coin not in markets for coin in coins):
            raise ValueError("market settings are required for every selected coin")
        if isinstance(metrics, str):
            raise ValueError("requested metrics must be a collection of names")
        metrics = tuple(dict.fromkeys(metrics))
        if not metrics or any(not isinstance(name, str) or not name for name in metrics):
            raise ValueError("requested metrics must be nonempty strings")
        if (eligibility is None) != (eligibility_rows is None):
            raise ValueError("Organillo eligibility and row mapping must be supplied together")
        if config["backtest"].get("organillo_csv") is not None and eligibility is None:
            raise ValueError("Organillo requires prepared eligibility buffers")
        if eligibility is not None:
            if config["backtest"].get("organillo_csv") is None:
                raise ValueError("Organillo prepared buffers require backtest.organillo_csv")
            digest = config["backtest"].get("organillo_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("Organillo requires a validated SHA-256 for prepared GPU eligibility")
            if (not isinstance(eligibility, SharedArraySpec) or not isinstance(eligibility_rows, SharedArraySpec)
                    or np.dtype(eligibility.dtype) != np.uint8 or len(eligibility.shape) != 2
                    or eligibility.shape[1] != len(indices) or eligibility.shape[0] < 1
                    or np.dtype(eligibility_rows.dtype) != np.int32
                    or eligibility_rows.shape != (timestamp_span[1] - timestamp_span[0],)):
                raise ValueError("Organillo buffers must match selected coin columns and candle rows")
        universe_snapshot = runtime_universe_snapshot(config, available_coins=coins)
        values = dict(universe_snapshot=universe_snapshot, hlcvs=hlcvs, btc=btc, timestamps=timestamps, exchange=exchange,
                      metrics=metrics, time_range=span, timestamp_range=timestamp_span,
                      coin_indices=indices, candle_coins=candle_coins,
                       eligibility=eligibility, eligibility_rows=eligibility_rows,
                      config_json=json.dumps(universe_source_view(config), allow_nan=False),
                      markets_json=json.dumps(markets, allow_nan=False))
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def policy_config(self):
        """Source authority plus its fixed prepared membership for planning."""
        return materialize_universe_snapshot(json.loads(self.config_json), self.universe_snapshot)

    def execution_config(self):
        """Ordinary JSON-native, fixed execution membership for the native worker."""
        return fixed_runtime_universe_view(json.loads(self.config_json), snapshot=self.universe_snapshot)

    @contextmanager
    def attach(self):
        """Borrow read-only views; close every attachment on failure and shutdown."""
        with ExitStack() as resources:
            arrays = []
            for spec, span in ((self.hlcvs, self.time_range), (self.btc, self.time_range),
                               (self.timestamps, self.timestamp_range)):
                array = resources.enter_context(_borrow_array(spec))
                array.flags.writeable = False
                arrays.append(array[slice(*span)])
            yield arrays

    @contextmanager
    def attach_selection(self):
        if self.eligibility is None:
            yield None
            return
        from organillo import Selection, validate_buffers
        with ExitStack() as resources:
            values = resources.enter_context(_borrow_array(self.eligibility))
            rows = resources.enter_context(_borrow_array(self.eligibility_rows))
            validate_buffers(values, rows, self.time_range[1] - self.time_range[0], len(self.coin_indices))
            values.flags.writeable = rows.flags.writeable = False
            yield Selection(values, rows, json.loads(self.config_json)["backtest"].get("organillo_sha256"))

    def with_config(self, config):
        """Prepare another execution view over the same immutable source arrays."""
        updated = config["backtest"]
        if self.eligibility is not None:
            original = json.loads(self.config_json)["backtest"]
            if (updated.get("organillo_csv") is None
                    or original.get("organillo_sha256") != updated.get("organillo_sha256")):
                raise ValueError("changing an Organillo schedule requires a newly prepared dataset")
        elif updated.get("organillo_csv") is not None:
            raise ValueError("changing an Organillo schedule requires a newly prepared dataset")
        return PreparedGpuDataset(
            config=config, markets=json.loads(self.markets_json), exchange=self.exchange,
            hlcvs=self.hlcvs, btc=self.btc, timestamps=self.timestamps,
            candle_coins=self.candle_coins, coin_indices=self.coin_indices,
            time_range=self.time_range, timestamp_range=self.timestamp_range, metrics=self.metrics,
            eligibility=self.eligibility, eligibility_rows=self.eligibility_rows,
        )
