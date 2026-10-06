"""CPU-prepared, immutable metadata and borrowed shared-array references.

The parent owns the shared segments and must keep them immutable and alive until
the service closes. Registration snapshots metadata, not large candle arrays.
No device runtime, simulation or search dependency is imported here.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import json
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
    config_json: str = field(repr=False)
    markets_json: str = field(repr=False)

    def __init__(self, *, config, markets, hlcvs, btc, timestamps, candle_coins, exchange, metrics,
                 time_range=None, coin_indices=None, timestamp_range=None):
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
        values = dict(hlcvs=hlcvs, btc=btc, timestamps=timestamps, exchange=exchange,
                      metrics=metrics, time_range=span, timestamp_range=timestamp_span,
                      coin_indices=indices, candle_coins=candle_coins,
                      config_json=json.dumps(config, allow_nan=False),
                      markets_json=json.dumps(markets, allow_nan=False))
        for name, value in values.items():
            object.__setattr__(self, name, value)

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
