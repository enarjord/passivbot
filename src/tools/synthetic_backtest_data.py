"""Deterministic NumPy-only candle fixtures shared by offline backtest tools."""

import numpy as np


def synthetic_hlcvs(bars: int, coins: int, seed: int):
    rng = np.random.default_rng(seed)
    steps = np.arange(bars, dtype=np.float64)
    values = np.empty((bars, coins, 4), dtype=np.float64)
    for coin in range(coins):
        phase = coin * 0.73
        trend = steps * (0.0004 + coin * 0.00002)
        wave = np.sin(steps / (37.0 + coin * 3.0) + phase) * (1.5 + coin * 0.1)
        noise = rng.normal(0.0, 0.03, size=bars).cumsum()
        close = 100.0 + coin * 15.0 + trend + wave + noise
        spread = 0.15 + np.abs(np.sin(steps / 19.0 + phase)) * 0.2
        values[:, coin, 0] = close + spread
        values[:, coin, 1] = close - spread
        values[:, coin, 2] = close
        values[:, coin, 3] = 100.0 + coin * 10.0
    timestamps = 1_700_000_000_000 + steps.astype(np.int64) * 60_000
    return values, timestamps
