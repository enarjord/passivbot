"""Versioned, typed inputs to candle-content cache identity.

This is not an optimizer evaluation identity or a migration of saved fitness.
"""
from __future__ import annotations

import math

from config.access import require_config_value


CACHE_REQUEST_SCHEMA_VERSION = 2


def gap_tolerance_minutes(config: dict) -> float:
    """Use one representation for the declared numeric minute quantity."""
    value = require_config_value(config, "backtest.gap_tolerance_ohlcvs_minutes")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("backtest.gap_tolerance_ohlcvs_minutes must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("backtest.gap_tolerance_ohlcvs_minutes must be a finite nonnegative number")
    return 0.0 if value == 0 else value
