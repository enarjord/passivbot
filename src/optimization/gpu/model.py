from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


GAP_BINS = 128
GAP_MAX_MINUTES = 4_000_000.0
MPS_MULTICOIN_MAX_COINS = 64
HSL_SIGNAL_MODES = {"unified", "pside", "coin"}


def validate_hsl_signal_topology(
    signal_mode: str,
    *,
    coin_count: int,
    enabled_side_count: int,
    shared_account_controller: bool = False,
) -> None:
    """Fail closed unless the selected HSL scope has the state it requires."""

    if signal_mode not in HSL_SIGNAL_MODES:
        raise ValueError(
            "GPU HSL requires live.hsl_signal_mode to be coin, pside, or "
            f"unified; got {signal_mode!r}"
        )
    if coin_count > 1:
        if (
            enabled_side_count > 1
            and signal_mode != "pside"
            and not shared_account_controller
        ):
            raise ValueError(
                "GPU dual-side multi-coin HSL currently supports only pside "
                "signal mode; coin requires a shared-balance denominator and "
                "unified requires one cross-side episode controller"
            )
        return


def validate_single_coin_hsl_signal_topology(
    signal_mode: str, *, enabled_side_count: int
) -> None:
    validate_hsl_signal_topology(
        signal_mode, coin_count=1, enabled_side_count=enabled_side_count
    )


def single_coin_shader_topology(
    *,
    long_enabled: bool,
    short_enabled: bool,
    hsl_enabled: bool,
    hsl_one_side_enabled: bool = False,
) -> str:
    """Select a one-side variant only when compile-time assumptions are exact."""

    if hsl_enabled and not hsl_one_side_enabled:
        return "generic"
    if long_enabled and not short_enabled:
        return "long_hsl" if hsl_enabled else "long_no_hsl"
    if short_enabled and not long_enabled:
        return "short_hsl" if hsl_enabled else "short_no_hsl"
    return "generic"


EMA_ANCHOR_PARAM_KEYS = (
    "base_qty_pct",
    "ema_span_0",
    "ema_span_1",
    "entry_double_down_factor",
    "offset",
    "offset_psize_weight",
    "offset_volatility_1h_weight",
    "offset_volatility_1m_weight",
    "offset_volatility_ema_span_1h",
    "offset_volatility_ema_span_1m",
    "entry_cooldown_minutes",
    "total_wallet_exposure_limit",
)

EXPOSURE_PARAM_KEYS = (
    "we_excess_allowance_pct",
    "we_excess_allowance_legacy_raw",
    "twel_entry_gate_enabled",
    "twel_enforcer_threshold",
)

POSITION_EXPOSURE_ENFORCER_PARAM_KEYS = (
    "wel_enforcer_enabled",
    "wel_enforcer_threshold",
)

TOTAL_EXPOSURE_ENFORCER_PARAM_KEYS = (
    "twel_enforcer_enabled",
)

UNSTUCK_PARAM_KEYS = (
    "unstuck_enabled",
    "unstuck_ema_gating_enabled",
    "unstuck_close_pct",
    "unstuck_ema_dist",
    "unstuck_loss_allowance_pct",
    "unstuck_threshold",
)

HSL_PARAM_KEYS = (
    "hsl_enabled",
    "hsl_red_threshold",
    "hsl_ema_span_minutes",
    "hsl_cooldown_minutes_after_red",
    "hsl_no_restart_drawdown_threshold",
    "hsl_restart_policy",
    "hsl_tier_ratio_yellow",
    "hsl_tier_ratio_orange",
    "hsl_orange_graceful_stop",
    "hsl_signal_mode",
    "hsl_slot_count",
)

HSL_COIN_OVERRIDE_PATHS = (
    ("hsl_enabled", ("enabled",)),
    ("hsl_red_threshold", ("red_threshold",)),
    ("hsl_ema_span_minutes", ("ema_span_minutes",)),
    ("hsl_cooldown_minutes_after_red", ("cooldown_minutes_after_red",)),
    ("hsl_no_restart_drawdown_threshold", ("no_restart_drawdown_threshold",)),
    ("hsl_restart_policy", ("restart_after_red_policy",)),
    ("hsl_tier_ratio_yellow", ("tier_ratios", "yellow")),
    ("hsl_tier_ratio_orange", ("tier_ratios", "orange")),
    ("hsl_orange_graceful_stop", ("orange_tier_mode",)),
    ("hsl_panic_market", ("panic_close_order_type",)),
)

EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS = tuple(EMA_ANCHOR_PARAM_KEYS[:-2])
EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN = len(
    EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS
)
EMA_ANCHOR_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN = (
    EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN + 1
)
EMA_ANCHOR_COIN_OVERRIDE_ALLOWANCE_PCT_COLUMN = (
    EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN + 2
)
EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_START_COLUMN = (
    EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN + 3
)
EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN = (
    EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_START_COLUMN + 6
)
EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN = (
    EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN + len(HSL_COIN_OVERRIDE_PATHS)
)
EMA_ANCHOR_COIN_OVERRIDE_COLS = EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN + 1


def encode_hsl_panic_order_type(value, *, field_name: str) -> float:
    if not isinstance(value, str):
        raise TypeError(f"GPU HSL requires {field_name} to be a string")
    if value not in {"limit", "market"}:
        raise ValueError(
            f"GPU HSL requires {field_name} to be limit or market, got "
            f"{value!r}"
        )
    return float(value == "market")


def validate_hsl_settings(settings: dict, *, field_name: str) -> dict:
    if not isinstance(settings, dict):
        raise TypeError(f"{field_name} must be a dictionary")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError(f"{field_name}.enabled must be a boolean")

    def finite_float(key: str, default: float) -> float:
        try:
            value = float(settings.get(key, default))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{field_name}.{key} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{field_name}.{key} must be finite")
        if abs(value) > float(np.finfo(np.float32).max):
            raise ValueError(
                f"{field_name}.{key} must be representable as float32"
            )
        return value

    red_threshold = finite_float("red_threshold", 0.15)
    ema_span_minutes = finite_float("ema_span_minutes", 720.0)
    cooldown_minutes = finite_float("cooldown_minutes_after_red", 0.0)
    no_restart_threshold = finite_float(
        "no_restart_drawdown_threshold", 1.0
    )
    if not 0.0 < red_threshold <= 1.0:
        raise ValueError(
            f"{field_name}.red_threshold must satisfy 0 < value <= 1"
        )
    if ema_span_minutes < 1.0:
        raise ValueError(f"{field_name}.ema_span_minutes must be >= 1")
    if cooldown_minutes < 0.0:
        raise ValueError(
            f"{field_name}.cooldown_minutes_after_red must be >= 0"
        )
    if not red_threshold <= no_restart_threshold <= 1.0:
        raise ValueError(
            f"{field_name}.no_restart_drawdown_threshold must satisfy "
            "red_threshold <= value <= 1"
        )

    tier_ratios = settings.get(
        "tier_ratios", {"yellow": 0.5, "orange": 0.75}
    )
    if not isinstance(tier_ratios, dict):
        raise TypeError(f"{field_name}.tier_ratios must be a dictionary")
    try:
        yellow = float(tier_ratios.get("yellow", 0.5))
        orange = float(tier_ratios.get("orange", 0.75))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name}.tier_ratios values must be numeric") from exc
    if not (math.isfinite(yellow) and math.isfinite(orange)):
        raise ValueError(f"{field_name}.tier_ratios values must be finite")
    if not 0.0 < yellow < orange < 1.0:
        raise ValueError(
            f"{field_name}.tier_ratios must satisfy 0 < yellow < orange < 1"
        )
    yellow_f32 = float(np.float32(yellow))
    orange_f32 = float(np.float32(orange))
    if not 0.0 < yellow_f32 < orange_f32 < 1.0:
        raise ValueError(
            f"{field_name}.tier_ratios must remain strictly ordered inside "
            "(0, 1) when represented as float32"
        )

    restart_policy = settings.get("restart_after_red_policy", "threshold")
    if not isinstance(restart_policy, str):
        raise TypeError(
            f"{field_name}.restart_after_red_policy must be a string"
        )
    if restart_policy not in {"always", "threshold", "never"}:
        raise ValueError(
            f"{field_name}.restart_after_red_policy must be always, threshold, "
            f"or never, got {restart_policy!r}"
        )
    orange_mode = settings.get(
        "orange_tier_mode", "tp_only_with_active_entry_cancellation"
    )
    if not isinstance(orange_mode, str):
        raise TypeError(f"{field_name}.orange_tier_mode must be a string")
    if orange_mode not in {
        "graceful_stop",
        "tp_only_with_active_entry_cancellation",
    }:
        raise ValueError(
            f"{field_name}.orange_tier_mode must be graceful_stop or "
            "tp_only_with_active_entry_cancellation, got "
            f"{orange_mode!r}"
        )
    panic_order_type = settings.get("panic_close_order_type", "limit")
    encode_hsl_panic_order_type(
        panic_order_type,
        field_name=f"{field_name}.panic_close_order_type",
    )
    return {
        "enabled": enabled,
        "red_threshold": red_threshold,
        "ema_span_minutes": ema_span_minutes,
        "cooldown_minutes_after_red": cooldown_minutes,
        "no_restart_drawdown_threshold": no_restart_threshold,
        "restart_after_red_policy": restart_policy,
        "tier_ratios": {"yellow": yellow, "orange": orange},
        "orange_tier_mode": orange_mode,
        "panic_close_order_type": panic_order_type,
    }


def validate_hsl_override_patch(
    base_hsl: dict, override_hsl: dict, *, field_name: str
) -> dict:
    if not isinstance(base_hsl, dict) or not isinstance(override_hsl, dict):
        raise TypeError(f"{field_name} must be a dictionary")
    effective = dict(base_hsl)
    base_ratios = base_hsl.get("tier_ratios", {})
    override_ratios = override_hsl.get("tier_ratios", {})
    if not isinstance(base_ratios, dict) or not isinstance(override_ratios, dict):
        raise TypeError(f"{field_name}.tier_ratios must be a dictionary")
    effective.update(
        {key: value for key, value in override_hsl.items() if key != "tier_ratios"}
    )
    effective["tier_ratios"] = {**base_ratios, **override_ratios}
    return validate_hsl_settings(effective, field_name=field_name)


MULTICOIN_TOTAL_EXPOSURE_ENFORCER_PARAM_KEYS = (
    *TOTAL_EXPOSURE_ENFORCER_PARAM_KEYS,
    "twel_enforcer_reduce_portfolio",
)

# Compatibility name retained for imports from the first exposure-policy slice.
SINGLE_COIN_EXPOSURE_PARAM_KEYS = EXPOSURE_PARAM_KEYS

EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS = (
    *EMA_ANCHOR_PARAM_KEYS,
    *SINGLE_COIN_EXPOSURE_PARAM_KEYS,
    *TOTAL_EXPOSURE_ENFORCER_PARAM_KEYS,
    *UNSTUCK_PARAM_KEYS,
    *HSL_PARAM_KEYS,
    "wallet_exposure_limit",
)

EMA_ANCHOR_MULTICOIN_PARAM_KEYS = (
    *EMA_ANCHOR_PARAM_KEYS,
    "forager_volume_ema_span_1m",
    "forager_volatility_ema_span_1m",
    "forager_volume_drop_pct",
    "forager_score_weights_volume",
    "forager_score_weights_ema_readiness",
    "forager_score_weights_volatility",
    "n_positions",
    *EXPOSURE_PARAM_KEYS,
    *MULTICOIN_TOTAL_EXPOSURE_ENFORCER_PARAM_KEYS,
    *UNSTUCK_PARAM_KEYS,
    *HSL_PARAM_KEYS,
)

TRAILING_MARTINGALE_PARAM_KEYS = (
    "ema_span_0",
    "ema_span_1",
    "volatility_ema_span_1h",
    "volatility_ema_span_1m",
    "entry_double_down_factor",
    "entry_initial_ema_dist",
    "entry_initial_qty_pct",
    "entry_threshold_base_pct",
    "entry_threshold_we_weight",
    "entry_threshold_volatility_1h_weight",
    "entry_threshold_volatility_1m_weight",
    "entry_retracement_base_pct",
    "entry_retracement_we_weight",
    "entry_retracement_volatility_1h_weight",
    "entry_retracement_volatility_1m_weight",
    "close_qty_pct",
    "close_threshold_base_pct",
    "close_threshold_we_weight",
    "close_threshold_volatility_1h_weight",
    "close_threshold_volatility_1m_weight",
    "close_retracement_base_pct",
    "close_retracement_volatility_1h_weight",
    "close_retracement_volatility_1m_weight",
    "entry_cooldown_minutes",
    "total_wallet_exposure_limit",
    "gate_initial",
    "gate_reentry",
)

TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS = (
    *TRAILING_MARTINGALE_PARAM_KEYS,
    *SINGLE_COIN_EXPOSURE_PARAM_KEYS,
    *POSITION_EXPOSURE_ENFORCER_PARAM_KEYS,
    *TOTAL_EXPOSURE_ENFORCER_PARAM_KEYS,
    *UNSTUCK_PARAM_KEYS,
    *HSL_PARAM_KEYS,
    "wallet_exposure_limit",
)

TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS = (
    *TRAILING_MARTINGALE_PARAM_KEYS,
    "forager_volume_ema_span_1m",
    "forager_volatility_ema_span_1m",
    "forager_volume_drop_pct",
    "forager_score_weights_volume",
    "forager_score_weights_ema_readiness",
    "forager_score_weights_volatility",
    "n_positions",
    *EXPOSURE_PARAM_KEYS,
    *POSITION_EXPOSURE_ENFORCER_PARAM_KEYS,
    *MULTICOIN_TOTAL_EXPOSURE_ENFORCER_PARAM_KEYS,
    *UNSTUCK_PARAM_KEYS,
    *HSL_PARAM_KEYS,
)

TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS = (
    ("ema_span_0", ("ema_span_0",)),
    ("ema_span_1", ("ema_span_1",)),
    ("volatility_ema_span_1h", ("volatility_ema_span_1h",)),
    ("volatility_ema_span_1m", ("volatility_ema_span_1m",)),
    ("entry_double_down_factor", ("entry", "double_down_factor")),
    ("entry_initial_ema_dist", ("entry", "initial_ema_dist")),
    ("entry_initial_qty_pct", ("entry", "initial_qty_pct")),
    ("entry_threshold_base_pct", ("entry", "threshold_base_pct")),
    ("entry_threshold_we_weight", ("entry", "threshold_we_weight")),
    (
        "entry_threshold_volatility_1h_weight",
        ("entry", "threshold_volatility_1h_weight"),
    ),
    (
        "entry_threshold_volatility_1m_weight",
        ("entry", "threshold_volatility_1m_weight"),
    ),
    ("entry_retracement_base_pct", ("entry", "retracement_base_pct")),
    ("entry_retracement_we_weight", ("entry", "retracement_we_weight")),
    (
        "entry_retracement_volatility_1h_weight",
        ("entry", "retracement_volatility_1h_weight"),
    ),
    (
        "entry_retracement_volatility_1m_weight",
        ("entry", "retracement_volatility_1m_weight"),
    ),
    ("close_qty_pct", ("close", "qty_pct")),
    ("close_threshold_base_pct", ("close", "threshold_base_pct")),
    ("close_threshold_we_weight", ("close", "threshold_we_weight")),
    (
        "close_threshold_volatility_1h_weight",
        ("close", "threshold_volatility_1h_weight"),
    ),
    (
        "close_threshold_volatility_1m_weight",
        ("close", "threshold_volatility_1m_weight"),
    ),
    ("close_retracement_base_pct", ("close", "retracement_base_pct")),
    (
        "close_retracement_volatility_1h_weight",
        ("close", "retracement_volatility_1h_weight"),
    ),
    (
        "close_retracement_volatility_1m_weight",
        ("close", "retracement_volatility_1m_weight"),
    ),
)

TRAILING_MARTINGALE_GATE_MODE_OVERRIDE_PATH = ("entry", "ema_gate_mode")
TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN = len(
    TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS
)
TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN = (
    TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN + 1
)
TRAILING_MARTINGALE_COIN_OVERRIDE_ALLOWANCE_PCT_COLUMN = (
    TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN + 1
)
TRAILING_MARTINGALE_COIN_OVERRIDE_WEL_ENFORCER_ENABLED_COLUMN = (
    TRAILING_MARTINGALE_COIN_OVERRIDE_ALLOWANCE_PCT_COLUMN + 1
)
TRAILING_MARTINGALE_COIN_OVERRIDE_WEL_ENFORCER_THRESHOLD_COLUMN = (
    TRAILING_MARTINGALE_COIN_OVERRIDE_WEL_ENFORCER_ENABLED_COLUMN + 1
)
TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_START_COLUMN = (
    TRAILING_MARTINGALE_COIN_OVERRIDE_WEL_ENFORCER_THRESHOLD_COLUMN + 1
)
TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN = (
    TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_START_COLUMN + 6
)
TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN = (
    TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN
    + len(HSL_COIN_OVERRIDE_PATHS)
)
TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_REENTRY_COLUMN = (
    TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN + 1
)
TRAILING_MARTINGALE_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN = (
    TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_REENTRY_COLUMN + 1
)
TRAILING_MARTINGALE_COIN_OVERRIDE_COLS = (
    TRAILING_MARTINGALE_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN + 1
)


def encode_tm_retracement_base_pct(value) -> np.float32:
    """Pack a TM retracement base without losing its recursive/trailing mode."""

    value = float(value)
    if not math.isfinite(value) or abs(value) > float(np.finfo(np.float32).max):
        raise ValueError(
            "Trailing Martingale retracement_base_pct must be finite and "
            "representable as float32"
        )
    encoded = np.float32(value)
    if value > 0.0 and encoded == np.float32(0.0):
        # Rust selects trailing mode from the float64 sign. Preserve that mode
        # in Metal even when an extremely small positive magnitude underflows
        # during float32 packing. The smallest normal is used because GPU
        # arithmetic may flush subnormal values to zero.
        encoded = np.float32(np.finfo(np.float32).tiny)
    return encoded


GPU_STRATEGY_PARAM_KEYS = {
    "ema_anchor": EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
    "trailing_martingale": TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
}


def flatten_trailing_martingale_params(strategy: dict, risk: dict) -> dict:
    """Flatten Rust's nested TM payload into the Metal row contract."""

    entry = strategy.get("entry", {})
    close = strategy.get("close", {})
    mode = str(entry.get("ema_gate_mode", "all")).strip().lower()
    if mode not in {"disabled", "all", "initial", "reentry"}:
        raise ValueError(f"unsupported trailing_martingale entry.ema_gate_mode={mode!r}")
    flattened = {
        "ema_span_0": strategy.get("ema_span_0"),
        "ema_span_1": strategy.get("ema_span_1"),
        "volatility_ema_span_1h": strategy.get("volatility_ema_span_1h"),
        "volatility_ema_span_1m": strategy.get("volatility_ema_span_1m"),
        "entry_cooldown_minutes": float(
            risk.get(
                "risk_entry_cooldown_minutes",
                risk.get("entry_cooldown_minutes", 0.0),
            )
            or 0.0
        ),
        "total_wallet_exposure_limit": float(
            risk["total_wallet_exposure_limit"]
        ),
        "gate_initial": float(mode in {"all", "initial"}),
        "gate_reentry": float(mode in {"all", "reentry"}),
    }
    for prefix, values, keys in (
        (
            "entry",
            entry,
            (
                "double_down_factor",
                "initial_ema_dist",
                "initial_qty_pct",
                "threshold_base_pct",
                "threshold_we_weight",
                "threshold_volatility_1h_weight",
                "threshold_volatility_1m_weight",
                "retracement_base_pct",
                "retracement_we_weight",
                "retracement_volatility_1h_weight",
                "retracement_volatility_1m_weight",
            ),
        ),
        (
            "close",
            close,
            (
                "qty_pct",
                "threshold_base_pct",
                "threshold_we_weight",
                "threshold_volatility_1h_weight",
                "threshold_volatility_1m_weight",
                "retracement_base_pct",
                "retracement_volatility_1h_weight",
                "retracement_volatility_1m_weight",
            ),
        ),
    ):
        for key in keys:
            flattened[f"{prefix}_{key}"] = values.get(key)
    return {key: flattened[key] for key in TRAILING_MARTINGALE_PARAM_KEYS}


def gpu_side_enabled(config: dict, side: str) -> bool:
    """Match Rust/backtest global side eligibility, including approved coins."""

    risk = config.get("bot", {}).get(side, {}).get("risk", {})
    total_exposure = float(risk.get("total_wallet_exposure_limit", 0.0) or 0.0)
    n_positions_raw = float(risk.get("n_positions", 0) or 0)
    if (
        not math.isfinite(total_exposure)
        or not math.isfinite(n_positions_raw)
        or total_exposure <= 0.0
        or int(round(n_positions_raw)) <= 0
    ):
        return False
    approved = config.get("live", {}).get("approved_coins", {})
    if isinstance(approved, dict):
        return bool(approved.get(side, []))
    return True


@dataclass(frozen=True)
class ProxyMarket:
    qty_step: float
    price_step: float
    min_qty: float
    min_cost: float
    c_mult: float
    maker_fee: float
    taker_fee: float = 0.0


@dataclass(frozen=True)
class ProxyRun:
    starting_balance: float
    warmup_bars: int
    trade_start_idx: int
    requested_start_ts_ms: int
    guard_ts_ms: int
    first_ts_ms: int
    interval_ms: int
    liquidation_threshold: float
    first_valid_idx: int
    last_valid_idx: int


def _build_hourly_log_range(high, low, timestamps, run: ProxyRun):
    n = len(timestamps)
    derived_timestamps = (
        int(timestamps[0]) + np.arange(n, dtype=np.int64) * run.interval_ms
    )
    hour_idx = derived_timestamps // 3_600_000
    boundary = np.zeros(n, dtype=bool)
    boundary[1:] = hour_idx[1:] > hour_idx[:-1]
    hour_log_range = np.zeros(n, dtype=np.float32)
    hour_valid = np.zeros(n, dtype=bool)
    last_hour_boundary_ms = (int(timestamps[0]) // 3_600_000) * 3_600_000
    first_valid = max(0, run.first_valid_idx)
    latest = None
    for k in range(1, n):
        if not boundary[k]:
            continue
        current_ts = int(derived_timestamps[k])
        window_start_ms = max(int(timestamps[0]), last_hour_boundary_ms)
        if current_ts > window_start_ms + run.interval_ms:
            start = max(
                int(
                    (window_start_ms - int(timestamps[0]))
                    // run.interval_ms
                ),
                first_valid,
            )
            end = min(k - 1, run.last_valid_idx)
            if end >= start:
                h_segment = high[start : end + 1]
                l_segment = low[start : end + 1]
                finite = np.isfinite(h_segment) & np.isfinite(l_segment)
                if finite.any():
                    highest = h_segment[finite].max()
                    lowest = l_segment[finite].min()
                    if highest > 0.0 and lowest > 0.0:
                        latest = math.log(highest / lowest)
        if latest is not None:
            hour_log_range[k] = latest
            hour_valid[k] = True
        last_hour_boundary_ms = (current_ts // 3_600_000) * 3_600_000
    return hour_log_range, hour_valid


def _strict_fill_tick_boundaries(
    high, low, price_step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Encode Rust's strict candle/order comparisons as integer tick boundaries."""

    if not np.isfinite(price_step) or price_step <= 0.0:
        raise ValueError(
            f"MPS proxy requires a positive finite price_step, got {price_step}"
        )
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    finite = np.isfinite(high) & np.isfinite(low)
    safe_high = np.where(finite, high, 0.0)
    safe_low = np.where(finite, low, 0.0)
    high_fill_max = np.floor(safe_high / price_step).astype(np.int64)
    low_nonfill_max = np.floor(safe_low / price_step).astype(np.int64)

    decimal_multiplier = None
    multiplier = 1.0
    for places in range(16):
        scaled_step = abs(price_step) * multiplier
        rounded_step = np.floor(scaled_step + 0.5)
        if rounded_step >= 1.0 and abs(scaled_step - rounded_step) <= max(
            abs(scaled_step), 1.0
        ) * 1e-12:
            decimal_multiplier = 10.0 ** max(places, 10)
            break
        multiplier *= 10.0

    def rust_tick_prices(ticks):
        prices = ticks.astype(np.float64) * price_step
        if decimal_multiplier is None:
            return prices
        scaled = prices * decimal_multiplier
        return (
            np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)
            / decimal_multiplier
        )

    # Division can land one tick to either side near an exact boundary. Repair
    # against Rust's step-decimal-preserving order-price rounding contract.
    for _ in range(2):
        high_fill_max -= (rust_tick_prices(high_fill_max) >= safe_high).astype(
            np.int64
        )
        high_fill_max += (rust_tick_prices(high_fill_max + 1) < safe_high).astype(
            np.int64
        )
        low_nonfill_max -= (rust_tick_prices(low_nonfill_max) > safe_low).astype(
            np.int64
        )
        low_nonfill_max += (
            rust_tick_prices(low_nonfill_max + 1) <= safe_low
        ).astype(np.int64)

    high_fill_max[~finite] = 0
    low_nonfill_max[~finite] = 0
    i32 = np.iinfo(np.int32)
    if (
        high_fill_max.min(initial=0) < i32.min
        or high_fill_max.max(initial=0) > i32.max
        or low_nonfill_max.min(initial=0) < i32.min
        or low_nonfill_max.max(initial=0) > i32.max
    ):
        raise ValueError("MPS proxy candle price ticks exceed signed 32-bit range")
    return high_fill_max.astype(np.int32), low_nonfill_max.astype(np.int32)


def _directional_touch_ticks(
    prices, price_step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode Rust-compatible down/up/nearest ticks for candle touch prices."""

    if not np.isfinite(price_step) or price_step <= 0.0:
        raise ValueError(
            f"MPS proxy requires a positive finite price_step, got {price_step}"
        )
    prices = np.asarray(prices, dtype=np.float64)
    finite = np.isfinite(prices) & (prices > 0.0)
    safe = np.where(finite, prices, 0.0)
    step_counts = safe / price_step
    nearest_ticks = np.floor(step_counts + 0.5).astype(np.int64)

    decimal_multiplier = None
    multiplier = 1.0
    for places in range(16):
        scaled_step = abs(price_step) * multiplier
        rounded_step = np.floor(scaled_step + 0.5)
        if rounded_step >= 1.0 and abs(scaled_step - rounded_step) <= max(
            abs(scaled_step), 1.0
        ) * 1e-12:
            decimal_multiplier = 10.0 ** max(places, 10)
            break
        multiplier *= 10.0

    nearest_prices = nearest_ticks.astype(np.float64) * price_step
    if decimal_multiplier is not None:
        scaled = nearest_prices * decimal_multiplier
        nearest_prices = (
            np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)
            / decimal_multiplier
        )
    tolerance = (
        np.finfo(np.float64).eps
        * np.maximum(np.abs(safe), np.abs(nearest_prices))
        * 4.0
    )
    aligned = finite & (np.abs(safe - nearest_prices) <= tolerance)
    down_ticks = np.where(aligned, nearest_ticks, np.floor(step_counts)).astype(
        np.int64
    )
    up_ticks = np.where(aligned, nearest_ticks, np.ceil(step_counts)).astype(np.int64)
    down_ticks[~finite] = 0
    up_ticks[~finite] = 0
    nearest_ticks[~finite] = 0
    i32 = np.iinfo(np.int32)
    if (
        down_ticks.min(initial=0) < i32.min
        or down_ticks.max(initial=0) > i32.max
        or up_ticks.min(initial=0) < i32.min
        or up_ticks.max(initial=0) > i32.max
        or nearest_ticks.min(initial=0) < i32.min
        or nearest_ticks.max(initial=0) > i32.max
    ):
        raise ValueError("MPS proxy candle touch ticks exceed signed 32-bit range")
    return (
        down_ticks.astype(np.int32),
        up_ticks.astype(np.int32),
        nearest_ticks.astype(np.int32),
    )


def _minimum_entry_qty_values(prices, market: ProxyMarket) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float64)
    finite = np.isfinite(prices) & (prices > 0.0)
    safe = np.where(finite, prices, 1.0)
    raw_min = np.maximum(
        float(market.min_qty),
        float(market.min_cost) / safe / float(market.c_mult),
    )
    raw_steps = raw_min / float(market.qty_step)
    nearest_steps = np.floor(raw_steps + 0.5)
    nearest_qty = nearest_steps * float(market.qty_step)
    representation_tolerance = (
        np.finfo(np.float64).eps
        * np.maximum(np.abs(raw_min), np.abs(nearest_qty))
        * 4.0
    )
    aligned = (
        (nearest_steps > 0.0)
        & (np.abs(raw_steps - nearest_steps) <= 1e-8)
        & (
            (nearest_qty >= raw_min)
            | (raw_min - nearest_qty <= representation_tolerance)
        )
    )
    minimum_qty = np.where(
        raw_min == 0.0,
        0.0,
        np.where(
            aligned,
            np.maximum(nearest_qty, raw_min),
            np.ceil(raw_steps) * float(market.qty_step),
        ),
    )
    minimum_qty = np.where(finite, minimum_qty, 0.0)
    return minimum_qty


def _float32_threshold_encoding(values) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    rounded = values.astype(np.float32)
    if (
        np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(~np.isfinite(rounded))
    ):
        raise ValueError("MPS proxy threshold values exceed float32 range")
    relation = np.sign(values - rounded.astype(np.float64)).astype(np.int32)
    rounded_bits = np.ascontiguousarray(rounded).view(np.int32)
    return rounded_bits, relation


def _minimum_entry_qty_encoding(
    prices, market: ProxyMarket
) -> tuple[np.ndarray, np.ndarray]:
    """Encode Rust-compatible float64 touch minima for float32 Metal comparisons."""

    return _float32_threshold_encoding(_minimum_entry_qty_values(prices, market))


def _maximum_effective_min_cost(prices, market: ProxyMarket) -> float:
    """Return the conservative executable minimum cost over a prepared window."""

    prices = np.asarray(prices, dtype=np.float64)
    minimum_qty = _minimum_entry_qty_values(prices, market)
    finite = np.isfinite(prices) & (prices > 0.0)
    effective_cost = np.where(
        finite,
        minimum_qty * prices * float(market.c_mult),
        0.0,
    )
    maximum = float(np.max(effective_cost, initial=0.0))
    encoded = np.float32(maximum)
    if float(encoded) < maximum:
        encoded = np.nextafter(encoded, np.float32(np.inf))
    if not np.isfinite(encoded):
        raise ValueError("MPS proxy maximum effective minimum cost exceeds float32")
    return float(encoded)


def build_mps_data(high, low, close, timestamps_ms, run: ProxyRun, market: ProxyMarket):
    """Prepare immutable minute data and keep it resident on Apple MPS.

    Torch is imported here, rather than at module import time, so normal CPU
    bot, backtest, and optimizer startup never depends on the optional GPU
    runtime.
    """

    try:
        import torch
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - exercised without the optional extra
        raise ModuleNotFoundError(
            "Apple MPS optimization requires the optional 'gpu-mps' dependencies; "
            "install Passivbot with `pip install -e '.[full,gpu-mps]'`"
        ) from exc

    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    if not (len(high) == len(low) == len(close) == len(timestamps)):
        raise ValueError("MPS price and timestamp arrays must have matching lengths")
    if len(close) < 3:
        raise ValueError("MPS proxy requires at least three candles")

    n = len(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_range = np.log(high / low)
    log_range = np.where(np.isfinite(log_range), log_range, 0.0).astype(np.float32)

    first_valid = max(0, run.first_valid_idx)
    hour_log_range, hour_valid = _build_hourly_log_range(
        high, low, timestamps, run
    )

    day_idx = ((timestamps // 86_400_000) - (timestamps[0] // 86_400_000)).astype(
        np.int32
    )
    valid = np.zeros(n, dtype=bool)
    last_valid = min(run.last_valid_idx, n - 1)
    valid[first_valid : last_valid + 1] = (
        np.isfinite(close[first_valid : last_valid + 1])
        & (close[first_valid : last_valid + 1] > 0.0)
        & np.isfinite(high[first_valid : last_valid + 1])
        & (high[first_valid : last_valid + 1] > 0.0)
        & np.isfinite(low[first_valid : last_valid + 1])
        & (low[first_valid : last_valid + 1] > 0.0)
    )
    indices = np.arange(n, dtype=np.int64)
    can_generate = (
        (indices > max(1, run.warmup_bars))
        & (timestamps[0] + indices * run.interval_ms >= run.guard_ts_ms)
        & (indices >= run.trade_start_idx)
        & valid
    )
    high_fill_max_tick, low_nonfill_max_tick = _strict_fill_tick_boundaries(
        high, low, market.price_step
    )
    touch_down_tick, touch_up_tick, touch_nearest_tick = _directional_touch_ticks(
        close, market.price_step
    )
    touch_min_qty_bits, touch_min_qty_relation = _minimum_entry_qty_encoding(
        close, market
    )
    max_effective_min_cost = _maximum_effective_min_cost(close, market)

    def tensor(values, *, dtype=None):
        return torch.as_tensor(values, dtype=dtype, device="mps")

    return {
        "high_f": tensor(np.where(np.isfinite(high), high, 0.0).astype(np.float32)),
        "low_f": tensor(np.where(np.isfinite(low), low, 0.0).astype(np.float32)),
        "close_f": tensor(np.where(np.isfinite(close), close, 0.0).astype(np.float32)),
        "log_range": tensor(log_range),
        "hour_log_range": tensor(hour_log_range),
        "hour_valid": tensor(hour_valid),
        "valid": tensor(valid),
        "can_gen": tensor(can_generate),
        "day_idx": tensor(day_idx),
        "high_fill_max_tick": tensor(high_fill_max_tick),
        "low_nonfill_max_tick": tensor(low_nonfill_max_tick),
        "touch_down_tick": tensor(touch_down_tick),
        "touch_up_tick": tensor(touch_up_tick),
        "touch_nearest_tick": tensor(touch_nearest_tick),
        "touch_min_qty_bits": tensor(touch_min_qty_bits),
        "touch_min_qty_relation": tensor(touch_min_qty_relation),
        "max_effective_min_cost": max_effective_min_cost,
        "n_days": int(day_idx[-1]) + 1,
        "ts0": int(timestamps[0]),
        "times_relative": True,
        "n": n,
    }


def build_mps_multicoin_data(
    hlcvs,
    timestamps_ms,
    runs: list[ProxyRun],
    markets: list[ProxyMarket],
    *,
    include_hourly_ranges: bool = True,
):
    """Pack compact multicoin inputs for the persistent Apple MPS kernel.

    Raw OHLCV stays float32 for unified-memory efficiency. Strict fill and
    order-book touch comparisons are encoded from the original float64 data as
    integer ticks, avoiding the most consequential float32 boundary collapse.
    """

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        raise ModuleNotFoundError(
            "Apple MPS optimization requires the optional 'gpu-mps' dependencies; "
            "install Passivbot with `pip install -e '.[full,gpu-mps]'`"
        ) from exc

    values = np.asarray(hlcvs)
    if values.ndim != 3 or values.shape[2] < 4:
        raise ValueError(
            "MPS multicoin proxy expects HLCVs shaped (candles, coins, >=4); "
            f"got {values.shape}"
        )
    candle_count, coin_count, _channels = values.shape
    if not 2 <= coin_count <= MPS_MULTICOIN_MAX_COINS:
        raise ValueError(
            "MPS multicoin proxy supports 2.."
            f"{MPS_MULTICOIN_MAX_COINS} coins; got {coin_count}"
        )
    if len(runs) != coin_count or len(markets) != coin_count:
        raise ValueError(
            "MPS multicoin run/market settings must match the prepared coin count"
        )
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    if len(timestamps) != candle_count:
        raise ValueError(
            f"MPS multicoin timestamps length {len(timestamps)} != {candle_count} candles"
        )
    if candle_count < 3:
        raise ValueError("MPS multicoin proxy requires at least three candles")
    intervals = np.diff(timestamps)
    interval_ms = int(runs[0].interval_ms)
    if interval_ms <= 0 or interval_ms % 60_000 != 0:
        raise ValueError(
            "MPS multicoin proxy requires a positive whole-minute candle interval"
        )
    if np.any(intervals != interval_ms):
        raise ValueError("MPS multicoin proxy requires a continuous candle timeline")

    bars = np.ascontiguousarray(values[:, :, :4], dtype=np.float32)
    # Preserve a non-finite close so portfolio-equity accumulation can mirror
    # exact Rust by omitting that coin's unrealized PnL. Other non-finite
    # fields use zero sentinels and remain blocked by candle validity.
    for field in (0, 1, 3):
        field_values = bars[:, :, field]
        field_values[~np.isfinite(field_values)] = 0.0
    fill_ticks = np.empty((candle_count, coin_count, 2), dtype=np.int32)
    touch_ticks = np.empty((candle_count, coin_count, 2), dtype=np.int32)
    touch_nearest_ticks = np.empty((candle_count, coin_count), dtype=np.int32)
    touch_min_qty_bits = np.empty((candle_count, coin_count), dtype=np.int32)
    touch_min_qty_relation = np.empty((candle_count, coin_count), dtype=np.int32)
    if include_hourly_ranges:
        # A valid log range is always non-negative, so -1.0 is an unambiguous
        # sentinel and avoids a second dense per-candle/per-coin validity tensor.
        hour_log_ranges = np.full(
            (candle_count, coin_count), -1.0, dtype=np.float32
        )
    else:
        hour_log_ranges = None
    coin_settings = np.empty((coin_count, 13), dtype=np.float32)
    for coin, (run, market) in enumerate(zip(runs, markets)):
        if run.interval_ms != interval_ms:
            raise ValueError("MPS multicoin runs must use one shared candle interval")
        high = values[:, coin, 0].astype(np.float64, copy=False)
        low = values[:, coin, 1].astype(np.float64, copy=False)
        close = values[:, coin, 2].astype(np.float64, copy=False)
        if hour_log_ranges is not None:
            coin_hour_log_range, coin_hour_valid = _build_hourly_log_range(
                high, low, timestamps, run
            )
            hour_log_ranges[coin_hour_valid, coin] = coin_hour_log_range[
                coin_hour_valid
            ]
        high_fill, low_nonfill = _strict_fill_tick_boundaries(
            high, low, market.price_step
        )
        touch_down, touch_up, touch_nearest = _directional_touch_ticks(
            close, market.price_step
        )
        fill_ticks[:, coin, 0] = high_fill
        fill_ticks[:, coin, 1] = low_nonfill
        touch_ticks[:, coin, 0] = touch_down
        touch_ticks[:, coin, 1] = touch_up
        touch_nearest_ticks[:, coin] = touch_nearest
        min_qty_bits, min_qty_relation = _minimum_entry_qty_encoding(close, market)
        touch_min_qty_bits[:, coin] = min_qty_bits
        touch_min_qty_relation[:, coin] = min_qty_relation
        max_effective_min_cost = _maximum_effective_min_cost(close, market)
        seed_index = min(max(int(run.first_valid_idx), 0), candle_count - 1)
        seed_close = float(close[seed_index])
        high_seed = float(values[seed_index, coin, 0])
        low_seed = float(values[seed_index, coin, 1])
        volume_seed = max(float(values[seed_index, coin, 3]), 0.0)
        typical_seed = (
            (high_seed + low_seed + seed_close) / 3.0
            if high_seed > 0.0 and low_seed > 0.0 and seed_close > 0.0
            else max(seed_close, 1.0)
        )
        coin_settings[coin] = (
            market.qty_step,
            market.price_step,
            market.min_qty,
            market.min_cost,
            market.c_mult,
            market.maker_fee,
            run.first_valid_idx,
            run.last_valid_idx,
            run.trade_start_idx,
            seed_close if np.isfinite(seed_close) and seed_close > 0.0 else 0.0,
            volume_seed * typical_seed,
            market.taker_fee,
            max_effective_min_cost,
        )

    invariant_bytes = (
        bars.nbytes
        + fill_ticks.nbytes
        + touch_ticks.nbytes
        + touch_nearest_ticks.nbytes
        + touch_min_qty_bits.nbytes
        + touch_min_qty_relation.nbytes
        + (hour_log_ranges.nbytes if hour_log_ranges is not None else 0)
    )
    recommended = None
    recommended_fn = getattr(torch.mps, "recommended_max_memory", None)
    if callable(recommended_fn):
        recommended = int(recommended_fn())
    if recommended and invariant_bytes > int(recommended * 0.45):
        raise MemoryError(
            "MPS multicoin invariant tensors would consume "
            f"{invariant_bytes / 2**30:.2f} GiB, above the 45% safety limit of "
            f"the device's {recommended / 2**30:.2f} GiB recommended working set"
        )

    def tensor(array, *, dtype=None):
        return torch.as_tensor(array, dtype=dtype, device="mps").contiguous()

    first_day = int(timestamps[0] // 86_400_000)
    last_day = int(timestamps[-1] // 86_400_000)
    packed = {
        "bars": tensor(bars, dtype=torch.float32),
        "fill_ticks": tensor(fill_ticks, dtype=torch.int32),
        "touch_ticks": tensor(touch_ticks, dtype=torch.int32),
        "touch_nearest_ticks": tensor(touch_nearest_ticks, dtype=torch.int32),
        "touch_min_qty_bits": tensor(touch_min_qty_bits, dtype=torch.int32),
        "touch_min_qty_relation": tensor(
            touch_min_qty_relation, dtype=torch.int32
        ),
        "coin_settings": tensor(coin_settings, dtype=torch.float32),
        "n": candle_count,
        "n_coins": coin_count,
        "n_days": last_day - first_day + 1,
        "ts0": int(timestamps[0]),
        "start_minute_of_day": int((timestamps[0] // 60_000) % 1440),
        "start_minute_of_hour": int((timestamps[0] // 60_000) % 60),
        "invariant_bytes": invariant_bytes,
    }
    if hour_log_ranges is not None:
        packed["hour_log_ranges"] = tensor(
            hour_log_ranges, dtype=torch.float32
        )
    return packed
