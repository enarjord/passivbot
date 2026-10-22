"""Canonical HSL policy transport for offline GPU screening.

The shared backtest payload owns effective policies, including coin overrides.
"""

import math


V9_NATIVE_SERVICE_REQUIRED = (
    "v9 GPU replay requires the shared-account NativeBacktestService "
    "(optimization.gpu.native), including one-coin datasets; "
    "the retired directional GPU replay cannot execute v9 risk"
)


def pack_cooldown_ms(minutes):
    """Encode the canonical f64 half-up duration as two exact f32 limbs."""
    if isinstance(minutes, bool):
        raise ValueError("GPU HSL cooldown must be finite nonnegative minutes")
    minutes = float(minutes)
    if not math.isfinite(minutes) or minutes < 0:
        raise ValueError("GPU HSL cooldown must be finite nonnegative minutes")
    scaled = minutes * 60000.0
    if not math.isfinite(scaled):
        raise ValueError("GPU HSL cooldown exceeds the exact 48-bit millisecond capability")
    duration = math.floor(scaled + 0.5)
    if duration > (1 << 48) - 1:
        raise ValueError("GPU HSL cooldown exceeds the exact 48-bit millisecond capability")
    return float(duration & ((1 << 24) - 1)), float(duration >> 24)


def project_bot(payload, coin_index, side, config, *, base=False):
    bot = payload.bot_params_list[coin_index][side]
    hsl = payload.backtest_params["equity_hard_stop_loss"]
    if hsl["engine"] != "hsl":
        raise ValueError("GPU HSL payload engine differs from selected config")
    index = ("long", "short").index(side)
    mode = hsl["mode"]
    if mode == "portfolio":
        policy = hsl["portfolio"]
    elif mode == "coin":
        coin = payload.backtest_params["coins"][coin_index]
        policy = config["bot"][side]["hsl"] if base else hsl["coins"][coin][index]
    else:
        policy = hsl["sides"][index]
    result = dict(bot)
    result.update({f"hsl_{key}": value for key, value in policy.items()})
    result["hsl_lookback_ema_span_minutes"] = payload.backtest_params["lookback_ema_span_minutes"]
    if mode == "coin":
        configured = int(bot["n_positions"])
        result["hsl_enabled"] = bool(policy["enabled"]) and configured > 0
        # One tradable coin has exactly one effective slot.
        result["hsl_slot_count"] = (
            1 if payload.backtest_params["dynamic_wel_by_tradability"] else configured
        )
    return result


def pack_params(bot, signal_mode):
    if signal_mode not in {"coin", "pside", "portfolio"}:
        raise ValueError("v9 GPU HSL mode must be coin, pside or portfolio")
    from config.risk_memory import normalize_lookback_ema_span_minutes

    enabled = bool(bot["hsl_enabled"])
    if "hsl_restart_after_red_policy" in bot:
        raise ValueError("v8 HSL restart policy requires explicit v9 migration")
    cooldown_lo, cooldown_hi = pack_cooldown_ms(bot["hsl_cooldown_minutes_after_red"])
    try:
        fast_span = normalize_lookback_ema_span_minutes(bot["hsl_ema_span_minutes"])
    except ValueError as error:
        raise ValueError("GPU HSL fast EMA span cannot encode the shared f32 decay") from error
    return {
        "hsl_enabled": float(enabled),
        "hsl_red_threshold": float(bot["hsl_red_threshold"]),
        "hsl_ema_span_minutes": fast_span,
        "hsl_cooldown_ms_lo": cooldown_lo,
        # The removed restart-policy positional word now transports shared
        # slow memory without expanding the strategy's compact replay ABI.
        "hsl_lookback_ema_span_minutes": normalize_lookback_ema_span_minutes(
            bot["hsl_lookback_ema_span_minutes"]
        ),
        "hsl_signal_mode": {"portfolio": 0.0, "pside": 1.0, "coin": 2.0}[signal_mode],
        "hsl_cooldown_ms_hi": cooldown_hi,
    }
