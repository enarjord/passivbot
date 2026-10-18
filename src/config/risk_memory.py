"""Dependency-light configuration for v9 risk memory and history acquisition."""

import math
import struct


DEFAULT_LOOKBACK_EMA_SPAN_MINUTES = 43_200.0
DEFAULT_RISK_BOOTSTRAP_MAX_LOOKBACK_MINUTES = 129_600.0
MAX_RISK_BOOTSTRAP_LOOKBACK_MINUTES = 129_600.0
DEFAULT_REALIZED_LOSS_GATE_LOOKBACK_MINUTES = 43_200.0
ALL_HISTORY_MINUTES = -1.0


def _finite_number(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def normalize_lookback_ema_span_minutes(value, *, field="live.lookback_ema_span_minutes"):
    span = _finite_number(value, field)
    if span < 1.0:
        raise ValueError(f"{field} must be >= 1.0 minutes")
    def f32(number):
        try:
            return struct.unpack("f", struct.pack("f", number))[0]
        except OverflowError:
            return math.inf

    rounded_span = f32(span)
    denominator = f32(rounded_span + 1.0)
    alpha = f32(2.0 / denominator)
    beta = f32(1.0 - alpha)
    if alpha <= 0.0 or beta >= 1.0:
        raise ValueError(f"{field} is too large for the shared f32 decay contract")
    return span


def normalize_risk_bootstrap_max_lookback_minutes(value):
    field = "live.risk_bootstrap_max_lookback_minutes"
    cap = _finite_number(value, field)
    if not 1.0 <= cap <= MAX_RISK_BOOTSTRAP_LOOKBACK_MINUTES:
        raise ValueError(
            f"{field} must be in [1.0, {MAX_RISK_BOOTSTRAP_LOOKBACK_MINUTES}] minutes"
        )
    return cap


def normalize_realized_loss_gate_lookback_minutes(value):
    field = "live.realized_loss_gate_lookback_minutes"
    minutes = _finite_number(value, field)
    if minutes != ALL_HISTORY_MINUTES and minutes < 0.0:
        raise ValueError(f"{field} must be >= 0.0, or -1.0 for all history")
    milliseconds = minutes * 60_000.0
    # Validate the same rounded duration used by live cutoff consumers against
    # the shared signed millisecond clock, before converting an infinity or an
    # out-of-domain duration into an integer at runtime.
    if not math.isfinite(milliseconds) or round(milliseconds) > (1 << 63) - 1:
        raise ValueError(f"{field} must fit in signed 64-bit milliseconds")
    return minutes


def risk_bootstrap_horizon_minutes(live):
    """Return (effective horizon, truncated) without deciding history readiness."""
    span = normalize_lookback_ema_span_minutes(live["lookback_ema_span_minutes"])
    cap = normalize_risk_bootstrap_max_lookback_minutes(
        live["risk_bootstrap_max_lookback_minutes"]
    )
    requested = 5.0 * span
    return min(requested, cap), requested > cap


def realized_loss_gate_start_ms(live, now_ms, *, minimum_ms=1):
    """Preserve all-history and each consumer's explicit minimum resolution."""
    minutes = normalize_realized_loss_gate_lookback_minutes(
        live["realized_loss_gate_lookback_minutes"]
    )
    if minutes == ALL_HISTORY_MINUTES:
        return None
    return int(now_ms) - max(int(minimum_ms), int(round(minutes * 60_000.0)))


def normalize_risk_memory_config(config):
    live = config["live"]
    live["lookback_ema_span_minutes"] = normalize_lookback_ema_span_minutes(
        live["lookback_ema_span_minutes"]
    )
    live["risk_bootstrap_max_lookback_minutes"] = (
        normalize_risk_bootstrap_max_lookback_minutes(
            live["risk_bootstrap_max_lookback_minutes"]
        )
    )
    live["realized_loss_gate_lookback_minutes"] = (
        normalize_realized_loss_gate_lookback_minutes(
            live["realized_loss_gate_lookback_minutes"]
        )
    )
