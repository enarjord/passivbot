"""Dependency-light configuration for v9 risk memory and history acquisition."""

import math
import struct


DEFAULT_LOOKBACK_EMA_SPAN_MINUTES = 43_200.0
DEFAULT_RISK_BOOTSTRAP_MAX_LOOKBACK_MINUTES = 129_600.0
MAX_RISK_BOOTSTRAP_LOOKBACK_MINUTES = 129_600.0


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


def normalize_memory_ema_span_minutes(value, *, field="memory_ema_span_minutes"):
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


def risk_memory_spans(config):
    """Return the independent memory spans of enabled factual consumers."""
    bot = config["bot"]
    spans = []
    scope = bot["risk"]["hsl"]["scope"]
    policies = ([bot["risk"]["hsl"]] if scope == "portfolio" else
                [bot[side]["hsl"] for side in ("long", "short")])
    spans.extend(policy["memory_ema_span_minutes"] for policy in policies
                 if policy["enabled"])
    if scope == "coin":
        for patch in config.get("coin_overrides", {}).values():
            for side in ("long", "short"):
                policy = {**bot[side]["hsl"], **patch.get("bot", {}).get(side, {}).get("hsl", {})}
                if policy["enabled"]:
                    spans.append(policy["memory_ema_span_minutes"])
    unstuck_enabled = False
    for side in ("long", "short"):
        if bot[side]["risk"]["total_wallet_exposure_limit"] <= 0:
            continue
        base_enabled = bot[side]["unstuck"]["enabled"]
        unstuck_enabled |= base_enabled or any(
            patch.get("bot", {}).get(side, {}).get("unstuck", {}).get("enabled", base_enabled)
            for patch in config.get("coin_overrides", {}).values()
        )
    if unstuck_enabled:
        spans.append(bot["unstuck"]["memory_ema_span_minutes"])
    gate = bot["risk"]["loss_gate"]
    if gate["max_drawdown_ratio"] < 1.0:
        spans.append(gate["memory_ema_span_minutes"])
    return tuple(sorted(set(spans)))


def risk_bootstrap_horizon_minutes(config):
    """Return (effective horizon, truncated) without deciding history readiness."""
    span = max(risk_memory_spans(config), default=1.0)
    cap = normalize_risk_bootstrap_max_lookback_minutes(
        config["live"]["risk_bootstrap_max_lookback_minutes"]
    )
    requested = 5.0 * span
    return min(requested, cap), requested > cap


def normalize_risk_memory_config(config):
    live = config["live"]
    bot = config["bot"]
    for path, block in (("bot.unstuck", bot["unstuck"]),
                        ("bot.risk.loss_gate", bot["risk"]["loss_gate"])):
        block["memory_ema_span_minutes"] = normalize_memory_ema_span_minutes(
            block["memory_ema_span_minutes"], field=f"{path}.memory_ema_span_minutes"
        )
    gate = bot["risk"]["loss_gate"]
    ratio = _finite_number(gate["max_drawdown_ratio"], "bot.risk.loss_gate.max_drawdown_ratio")
    if ratio < 0:
        raise ValueError("bot.risk.loss_gate.max_drawdown_ratio must be >= 0")
    gate["max_drawdown_ratio"] = ratio
    live["risk_bootstrap_max_lookback_minutes"] = (
        normalize_risk_bootstrap_max_lookback_minutes(
            live["risk_bootstrap_max_lookback_minutes"]
        )
    )


# Explicit v8 conversion retains its old command argument name. Runtime consumers
# use the generic policy-local helper above. This is not a config-path alias.
normalize_lookback_ema_span_minutes = normalize_memory_ema_span_minutes
