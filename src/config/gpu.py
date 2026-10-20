"""Canonical GPU configuration policy, independent of optional runtime imports."""

import json
import math

GPU_SCREENING_DEFAULTS = {"scenarios": [], "survival_fraction": 0.1, "min_survivors": 64}


def resolve_gpu_sizing(value, *, field):
    """Normalize automatic intent and positive integer-valued sizing inputs."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 1 or value != int(value)):
        raise ValueError(f"optimize.gpu.{field} must be a positive integer or auto")
    return int(value)


def resolve_gpu_screening(value=None) -> dict:
    screening = dict(GPU_SCREENING_DEFAULTS)
    configured_screening = value
    if configured_screening is not None and not isinstance(configured_screening, dict):
        raise TypeError("optimize.gpu.screening must be an object")
    screening.update(configured_screening or {})
    unknown = sorted(set(screening) - set(GPU_SCREENING_DEFAULTS))
    if unknown:
        raise ValueError("unknown optimize.gpu.screening settings: " + ", ".join(unknown))
    labels = screening["scenarios"]
    if (
        not isinstance(labels, list)
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError(
            "optimize.gpu.screening.scenarios must be an array of unique non-empty scenario labels"
        )
    survival_fraction = float(screening["survival_fraction"])
    if not math.isfinite(survival_fraction) or not 0.0 < survival_fraction <= 1.0:
        raise ValueError("optimize.gpu.screening.survival_fraction must be in (0, 1]")
    min_survivors = int(screening["min_survivors"])
    if min_survivors <= 0:
        raise ValueError("optimize.gpu.screening.min_survivors must be greater than zero")
    screening["scenarios"] = list(labels)
    screening.update(survival_fraction=survival_fraction, min_survivors=min_survivors)
    return screening


def parse_screening_scenarios(value: str) -> list[str]:
    """CLI accepts comma-separated labels or a JSON array, including [] to disable."""
    raw = value.strip()
    labels = json.loads(raw) if raw.startswith("[") else [label.strip() for label in raw.split(",")]
    return resolve_gpu_screening({"scenarios": labels})["scenarios"]


def gpu_hsl_policy(config: dict, side: str) -> dict:

    if config["bot"]["global"]["hsl"]["scope"] == "portfolio":
        return config["bot"]["global"]["hsl"]
    return config.get("bot", {}).get(side, {}).get("hsl", {})


def gpu_hsl_side_enabled(config: dict, side: str, markets_by_exchange=None) -> bool:
    from config.hsl import _side_has_enabled_policy

    if config["bot"]["global"]["hsl"]["scope"] == "coin":
        return _side_has_enabled_policy(config, side, markets_by_exchange)
    globally_enabled = bool(gpu_hsl_policy(config, side).get("enabled", False))
    if globally_enabled:
        return True
    for patch in (config.get("coin_overrides") or {}).values():
        if not isinstance(patch, dict):
            continue
        hsl_patch = patch.get("bot", {}).get(side, {}).get("hsl", {}) or {}
        if isinstance(hsl_patch, dict) and bool(hsl_patch.get("enabled", False)):
            return True
    return False


def validate_hsl_gpu_inputs(config: dict) -> None:

    interval = float(config.get("backtest", {}).get("candle_interval_minutes", 1))
    if interval != 1 and any(
        gpu_hsl_side_enabled(config, side) for side in ("long", "short")
    ):
        raise ValueError("GPU HSL requires 1m candles")


def validate_gpu_backtest_config(config: dict) -> None:
    """Reject unsupported simulation contracts without importing a device runtime."""
    from config.validate import validate_limit_order_fill_buffer_pct

    strategy = config["live"]["strategy_kind"]
    if strategy not in {"ema_anchor", "trailing_martingale"}:
        raise ValueError(
            f"GPU backtests do not support strategy {strategy!r}; use a CPU backend"
        )
    if strategy == "ema_anchor":
        from optimization.gpu.model import gpu_side_enabled

        for side in ("long", "short"):
            if not gpu_side_enabled(config, side):
                continue
            risks = [config["bot"][side]["risk"]]
            risks.extend(patch.get("bot", {}).get(side, {}).get("risk", {})
                         for patch in (config.get("coin_overrides") or {}).values())
            if any(risk.get("position_exposure_enforcer_enabled", False) for risk in risks):
                raise ValueError(
                    f"GPU EMA Anchor does not support bot.{side}.risk."
                    "position_exposure_enforcer_enabled; disable it or use a CPU backend"
                )
    backtest = config["backtest"]
    collateral = float(backtest["btc_collateral_cap"])
    if not math.isfinite(collateral) or collateral != 0.0:
        raise ValueError("GPU backtests require backtest.btc_collateral_cap=0; use a CPU backend")
    validate_limit_order_fill_buffer_pct(backtest["limit_order_fill_buffer_pct"])
    validate_hsl_gpu_inputs(config)
