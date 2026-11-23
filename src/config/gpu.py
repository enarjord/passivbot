"""Canonical GPU configuration policy, independent of optional runtime imports."""

import json
import math

GPU_TUNING_DEFAULT = "conservative"
GPU_TUNING_INPUTS = (GPU_TUNING_DEFAULT, "max", "refresh", "off", "auto")
GPU_SCREENING_DEFAULTS = {"scenarios": [], "survival_fraction": 0.1, "min_survivors": 64}


def resolve_gpu_tuning_mode(value=GPU_TUNING_DEFAULT) -> str:
    """Resolve the legacy automatic mode to the current conservative policy."""
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in GPU_TUNING_INPUTS:
            return GPU_TUNING_DEFAULT if mode == "auto" else mode
    raise ValueError("optimize.gpu.tuning_mode must be conservative, max, refresh, off, or auto")


def resolve_gpu_sizing(value, *, field):
    """Normalize automatic intent and positive integer-valued sizing inputs."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 1 or value != int(value)):
        raise ValueError(f"optimize.gpu.{field} must be a positive integer or auto")
    return int(value)


def resolve_gpu_search_cohort_size(value, *, tuning_mode):
    size = resolve_gpu_sizing(value, field="search_cohort_size")
    if size is not None and tuning_mode != "max":
        raise ValueError("optimize.gpu.search_cohort_size requires tuning_mode='max'")
    return size


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

    if config["bot"]["portfolio"]["risk"]["hsl"]["scope"] == "portfolio":
        return config["bot"]["portfolio"]["risk"]["hsl"]
    return config.get("bot", {}).get(side, {}).get("risk", {}).get("hsl", {})


def gpu_hsl_side_enabled(config: dict, side: str, markets_by_exchange=None) -> bool:
    from config.hsl import _side_has_enabled_policy

    if config["bot"]["portfolio"]["risk"]["hsl"]["scope"] == "coin":
        return _side_has_enabled_policy(config, side, markets_by_exchange)
    globally_enabled = bool(gpu_hsl_policy(config, side).get("enabled", False))
    if globally_enabled:
        return True
    for patch in (config.get("coin_overrides") or {}).values():
        if not isinstance(patch, dict):
            continue
        hsl_patch = patch.get("bot", {}).get(side, {}).get("risk", {}).get("hsl", {}) or {}
        if isinstance(hsl_patch, dict) and bool(hsl_patch.get("enabled", False)):
            return True
    return False


def validate_hsl_gpu_inputs(config: dict) -> None:

    interval = float(config.get("backtest", {}).get("candle_interval_minutes", 1))
    if interval != 1 and any(
        gpu_hsl_side_enabled(config, side) for side in ("long", "short")
    ):
        raise ValueError("GPU HSL requires 1m candles")


def validate_gpu_loss_gate_policy(enabled: bool, cap: float | None) -> None:
    if type(enabled) is not bool:
        raise ValueError("GPU loss_gate_enabled must be an explicit boolean")
    if cap is not None and (
        type(cap) not in (int, float) or not math.isfinite(cap) or not 0.0 <= cap < 1.0
    ):
        raise ValueError("GPU loss_gate_max_drawdown_fraction must be in [0, 1) or null")
    if enabled and cap is None:
        raise ValueError("GPU loss_gate_max_drawdown_fraction is required when the gate is enabled")


def gpu_loss_gate_transport_cap(enabled: bool, cap: float | None) -> float:
    """Encode a dormant gate as zero while preserving active zero via its flag."""
    validate_gpu_loss_gate_policy(enabled, cap)
    return float(cap) if enabled else 0.0


def validate_gpu_backtest_config(
    config: dict, *, check_realized_close_policies: bool = True, target_vendor: str | None = None
) -> None:
    """Reject unsupported simulation contracts without importing a device runtime."""
    from config.validate import validate_limit_order_fill_buffer_fraction

    strategy = config['bot']["portfolio"]['strategy_kind']
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
            if any(risk.get('position_exposure_enforcer', {}).get('enabled', False) for risk in risks):
                raise ValueError(
                    f"GPU EMA Anchor does not support bot.{side}.risk."
                    "position_exposure_enforcer.enabled; disable it or use a CPU backend"
                )
    backtest = config["backtest"]
    collateral = float(backtest["btc_collateral_target_ratio"])
    if not math.isfinite(collateral) or collateral != 0.0:
        raise ValueError("GPU backtests require backtest.btc_collateral_target_ratio=0; use a CPU backend")
    validate_limit_order_fill_buffer_fraction(backtest["limit_order_fill_buffer_fraction"])
    validate_hsl_gpu_inputs(config)
    if check_realized_close_policies:
        validate_gpu_realized_close_policies(config, target_vendor=target_vendor)


_CLOSE_SIZING_GPU_VENDORS = frozenset({"apple", "nvidia", "amd"})


def require_gpu_close_sizing_support(
    *, target_vendor: str, loss_gate_enabled: bool, unstuck_enabled: bool
) -> None:
    """Keep active close sizing on qualified shader runtimes."""
    if target_vendor in _CLOSE_SIZING_GPU_VENDORS or not (loss_gate_enabled or unstuck_enabled):
        return
    platform = "AMD ROCm" if target_vendor == "amd" else "the selected GPU backend"
    feature = "realized-loss close sizing" if loss_gate_enabled else "unstuck exposure sizing"
    raise ValueError(f"GPU replay on {platform} does not support v9 {feature}; use a CPU backend")


def validate_gpu_realized_close_policies(config: dict, *, target_vendor: str | None = None) -> None:
    from config.bot_inheritance import get_bot_policy, get_side_field_catalog, resolve_bot_policy
    from config.risk_memory import require_loss_realization_policy
    from optimization.gpu.model import gpu_side_enabled

    _, gate_enabled, gate_cap = require_loss_realization_policy(config)
    validate_gpu_loss_gate_policy(gate_enabled, gate_cap)
    # Configuration preparation is also used without a runtime. The physical
    # replay entry points always enforce the known device and packed features.
    if target_vendor is None or target_vendor in _CLOSE_SIZING_GPU_VENDORS:
        return

    bot = config["bot"]
    if get_bot_policy(config) is None:
        bot = resolve_bot_policy(bot, get_side_field_catalog()).effective_view()
    effective = {**config, "bot": bot}
    overrides = config.get("coin_overrides") or {}
    unstuck_enabled = any(
        gpu_side_enabled(effective, side)
        and (
            bot[side]["risk"]["unstuck"]["enabled"]
            or any(
                patch.get("bot", {}).get(side, {}).get("risk", {}).get("unstuck", {}).get("enabled")
                for patch in overrides.values()
            )
        )
        for side in ("long", "short")
    )
    require_gpu_close_sizing_support(
        target_vendor=target_vendor, loss_gate_enabled=gate_enabled, unstuck_enabled=unstuck_enabled,
    )
