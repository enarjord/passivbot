"""Dependency-light configuration for v9 risk memory and history acquisition."""

from copy import deepcopy
import math
import struct


DEFAULT_REALIZED_LOSS_MEMORY_EMA_SPAN_MINUTES = 85_511.137
LOSS_REALIZATION_PATH = ("bot", "portfolio", "risk", "loss_realization")
LOSS_GATE_CAP_PATH = (*LOSS_REALIZATION_PATH, "loss_gate", "max_realized_drawdown_fraction")
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
    field = "live.startup.risk_bootstrap_max_lookback_minutes"
    cap = _finite_number(value, field)
    if not 1.0 <= cap <= MAX_RISK_BOOTSTRAP_LOOKBACK_MINUTES:
        raise ValueError(
            f"{field} must be in [1.0, {MAX_RISK_BOOTSTRAP_LOOKBACK_MINUTES}] minutes"
        )
    return cap


def normalize_loss_gate_cap(value, *, field=".".join(LOSS_GATE_CAP_PATH)):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    cap = _finite_number(value, field)
    if not 0.0 <= cap < 1.0:
        raise ValueError(f"{field} must be in [0, 1)")
    return cap


def require_loss_realization_policy(config):
    """Read the resolved portfolio policy, including explicit gate enablement."""
    policy = config["bot"]["portfolio"]["risk"]["loss_realization"]
    path = ".".join(LOSS_REALIZATION_PATH)
    span = normalize_memory_ema_span_minutes(
        policy["memory_ema_span_minutes"], field=f"{path}.memory_ema_span_minutes"
    )
    gate = policy["loss_gate"]
    if not isinstance(gate, dict) or type(gate.get("enabled")) is not bool:
        raise ValueError(f"{path}.loss_gate.enabled must be an explicit boolean")
    enabled = gate["enabled"]
    if "max_realized_drawdown_fraction" in gate:
        cap = normalize_loss_gate_cap(gate["max_realized_drawdown_fraction"])
    elif enabled:
        raise ValueError(f"{'.'.join(LOSS_GATE_CAP_PATH)} is required when the loss gate is enabled")
    else:
        cap = None
    return span, enabled, cap


def _project_loss_gate(base, overrides):
    from .scenario_overrides import normalize_scenario_overrides

    gate_path = LOSS_GATE_CAP_PATH[:-1]
    gate = deepcopy(base)
    for selector, value in normalize_scenario_overrides(overrides).items():
        path = tuple(selector.split("."))
        if path == gate_path:
            gate = deepcopy(value)
        elif path[:-1] == gate_path:
            gate[path[-1]] = value
        elif gate_path[:len(path)] == path:
            gate = value
            for part in gate_path[len(path):]:
                gate = gate.get(part, {})
            gate = deepcopy(gate)
    gate.setdefault("enabled", False)
    if type(gate["enabled"]) is not bool:
        raise ValueError(f"{'.'.join(gate_path)}.enabled must be a boolean")
    if LOSS_GATE_CAP_PATH[-1] in gate:
        normalize_loss_gate_cap(gate[LOSS_GATE_CAP_PATH[-1]])
    elif gate["enabled"]:
        raise ValueError(f"{'.'.join(LOSS_GATE_CAP_PATH)} is required when the loss gate is enabled")
    return gate


def optimizer_loss_gate_policy(config, scenario_overrides=None):
    """Resolve authored loss policy independently of the optimizer's genome."""
    _, enabled, cap = require_loss_realization_policy(config)
    root = {"enabled": enabled}
    if cap is not None:
        root[LOSS_GATE_CAP_PATH[-1]] = cap
    fixed = _project_loss_gate(root, config.get("optimize", {}).get("fixed_runtime_overrides"))
    return _project_loss_gate(fixed, scenario_overrides) if scenario_overrides else fixed


def optimizer_loss_gate_policies(config):
    """Determine gate reachability before reading numeric bounds."""
    fixed = optimizer_loss_gate_policy(config)
    backtest = config.get("backtest", {})
    if backtest.get("suite_enabled") and backtest.get("scenarios"):
        return tuple(_project_loss_gate(fixed, scenario.get("overrides"))
                     for scenario in backtest["scenarios"])
    return (fixed,)


def unstuck_memory_required(config):
    bot = config["bot"]
    overrides = config.get("coin_overrides", {}).values()
    for side in ("long", "short"):
        risk = bot[side]["risk"]
        if risk["total_wallet_exposure_limit_ratio"] <= 0:
            continue
        if risk["unstuck"]["enabled"] or any(
            patch.get("bot", {}).get(side, {}).get("risk", {}).get("unstuck", {}).get("enabled", False)
            for patch in overrides
        ):
            return True
    return False


def risk_memory_spans(config):
    """Return the enabled equity memory spans and at most one realized span."""
    bot = config["bot"]
    hsl = bot["portfolio"]["risk"]["hsl"]
    scope = hsl["scope"]
    policies = ([hsl] if scope == "portfolio"
                else [bot[side]["risk"]["hsl"] for side in ("long", "short")])
    spans = [policy["memory_ema_span_minutes"] for policy in policies if policy["enabled"]]
    if scope == "coin":
        for patch in config.get("coin_overrides", {}).values():
            for side in ("long", "short"):
                override = patch.get("bot", {}).get(side, {}).get("risk", {}).get("hsl", {})
                policy = {**bot[side]["risk"]["hsl"], **override}
                if policy["enabled"]:
                    spans.append(policy["memory_ema_span_minutes"])
    span, gate_enabled, _ = require_loss_realization_policy(config)
    if gate_enabled or unstuck_memory_required(config):
        spans.append(span)
    return tuple(sorted(set(spans)))


def risk_bootstrap_horizon_minutes(config):
    """Return (effective horizon, truncated) without deciding history readiness."""
    span = max(risk_memory_spans(config), default=1.0)
    cap = normalize_risk_bootstrap_max_lookback_minutes(
        config["live"]["startup"]["risk_bootstrap_max_lookback_minutes"]
    )
    requested = 5.0 * span
    return min(requested, cap), requested > cap


def normalize_risk_memory_config(config):
    live = config["live"]
    span, _, cap = require_loss_realization_policy(config)
    policy = config["bot"]["portfolio"]["risk"]["loss_realization"]
    policy["memory_ema_span_minutes"] = span
    if cap is not None:
        policy["loss_gate"]["max_realized_drawdown_fraction"] = cap
    live["startup"]["risk_bootstrap_max_lookback_minutes"] = (
        normalize_risk_bootstrap_max_lookback_minutes(
            live["startup"]["risk_bootstrap_max_lookback_minutes"]
        )
    )


# Explicit v8 conversion retains its old command argument name. Runtime consumers
# use the generic policy-local helper above. This is not a config-path alias.
normalize_lookback_ema_span_minutes = normalize_memory_ema_span_minutes
