"""Strict directional permission to increase exposure, independent of capacity."""

from copy import deepcopy
from dataclasses import dataclass, replace

from .access import require_config_value

POSITION_SIDES = ("long", "short")
ACTIVATION_PATHS = tuple(("bot", owner, "enabled") for owner in ("global", *POSITION_SIDES))


@dataclass(frozen=True)
class ActivationSource:
    global_enabled: bool | None
    long: bool | None
    short: bool | None

    def effective(self, owner):
        default = False if self.global_enabled is None else self.global_enabled
        pin = default if owner == "global" else getattr(self, owner)
        return default if pin is None else pin


class PreparedActivationBot(dict):
    """Effective bot values plus immutable, non-serialized activation declarations."""

    __slots__ = ("_activation_source",)

    def __init__(self, values, source):
        super().__init__(values)
        self._activation_source = source


def _authored_flag(bot, owner):
    node = bot.get(owner, {})
    if not isinstance(node, dict):
        raise TypeError(f"bot.{owner} must be a mapping")
    if "enabled" not in node:
        return None
    value = node["enabled"]
    if type(value) is not bool:
        raise TypeError(f"bot.{owner}.enabled must be a boolean")
    return value


def get_activation_source(config):
    bot = config.get("bot", {})
    if not isinstance(bot, dict):
        raise TypeError("bot must be a mapping")
    if type(bot) is PreparedActivationBot and type(bot._activation_source) is ActivationSource:
        source = bot._activation_source
        # Preserve ordinary direct value edits when a prepared config is copied
        # and re-prepared. Equal-value authored declarations use the patch API.
        updates = {}
        for owner, field in (("global", "global_enabled"), ("long", "long"), ("short", "short")):
            current = _authored_flag(bot, owner)
            if current is None or current != source.effective(owner):
                updates[field] = current
        return replace(source, **updates)
    return ActivationSource(_authored_flag(bot, "global"),
                            _authored_flag(bot, "long"), _authored_flag(bot, "short"))


def activation_source_view(config):
    """Project actual declarations, keeping omission distinct from explicit false."""
    source = get_activation_source(config)
    result = deepcopy(dict(config))
    bot = dict(result.get("bot", {}))
    for owner, value in (("global", source.global_enabled), ("long", source.long), ("short", source.short)):
        node = bot.get(owner)
        if node is None:
            continue
        if value is None:
            node.pop("enabled", None)
        else:
            node["enabled"] = value
    if "bot" in result:
        result["bot"] = bot
    return result


def resolve_side_activation(config, source=None, *, tracker=None):
    source = get_activation_source(config) if source is None else source
    bot = dict(config.setdefault("bot", {}))
    for owner in ("global", *POSITION_SIDES):
        node = bot.setdefault(owner, {})
        value = source.effective(owner)
        if tracker is not None and "enabled" not in node:
            tracker.add(["bot", owner, "enabled"], value)
        node["enabled"] = value
    config["bot"] = PreparedActivationBot(bot, source)
    return config


def apply_activation_patch(config, overrides):
    """Retain activation declarations after the real fixed/scenario patch owner.

    Callers supply their validated canonical selectors. Parent replacement and
    equal-value false declarations carry the same authority as leaf edits.
    """
    relevant = []
    for selector, value in overrides.items():
        if isinstance(selector, str):
            path = tuple(selector.split("."))
            if any(target[:len(path)] == path for target in ACTIVATION_PATHS):
                relevant.append((path, value))
    if not relevant:
        return
    source = get_activation_source(config)
    updates = {}
    for path, value in relevant:
        for target in ACTIVATION_PATHS:
            if target[:len(path)] != path:
                continue
            projected = value
            for part in target[len(path):]:
                if not isinstance(projected, dict) or part not in projected:
                    projected = None
                    break
                projected = projected[part]
            if projected is not None and type(projected) is not bool:
                raise TypeError(f"{'.'.join(target)} must be a boolean")
            # A null leaf is invalid; only absence in a parent replacement
            # declares inheritance.
            if len(path) == len(target) and projected is None:
                raise TypeError(f"{'.'.join(target)} must be a boolean")
            updates["global_enabled" if target[1] == "global" else target[1]] = projected
    resolve_side_activation(config, replace(source, **updates))


def validate_side_activation(value):
    """Validate the complete native policy pair without truthiness coercion."""
    if not isinstance(value, dict) or set(value) != set(POSITION_SIDES):
        raise TypeError("side_activation must contain exactly long and short booleans")
    for side in POSITION_SIDES:
        if type(value[side]) is not bool:
            raise TypeError(f"side_activation.{side} must be a boolean")
    return {side: value[side] for side in POSITION_SIDES}


def side_enabled(config, side):
    """Read the canonical resolved permission, never infer it from sizing or mode."""
    if side not in POSITION_SIDES:
        raise ValueError(f"invalid position side: {side!r}")
    value = require_config_value(config, f"bot.{side}.enabled")
    if type(value) is not bool:
        raise TypeError(f"bot.{side}.enabled must be a boolean")
    return value


def side_activation(config):
    return {side: side_enabled(config, side) for side in POSITION_SIDES}


def optimizer_side_may_activate(config, side):
    """Boolean reachability; numeric search ranges never grant permission.

    Scenario declarations are already parsed into the optimizer run contract.
    Only activation selectors are projected here, without resolving coin files
    or changing the base policy seen by scenarios that omit a declaration.
    """
    effective = deepcopy(config)
    apply_activation_patch(effective, config.get("optimize", {}).get("fixed_runtime_overrides", {}))
    backtest = config.get("backtest", {})
    scenarios = backtest.get("scenarios") or []
    if backtest.get("suite_enabled") and scenarios:
        from .scenario_overrides import normalize_scenario_overrides

        return any(
            _scenario_side_enabled(effective, normalize_scenario_overrides(scenario.get("overrides", {})), side)
            for scenario in scenarios
        )
    return side_enabled(effective, side)


def _scenario_side_enabled(config, overrides, side):
    candidate = deepcopy(config)
    apply_activation_patch(candidate, overrides)
    return side_enabled(candidate, side)
