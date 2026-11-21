"""Translate released-v8 risk policy after frozen legacy normalization."""

from copy import deepcopy
import math
import re

from config.risk_memory import LOSS_REALIZATION_PATH


_GATE_CAP = "max_realized_drawdown_fraction"
_SCOPES = ("global", "long", "short", "portfolio")
_SCOPE_PATH = re.compile(r"(^|\.)(long|short|global|portfolio|\*)\.(hsl|unstuck)(?=\.|$)")
_OLD_GATE_CAP = ".".join(("bot", "portfolio", "loss_gate", "max_drawdown_fraction"))
_OLD_MEMORY = (
    "bot.portfolio.unstuck.memory_ema_span_minutes",
    "bot.portfolio.loss_gate.memory_ema_span_minutes",
)


def released_risk_path(path):
    if path in _OLD_MEMORY:
        return ".".join((*LOSS_REALIZATION_PATH, "memory_ema_span_minutes"))
    return _SCOPE_PATH.sub(r"\1\2.risk.\3", path)


def _gate(fraction, path):
    if isinstance(fraction, bool):
        raise ValueError(f"{path}: expected a finite nonnegative v8 gate fraction")
    try:
        fraction = float(fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{path}: expected a finite nonnegative v8 gate fraction") from exc
    if not math.isfinite(fraction) or fraction < 0:
        raise ValueError(f"{path}: expected a finite nonnegative v8 gate fraction")
    return {"enabled": True, _GATE_CAP: fraction} if fraction < 1 else {"enabled": False}


def _close_enabled(number, path):
    if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or number < 0:
        raise ValueError(f"{path}: choose an explicit v9 unstuck enabled policy")
    return number > 0


def _discard_size_bound(value, path, record):
    endpoints = value[:2] if isinstance(value, (list, tuple)) else [value]
    if not endpoints:
        raise ValueError(f"{path}: empty released unstuck size bound")
    permissions = {_close_enabled(endpoint, path) for endpoint in endpoints}
    if len(permissions) != 1:
        raise ValueError(f"{path} searched both disabled and enabled unstuck sizing. Choose an explicit unstuck.enabled policy and remove the size bound.")
    enabled = permissions.pop()
    record(path, value, None, "discarded the retired unstuck size search coordinate")
    return enabled


def finalize_risk_report(changes):
    """Keep authored source names and report the final public target paths."""
    for change in changes:
        new = change.get("new")
        if not isinstance(new, dict):
            continue
        projected = {}
        for path, value in new.items():
            if path in {_OLD_GATE_CAP, _OLD_GATE_CAP.removesuffix("_fraction") + "_ratio"}:
                projected.update({
                    ".".join((*LOSS_REALIZATION_PATH, "loss_gate", leaf)): resolved
                    for leaf, resolved in _gate(value, path).items()
                })
            else:
                projected[released_risk_path(path) if isinstance(path, str) else path] = value
        change["new"] = projected


def finalize_released_risk(config, record):
    root_bot = config.get("bot", {})
    root_positive_size = {}
    for side in ("long", "short"):
        old_policy = root_bot.get(side, {}).get("unstuck", {})
        if isinstance(old_policy, dict) and "close_fraction" in old_policy:
            source = f"bot.{side}.unstuck.close_fraction"
            root_positive_size[side] = _close_enabled(old_policy["close_fraction"], source)

    def normalize_tree(node, path, *, bounds=False):
        if not isinstance(node, dict):
            return
        for scope in _SCOPES:
            owner = node.get(scope)
            if not isinstance(owner, dict):
                continue
            risk = owner.get("risk")
            if not isinstance(risk, dict):
                if "risk" in owner:
                    continue  # Retain malformed authority for schema validation.
                risk = {}
            for group in ("hsl", "unstuck"):
                if group not in owner or (scope == "portfolio" and group == "unstuck"):
                    continue
                old = owner.pop(group)
                if group in risk and isinstance(risk[group], dict) and isinstance(old, dict):
                    overlapping = risk[group].keys() & old.keys()
                    if any(risk[group][key] != old[key] for key in overlapping):
                        raise ValueError(f"{path}.{scope}.{group} conflicts with risk.{group}")
                    old = {**risk[group], **old}
                risk[group] = old
                if group == "unstuck" and isinstance(old, dict):
                    convert_size(old, f"{path}.{scope}.unstuck", bounds=bounds)
                record(f"{path}.{scope}.{group}", old, {f"risk.{group}": old},
                       "grouped the directional or portfolio risk policy")
            if scope == "portfolio":
                convert_portfolio(owner, risk, f"{path}.portfolio", bounds=bounds)
            if risk:
                owner["risk"] = risk

    def convert_size(block, path, *, bounds):
        if "close_fraction" not in block:
            return
        value = block.pop("close_fraction")
        source = f"{path}.close_fraction"
        if bounds:
            enabled = _discard_size_bound(value, source, record)
            side = path.split(".")[-2]
            policy = config.get("bot", {}).get(side, {}).get("risk", {}).get("unstuck", {})
            if not enabled and policy.get("enabled", True):
                raise ValueError(f"{source} disagrees with unstuck enablement. Choose an explicit policy and remove the size bound.")
            return
        enabled = _close_enabled(value, source)
        if not enabled:
            block["enabled"] = False
        replacement = {"enabled": block["enabled"]} if "enabled" in block else None
        record(source, value, replacement, "retired quantity sizing and preserved effective unstuck permission")

    def convert_portfolio(owner, risk, path, *, bounds):
        old_unstuck = owner.pop("unstuck", {})
        old_gate = owner.pop("loss_gate", {})
        if not old_unstuck and not old_gate:
            return
        memory = risk.setdefault("loss_realization", {})
        spans = [block["memory_ema_span_minutes"] for block in (old_unstuck, old_gate)
                 if "memory_ema_span_minutes" in block]
        if spans:
            if any(span != spans[0] for span in spans[1:]):
                raise ValueError(f"{path}: choose one explicit realized-loss memory EMA span")
            memory["memory_ema_span_minutes"] = spans[0]
        if bounds and "max_drawdown_fraction" in old_gate:
            raise ValueError(f"{path}: choose new loss gate bounds in [0, 1) explicitly")
        if "max_drawdown_fraction" in old_gate:
            source = f"{path}.loss_gate.max_drawdown_fraction"
            value = old_gate["max_drawdown_fraction"]
            memory["loss_gate"] = _gate(value, source)
            record(source, value, memory["loss_gate"], "explicit portfolio gate enablement and cap")

    def normalize_patch(patch, path):
        if not isinstance(patch, dict):
            return
        converted = {}
        replaced_permissions = {}
        for selector, value in patch.items():
            source = f"{path}.{selector}"
            parts = selector.split(".")
            if selector == _OLD_GATE_CAP:
                gate = _gate(value, source)
                target = {".".join((*LOSS_REALIZATION_PATH, "loss_gate", key)): item
                          for key, item in gate.items()}
                converted.update(target)
                record(source, value, target, "explicit portfolio gate enablement and cap")
                continue
            if parts[-2:] == ["unstuck", "close_fraction"]:
                flag = ".".join((*parts[:-1], "enabled"))
                target = released_risk_path(flag)
                positive = _close_enabled(value, source)
                if positive and parts[:1] == ["bot"] and root_positive_size.get(parts[1]) is False:
                    if flag not in patch:
                        raise ValueError(
                            f"{source} overrides disabled v8 unstuck sizing. "
                            "Choose an explicit unstuck.enabled policy and remove the size override."
                        )
                if not positive:
                    replaced_permissions[target] = False
                record(source, value, {target: False} if not positive else None,
                       "retired quantity sizing and preserved effective unstuck permission")
                continue
            target = released_risk_path(selector)
            if target in converted and converted[target] != value:
                raise ValueError(f"{source} conflicts with {target}")
            converted[target] = value
        patch.clear()
        patch.update(converted)
        patch.update(replaced_permissions)

    def visit(node, path):
        if isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{path}[{index}]")
        elif isinstance(node, dict):
            for key, child in list(node.items()):
                here = f"{path}.{key}" if path else key
                if (key == "bot" and here != "bot") or here == "optimize.bounds":
                    normalize_tree(child, here, bounds=here == "optimize.bounds")
                elif key in {"overrides", "fixed_runtime_overrides"}:
                    normalize_patch(child, here)
                elif key == "fixed_params" and isinstance(child, list):
                    translated = []
                    for selector in child:
                        if isinstance(selector, str) and (
                            selector in {"close_fraction", "close_pct"}
                            or "unstuck.close_fraction" in selector or "unstuck_close_fraction" in selector
                        ):
                            record(here, selector, None, "discarded the retired unstuck size selector")
                        else:
                            translated.append(released_risk_path(selector) if isinstance(selector, str) else selector)
                    node[key] = translated
                visit(child, here)

    normalize_tree(root_bot, "bot")
    visit(config, "")
