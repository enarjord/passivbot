"""One optimizer coordinate per authored owner, with explicit effective fanout.

This pure plan is built before bound/value hydration. Recommended search ranges
never supply missing authoring authority. Native parameter identifiers remain
the registry's existing identifiers, not new public config aliases.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Mapping, Sequence

from optimization.bounds import Bound
from .bot_inheritance import ABSENT, SIDES, BotPolicy, Path, _get, apply_bot_policy_patch
from .field_spec import FieldType


@dataclass(frozen=True)
class GeneBinding:
    authoring_path: Path
    bound_path: Path
    bound: Bound
    effective_targets: tuple[Path, ...]
    native_ids: tuple[str, ...]
    source: str

    @property
    def key(self):
        return ".".join(self.authoring_path)


def _remove(node, path):
    for part in path[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(path[-1], None)


def _without_empty_objects(node):
    if isinstance(node, dict):
        return {key: selected for key, value in node.items()
                if (selected := _without_empty_objects(value)) != {}}
    return node


def _source_signature(policy, bindings):
    source = policy.source_view(include_defaults=True)
    effective = replace(policy, _derived=()).effective_view()
    for binding in bindings:
        _remove(source, binding.authoring_path[1:])
    return json.dumps({
        "policy": _without_empty_objects(source),
        "prepared_membership": [
            (entry.path, entry.members, entry.source_present) for entry in policy._prepared
        ],
        "omitted_fixed_values": [
            (binding.authoring_path, _get(effective, binding.authoring_path[1:]))
            for binding in bindings if binding.source.startswith("omitted_fixed_")
        ],
    }, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class BoundPlan:
    bindings: tuple[GeneBinding, ...]
    unused_bounds: tuple[Path, ...]
    source_signature: str
    strategy_kind: str
    hsl_scope: str

    def identity(self):
        """Capture coupling/pin authority even when current values are equal."""
        payload = {
            "strategy_kind": self.strategy_kind,
            "hsl_scope": self.hsl_scope,
            "source": self.source_signature,
            "bindings": [
                (binding.authoring_path, binding.bound_path,
                 (binding.bound.low, binding.bound.high, binding.bound.step),
                 binding.effective_targets, binding.native_ids, binding.source)
                for binding in self.bindings
            ],
            "unused_bounds": self.unused_bounds,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def apply_vector(self, policy: BotPolicy, vector: Sequence[float]):
        if type(policy) is not BotPolicy:
            raise TypeError("a typed BotPolicy is required")
        if len(vector) != len(self.bindings):
            raise ValueError("optimizer vector length differs from its inheritance plan")
        if _source_signature(policy, self.bindings) != self.source_signature:
            raise ValueError("optimizer binding plan does not match source authority")
        values = {}
        for binding, value in zip(self.bindings, vector):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise TypeError(f"{binding.key} optimizer value must be finite numeric")
            if not binding.bound.low <= value <= binding.bound.high:
                raise ValueError(f"{binding.key} optimizer value is outside its declared bound")
            values[binding.key] = value
        return apply_bot_policy_patch(policy, values, channel="gene")


def _bound(value, path):
    values = value if isinstance(value, (tuple, list)) else (value,)
    if not values or any(isinstance(number, bool) or not isinstance(number, (int, float))
                         or not math.isfinite(number) for number in values if number is not None):
        raise TypeError(f"{'.'.join(path)} requires finite numeric bound values")
    if any(number is None for number in values[:2]):
        raise TypeError(f"{'.'.join(path)} bound endpoints cannot be null")
    return Bound.from_config(".".join(path), value)


def build_bound_plan(
    policy: BotPolicy, authored_bounds: Mapping, *, strategy_kind: str, hsl_scope: str,
) -> BoundPlan:
    if type(policy) is not BotPolicy:
        raise TypeError("a typed BotPolicy is required")
    if not isinstance(authored_bounds, Mapping) or set(authored_bounds) - {"global", "portfolio", *SIDES}:
        raise ValueError("optimizer bounds accept only portfolio/global/long/short scopes")
    if hsl_scope not in {"coin", "position_side", "portfolio"}:
        raise ValueError("an explicit canonical HSL scope is required")
    fields = {field.suffix: field for field in policy.catalog.fields}
    portfolio = {spec.path[2:]: spec for spec in policy.catalog.portfolio_fields}

    def active_hsl(scope, suffix):
        return suffix[:1] != ("hsl",) or (scope == "portfolio") == (hsl_scope == "portfolio")

    def check(node, prefix):
        if not isinstance(node, Mapping):
            raise TypeError(f"{'.'.join(prefix)} must be an object")
        def visit(value, suffix):
            field = (portfolio if prefix[-1] == "portfolio" else fields).get(suffix)
            if field is not None:
                if not active_hsl(prefix[-1], suffix):
                    raise ValueError(f"{'.'.join((*prefix, *suffix))} is inactive in {hsl_scope} HSL scope")
                numeric = FieldType.FLOAT in field.types or FieldType.INTEGER in field.types
                if not numeric or not field.optimizable:
                    raise ValueError(f"{'.'.join((*prefix, *suffix))} is not optimizer eligible")
                _bound(value, (*prefix, *suffix))
                return
            if not isinstance(value, Mapping) or not any(
                path[:len(suffix)] == suffix for path in (portfolio if prefix[-1] == "portfolio" else fields)
            ):
                raise ValueError(f"unknown optimizer field: {'.'.join((*prefix, *suffix))}")
            for key, child in value.items():
                visit(child, (*suffix, key))
        for key, value in node.items():
            visit(value, (key,))
    for scope, values in authored_bounds.items():
        check(values, ("optimize", "bounds", scope))

    # Transforms own an effective overlay, not the authored fixed values or
    # search authority from which this plan is built.
    effective = replace(policy, _derived=()).effective_view()
    bindings, unused = [], []
    for field in policy.catalog.fields:
        if (not field.optimizable or not active_hsl("global", field.suffix)
                or field.suffix[:1] == ("strategy",) and field.suffix[1] != strategy_kind):
            continue
        shared_bound_path = ("optimize", "bounds", "global", *field.suffix)
        shared_bound = _get(authored_bounds.get("global", {}), field.suffix)
        inherited = []
        for side in SIDES:
            side_bound = _get(authored_bounds.get(side, {}), field.suffix)
            target = ("bot", side, *field.suffix)
            if side_bound is not ABSENT:
                bound = _bound(side_bound, ("optimize", "bounds", side, *field.suffix))
                owner = "authored_side_bound"
            elif policy.has_side_pin(side, field.suffix):
                value = _get(effective[side], field.suffix)
                if value is None:
                    continue  # Nullable policy joins a numeric genome only with an explicit domain.
                bound = _bound(value, target)
                owner = "omitted_fixed_side_pin"
            else:
                inherited.append(side)
                continue
            native_id = field.native_key(side)
            if native_id is None:
                raise ValueError(f"missing native optimizer binding for {target}")
            bindings.append(GeneBinding(target, ("optimize", "bounds", side, *field.suffix),
                                        bound, (target,), (native_id,), owner))
        if inherited:
            value = _get(effective["global"], field.suffix)
            if shared_bound is ABSENT and value is None:
                continue
            bound = _bound(value if shared_bound is ABSENT else shared_bound, shared_bound_path)
            ids = tuple(field.native_key(side) for side in inherited)
            if None in ids:
                raise ValueError(f"missing native optimizer binding for {field.suffix}")
            bindings.append(GeneBinding(("bot", "global", *field.suffix), shared_bound_path, bound,
                                        tuple(("bot", side, *field.suffix) for side in inherited), ids,
                                        "omitted_fixed_global" if shared_bound is ABSENT else "authored_global_bound"))
        elif shared_bound is not ABSENT:
            unused.append(shared_bound_path)
    for suffix, spec in portfolio.items():
        if not spec.optimizable or not active_hsl("portfolio", suffix):
            continue
        value = _get(authored_bounds.get("portfolio", {}), suffix)
        bound_path = ("optimize", "bounds", "portfolio", *suffix)
        if value is ABSENT:
            value = _get(effective["portfolio"], suffix)
            if value is ABSENT or value is None:
                continue
            owner = "omitted_fixed_portfolio"
        else:
            owner = "authored_portfolio_bound"
        if spec.optimize_key is None:
            raise ValueError(f"missing native optimizer binding for {spec.path}")
        bindings.append(GeneBinding(spec.path, bound_path, _bound(value, bound_path),
                                    (spec.path,), (spec.optimize_key,), owner))
    bindings = tuple(sorted(bindings, key=lambda binding: binding.authoring_path))
    return BoundPlan(
        bindings, tuple(sorted(unused)), _source_signature(policy, bindings),
        strategy_kind, hsl_scope,
    )
