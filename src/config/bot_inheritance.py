"""Pure, presence-aware bot policy resolution.

Authored policy and effective values are separate projections.  The immutable
internal objects here are not configuration metadata and cannot be constructed
by loading JSON.  Source resolution, field-domain validation and trading remain
with their existing owners; this module never opens a source file.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Any, Mapping

from .field_spec import FieldRegistry, FieldSpec, FieldType
from .universe import ResolvedCoinList


SIDES = ("long", "short")
SOURCE_FIELDS = frozenset({("universe", "approved_coins"), ("universe", "ignored_coins")})
Path = tuple[str, ...]


class _Absent(Enum):
    VALUE = "absent"


ABSENT = _Absent.VALUE


@dataclass(frozen=True)
class _Object:
    items: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class _Array:
    items: tuple[Any, ...]


def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("bot policy object keys must be strings")
        return _Object(tuple((key, _freeze(child)) for key, child in value.items()))
    if isinstance(value, (list, tuple)):
        return _Array(tuple(_freeze(child) for child in value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported bot policy value: {type(value).__name__}")


def _thaw(value):
    if isinstance(value, _Object):
        return {key: _thaw(child) for key, child in value.items}
    if isinstance(value, _Array):
        return [_thaw(child) for child in value.items]
    return value


def _get(node, path, default=ABSENT):
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def _set(node, path, value):
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = value


def _intersects(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


@dataclass(frozen=True)
class SideField:
    suffix: Path
    types: tuple[FieldType, ...]
    nullable: bool
    default: Any
    native_keys: tuple[str | None, str | None]
    optimizable: bool

    def default_value(self):
        return _thaw(self.default)

    def native_key(self, side):
        return self.native_keys[SIDES.index(side)]


@dataclass(frozen=True)
class SideFieldCatalog:
    fields: tuple[SideField, ...]
    portfolio_fields: tuple[FieldSpec, ...] = ()

    @classmethod
    def from_registry(cls, registry: FieldRegistry, *, common_defaults: Mapping):
        """Require deliberately supplied common defaults, never choose a side.

        Registry paths determine inheritability.  Coin-override permissions do
        not affect this catalog.  Asymmetric native/schema defaults remain the
        released converter's concern, rather than an implicit merge rule.
        """
        by_side = {side: {} for side in SIDES}
        portfolio = []
        for spec in registry:
            if "*" in spec.path or spec.path[:1] != ("bot",):
                continue
            if spec.path[1] in SIDES:
                by_side[spec.path[1]][spec.path[2:]] = spec
            elif spec.path[1] == "portfolio":
                portfolio.append(spec)
        if set(by_side["long"]) != set(by_side["short"]):
            raise ValueError("side-capable registry paths must exist for both sides")
        fields = []
        for suffix, long in sorted(by_side["long"].items()):
            short = by_side["short"][suffix]
            if (long.types, long.nullable) != (short.types, short.nullable):
                raise ValueError(f"incompatible side declarations for {'.'.join(suffix)}")
            default = _get(common_defaults, suffix)
            if default is ABSENT:
                raise ValueError(f"missing explicit common default: {'.'.join(suffix)}")
            field = SideField(suffix, long.types, long.nullable, _freeze(default),
                              (long.optimize_key, short.optimize_key),
                              bool(long.optimizable and short.optimizable))
            _validate_value(field, default, ("bot", "global", *suffix), shared=True)
            fields.append(field)
        result = cls(tuple(fields), tuple(portfolio))
        result.validate_tree(common_defaults, ("bot", "global"))
        return result

    def find(self, suffix):
        return next((field for field in self.fields if field.suffix == tuple(suffix)), None)

    def validate_tree(self, node, prefix):
        if not isinstance(node, Mapping):
            raise TypeError(f"{'.'.join(prefix)} must be an object")
        def visit(value, suffix):
            field = self.find(suffix)
            if field is not None:
                _validate_value(field, value, (*prefix, *suffix), shared=prefix[1] == "global")
                return
            children = [field for field in self.fields if field.suffix[:len(suffix)] == suffix]
            if not children:
                raise ValueError(f"unknown inherited bot field: {'.'.join((*prefix, *suffix))}")
            if not isinstance(value, Mapping):
                raise TypeError(f"{'.'.join((*prefix, *suffix))} must be an object")
            for name, child in value.items():
                if not isinstance(name, str):
                    raise TypeError("bot policy keys must be strings")
                visit(child, (*suffix, name))
        for name, child in node.items():
            visit(child, (name,))


def _validate_value(field, value, path, *, shared=False):
    name = ".".join(path)
    if value is None:
        if not field.nullable or shared and field.suffix == ("universe", "approved_coins"):
            raise TypeError(f"{name} does not accept null")
        return
    valid = (
        FieldType.BOOLEAN in field.types and type(value) is bool
        or FieldType.FLOAT in field.types and type(value) in (int, float) and math.isfinite(value)
        # Existing schema/native defaults may spell an integral value as 7.0.
        # Export typing remains the serialization owner's responsibility.
        or FieldType.INTEGER in field.types and type(value) in (int, float)
        and math.isfinite(value) and float(value).is_integer()
        or FieldType.STRING in field.types and isinstance(value, str)
        or FieldType.ARRAY in field.types and isinstance(value, list)
        or FieldType.OBJECT in field.types and isinstance(value, Mapping)
    )
    if not valid:
        raise TypeError(f"{name} does not match its declared serialization type")
    if field.suffix in SOURCE_FIELDS or field.suffix == ("forager", "always_include"):
        if isinstance(value, list) and any(not isinstance(item, str) for item in value):
            raise TypeError(f"{name} must contain coin identifiers")


@dataclass(frozen=True)
class LeafOrigin:
    target: Path
    source: Path | None
    kind: str
    declaration: Path | None = None


@dataclass(frozen=True)
class _PreparedSource:
    path: Path
    members: tuple[str, ...]
    source: Any
    shared_source: Any
    source_present: bool = True


@dataclass(frozen=True)
class _DerivedValue:
    path: Path
    value: Any
    owner: str


@dataclass(frozen=True)
class BotPolicy:
    """Trusted internal policy; each public projection is an independent copy."""
    catalog: SideFieldCatalog
    _source: _Object
    _prepared: tuple[_PreparedSource, ...] = ()
    _derived: tuple[_DerivedValue, ...] = ()
    _genes: frozenset[Path] = frozenset()

    def source_view(self, *, include_defaults=False):
        result = _thaw(self._source)
        if include_defaults:
            common = {}
            for field in self.catalog.fields:
                _set(common, field.suffix, field.default_value())
            authored = result.get("global", {})
            for field in self.catalog.fields:
                value = _get(authored, field.suffix)
                if value is not ABSENT:
                    _set(common, field.suffix, deepcopy(value))
            result["global"] = common
        return result

    def has_side_pin(self, side, suffix):
        value = _get(self.source_view(), (side, *suffix))
        return value is not ABSENT and not (suffix == ("universe", "approved_coins") and value is None)

    def resolve(self):
        source = self.source_view()
        common = source.get("global", {})
        result = {"portfolio": deepcopy(source.get("portfolio", {})), "global": {}, "long": {}, "short": {}}
        origins = []
        for field in self.catalog.fields:
            shared = _get(common, field.suffix)
            shared_path = ("bot", "global", *field.suffix)
            shared_kind = "schema_default" if shared is ABSENT else "global"
            if shared is ABSENT:
                shared = field.default_value()
            elif field.suffix == ("universe", "ignored_coins") and shared is None:
                shared = []
            _set(result["global"], field.suffix, deepcopy(shared))
            for side in SIDES:
                path = ("bot", side, *field.suffix)
                value = _get(source.get(side, {}), field.suffix)
                declaration = path if value is not ABSENT else None
                inherited = value is ABSENT or field.suffix == ("universe", "approved_coins") and value is None
                owner = shared_path if inherited and shared_kind != "schema_default" else None if inherited else path
                kind = shared_kind if inherited else "side"
                if inherited:
                    value = deepcopy(shared)
                elif field.suffix == ("universe", "ignored_coins") and value is None:
                    value = []
                for prepared in self._prepared:
                    if prepared.path != path:
                        continue
                    authored = _get(source, (side, *field.suffix))
                    expected = _thaw(prepared.source) if prepared.source_present else ABSENT
                    if authored == expected and (not inherited or shared == _thaw(prepared.shared_source)):
                        value = ResolvedCoinList(prepared.members, _thaw(prepared.source), _thaw(prepared.shared_source))
                _set(result[side], field.suffix, value)
                origins.append(LeafOrigin(path, owner, "gene" if owner in self._genes else kind, declaration))
        for derived in self._derived:
            _set(result, derived.path[1:], _thaw(derived.value))
            origins = [origin for origin in origins if origin.target != derived.path]
            origins.append(LeafOrigin(derived.path, None, derived.owner))
        return result, tuple(origins)

    def effective_view(self):
        return self.resolve()[0]

    def candidate_view(self):
        """Materialize only derived result leaves needed by ordinary replay.

        These explicit result pins are a candidate snapshot, not the original
        authored mask. Resume owners retain source and validated bindings apart.
        """
        result = self.source_view(include_defaults=True)
        for derived in self._derived:
            _set(result, derived.path[1:], _thaw(derived.value))
        return result


def resolve_bot_policy(source: Mapping, catalog: SideFieldCatalog) -> BotPolicy:
    if not isinstance(source, Mapping) or set(source) - {"portfolio", "global", *SIDES}:
        raise ValueError("bot policy accepts only portfolio/global/long/short scopes")
    source = deepcopy(source)
    if "portfolio" in source and not isinstance(source["portfolio"], Mapping):
        raise TypeError("bot.portfolio must be an object")
    prepared = []
    for scope in ("global", *SIDES):
        values = source.get(scope, {})
        catalog.validate_tree(values, ("bot", scope))
        for suffix in SOURCE_FIELDS:
            value = _get(values, suffix)
            if type(value) is ResolvedCoinList:
                if scope == "global":
                    raise TypeError("prepared source membership belongs to a side")
                prepared.append(_PreparedSource(("bot", scope, *suffix), tuple(value), _freeze(value.source), _freeze(value.shared_source)))
                _set(values, suffix, deepcopy(value.source))
    return BotPolicy(catalog, _freeze(source), tuple(prepared))


class PreparedBotPolicy(dict):
    """Runtime bot view carrying one non-JSON, typed source authority."""

    def __init__(self, values: Mapping, policy: BotPolicy):
        if type(policy) is not BotPolicy:
            raise TypeError("a typed BotPolicy is required")
        super().__init__(deepcopy(values))
        self._bot_policy = policy


def get_bot_policy(config: Mapping) -> BotPolicy | None:
    bot = config.get("bot")
    if type(bot) is not PreparedBotPolicy:
        return None
    policy = getattr(bot, "_bot_policy", None)
    if type(policy) is not BotPolicy:
        raise TypeError("prepared bot policy has invalid source authority")
    return policy


def install_bot_policy(config: dict, policy: BotPolicy, *, effective_bot=None) -> None:
    """Install a defensive runtime projection after its owning preparation."""
    if type(policy) is not BotPolicy:
        raise TypeError("a typed BotPolicy is required")
    config["bot"] = PreparedBotPolicy(
        policy.effective_view() if effective_bot is None else effective_bot, policy,
    )


def bot_policy_source_view(config: Mapping, *, include_defaults=False, candidate=False):
    """Project trusted source, or the explicitly materialized candidate output.

    Callers apply declared changes through ``apply_bot_policy_patch`` before
    exporting. Plain JSON fields never become a retained internal snapshot.
    """
    result = deepcopy(dict(config))
    policy = get_bot_policy(config)
    if policy is not None:
        result["bot"] = (policy.candidate_view() if candidate else
                         policy.source_view(include_defaults=include_defaults))
    return result


def with_prepared_bot_sources(policy: BotPolicy, prepared_bot: Mapping) -> BotPolicy:
    """Capture only exact prepared source objects, preserving raw omission.

    The universe owner resolves each field's shared binding before this call.
    In particular an inherited ignored list uses the ignored shared source,
    rather than the approved source. No file or metadata value is consulted.
    """
    if type(policy) is not BotPolicy:
        raise TypeError("a typed BotPolicy is required")
    source = policy.source_view()
    prepared = {entry.path: entry for entry in policy._prepared}
    for side in SIDES:
        for suffix in SOURCE_FIELDS:
            path = ("bot", side, *suffix)
            prepared.pop(path, None)
            value = _get(prepared_bot, (side, *suffix))
            if type(value) is not ResolvedCoinList:
                continue
            raw = _get(source, (side, *suffix))
            present = raw is not ABSENT
            inherited = not present or suffix[-1] == "approved_coins" and raw is None
            expected_source = raw if present else None
            field = policy.catalog.find(suffix)
            shared = _get(source, ("global", *suffix), field.default_value())
            if suffix[-1] == "ignored_coins" and shared is None:
                shared = []
            if value.source != expected_source or inherited and value.shared_source != shared:
                raise ValueError(f"prepared membership does not match {'.'.join(('bot', side, *suffix))} source")
            prepared[path] = _PreparedSource(
                path, tuple(value), _freeze(expected_source), _freeze(shared), present,
            )
    return replace(policy, _prepared=tuple(prepared.values()))


def apply_bot_policy_patch(policy: BotPolicy, replacements: Mapping[str, Any], *, channel="authored") -> BotPolicy:
    """Apply declared replacements, retaining atomic object replacement semantics.

    Gene writes occur at the binding's authoring path. Derived writes affect
    only effective values and never create authored side pins or search axes.
    """
    if type(policy) is not BotPolicy:
        raise TypeError("a typed BotPolicy is required")
    if channel not in {"authored", "gene", "derived"}:
        raise ValueError("unknown bot policy patch channel")
    paths = {}
    for selector, value in replacements.items():
        if not isinstance(selector, str):
            raise TypeError("bot policy selectors must be dotted strings")
        path = tuple(selector.split("."))
        if len(path) < 2 or path[0] != "bot" or path[1] not in {"portfolio", "global", *SIDES} or any(not part for part in path):
            raise ValueError(f"unsupported bot policy selector: {selector}")
        if any(_intersects(path, other) for other in paths):
            raise ValueError("overlapping bot policy replacements")
        paths[path] = value
    source = policy.source_view()
    prepared = policy._prepared
    # Derived values must be recalculated by their owner after source/gene edits.
    derived = list(policy._derived) if channel == "derived" else []
    genes = set(policy._genes)
    for path, value in paths.items():
        if path[1] == "portfolio":
            spec = next((spec for spec in policy.catalog.portfolio_fields if spec.path == path), None)
            field = None if spec is None else SideField(spec.path[2:], spec.types, spec.nullable,
                                                        None, (None, None), bool(spec.optimizable))
        else:
            field = policy.catalog.find(path[2:])
        if channel in {"gene", "derived"} and field is None:
            raise ValueError("gene and derived writes must target registered leaves")
        if channel == "gene" and (not field.optimizable or not any(
            kind in field.types for kind in (FieldType.FLOAT, FieldType.INTEGER)
        )):
            raise ValueError("gene writes must target optimizer-eligible numeric leaves")
        if channel == "derived":
            if path[1] not in SIDES or field.suffix in SOURCE_FIELDS:
                raise ValueError("derived writes cannot replace source authority")
            _validate_value(field, value, path)
            derived = [entry for entry in derived if entry.path != path]
            derived.append(_DerivedValue(path, _freeze(value), "derived"))
            continue
        if field is None:
            if path[1] == "portfolio":
                raise ValueError("portfolio object replacements belong to their owning validator")
            children = [entry for entry in policy.catalog.fields if entry.suffix[:len(path[2:])] == path[2:]]
            if not children:
                raise ValueError(f"unknown bot policy selector: {'.'.join(path)}")
            # General parent validation stays with the existing patch owner;
            # do not require unrelated/inactive registry descendants here.
            # Weight objects already have an atomic completeness contract.
            weights_path = ("forager", "score_weights")
            if weights_path[:len(path[2:])] == path[2:]:
                weights = _get(value, weights_path[len(path[2:]):])
                if weights is not ABSENT:
                    from .bot import normalize_forager_score_weights
                    normalize_forager_score_weights(
                        weights, path=".".join(("bot", path[1], *weights_path)),
                    )
        _set(source, path[1:], deepcopy(value))
        prepared = tuple(entry for entry in prepared if not _intersects(path, entry.path)
                         and not (path[:2] == ("bot", "global") and _intersects(path[2:], entry.path[2:])))
        derived = [entry for entry in derived if not _intersects(path, entry.path)]
        if channel == "gene":
            genes.add(path)
        else:
            genes = {entry for entry in genes if not _intersects(path, entry)}
    result = resolve_bot_policy(source, policy.catalog)
    return replace(result, _prepared=prepared, _derived=tuple(derived), _genes=frozenset(genes))


def resolve_coin_bot_policy(policy: BotPolicy, patch: Mapping, *, coin: str,
                            strategy_kind: str, hsl_signal_mode="coin"):
    """Apply only permissions granted by the existing coin-patch owner.

    File materialization precedes this operation. Operational live patches have
    their own consumer; this function projects only the bot policy and origins.
    """
    if type(policy) is not BotPolicy:
        raise TypeError("a typed BotPolicy is required")
    if patch.get("override_config_path"):
        raise ValueError("coin bot resolution requires materialized override files")
    from .overrides import _extract_allowed_patch, nested_update
    allowed = _extract_allowed_patch(
        patch, coin=coin, source=f"coin_overrides.{coin}", strict=True,
        strategy_kind=strategy_kind, hsl_signal_mode=hsl_signal_mode,
    )
    effective, origins = policy.resolve()
    origins = list(origins)
    for side, values in allowed.get("bot", {}).items():
        if side not in SIDES:
            continue
        def visit(node, suffix=()):
            for key, value in node.items():
                path = (*suffix, key)
                if isinstance(value, Mapping):
                    visit(value, path)
                else:
                    field = policy.catalog.find(path)
                    target = ("bot", side, *path)
                    if field is not None:
                        _validate_value(field, value, target)
                    origins[:] = [origin for origin in origins if origin.target != target]
                    origins.append(LeafOrigin(target, ("coin_overrides", coin, *target), "coin"))
        visit(values)
        nested_update(effective[side], deepcopy(values))
    return effective, tuple(origins)
