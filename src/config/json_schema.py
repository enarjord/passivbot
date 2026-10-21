"""Generate editor assistance for canonical authored v9 JSON.

The schema never loads files, hydrates defaults, unwraps result records or grants
runtime permissions. Existing owners enforce context-dependent and trading rules.
"""

from copy import deepcopy
import re

from .field_registry import LIVE_OPERATIONAL_PATHS, get_field_registry
from .field_spec import CoinOverridePolicy, FieldType, NO_DEFAULT


def field_schema(field, *, include_default=True):
    """Project declared metadata without inferring a domain from recommendations."""
    types = ["number" if kind is FieldType.FLOAT else kind.value for kind in field.types]
    if field.nullable:
        types.append("null")
    result = {"type": types[0] if len(types) == 1 else types}
    if include_default and field.default is not NO_DEFAULT:
        result["default"] = deepcopy(field.default)
    if field.description:
        result["description"] = field.description
    if field.domain:
        domain = field.domain
        if domain.minimum is not None:
            result["exclusiveMinimum" if domain.exclusive_minimum else "minimum"] = domain.minimum
        if domain.maximum is not None:
            result["exclusiveMaximum" if domain.exclusive_maximum else "maximum"] = domain.maximum
        if domain.string_pattern is not None:
            result["pattern"] = domain.string_pattern
            if domain.choices:
                result["examples"] = [*domain.choices, *([None] if field.nullable else [])]
        elif domain.choices:
            numeric_types = [kind for kind in types if kind in {"integer", "number"}]
            if numeric_types and "string" in types and all(isinstance(choice, str) for choice in domain.choices):
                # Mixed numeric controls can name an automatic policy without
                # restricting their separate positive numeric domain to strings.
                result["anyOf"] = [{"type": numeric_types}, {"enum": list(domain.choices)}]
                if field.nullable:
                    result["anyOf"].append({"type": "null"})
            else:
                result["enum"] = [*domain.choices, *([None] if field.nullable else [])]
    annotations = {
        "unit": field.unit, "denominator": field.denominator,
        "scope": field.scope, "consumers": field.consumers,
        "optimizer-key": field.optimize_key, "optimizable": field.optimizable,
        "optimize-when": field.optimize_when,
        "recommended-bounds": field.recommended_bounds,
        "coin-override": field.coin_override.value if field.coin_override else None,
    }
    for name, value in annotations.items():
        if value is not None and value != ():
            result[f"x-passivbot-{name}"] = list(value) if isinstance(value, tuple) else value
    return result


def _insert(tree, path, field):
    for part in path:
        tree = tree.setdefault(part, {})
    tree[None] = field


def _coin_source(*, sides=True):
    types = [{"type": "string"}, {"type": "array", "items": {"type": "string"}}, {"type": "null"}]
    if sides:
        types.append({"type": "object", "properties": {
            side: _coin_source(sides=False) for side in ("long", "short")
        }, "additionalProperties": False})
    return {"anyOf": types}


def generate_schema():
    from .schema import CONFIG_SCHEMA_VERSION

    registry = get_field_registry()
    tree = {}
    for field in registry:
        if field.path[:1] != ("coin_overrides",):
            _insert(tree, field.path, field)
    definitions = {}
    root = _object_schema(tree, (), definitions)
    root["properties"]["config_version"]["const"] = CONFIG_SCHEMA_VERSION
    root["required"] = ["config_version"]
    root["properties"]["coin_overrides"] = {
        "type": "object", "additionalProperties": {"$ref": "#/$defs/coinPatch"}
    }
    root["properties"]["optimize"]["properties"]["bounds"] = _bounds_schema(registry)
    definitions["config"] = root
    definitions["coinPatch"] = _coin_patch_schema(registry)
    definitions["coinPatchWithoutHsl"] = _coin_patch_schema(registry, hsl=False)
    for mode in ("scenario", "fixed"):
        _patch_schemas(root, definitions, mode, registry)
    # Context-specific scenarios and fixed runtime policies are revalidated by
    # their owners. At the root, the mode can also restrict inline coin HSL.
    root["allOf"] = [{
        "if": {"required": ["bot"], "properties": {"bot": {
            "required": ["global"], "properties": {"global": {
                "required": ["hsl"], "properties": {"hsl": {
                    "required": ["scope"], "properties": {"scope": {"enum": ["position_side", "portfolio"]}}
                }}
            }}
        }}},
        "then": {"properties": {"coin_overrides": {
            "additionalProperties": {"$ref": "#/$defs/coinPatchWithoutHsl"}
        }}},
    }]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Passivbot authored v9 configuration",
        "description": "Canonical authored JSON, including sparse policy. Defaults are annotations, never inserted. "
        "Runtime validators own cross-field, strategy activation, native numeric, market and dataset rules. "
        "Result records, runtime metadata and configuration aliases are outside this schema.",
        "$ref": "#/$defs/config", "$defs": definitions,
    }


def _object_schema(tree, path, definitions):
    field = tree.get(None)
    result = field_schema(field) if field else {"type": "object"}
    children = {name: node for name, node in tree.items() if name is not None}
    if (len(path) == 4 and path[:1] == ("bot",) and path[1] in {"long", "short"}
            and path[2] == "universe" and path[3] in {"approved_coins", "ignored_coins"}):
        return {**result, **_coin_source(sides=False)}
    if path in {
                ("backtest", "scenarios", "*", "coins"),
                ("backtest", "scenarios", "*", "ignored_coins")}:
        # The registry declares the union; this structural projection closes
        # its long/short object and declares array element types.
        return {**result, **_coin_source()}
    if path == ("optimize", "fixed_runtime_overrides"):
        return {"$ref": "#/$defs/fixedPatch"}
    if path == ("backtest", "scenarios", "*", "overrides"):
        return {"$ref": "#/$defs/scenarioPatch"}
    if field and FieldType.ARRAY in field.types:
        if "*" in children:
            result["items"] = _object_schema(children.pop("*"), (*path, "*"), definitions)
        else:
            result["items"] = {"type": "string"}
        if path == ("optimize", "scoring"):
            result["items"] = {"anyOf": [result["items"], {"type": "string", "minLength": 1}]}
        if path == ("backtest", "scenarios", "*", "exchanges"):
            result["items"] = {"type": "string"}
        return result
    properties = {name: _object_schema(child, (*path, name), definitions)
                  for name, child in sorted(children.items()) if name != "*"}
    if properties or not field or field and FieldType.OBJECT in field.types:
        result["properties"] = properties
        result["additionalProperties"] = (_object_schema(children["*"], (*path, "*"), definitions)
                                          if "*" in children else False)
    if path == LIVE_OPERATIONAL_PATHS["startup_phase_budgets"]:
        from live.event_bus import STARTUP_TIMING_PHASES

        result["propertyNames"] = {"enum": sorted(STARTUP_TIMING_PHASES)}
        result["additionalProperties"]["minProperties"] = 1
    return result


def _bounds_schema(registry):
    tree = {}
    for field in registry:
        if field.optimizable and field.path[:1] == ("bot",):
            endpoint = {"type": "number"}  # Fractional genes may target integer counts.
            schema = {"anyOf": [endpoint, {"type": "array", "minItems": 1, "maxItems": 3,
                      "prefixItems": [endpoint, endpoint, {"type": ["number", "null"]}],
                      "items": False}], "x-passivbot-target": ".".join(field.path)}
            if field.recommended_bounds:
                schema["examples"] = [list(field.recommended_bounds)]
            target = tree
            for part in field.path[1:-1]:
                target = target.setdefault("properties", {}).setdefault(part, {
                    "type": "object", "additionalProperties": False,
                })
            target.setdefault("properties", {})[field.path[-1]] = schema
    return {"type": "object", "additionalProperties": False, **tree}


def _coin_patch_schema(registry, *, hsl=True):
    tree = {}
    for field in registry:
        if field.path[:1] == ("coin_overrides",):
            path = field.path[2:]
            if not path:
                continue
        elif field.coin_override in {CoinOverridePolicy.ALLOWED, CoinOverridePolicy.COIN_HSL_ONLY}:
            if not hsl and field.coin_override is CoinOverridePolicy.COIN_HSL_ONLY:
                continue
            path = field.path
        else:
            continue
        current = tree
        for part in path[:-1]:
            current = current.setdefault("properties", {}).setdefault(part, {
                "type": "object", "additionalProperties": False,
            })
        current.setdefault("properties", {})[path[-1]] = field_schema(field, include_default=False)
    return {"type": "object", "additionalProperties": False, **tree}


def _pointer(path):
    return "#/$defs/config" + "".join("/properties/" + part.replace("~", "~0").replace("/", "~1") for part in path)


def _patch_schemas(root, definitions, mode, registry):
    """Reuse the addressed schemas for literal dotted and nested sparse paths."""
    nodes = {}

    def visit(node, path=()):
        nodes[path] = node
        for name, child in node.get("properties", {}).items():
            visit(child, (*path, name))

    visit(root)
    for prefix, node in nodes.items():
        if "properties" not in node:
            continue
        properties = {}
        forbidden = []
        for path, child in nodes.items():
            if path[:len(prefix)] != prefix or len(path) <= len(prefix):
                continue
            if mode == "fixed" and path[:2] == ("optimize", "prefilter"):
                forbidden.append(".".join(path[len(prefix):]))
                continue
            relative = path[len(prefix):]
            name = ".".join(relative)
            if mode == "fixed" and "properties" in child:
                forbidden.append(name)
                continue  # Fixed runtime pins must target leaf settings.
            if path == ("coin_overrides",):
                if mode != "fixed":
                    properties[name] = {"$ref": "#/$defs/config/properties/coin_overrides"}
                else:
                    forbidden.append(name)
                continue
            if len(relative) == 1 and "properties" in child:
                properties[name] = {"$ref": "#/$defs/" + _patch_name(mode, path)}
            elif mode == "fixed":
                continue  # The common scenario view already supplies leaf types.
            elif len(relative) == 1 and isinstance(child.get("additionalProperties"), dict):
                properties[name] = {"type": "object", "maxProperties": 0}
            else:
                properties[name] = {"$ref": _pointer(path)}
        patterns = {}
        # Dotted per-coin settings retain the same narrow permission set.
        coin_patch = definitions["coinPatch"]
        def coin_paths(node, path=()):
            for name, child in node.get("properties", {}).items():
                yield (*path, name), child
                yield from coin_paths(child, (*path, name))
        if not prefix:
            patterns[r"^coin_overrides\.[^.]+$"] = (
                False if mode == "fixed" else {"$ref": "#/$defs/coinPatch"}
            )
            for path, schema in coin_paths(coin_patch):
                if path == ("override_config_path",):
                    continue  # File references belong to atomic coin maps only.
                pattern = r"^coin_overrides\.[^.]+\." + re.escape(".".join(path)) + "$"
                if mode == "fixed":
                    if "properties" in schema:
                        patterns[pattern] = False
                else:
                    patterns[pattern] = {"$ref": "#/$defs/coinPatch" + "".join(
                        "/properties/" + part for part in path
                    )}
        patch = {
            "type": "object", "properties": properties,
            "patternProperties": patterns,
        }
        if mode == "fixed":
            patch["propertyNames"] = {"not": {"enum": forbidden}} if forbidden else {}
            # Share all canonical path/type definitions. This overlay only
            # narrows fixed pins to literal dotted leaf keys.
            patch = {"allOf": [{"$ref": "#/$defs/" + _patch_name("scenario", prefix)}, patch]}
        else:
            patch["additionalProperties"] = False
        definitions[_patch_name(mode, prefix)] = patch


def _patch_name(mode, path):
    return mode + "Patch" + (":" + ".".join(path) if path else "")
