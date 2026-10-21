"""Shared approval binding, side-owned sources and persistent selections."""

from copy import deepcopy
from pathlib import Path
from typing import TypeAlias

from .access import require_config_value

COIN_SOURCE_FIELDS = frozenset({"approved_coins", "ignored_coins"})
PreparedUniverseSnapshot: TypeAlias = tuple[tuple[str, str, tuple[str, ...]], ...]


class ResolvedCoinList(list):
    """A runtime membership view retaining its validated authored source.

    This is not a cache: each refresh reads the source again. JSON input cannot
    construct this internal view, and assigning a plain public value replaces it.
    """
    def __init__(self, values, source, shared_source=None):
        super().__init__(values)
        self.source = deepcopy(source)
        self.shared_source = deepcopy(shared_source)


def restore_universe_sources(config):
    """Restore source bindings before patching, reloading or authored export."""
    bot = config.get("bot", {})
    if not isinstance(bot, dict):
        return config  # The authored validator owns malformed structure errors.
    for side in ("long", "short"):
        node = bot.get(side, {})
        universe = node.get("universe", {}) if isinstance(node, dict) else {}
        if not isinstance(universe, dict):
            continue
        for field in COIN_SOURCE_FIELDS:
            value = universe.get(field)
            if isinstance(value, ResolvedCoinList):
                universe[field] = deepcopy(value.source)
    return config


def universe_source_view(config):
    return restore_universe_sources(deepcopy(config))


def raw_side_universe_sources(config, field):
    if field not in COIN_SOURCE_FIELDS:
        raise KeyError(field)
    result = {}
    for side in ("long", "short"):
        universe = config.get("bot", {}).get(side, {}).get("universe", {})
        if field in universe:
            value = universe[field]
            result[side] = deepcopy(value.source if isinstance(value, ResolvedCoinList) else value)
    return result


def get_side_universe_sources(config, field, default=None):
    if field not in COIN_SOURCE_FIELDS:
        raise KeyError(field)
    result = {}
    shared = config.get("bot", {}).get("global", {}).get("universe", {}).get("approved_coins", [])
    for side in ("long", "short"):
        universe = config.get("bot", {}).get(side, {}).get("universe", {})
        if field in universe:
            value = universe[field]
            if field == "approved_coins" and isinstance(value, ResolvedCoinList) and value.source is None and value.shared_source != shared:
                value = None
            result[side] = shared if field == "approved_coins" and value is None else value
    return result if result else default


def require_side_universe_sources(config, field):
    result = get_side_universe_sources(config, field, {})
    for side in ("long", "short"):
        if side not in result:
            raise KeyError(f"bot.{side}.universe.{field}")
    return result


def set_side_universe_sources(config, field, sources):
    """Assign explicit replacement authority, never inherited or merged membership."""
    if field not in COIN_SOURCE_FIELDS:
        raise KeyError(field)
    by_side = sources if isinstance(sources, dict) else {side: sources for side in ("long", "short")}
    if set(by_side) - {"long", "short"}:
        raise ValueError(f"{field} source mapping must contain only long/short")
    for side in ("long", "short"):
        config.setdefault("bot", {}).setdefault(side, {}).setdefault("universe", {})[field] = deepcopy(by_side.get(side, []))


def materialize_side_universe_sources(config, field, values, *, sources=None):
    """Build the existing runtime list view without losing source bindings."""
    sources = raw_side_universe_sources(config, field) if sources is None else sources
    set_side_universe_sources(config, field, {
        side: ResolvedCoinList(value, sources.get(side), config.get("bot", {}).get("global", {}).get("universe", {}).get("approved_coins", [])) for side, value in values.items()
    })


def universe_sources_affected(overrides):
    """Whether replacement authority intersects a source leaf, not other filters."""
    paths = [("bot", side, "universe", field)
             for side in ("long", "short") for field in COIN_SOURCE_FIELDS]
    paths.append(("bot", "global", "universe", "approved_coins"))
    for selector in overrides or ():
        if not isinstance(selector, str):
            continue  # The override owner reports malformed selector types.
        selected = tuple(selector.split("."))
        if any(selected[:len(path)] == path or path[:len(selected)] == selected for path in paths):
            return True
    return False


def resolve_universe_source_path(source, *, base_config_path="", discover_ambiguous=False, allow_all=True):
    """Select the same local file for resolution and provenance reporting."""
    from utils import _resolve_coins_file_path

    candidate = source[0] if isinstance(source, list) and len(source) == 1 else source
    if not isinstance(candidate, str) or allow_all and candidate.strip().lower() == "all":
        return None
    candidate = candidate.strip()
    path = Path(candidate).expanduser()
    file_like = candidate.startswith(("./", "../", "/", "~")) or path.suffix.lower() in {".json", ".hjson", ".txt"}
    if not discover_ambiguous and not file_like:
        return None
    base = Path(base_config_path).absolute().parent.resolve() if base_config_path else None
    if not path.is_absolute() and base is not None and (base / path).is_file():
        return base / path
    if path.is_absolute():
        return path if path.is_file() else None  # A bound absolute file has no alias fallback.
    if base is None or not file_like or discover_ambiguous:
        return _resolve_coins_file_path(candidate, preserve_selected_path=True)
    return None


def bind_universe_source(value, *, field, base_config_path="", anchor_all=False, released_wrapped_files=False):
    """Bind file syntax at preparation; never rediscover literals during replay."""
    if isinstance(value, ResolvedCoinList):
        return value
    if not anchor_all and isinstance(value, list) and len(value) > 1:
        for item in value:
            if isinstance(item, str) and resolve_universe_source_path(
                item, base_config_path=base_config_path,
                allow_all=field == "approved_coins",
            ) is not None:
                raise ValueError("a coin-list file remains unresolved inside a multi-item source; use a single list-file source or explicit coin identifiers")
    candidate = value[0] if isinstance(value, list) and len(value) == 1 else value
    if not isinstance(candidate, str) or field == "approved_coins" and candidate.strip().lower() == "all":
        return value
    candidate = candidate.strip()
    path = Path(candidate).expanduser()
    explicit = candidate.startswith(("./", "../", "/", "~")) or path.suffix.lower() in {".json", ".hjson", ".txt"}
    if anchor_all:
        if not explicit:
            return value  # Preparation has already made genuine bare files explicit.
        base = Path(base_config_path).absolute().parent.resolve() if base_config_path else Path.cwd()
        bound = str(path if path.is_absolute() else base / path)
    else:
        if not explicit and isinstance(value, list) and not released_wrapped_files:
            return value  # V9 lists contain literal tokens unless file syntax is explicit.
        selected = resolve_universe_source_path(
            candidate, base_config_path=base_config_path, discover_ambiguous=True,
            allow_all=field == "approved_coins",
        )
        if selected is None:
            if explicit:
                base = Path(base_config_path).absolute().parent.resolve() if base_config_path else Path.cwd()
                selected = path if path.is_absolute() else base / path
            else:
                if isinstance(value, str):
                    from utils import normalize_coins_source
                    tokens = normalize_coins_source(value, allow_all=field == "approved_coins", resolve_files=False, parse_jsonish=False)
                    if tokens["long"] == tokens["short"]:
                        return tokens["long"]  # Scalar literals cannot become files on a later reload.
                return value
        bound = str(selected.absolute())  # Freeze the selected location, never mutable contents.
    return [bound] if isinstance(value, list) else bound


def normalize_universe_source_bindings(config, *, anchor_all=False, base_config_path=None, on_change=None, released_wrapped_files=False):
    """Bind ambiguous files once; result writers anchor known path syntax only.

    No file contents or membership are cached. Refresh always reads explicit
    bound files, while later local-file creation cannot reinterpret literal coins.
    Only the released-v8 converter discovers historical singleton-list file refs.
    """
    base_path = (config.get("live", {}).get("base_config_path", "")
                 if base_config_path is None else base_config_path)

    def visit(node, parts=()):
        if isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, (*parts, index))
        elif isinstance(node, dict):
            for key, value in list(node.items()):
                if key.startswith("_"):
                    continue
                field = key.split(".")[-1]
                if field in COIN_SOURCE_FIELDS:
                    bound = bind_universe_source(value, field=field,
                        base_config_path=base_path, anchor_all=anchor_all,
                        released_wrapped_files=released_wrapped_files)
                    if bound != value:
                        node[key] = bound
                        if on_change is not None:
                            on_change((*parts, key), value, bound)
                else:
                    visit(value, (*parts, key))

    visit(config)
    return config


def anchored_universe_source_view(config):
    """Reloadable semantic output with source files bound before context is removed."""
    return normalize_universe_source_bindings(universe_source_view(config), anchor_all=True)


def resolve_universe_sources(config, field, *, allow_partial=False, position_sides=None):
    """Read each effective source in this context; shared sources are read once.

    Partial mode returns unavailable sides separately for the live entry gate;
    an unreadable source is never represented as a successful empty selection.
    A prepared scenario may resolve only the sides whose source was replaced.
    """
    import hjson
    from utils import normalize_coins_source, read_external_coins_lists, _parse_plain_coins_list
    positions = ("long", "short") if position_sides is None else tuple(position_sides)
    if any(side not in {"long", "short"} for side in positions):
        raise ValueError("position_sides must contain only long/short")
    sources = raw_side_universe_sources(config, field)
    shared = config.get("bot", {}).get("global", {}).get("universe", {}).get("approved_coins", [])
    base = config.get("live", {}).get("base_config_path", "")

    def parse(source, *, common=False):
        if isinstance(source, list) and len(source) > 1:
            for item in source:
                if not isinstance(item, str):
                    continue
                child = Path(item)
                file_like = item.startswith(("./", "../", "/", "~")) or child.suffix.lower() in {".json", ".hjson", ".txt"}
                if file_like or resolve_universe_source_path(item, base_config_path=base, allow_all=field == "approved_coins") is not None:
                    if resolve_universe_source_path(item, base_config_path=base, allow_all=field == "approved_coins") is None:
                        raise FileNotFoundError(f"coin-list file not found: {item}")
                    raise ValueError("a coin-list file remains unresolved inside a multi-item source; use a single list-file source or explicit coin identifiers")
        candidate = source[0] if isinstance(source, list) and len(source) == 1 else source
        if field == "approved_coins" and isinstance(candidate, str) and candidate.strip().lower() == "all":
            return {side: ["all"] for side in ("long", "short")}
        if isinstance(candidate, str):
            path = Path(candidate).expanduser()
            file_like = candidate.startswith(("./", "../", "/", "~")) or path.suffix.lower() in {".json", ".hjson", ".txt"}
            selected = resolve_universe_source_path(candidate, base_config_path=base, allow_all=field == "approved_coins")
            if selected is not None:
                if common:
                    message = "shared approved_coins file must contain one coin list; use bot.long.universe.approved_coins and bot.short.universe.approved_coins for directional files"
                    with selected.open() as stream:
                        text = stream.read()
                    try:
                        content = hjson.loads(text)
                    except ValueError as error:
                        if text.lstrip().startswith(("{", "[")):
                            raise ValueError(message) from error
                        content = None  # The same literal text fallback as side-owned lists.
                    if isinstance(content, dict):
                        raise ValueError(message)  # Equal directional arrays remain an object.
                    if isinstance(content, list):
                        if not all(isinstance(x, str) for x in content):
                            raise ValueError(message)
                        source = content
                    else:
                        if "{" in text or "}" in text:
                            raise ValueError(message)  # Never demote a malformed/encoded object to common text.
                        source = _parse_plain_coins_list(text)
                else:
                    source = read_external_coins_lists(str(selected))
            elif file_like:
                raise FileNotFoundError(f"coin-list file not found: {candidate}")
        candidate = source[0] if isinstance(source, list) and len(source) == 1 else source
        if field == "approved_coins" and isinstance(candidate, str) and candidate.strip().lower() == "all":
            return {side: ["all"] for side in ("long", "short")}
        return normalize_coins_source(source, allow_all=field == "approved_coins", resolve_files=False, parse_jsonish=False)

    values, errors = {}, {}
    inherited = [side for side in positions if field == "approved_coins" and sources.get(side) is None]
    if inherited:
        try:
            common = parse(shared, common=True)
        except (OSError, ValueError, TypeError) as exc:
            for side in inherited:
                errors[side] = exc
        else:
            for side in inherited:
                values[side] = common[side]
    for side in positions:
        if side in inherited:
            continue
        try:
            values[side] = parse(sources.get(side, []))[side]
        except (OSError, ValueError, TypeError) as exc:
            errors[side] = exc
    if errors and not allow_partial:
        raise next(iter(errors.values()))
    return values, errors


def normalize_always_include(value, *, path):
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a list of coin identifiers")
    result = []
    seen = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise TypeError(f"{path} must contain non-empty coin identifiers")
        item = item.strip()
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def always_include_matches(config, side, coin, *, exchange=None, market_symbol=None):
    """Match persistent selections over existing market maps, without exchange I/O."""
    from utils import (
        MarketIdentifierExchangeMismatch,
        UnknownMarketIdentifier,
        coin_to_symbol,
        heuristic_symbol_to_coin,
        looks_like_exact_market_identifier,
        split_exchange_qualified_market_identifier,
        to_standard_exchange_name,
    )

    selected = require_config_value(config, f"bot.{side}.forager.always_include")
    coin_venue, coin_value = split_exchange_qualified_market_identifier(str(coin))
    venue = coin_venue or exchange
    for identifier in selected:
        identifier_venue, identifier_value = split_exchange_qualified_market_identifier(
            identifier
        )
        if identifier_venue is not None and (
            venue is None or identifier_venue != to_standard_exchange_name(venue)
        ):
            continue
        if identifier_value == coin_value:
            return True
        if venue is not None and market_symbol is not None:
            try:
                resolved = coin_to_symbol(identifier, venue, verbose=False)
            except (MarketIdentifierExchangeMismatch, UnknownMarketIdentifier):
                continue
            if resolved == market_symbol:
                return True
            continue
        if (
            not looks_like_exact_market_identifier(identifier_value)
            and not looks_like_exact_market_identifier(coin_value)
            and heuristic_symbol_to_coin(identifier_value)
            == heuristic_symbol_to_coin(coin_value)
        ):
            return True
    return False


def runtime_universe_snapshot(config, *, available_coins=None) -> PreparedUniverseSnapshot:
    """Freeze already prepared membership for one immutable dataset execution.

    This typed snapshot never grants authored permission and never reloads a file
    during candidate replay. The source policy remains separate for identity and
    reloadable results. Only an explicit all sentinel can use prepared columns.
    """
    snapshot = []
    for field in ("approved_coins", "ignored_coins"):
        values = require_side_universe_sources(config, field)
        for side in ("long", "short"):
            value = values[side]
            if field == "approved_coins" and (value == "all" or value == ["all"]):
                if available_coins is None:
                    raise ValueError("all approval requires prepared dataset coin membership")
                value = list(available_coins)
            if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                raise ValueError(f"prepared {side} {field} membership is required before dataset registration")
            snapshot.append((field, side, tuple(value)))
    return tuple(snapshot)


def materialize_universe_snapshot(config, snapshot):
    result = universe_source_view(config)
    for field in ("approved_coins", "ignored_coins"):
        values = {side: list(value) for name, side, value in snapshot if name == field}
        materialize_side_universe_sources(result, field, values)
    return result


def fixed_runtime_universe_view(config, *, snapshot=None, available_coins=None):
    snapshot = runtime_universe_snapshot(config, available_coins=available_coins) if snapshot is None else snapshot
    result = universe_source_view(config)
    for field in ("approved_coins", "ignored_coins"):
        set_side_universe_sources(result, field, {side: list(value) for name, side, value in snapshot if name == field})
    return result
