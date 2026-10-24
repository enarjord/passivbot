"""Inherited source bindings, prepared membership and persistent selections."""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple, TypeAlias

from .access import require_config_value

COIN_SOURCE_FIELDS = frozenset({"approved_coins", "ignored_coins"})


@dataclass(frozen=True)
class PreparedMarketAliases:
    """Immutable market routing facts acquired with the approval membership."""
    venues: tuple
    unavailable_venues: tuple[str, ...] = ()
    fingerprint: str = field(init=False)
    _lookups: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (type(self.venues) is not tuple or type(self.unavailable_venues) is not tuple
                or any(type(exchange) is not str for exchange in self.unavailable_venues)):
            raise TypeError("prepared market aliases must be immutable")
        lookups = []
        for venue in self.venues:
            if type(venue) is not tuple or len(venue) != 3:
                raise TypeError("prepared market venue must be an immutable triple")
            exchange, aliases, native_ids = venue
            if (type(exchange) is not str or type(aliases) is not tuple or type(native_ids) is not tuple
                    or any(type(pair) is not tuple or len(pair) != 2 for pair in (*aliases, *native_ids))
                    or any(type(alias) is not str or type(symbols) is not tuple
                           or any(type(symbol) is not str for symbol in symbols) for alias, symbols in aliases)
                    or any(type(symbol) is not str or type(native) is not str for symbol, native in native_ids)):
                raise TypeError("prepared market aliases must contain immutable string facts")
            lookups.append((exchange, MappingProxyType(dict(aliases)), MappingProxyType(dict(native_ids))))
        object.__setattr__(self, "_lookups", tuple(lookups))
        encoded = json.dumps((self.venues, self.unavailable_venues), separators=(",", ":"), ensure_ascii=True).encode()
        object.__setattr__(self, "fingerprint", hashlib.sha256(encoded).hexdigest())

    def __deepcopy__(self, memo):
        return self  # All authoritative contents are immutable.

    def __reduce__(self):
        return (type(self), (self.venues, self.unavailable_venues))

    @staticmethod
    def _resolve(identifier, exchange, aliases):
        from utils import _resolve_market_candidates, split_exchange_qualified_market_identifier

        venue, raw = split_exchange_qualified_market_identifier(identifier)
        if venue is not None and venue != exchange:
            return None
        for key in (identifier, raw) if venue is not None else (raw,):
            symbol = _resolve_market_candidates(identifier, aliases.get(key, ()), exchange)
            if symbol is not None:
                return symbol
        return None

    def for_bound_inputs(self, members, ignored=(), *, complete_venues=()):
        """Keep permission and declared-ignore lookup facts, without granting membership."""
        from utils import split_exchange_qualified_market_identifier

        venues, unavailable = [], set(self.unavailable_venues)
        for exchange, aliases, native in self._lookups:
            targets = {symbol for member in members
                       if (symbol := self._resolve(member, exchange, aliases)) is not None}
            if exchange not in complete_venues and any(
                split_exchange_qualified_market_identifier(member)[0] in (None, exchange)
                and self._resolve(member, exchange, aliases) is None for member in members
            ):
                unavailable.add(exchange)
            for identifier in ignored:
                venue, raw = split_exchange_qualified_market_identifier(identifier)
                if venue is None or venue == exchange:
                    for key in (identifier, raw) if venue is not None else (raw,):
                        if key in aliases:
                            targets.update(aliases[key])
                            break
            selected = tuple((alias, symbols) for alias, symbols in aliases.items() if targets.intersection(symbols))
            venues.append((exchange, selected, tuple((symbol, native[symbol]) for symbol in sorted(targets))))
        return type(self)(tuple(venues), tuple(sorted(unavailable)))

    def subtract(self, approved, ignored):
        """Apply the existing per-venue alias and partial-survivor rules, without I/O."""
        from utils import (split_exchange_qualified_market_identifier,
                           looks_like_exact_market_identifier, heuristic_symbol_to_coin)

        potential = any(a != i and (looks_like_exact_market_identifier(a)
                        or looks_like_exact_market_identifier(i)
                        or heuristic_symbol_to_coin(a) == heuristic_symbol_to_coin(i))
                        for a in approved for i in ignored)
        if not potential:
            return [coin for coin in approved if coin not in ignored]
        for exchange in self.unavailable_venues:
            if (any(split_exchange_qualified_market_identifier(coin)[0] in (None, exchange) for coin in approved)
                    and any(split_exchange_qualified_market_identifier(coin)[0] in (None, exchange) for coin in ignored)):
                raise ValueError(f"prepared market alias facts unavailable on {exchange}; prepare universe membership before changing alias-based ignores")
        ignored_symbols = {
            exchange: {symbol for value in ignored if (symbol := self._resolve(value, exchange, aliases)) is not None}
            for exchange, aliases, _native in self._lookups
        }
        result = []
        for identifier in approved:
            if identifier in ignored:
                continue
            resolved = [(exchange, symbol, native_ids) for exchange, aliases, native_ids in self._lookups
                        if (symbol := self._resolve(identifier, exchange, aliases)) is not None]
            if not any(symbol in ignored_symbols[exchange] for exchange, symbol, _native in resolved):
                result.append(identifier)
            else:
                result.extend(f"{exchange}::{native_ids[symbol]}" for exchange, symbol, native_ids in resolved
                              if symbol not in ignored_symbols[exchange])
        return list(dict.fromkeys(result))

    def within_columns(self, members, columns):
        """Intersect aliases with the exact prepared venue/column identities."""
        def identities(identifier):
            return {(exchange, symbol) for exchange, aliases, _native in self._lookups
                    if (symbol := self._resolve(identifier, exchange, aliases)) is not None}

        available = set().union(*(identities(value) for value in columns))
        result = []
        for member in members:
            admitted = identities(member)
            surviving = admitted & available
            if admitted and surviving == admitted or not admitted and member in columns:
                result.append(member)
            elif surviving:
                result.extend(f"{exchange}::{native_ids[symbol]}"
                              for exchange, _aliases, native_ids in self._lookups
                              for venue, symbol in sorted(surviving) if venue == exchange)
        return list(dict.fromkeys(result))

    def venue_columns(self, columns, exchanges):
        result = []
        for column in columns:
            qualified = [f"{exchange}::{native[symbol]}" for exchange, aliases, native in self._lookups
                         if exchange in exchanges and (symbol := self._resolve(column, exchange, aliases)) in native]
            result.extend(qualified or [column])
        return tuple(dict.fromkeys(result))

    def permits(self, members, coin, exchange, market_symbol):
        from utils import to_standard_exchange_name

        exchange = to_standard_exchange_name(exchange)
        if exchange in self.unavailable_venues:
            return None  # The existing literal admission still applies.
        for venue, aliases, _native in self._lookups:
            if venue == exchange:
                target = market_symbol or self._resolve(coin, exchange, aliases)
                return target is not None and any(self._resolve(value, exchange, aliases) == target for value in members)
        return False

    def resolve_identifier(self, identifier, exchange):
        from utils import UnknownMarketIdentifier, to_standard_exchange_name

        exchange = to_standard_exchange_name(exchange)
        if exchange in self.unavailable_venues:
            raise ValueError(f"prepared market alias facts unavailable on {exchange}; prepare universe membership")
        for venue, aliases, _native in self._lookups:
            if venue == exchange:
                symbol = self._resolve(identifier, exchange, aliases)
                if symbol is not None:
                    return symbol
        raise UnknownMarketIdentifier(f"market identifier {identifier!r} is absent from prepared alias facts on {exchange}")


class PreparedUniverseMembership(NamedTuple):
    field: str
    side: str
    members: tuple[str, ...]
    acquired_members: tuple[str, ...]
    market_aliases: PreparedMarketAliases | None
    available_members: tuple[str, ...] | None


PreparedUniverseSnapshot: TypeAlias = tuple[PreparedUniverseMembership, ...]


class ResolvedCoinList(list):
    """A runtime membership view retaining its validated authored source.

    Explicit refresh reads the source again. Internal replay retains acquisition
    facts separately from filtered eligibility; JSON cannot construct this view.
    """
    def __init__(self, values, source, shared_source=None, *, source_present=True, acquired_members=None, market_aliases=None, available_members=None):
        if type(source_present) is not bool:
            raise TypeError("prepared source presence must be a boolean")
        super().__init__(values)
        self.source = deepcopy(source)
        self.shared_source = deepcopy(shared_source)
        self.source_present = source_present
        if acquired_members is None:
            acquired_members = values.acquired_members if type(values) is ResolvedCoinList else tuple(self)
        if type(acquired_members) is not tuple or any(type(coin) is not str for coin in acquired_members):
            raise TypeError("acquired source membership must be a tuple of strings")
        self.acquired_members = acquired_members
        if market_aliases is None and type(values) is ResolvedCoinList:
            market_aliases = values.market_aliases
        if market_aliases is not None and type(market_aliases) is not PreparedMarketAliases:
            raise TypeError("prepared market aliases require an exact internal projection")
        self.market_aliases = market_aliases
        if available_members is None and type(values) is ResolvedCoinList:
            available_members = values.available_members
        if available_members is not None and (type(available_members) is not tuple
                or any(type(coin) is not str for coin in available_members)):
            raise TypeError("prepared column membership must be an immutable tuple of strings")
        self.available_members = available_members


def restore_universe_sources(config):
    """Restore source bindings before patching, reloading or authored export."""
    bot = config.get("bot", {})
    if not isinstance(bot, dict):
        return config  # The authored validator owns malformed structure errors.
    from .bot_inheritance import reconciled_bot_policy

    policy = reconciled_bot_policy(config)
    authored = policy.source_view() if policy is not None else None
    for side in ("global", "long", "short"):
        node = bot.get(side, {})
        universe = node.get("universe", {}) if isinstance(node, dict) else {}
        if not isinstance(universe, dict):
            continue
        for field in COIN_SOURCE_FIELDS:
            if authored is not None:
                source = authored.get(side, {}).get("universe", {})
                if field in source:
                    universe[field] = deepcopy(source[field])
                else:
                    universe.pop(field, None)
                continue
            value = universe.get(field)
            if type(value) is ResolvedCoinList:
                if value.source_present:
                    universe[field] = deepcopy(value.source)
                else:
                    universe.pop(field, None)
    return config


def universe_source_view(config, *, candidate=False, include_defaults=False):
    from .bot_inheritance import bot_policy_source_view

    return restore_universe_sources(bot_policy_source_view(
        config, candidate=candidate, include_defaults=include_defaults,
    ))


def shared_universe_source(config, field):
    """Return this field's common binding, without opening files or granting pins."""
    if field not in COIN_SOURCE_FIELDS:
        raise KeyError(field)
    from .bot_inheritance import reconciled_bot_policy

    policy = reconciled_bot_policy(config)
    bot = policy.source_view() if policy is not None else config.get("bot", {})
    common = bot.get("global", {}).get("universe", {})
    if field in common:
        value = common[field]
    elif policy is not None:
        value = policy.catalog.find(("universe", field)).default_value()
    else:
        value = []
    if field == "approved_coins" and value is None:
        raise TypeError("bot.global.universe.approved_coins does not accept null")
    return [] if field == "ignored_coins" and value is None else deepcopy(value)


def source_inherits_shared(field, sources, side):
    """Omission inherits both sources; approved null also inherits, ignored null clears."""
    if field not in COIN_SOURCE_FIELDS:
        raise KeyError(field)
    return side not in sources or field == "approved_coins" and sources[side] is None


def raw_side_universe_sources(config, field):
    if field not in COIN_SOURCE_FIELDS:
        raise KeyError(field)
    from .bot_inheritance import reconciled_bot_policy

    policy = reconciled_bot_policy(config)
    bot = policy.source_view() if policy is not None else config.get("bot", {})
    result = {}
    for side in ("long", "short"):
        universe = bot.get(side, {}).get("universe", {})
        if field in universe:
            value = universe[field]
            if type(value) is ResolvedCoinList:
                if value.source_present:
                    result[side] = deepcopy(value.source)
            else:
                result[side] = deepcopy(value)
    return result


def get_side_universe_sources(config, field, default=None):
    if field not in COIN_SOURCE_FIELDS:
        raise KeyError(field)
    result = {}
    shared = shared_universe_source(config, field)
    sources = raw_side_universe_sources(config, field)
    for side in ("long", "short"):
        universe = config.get("bot", {}).get(side, {}).get("universe", {})
        value = universe.get(field)
        inherited = source_inherits_shared(field, sources, side)
        if type(value) is ResolvedCoinList and (
            value.source_present == (side in sources)
            and value.source == sources.get(side)
            and (not inherited or value.shared_source == shared)
        ):
            result[side] = value
        elif inherited:
            result[side] = deepcopy(shared)
        else:
            result[side] = [] if field == "ignored_coins" and sources[side] is None else deepcopy(sources[side])
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


def copy_universe_source_policy(config, source):
    """Copy declared source bindings, preserving omission and numeric authority.

    Dataset/context owners call this before installing their frozen membership.
    No files are read and no concrete list can become a new source permission.
    """
    from dataclasses import replace
    from .bot_inheritance import reconciled_bot_policy, apply_bot_policy_patch, install_bot_policy

    incoming = universe_source_view(source).get("bot", {})
    policy = reconciled_bot_policy(config)
    target = policy.source_view() if policy is not None else config.setdefault("bot", {})
    patches = {}
    for scope in ("global", "long", "short"):
        original = target.get(scope, {}).get("universe", {})
        updated = deepcopy(original)
        declarations = incoming.get(scope, {}).get("universe", {})
        for field in COIN_SOURCE_FIELDS:
            if field in declarations:
                updated[field] = deepcopy(declarations[field])
            else:
                updated.pop(field, None)
        if updated != original or set(updated) != set(original):
            if policy is None:
                target.setdefault(scope, {})["universe"] = updated
            else:
                patches[f"bot.{scope}.universe"] = updated
    if patches:
        # Source-only changes cannot invalidate numeric transforms/formatting.
        # The owned parent patch discards the replaced prepared memberships.
        updated = apply_bot_policy_patch(policy, patches)
        install_bot_policy(config, replace(updated, _derived=policy._derived,
                                           _normalized=policy._normalized))


def materialize_side_universe_sources(config, field, values, *, sources=None, acquired_members=None, market_aliases=None, available_members=None):
    """Build the existing runtime list view without losing source bindings."""
    from .bot_inheritance import reconciled_bot_policy, install_bot_policy, with_prepared_bot_sources

    policy = reconciled_bot_policy(config)
    sources = raw_side_universe_sources(config, field) if sources is None else sources
    shared = shared_universe_source(config, field)
    for side, value in values.items():
        if side not in {"long", "short"}:
            raise ValueError(f"{field} membership must contain only long/short")
        config.setdefault("bot", {}).setdefault(side, {}).setdefault("universe", {})[field] = ResolvedCoinList(
            value, sources.get(side), shared, source_present=side in sources,
            acquired_members=None if acquired_members is None else acquired_members[side],
            market_aliases=None if market_aliases is None else market_aliases[side],
            available_members=None if available_members is None else available_members[side],
        )
    if policy is not None:
        captured = with_prepared_bot_sources(policy, config["bot"])
        install_bot_policy(config, captured)


def filter_prepared_approvals(config):
    """Derive eligibility from acquired approvals, without reopening their sources."""
    approved = require_side_universe_sources(config, "approved_coins")
    ignored = require_side_universe_sources(config, "ignored_coins")
    values, acquired, aliases, available = {}, {}, {}, {}
    for side, value in approved.items():
        members = value.acquired_members if type(value) is ResolvedCoinList else tuple(value)
        acquired[side] = members
        aliases[side] = value.market_aliases if type(value) is ResolvedCoinList else None
        values[side] = (aliases[side].subtract(members, ignored[side]) if aliases[side] is not None
                        else [coin for coin in members if coin not in ignored[side]])
        available[side] = value.available_members if type(value) is ResolvedCoinList else None
        if available[side] is not None:
            values[side] = (aliases[side].within_columns(values[side], available[side]) if aliases[side] is not None
                            else [coin for coin in values[side] if coin in available[side]])
    materialize_side_universe_sources(config, "approved_coins", values, acquired_members=acquired,
                                    market_aliases=aliases, available_members=available)


def restrict_prepared_columns(previous, columns, aliases):
    """A narrower execution context may restrict, but cannot expand, its ceiling."""
    if columns is None:
        return previous
    if previous is None:
        return tuple(columns)
    return tuple(aliases.within_columns(previous, columns) if aliases is not None
                 else [coin for coin in previous if coin in columns])


def approval_allows_coin(value, coin, *, exchange, market_symbol=None, members=None):
    """Lower frozen permission against one factual dataset venue/market row."""
    from backtest_universe import normalize_backtest_coin
    from utils import heuristic_symbol_to_coin, looks_like_exact_market_identifier

    members = value if members is None else members
    if type(value) is ResolvedCoinList and value.market_aliases is not None:
        matched = value.market_aliases.permits(members, coin, exchange, market_symbol)
        if matched is not None:
            return matched
    coin = normalize_backtest_coin(coin)
    return coin in members or any(not looks_like_exact_market_identifier(identifier)
        and heuristic_symbol_to_coin(identifier) == heuristic_symbol_to_coin(coin) for identifier in members)


def universe_sources_affected(overrides):
    """Whether replacement authority intersects a source leaf, not other filters."""
    paths = [("bot", side, "universe", field)
             for side in ("long", "short") for field in COIN_SOURCE_FIELDS]
    paths.extend(("bot", "global", "universe", field) for field in COIN_SOURCE_FIELDS)
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


def anchored_universe_source_view(config, *, candidate=False, include_defaults=False):
    """Reloadable semantic output with source files bound before context is removed."""
    return normalize_universe_source_bindings(universe_source_view(
        config, candidate=candidate, include_defaults=include_defaults,
    ), anchor_all=True)


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
    shared = shared_universe_source(config, field)
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
                    message = (f"shared {field} file must contain one coin list; use "
                               f"bot.long.universe.{field} and bot.short.universe.{field} for directional files")
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
    inherited = [side for side in positions if source_inherits_shared(field, sources, side)]
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


def runtime_universe_snapshot(config, *, available_coins=None, available_exchanges=None) -> PreparedUniverseSnapshot:
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
            acquired = value.acquired_members if type(value) is ResolvedCoinList else tuple(value)
            if field == "approved_coins":
                if acquired == ("all",):
                    if available_coins is None:
                        raise ValueError("all approval requires prepared dataset coin membership")
                    acquired = tuple(available_coins)
            aliases = value.market_aliases if type(value) is ResolvedCoinList else None
            columns = value.available_members if type(value) is ResolvedCoinList else None
            if field == "approved_coins" and available_coins is not None:
                selected = (aliases.venue_columns(available_coins, available_exchanges)
                            if aliases is not None and available_exchanges is not None else available_coins)
                columns = restrict_prepared_columns(columns, selected, aliases)
            snapshot.append(PreparedUniverseMembership(field, side, tuple(value), acquired, aliases, columns))
    return tuple(snapshot)


def materialize_universe_snapshot(config, snapshot):
    # Execution retains the expanded parameter view. Only the source owner
    # replaces membership; authored serialization uses universe_source_view.
    result = deepcopy(config)
    for field in ("approved_coins", "ignored_coins"):
        values = {entry.side: list(entry.members) for entry in snapshot if entry.field == field}
        acquired = {entry.side: entry.acquired_members for entry in snapshot if entry.field == field}
        aliases = {entry.side: entry.market_aliases for entry in snapshot if entry.field == field}
        columns = {entry.side: entry.available_members for entry in snapshot if entry.field == field}
        materialize_side_universe_sources(result, field, values, acquired_members=acquired, market_aliases=aliases,
                                         available_members=columns)
    return result


def universe_snapshot_signature(snapshot):
    """Bind immutable acquired routing facts without copying their complete maps."""
    return [{"field": entry.field, "side": entry.side, "members": list(entry.members),
             "acquired_members": list(entry.acquired_members),
             "market_aliases_sha256": None if entry.market_aliases is None else entry.market_aliases.fingerprint,
             "available_members": None if entry.available_members is None else list(entry.available_members)}
            for entry in snapshot]


def fixed_runtime_universe_view(config, *, snapshot=None, available_coins=None):
    snapshot = runtime_universe_snapshot(config, available_coins=available_coins) if snapshot is None else snapshot
    result = restore_universe_sources(deepcopy(config))
    # This deliberately fixed execution view is not reloadable source policy.
    # Keep the expanded numeric tree, without a policy envelope which would
    # restore inherited bindings during its internal formatting.
    result["bot"] = dict(result["bot"])
    for field in ("approved_coins", "ignored_coins"):
        set_side_universe_sources(result, field, {entry.side: list(entry.members) for entry in snapshot if entry.field == field})
    return result
