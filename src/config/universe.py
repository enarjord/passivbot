"""Side-owned coin-source access and persistent forager selections."""

from copy import deepcopy

from .access import require_config_value


COIN_SOURCE_FIELDS = frozenset({"approved_coins", "ignored_coins"})


def require_side_universe_sources(config, field):
    if field not in COIN_SOURCE_FIELDS:
        raise KeyError(field)
    return {
        side: require_config_value(config, f"bot.{side}.universe.{field}")
        for side in ("long", "short")
    }


def get_side_universe_sources(config, field, default=None):
    if field not in COIN_SOURCE_FIELDS:
        raise KeyError(field)
    result = {}
    for side in ("long", "short"):
        universe = config.get("bot", {}).get(side, {}).get("universe", {})
        if field in universe:
            result[side] = universe[field]
    return result if result else default


def set_side_universe_sources(config, field, sources):
    """Assign a CLI/resolved source mapping into its canonical side scopes."""
    if field not in COIN_SOURCE_FIELDS:
        raise KeyError(field)
    by_side = (
        sources
        if isinstance(sources, dict)
        else {side: sources for side in ("long", "short")}
    )
    if set(by_side) - {"long", "short"}:
        raise ValueError(f"{field} source mapping must contain only long/short")
    for side in ("long", "short"):
        config.setdefault("bot", {}).setdefault(side, {}).setdefault("universe", {})[
            field
        ] = deepcopy(by_side.get(side, []))


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
