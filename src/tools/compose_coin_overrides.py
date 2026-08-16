from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from config.load import load_input_config, prepare_config
from config.overrides import get_allowed_modifications, parse_overrides
from config.schema import get_template_config
from config_utils import clean_config
from pure_funcs import sort_dict_keys
from utils import (
    MarketIdentifierResolutionError,
    coin_to_symbol,
    heuristic_symbol_to_coin,
    json_dumps_streamlined,
    looks_like_exact_market_identifier,
    market_denomination_identity,
    split_exchange_qualified_market_identifier,
    to_standard_exchange_name,
)


CONFIG_SUFFIXES = frozenset({".json", ".hjson"})
POSITION_SIDES = ("long", "short")
OUTPUT_ROOTS = ("config_version", "bot", "coin_overrides", "live", "logging", "monitor")
HYPERLIQUID_MARKET_PREFIXES = ("xyz:", "xyz-", "xyz_")


@dataclass
class SingleCoinConfig:
    path: Path
    coin: str
    approved_sides: frozenset[str]
    config: dict


@dataclass
class CompositionReport:
    source_paths: list[Path]
    master_path: Path
    master_was_selected: bool
    coins: list[str]
    canonicalized_features: list[str]
    account_wide_conflicts: dict[str, list[str]]
    override_leaf_counts: dict[str, int]
    included_backtest_optimize: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool compose-coin-overrides",
        description=(
            "Compose a directory of single-coin configs into one config with minimal inline "
            "coin_overrides."
        ),
    )
    parser.add_argument(
        "input_directory",
        type=Path,
        help="Directory containing the single-coin JSON/HJSON configs",
    )
    parser.add_argument("output_config", type=Path, help="Path for the composed JSON config")
    parser.add_argument(
        "--master-config",
        type=Path,
        default=None,
        help=(
            "Input config to use for master/global values. May be a filename in input_directory "
            "or its path. Defaults to the alphabetically first input."
        ),
    )
    parser.add_argument(
        "--include-backtest-optimize",
        action="store_true",
        help=(
            "Include backtest and optimize sections from the master input. They are omitted by "
            "default for a lean live config."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output_config if it already exists",
    )
    return parser


def discover_config_paths(
    input_directory: Path, *, output_config: Path | None = None
) -> list[Path]:
    directory = input_directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(
            f"input directory does not exist or is not a directory: {directory}"
        )
    output_resolved = output_config.expanduser().resolve() if output_config is not None else None
    paths = sorted(
        (
            path.resolve()
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in CONFIG_SUFFIXES
            and path.resolve() != output_resolved
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if len(paths) < 2:
        raise ValueError(
            f"expected at least two JSON/HJSON configs in {directory}; found {len(paths)}"
        )
    return paths


def _side_coin_lists(config: dict, key: str, *, path: Path) -> dict[str, list[str]]:
    value = config.get("live", {}).get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: live.{key} must be a per-side mapping")
    result = {}
    for side in POSITION_SIDES:
        coins = value.get(side)
        if not isinstance(coins, list) or not all(isinstance(coin, str) for coin in coins):
            raise ValueError(f"{path}: live.{key}.{side} must be a list of coin names")
        result[side] = coins
    return result


def load_single_coin_config(path: Path) -> SingleCoinConfig:
    source, base_config_path, raw_snapshot = load_input_config(str(path), log_info=False)
    if not isinstance(source, dict):
        raise TypeError(f"{path}: configuration root must be an object")
    if source.get("coin_overrides"):
        raise ValueError(f"{path}: single-coin input must not contain coin_overrides")
    prepared = prepare_config(
        source,
        base_config_path=base_config_path,
        verbose=False,
        log_config_transforms=False,
        raw_snapshot=raw_snapshot,
    )
    config = clean_config(prepared)
    if config.get("coin_overrides"):
        raise ValueError(f"{path}: single-coin input must not contain coin_overrides")
    approved = _side_coin_lists(config, "approved_coins", path=path)
    ignored = _side_coin_lists(config, "ignored_coins", path=path)
    approved_union = {coin for coins in approved.values() for coin in coins}
    if len(approved_union) != 1:
        formatted = ", ".join(sorted(approved_union)) or "none"
        raise ValueError(
            f"{path}: expected exactly one approved coin across long/short; found {formatted}"
        )
    coin = next(iter(approved_union))
    if coin.strip().casefold() == "all":
        raise ValueError(
            f"{path}: live.approved_coins must name one coin; the 'all' sentinel is not valid "
            "for a single-coin input"
        )
    approved_sides = frozenset(side for side, coins in approved.items() if coin in coins)
    for side in POSITION_SIDES:
        if coin in ignored[side]:
            raise ValueError(
                f"{path}: single approved coin {coin} is also present in live.ignored_coins.{side}"
            )
        extra = sorted(set(approved[side]) - {coin})
        if extra:
            raise ValueError(
                f"{path}: live.approved_coins.{side} contains additional coin(s): "
                + ", ".join(extra)
            )
    return SingleCoinConfig(path=path, coin=coin, approved_sides=approved_sides, config=config)


def _market_resolution_exchanges(configs: list[SingleCoinConfig]) -> tuple[str, ...]:
    exchanges = set()
    for item in configs:
        backtest = item.config.get("backtest", {})
        configured = backtest.get("exchanges", [])
        if isinstance(configured, str):
            configured = [configured]
        coin_sources = backtest.get("coin_sources", {})
        source_exchanges = coin_sources.values() if isinstance(coin_sources, dict) else []
        for exchange in [*configured, *source_exchanges]:
            normalized = to_standard_exchange_name(str(exchange))
            if normalized and normalized != "fake":
                exchanges.add(normalized)
        for key in ("approved_coins", "ignored_coins"):
            for identifiers in item.config.get("live", {}).get(key, {}).values():
                for identifier in identifiers:
                    qualified_exchange, _unqualified = (
                        split_exchange_qualified_market_identifier(identifier)
                    )
                    if qualified_exchange and qualified_exchange != "fake":
                        exchanges.add(qualified_exchange)
                    if str(identifier).strip().casefold().startswith(
                        HYPERLIQUID_MARKET_PREFIXES
                    ):
                        exchanges.add("hyperliquid")
    return tuple(sorted(exchanges))


def _market_identity(
    identifier: str, exchanges: tuple[str, ...]
) -> tuple[frozenset[tuple[str, str]], str]:
    resolved = set()
    for exchange in exchanges:
        try:
            symbol = coin_to_symbol(identifier, exchange, verbose=False)
        except MarketIdentifierResolutionError:
            continue
        resolved.add((exchange, symbol))
    if looks_like_exact_market_identifier(identifier):
        if not resolved:
            formatted_exchanges = ", ".join(exchanges) or "none"
            raise ValueError(
                f"could not resolve exact market identifier {identifier!r} on configured "
                f"venue(s): {formatted_exchanges}; refresh market metadata before composing"
            )
        resolved_contracts = {
            market_denomination_identity(symbol, exchange=exchange)
            for exchange, symbol in resolved
        }
        if len(resolved_contracts) > 1:
            formatted_resolutions = ", ".join(
                f"{exchange}={symbol}"
                for exchange, symbol in sorted(resolved)
            )
            raise ValueError(
                f"exact market identifier {identifier!r} resolves to different contracts "
                f"across configured venues ({formatted_resolutions}); use "
                "exchange::<native-id>"
            )
    _qualified_exchange, unqualified = split_exchange_qualified_market_identifier(identifier)
    fallback = heuristic_symbol_to_coin(unqualified).strip().casefold()
    return frozenset(resolved), fallback


def _identifiers_refer_to_same_market(
    left: str, right: str, exchanges: tuple[str, ...]
) -> bool:
    if left == right:
        return True
    left_resolved, left_fallback = _market_identity(left, exchanges)
    right_resolved, right_fallback = _market_identity(right, exchanges)
    if left_resolved & right_resolved:
        return True
    # When cached venue metadata cannot resolve one or both identifiers, fail closed
    # on the deterministic underlying-name fallback instead of permitting a duplicate.
    return (
        not left_resolved or not right_resolved
    ) and left_fallback == right_fallback


def _validate_market_identifiers(
    configs: list[SingleCoinConfig], exchanges: tuple[str, ...]
) -> None:
    for item in configs:
        for key in ("approved_coins", "ignored_coins"):
            for side, identifiers in item.config.get("live", {}).get(key, {}).items():
                for identifier in identifiers:
                    try:
                        _market_identity(identifier, exchanges)
                    except ValueError as exc:
                        raise ValueError(
                            f"{item.path}: live.{key}.{side}: {exc}"
                        ) from exc


def _resolve_master_path(
    input_directory: Path, paths: list[Path], master_config: Path | None
) -> tuple[Path, bool]:
    if master_config is None:
        return paths[0], False
    selected = master_config.expanduser()
    if not selected.is_absolute():
        directory_candidate = input_directory.expanduser().resolve() / selected
        selected = directory_candidate if directory_candidate.exists() else selected.resolve()
    selected = selected.resolve()
    if selected not in paths:
        raise ValueError(
            f"selected master config is not a JSON/HJSON input in {input_directory}: {selected}"
        )
    return selected, True


def load_single_coin_directory(
    input_directory: Path,
    *,
    output_config: Path | None = None,
    master_config: Path | None = None,
) -> tuple[list[SingleCoinConfig], bool]:
    paths = discover_config_paths(input_directory, output_config=output_config)
    master_path, master_was_selected = _resolve_master_path(
        input_directory, paths, master_config
    )
    configs = [load_single_coin_config(path) for path in paths]
    by_coin: dict[str, Path] = {}
    for item in configs:
        if item.coin in by_coin:
            raise ValueError(
                f"duplicate single-coin config for {item.coin}: "
                f"{by_coin[item.coin]} and {item.path}"
            )
        by_coin[item.coin] = item.path
    resolution_exchanges = _market_resolution_exchanges(configs)
    _validate_market_identifiers(configs, resolution_exchanges)
    for index, item in enumerate(configs):
        for previous in configs[:index]:
            if _identifiers_refer_to_same_market(
                previous.coin, item.coin, resolution_exchanges
            ):
                raise ValueError(
                    "duplicate single-coin configs resolve to the same market: "
                    f"{previous.coin} ({previous.path}) and {item.coin} ({item.path})"
                )
    strategy_kinds = {
        str(item.config.get("live", {}).get("strategy_kind")) for item in configs
    }
    if len(strategy_kinds) != 1:
        raise ValueError(
            "all single-coin configs must use the same live.strategy_kind; found "
            + ", ".join(sorted(strategy_kinds))
        )
    signal_modes = {
        str(item.config.get("live", {}).get("hsl_signal_mode")) for item in configs
    }
    if len(signal_modes) != 1:
        raise ValueError(
            "all single-coin configs must use the same live.hsl_signal_mode; found "
            + ", ".join(sorted(signal_modes))
        )
    master = next(item for item in configs if item.path == master_path)
    ordered = [master, *(item for item in configs if item.path != master_path)]
    return ordered, master_was_selected


def _get_path(config: dict, path: Iterable[str]) -> Any:
    current: Any = config
    parts = tuple(path)
    for key in parts:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(parts))
        current = current[key]
    return current


def _set_path(config: dict, path: tuple[str, ...], value: Any) -> None:
    current = config
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = deepcopy(value)


def _bound_lower(config: dict, side: str, group: str, key: str) -> float | int:
    raw = _get_path(config, ("optimize", "bounds", side, group, key))
    if isinstance(raw, (list, tuple)) and raw:
        candidates = raw[:2] if len(raw) >= 2 else raw
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in candidates
        ):
            return min(candidates)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return raw
    raise ValueError(
        f"optimize.bounds.{side}.{group}.{key} must contain a numeric lower bound"
    )


def _canonical_disabled_group(master: dict, side: str, group: str) -> dict:
    result = deepcopy(get_template_config()["bot"][side][group])
    bounds = master["optimize"]["bounds"][side].get(group, {})
    for key in bounds:
        if key in result:
            result[key] = _bound_lower(master, side, group, key)
    result["enabled"] = False
    return result


def canonicalize_inactive_features(configs: list[SingleCoinConfig]) -> list[str]:
    master = configs[0].config
    actions: list[str] = []
    schema = get_template_config()
    for side in POSITION_SIDES:
        hsl_disabled = [
            not bool(item.config["bot"][side]["hsl"]["enabled"]) for item in configs
        ]
        if all(hsl_disabled):
            canonical = _canonical_disabled_group(master, side, "hsl")
            for item in configs:
                item.config["bot"][side]["hsl"] = deepcopy(canonical)
            actions.append(f"bot.{side}.hsl (disabled in every input)")
        else:
            master_hsl = master["bot"][side]["hsl"]
            for item, disabled in zip(configs, hsl_disabled):
                if disabled:
                    item.config["bot"][side]["hsl"] = {
                        **deepcopy(master_hsl),
                        "enabled": False,
                    }

        unstuck_disabled = [
            not bool(item.config["bot"][side]["unstuck"]["enabled"]) for item in configs
        ]
        unstuck_all_disabled = all(unstuck_disabled)
        if unstuck_all_disabled:
            canonical = _canonical_disabled_group(master, side, "unstuck")
            for item in configs:
                item.config["bot"][side]["unstuck"] = deepcopy(canonical)
            actions.append(f"bot.{side}.unstuck (disabled in every input)")
        else:
            master_unstuck = master["bot"][side]["unstuck"]
            for item, disabled in zip(configs, unstuck_disabled):
                if disabled:
                    item.config["bot"][side]["unstuck"] = {
                        **deepcopy(master_unstuck),
                        "enabled": False,
                    }
            ema_gating_disabled = [
                not bool(item.config["bot"][side]["unstuck"]["ema_gating_enabled"])
                for item in configs
            ]
            if all(ema_gating_disabled):
                lower = _bound_lower(master, side, "unstuck", "ema_dist")
                for item in configs:
                    item.config["bot"][side]["unstuck"]["ema_dist"] = lower
                actions.append(
                    f"bot.{side}.unstuck.ema_dist (EMA gating disabled in every input)"
                )
            else:
                master_ema_dist = master["bot"][side]["unstuck"]["ema_dist"]
                for item, disabled in zip(configs, ema_gating_disabled):
                    if disabled:
                        item.config["bot"][side]["unstuck"]["ema_dist"] = master_ema_dist

        wel_enforcer_disabled = [
            not bool(
                item.config["bot"][side]["risk"][
                    "position_exposure_enforcer_enabled"
                ]
            )
            for item in configs
        ]
        if all(wel_enforcer_disabled):
            lower = _bound_lower(master, side, "risk", "position_exposure_enforcer_threshold")
            for item in configs:
                risk = item.config["bot"][side]["risk"]
                risk["position_exposure_enforcer_enabled"] = False
                risk["position_exposure_enforcer_threshold"] = lower
            actions.append(f"bot.{side}.risk.position_exposure_enforcer (disabled in every input)")
        else:
            master_threshold = master["bot"][side]["risk"][
                "position_exposure_enforcer_threshold"
            ]
            for item, disabled in zip(configs, wel_enforcer_disabled):
                if disabled:
                    item.config["bot"][side]["risk"][
                        "position_exposure_enforcer_threshold"
                    ] = master_threshold

        twel_enforcer_disabled = all(
            not bool(item.config["bot"][side]["risk"]["total_exposure_enforcer_enabled"])
            for item in configs
        )
        twel_entry_gate_disabled = all(
            not bool(item.config["bot"][side]["risk"]["total_exposure_entry_gate_enabled"])
            for item in configs
        )
        if twel_enforcer_disabled:
            policy_default = schema["bot"][side]["risk"]["total_exposure_enforcer_policy"]
            for item in configs:
                risk = item.config["bot"][side]["risk"]
                risk["total_exposure_enforcer_enabled"] = False
                risk["total_exposure_enforcer_policy"] = policy_default
            actions.append(f"bot.{side}.risk.total_exposure_enforcer (disabled in every input)")
            if twel_entry_gate_disabled:
                lower = _bound_lower(
                    master, side, "risk", "total_exposure_enforcer_threshold"
                )
                for item in configs:
                    item.config["bot"][side]["risk"][
                        "total_exposure_enforcer_threshold"
                    ] = lower
                actions.append(
                    f"bot.{side}.risk.total_exposure_enforcer_threshold "
                    "(entry gate and enforcer disabled in every input)"
                )
    return actions


def _iter_leaves(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _iter_leaves(value[key], (*path, key))
        return
    yield path, value


def _policy_allows(policy: dict, path: tuple[str, ...]) -> bool:
    current: Any = policy
    for key in path:
        if current is True:
            return True
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return current is True


def _count_leaves(value: Any) -> int:
    return sum(1 for _path, _value in _iter_leaves(value))


def _format_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def compose_configs(
    configs: list[SingleCoinConfig],
    *,
    include_backtest_optimize: bool = False,
    master_was_selected: bool = False,
) -> tuple[dict, CompositionReport]:
    if len(configs) < 2:
        raise ValueError("at least two single-coin configs are required")
    if (
        include_backtest_optimize
        and str(configs[0].config.get("optimize", {}).get("backend", "")).casefold()
        == "gpu"
    ):
        raise ValueError(
            "--include-backtest-optimize cannot retain optimize.backend='gpu': the GPU "
            "optimizer supports neither multi-coin datasets nor coin_overrides; select a "
            "CPU optimizer backend in the master input"
        )
    canonicalized_features = canonicalize_inactive_features(configs)
    master_source = configs[0]
    master = deepcopy(master_source.config)

    approved = {
        side: sorted(item.coin for item in configs if side in item.approved_sides)
        for side in POSITION_SIDES
    }
    master["live"]["approved_coins"] = approved
    resolution_exchanges = _market_resolution_exchanges(configs)
    for side in POSITION_SIDES:
        master["live"]["ignored_coins"][side] = sorted(
            ignored
            for ignored in set(master["live"]["ignored_coins"][side])
            if not any(
                _identifiers_refer_to_same_market(ignored, coin, resolution_exchanges)
                for coin in approved[side]
            )
        )
        if approved[side] and float(master["bot"][side]["risk"]["total_wallet_exposure_limit"]) > 0:
            master["bot"][side]["risk"]["n_positions"] = float(len(approved[side]))

    if not include_backtest_optimize:
        master.pop("backtest", None)
        master.pop("optimize", None)
    master["coin_overrides"] = {}

    policy = get_allowed_modifications(hsl_signal_mode=master["live"]["hsl_signal_mode"])
    conflicts: dict[str, list[str]] = {}
    skip_paths = {
        ("live", "approved_coins", "long"),
        ("live", "approved_coins", "short"),
        ("bot", "long", "risk", "n_positions"),
        ("bot", "short", "risk", "n_positions"),
    }
    comparable_master = master_source.config
    for item in configs:
        patch: dict = {}
        for root in ("bot", "live", "logging", "monitor"):
            master_leaves = dict(_iter_leaves(comparable_master[root], (root,)))
            source_leaves = dict(_iter_leaves(item.config[root], (root,)))
            for path in sorted(set(master_leaves) | set(source_leaves)):
                if path in skip_paths or master_leaves.get(path) == source_leaves.get(path):
                    continue
                source_value = source_leaves.get(path)
                if _policy_allows(policy, path):
                    _set_path(patch, path, source_value)
                else:
                    description = (
                        f"{item.path.name}={_format_value(source_value)}; "
                        f"master={_format_value(master_leaves.get(path))}"
                    )
                    conflicts.setdefault(".".join(path), []).append(description)
        if patch:
            master["coin_overrides"][item.coin] = patch

    expected_roots = set(OUTPUT_ROOTS)
    if include_backtest_optimize:
        expected_roots.update({"backtest", "optimize"})
    master = sort_dict_keys({key: master[key] for key in expected_roots})

    validated = prepare_config(
        master,
        verbose=False,
        log_config_transforms=False,
        raw_snapshot=master,
    )
    parse_overrides(validated, verbose=False)

    report = CompositionReport(
        source_paths=sorted((item.path for item in configs), key=lambda path: path.name.casefold()),
        master_path=master_source.path,
        master_was_selected=master_was_selected,
        coins=sorted(item.coin for item in configs),
        canonicalized_features=canonicalized_features,
        account_wide_conflicts=conflicts,
        override_leaf_counts={
            coin: _count_leaves(patch) for coin, patch in master["coin_overrides"].items()
        },
        included_backtest_optimize=include_backtest_optimize,
    )
    return master, report


def compose_directory(
    input_directory: Path,
    *,
    output_config: Path | None = None,
    master_config: Path | None = None,
    include_backtest_optimize: bool = False,
) -> tuple[dict, CompositionReport]:
    configs, master_was_selected = load_single_coin_directory(
        input_directory,
        output_config=output_config,
        master_config=master_config,
    )
    return compose_configs(
        configs,
        include_backtest_optimize=include_backtest_optimize,
        master_was_selected=master_was_selected,
    )


def write_config(config: dict, output_config: Path, *, overwrite: bool = False) -> None:
    output = output_config.expanduser()
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output config already exists: {output}; pass --overwrite to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json_dumps_streamlined(config, indent=4, max_inline=72, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def print_report(report: CompositionReport, output_config: Path) -> None:
    print(
        f"Validated {len(report.source_paths)} single-coin configs: "
        + ", ".join(report.coins)
    )
    selection = "selected" if report.master_was_selected else "alphabetically first"
    print(f"Master source ({selection}): {report.master_path}")
    if report.included_backtest_optimize:
        print(f"Included backtest and optimize sections from: {report.master_path}")
    else:
        print("Omitted backtest and optimize sections for a lean live config")
    for description in report.canonicalized_features:
        print(f"Canonicalized inactive feature: {description}")
    if report.account_wide_conflicts:
        print("Kept master values for differing non-overridable/account-wide parameters:")
        for path, descriptions in sorted(report.account_wide_conflicts.items()):
            print(f"  {path}")
            for description in descriptions:
                print(f"    {description}")
    else:
        print("No differing non-overridable/account-wide parameters")
    if report.override_leaf_counts:
        rendered = ", ".join(
            f"{coin}={count}" for coin, count in sorted(report.override_leaf_counts.items())
        )
        print(f"Coin override leaves: {rendered}")
    else:
        print("Coin override leaves: none")
    print(f"Wrote composed config: {output_config}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, report = compose_directory(
            args.input_directory,
            output_config=args.output_config,
            master_config=args.master_config,
            include_backtest_optimize=args.include_backtest_optimize,
        )
        write_config(config, args.output_config, overwrite=args.overwrite)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"compose-coin-overrides: {exc}", file=sys.stderr)
        return 2
    print_report(report, args.output_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
