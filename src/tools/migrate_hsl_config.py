"""Offline, explicit configuration migration to revised HSL; never deploys a bot."""
from argparse import ArgumentParser
from copy import deepcopy
import json
import logging
from pathlib import Path
import sys

from config import prepare_config
from config.hsl_revised import _mode, validate_parameter_path
from config.load import load_input_config
from config.optimize_bounds import flatten_optimize_bounds
from config.overrides import parse_overrides
from config.param_paths import require_existing_config_path, resolve_bound_selectors
from config_utils import strip_config_metadata
from optimization.warmup import _finalize_optimizer_vector_config
from suite_runner import apply_scenario_overrides, build_scenarios


def migrate(source, *, restart_policies=None, portfolio=None, base_config_path=""):
    """Use the canonical loader after applying only explicit operator choices.

    The loader owns retired-field diagnostics, optimizer-path rejection, lookback
    bounds and policy validation. Do not duplicate those rules in this utility.
    """
    result = deepcopy(source)
    if not isinstance(result, dict):
        raise ValueError("configuration must be a mapping")
    if "config" in result and "bot" not in result:
        result = result["config"]
    if not isinstance(result, dict) or not isinstance(result.get("bot"), dict):
        raise ValueError("expected a current bot configuration; migrate older config formats first")
    live = result.setdefault("live", {})
    if not isinstance(live, dict):
        raise ValueError("live must be a mapping")
    live["hsl_engine"] = "revised"
    mode = live["hsl_signal_mode"] = _mode(result)
    bot = result["bot"]
    chosen_paths = {}
    if portfolio is not None:
        if mode != "unified":
            raise ValueError("--portfolio-policy applies only to unified signal mode")
        if "hsl" in bot:
            raise ValueError("bot.hsl already exists; edit it explicitly instead of replacing it")
        bot["hsl"] = deepcopy(portfolio)
    for scope, policy in (restart_policies or {}).items():
        if scope not in {"long", "short", "portfolio"} or policy not in {"always", "never"}:
            raise ValueError("restart choices must be long|short|portfolio=always|never")
        if scope == "portfolio":
            if mode != "unified" or "hsl" not in bot:
                raise ValueError("portfolio restart choice requires an explicit unified bot.hsl block")
            block = bot["hsl"]
        else:
            if mode == "unified":
                raise ValueError("side restart choices are inactive in unified mode; choose portfolio")
            block = bot.setdefault(scope, {}).setdefault("hsl", {})
        block["restart_after_red_policy"] = policy
        path = ("bot", "hsl") if scope == "portfolio" else ("bot", scope, "hsl")
        chosen_paths[(*path, "restart_after_red_policy")] = policy
    prepared = prepare_config(result, verbose=True, target="canonical", runtime=None,
                              base_config_path=base_config_path)
    # An explicit migration choice must also survive optimizer policy application.
    fixed = prepared.get("optimize", {}).get("fixed_runtime_overrides", {})
    for selector in fixed:
        path = require_existing_config_path(prepared, selector)
        if path in chosen_paths and fixed[selector] != chosen_paths[path]:
            logging.warning("Updating optimize.fixed_runtime_overrides[%s] to explicit choice %s",
                            selector, chosen_paths[path])
            fixed[selector] = chosen_paths[path]
    bounds = flatten_optimize_bounds(prepared["optimize"]["bounds"],
                                     strategy_kind=prepared["live"]["strategy_kind"])
    for selector in prepared["optimize"].get("fixed_params", []):
        validate_parameter_path(selector, mode)
        if not resolve_bound_selectors(prepared, [selector], bounds):
            raise ValueError(f"optimize.fixed_params selector {selector!r} matches no active bounds")
    # Runtime's canonical override stage validates files as well as inline patches.
    # Materializing its result preserves file-then-inline precedence when output moves.
    scenario_base = deepcopy(prepared)
    prepared = parse_overrides(prepared, verbose=True)
    # Validate the actual optimizer policy on a copy. Fixed enablement can activate
    # a restart policy that was valid only while the base scope was disabled.
    optimized = _finalize_optimizer_vector_config(deepcopy(prepared))
    optimized = prepare_config(optimized, verbose=False, target="canonical", runtime=None)
    optimized = parse_overrides(optimized, verbose=False)
    for path, chosen in chosen_paths.items():
        effective_choice = optimized
        for key in path:
            effective_choice = effective_choice[key]
        if effective_choice != chosen:
            raise ValueError(
                f"optimizer changes explicit {'.'.join(path)} choice from {chosen!r} "
                f"to {effective_choice!r}; reconcile optimize.enable_overrides "
                "(including mirror_short_from_long) with the restart choices"
            )
    if prepared["backtest"].get("scenarios"):
        scenarios, _ = build_scenarios(prepared["backtest"])
        for raw, scenario in zip(prepared["backtest"]["scenarios"], scenarios):
            try:
                for path in scenario.overrides or {}:
                    if require_existing_config_path(scenario_base, path)[0] == "optimize":
                        raise ValueError("scenario optimizer controls are not applied; move them to top-level optimize")
                effective = deepcopy(scenario_base)
                apply_scenario_overrides(effective, scenario.overrides)
                effective = parse_overrides(effective, verbose=False)
                # Atomic coin mappings replace the base mapping, including {}.
                # Persist only the patch, not unrelated scenario-effective fields.
                overrides = deepcopy(scenario.overrides or {})
                coin_keys = [key for key in overrides
                             if key == "coin_overrides" or key.startswith("coin_overrides.")]
                if coin_keys:
                    for key in coin_keys:
                        del overrides[key]
                    overrides["coin_overrides"] = effective["coin_overrides"]
                optimized_scenario = deepcopy(optimized)
                apply_scenario_overrides(optimized_scenario, overrides)
                parse_overrides(optimized_scenario, verbose=False)
                if scenario.overrides is not None:
                    raw["overrides"] = overrides
            except (ValueError, TypeError, KeyError, OSError) as exc:
                raise ValueError(f"backtest scenario {scenario.label!r}: {exc}") from exc
    return strip_config_metadata(prepared)


def main(argv=None):
    parser = ArgumentParser(prog="passivbot tool migrate-hsl",
        description="Write a separate, canonically validated revised-HSL config. No exchange access.")
    parser.add_argument("input_config", type=Path)
    parser.add_argument("output_config", type=Path)
    parser.add_argument("--restart-policy", action="append", default=[], metavar="SCOPE=POLICY",
                        help="Explicit long, short or portfolio choice: always or never. Repeat per scope.")
    parser.add_argument("--portfolio-policy", type=Path,
                        help="JSON object containing the explicit unified bot.hsl policy")
    args = parser.parse_args(argv)
    try:
        if args.input_config.resolve() == args.output_config.resolve():
            raise ValueError("input and output must be different files")
        if args.output_config.exists():
            raise ValueError("output already exists; choose a new output path")
        choices = {}
        for item in args.restart_policy:
            parts = item.split("=")
            if len(parts) != 2 or parts[0] in choices:
                raise ValueError("supply each restart scope once as SCOPE=always or SCOPE=never")
            choices[parts[0]] = parts[1]
        source, _, _ = load_input_config(str(args.input_config), log_info=False)
        portfolio = (json.loads(args.portfolio_policy.read_text())
                     if args.portfolio_policy is not None else None)
        output = migrate(source, restart_policies=choices, portfolio=portfolio,
                         base_config_path=str(args.input_config))
        serialized = json.dumps(output, indent=4, allow_nan=False) + "\n"
        # Exclusive creation also closes the race after the existence check.
        with args.output_config.open("x") as stream:
            stream.write(serialized)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print("Wrote a revised-HSL configuration. No bot was started or changed.")
    print("Re-backtest thresholds: revised equity-peak drawdown, current-only RED and lookback-bounded "
          "restart semantics are not equivalent to legacy HSL. Do not reuse legacy optimizer fitness.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
