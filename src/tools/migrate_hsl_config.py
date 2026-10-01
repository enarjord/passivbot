"""Offline, explicit configuration migration to HSL; never deploys a bot."""

from argparse import ArgumentParser
from copy import deepcopy
import json
import logging
from pathlib import Path
import sys

from config import prepare_config
from config.hsl import _mode, validate_parameter_path, validate_optimizer_metrics
from config.load import load_input_config
from config.migrations import migrate_config_version
from config.optimize_bounds import flatten_optimize_bounds
from config.overrides import normalize_coin_override_keys, parse_overrides
from config.param_paths import (
    require_existing_config_path,
    resolve_bound_selectors,
    resolve_dotted_config_path,
)
from config.scoring import (
    extract_objective_specs,
    default_scoring_weights,
    objective_index_map,
    resolve_objective_basis,
)
from limit_utils import expand_limit_checks
from config_utils import strip_config_metadata
from optimization.warmup import _finalize_optimizer_vector_config
from suite_runner import apply_scenario_overrides, build_scenarios


def _validate_optimizer_inputs(candidate, authored, *, label=None, reducer_cfg=None):
    """Reuse runtime metric selection and static GPU guards without hardware access."""
    specs = extract_objective_specs(authored)
    optimize = authored["optimize"]
    checks = expand_limit_checks(
        optimize.get("limits", []),
        default_scoring_weights(),
        penalty_weight=1e6,
        objective_index_map=objective_index_map(specs),
        reducer_cfg=reducer_cfg,
    )
    if label is None:
        metrics = [
            *(spec.metric for spec in specs),
            *(check["metric"] for check in checks),
        ]
    else:
        default_scenario = optimize.get("objective_scenario")
        default_scenario = (
            str(default_scenario).strip() or None
            if default_scenario is not None
            else None
        )
        bases = [
            resolve_objective_basis(
                spec, default_scenario=default_scenario, reducer_cfg=reducer_cfg
            )
            for spec in specs
        ]
        metrics = [
            *(
                spec.metric
                for spec, basis in zip(specs, bases)
                if basis.scenario is None or basis.scenario == label
            ),
            *(
                check["metric"]
                for check in checks
                if check.get("scenario") is None or check["scenario"] == label
            ),
        ]
    validate_optimizer_metrics(candidate, metrics)
    if optimize.get("backend") == "gpu":
        from optimization.backends.gpu_backend import _validate_hsl_gpu_inputs

        _validate_hsl_gpu_inputs(candidate)


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
        raise ValueError(
            "expected a current bot configuration; migrate older config formats first"
        )
    live = result.setdefault("live", {})
    if not isinstance(live, dict):
        raise ValueError("live must be a mapping")
    live.pop("hsl_engine", None)
    mode = live["hsl_signal_mode"] = _mode(result)
    bot = result["bot"]
    chosen_paths = {}
    if portfolio is not None:
        if mode != "unified":
            raise ValueError("--portfolio-policy applies only to unified signal mode")
        if "hsl" in bot:
            raise ValueError(
                "bot.hsl already exists; edit it explicitly instead of replacing it"
            )
        bot["hsl"] = deepcopy(portfolio)
        if isinstance(portfolio, dict) and "restart_after_red_policy" in portfolio:
            chosen_paths[("bot", "hsl", "restart_after_red_policy")] = portfolio[
                "restart_after_red_policy"
            ]
    for scope, policy in (restart_policies or {}).items():
        if scope not in {"long", "short", "portfolio"} or policy not in {
            "always",
            "never",
        }:
            raise ValueError(
                "restart choices must be long|short|portfolio=always|never"
            )
        if scope == "portfolio":
            if mode != "unified" or "hsl" not in bot:
                raise ValueError(
                    "portfolio restart choice requires an explicit unified bot.hsl block"
                )
            block = bot["hsl"]
        else:
            if mode == "unified":
                raise ValueError(
                    "side restart choices are inactive in unified mode; choose portfolio"
                )
            block = bot.setdefault(scope, {}).setdefault("hsl", {})
        block["restart_after_red_policy"] = policy
        path = ("bot", "hsl") if scope == "portfolio" else ("bot", scope, "hsl")
        chosen_paths[(*path, "restart_after_red_policy")] = policy
    # This command is the explicit semantic migration boundary. Validate and
    # upgrade the schema here before the ordinary loader checks old HSL inputs;
    # never bypass malformed/future/unsupported schema rejection.
    migrate_config_version(result, verbose=True)
    prepared = prepare_config(
        result,
        verbose=True,
        target="canonical",
        runtime=None,
        base_config_path=base_config_path,
    )
    # Compare canonical choices, not accepted input spellings such as " Never ".
    for path in chosen_paths:
        value = prepared
        for key in path:
            value = value[key]
        chosen_paths[path] = value
    # An explicit migration choice must also survive optimizer policy application.
    fixed = prepared.get("optimize", {}).get("fixed_runtime_overrides", {})
    for selector in fixed:
        path = require_existing_config_path(prepared, selector)
        if path in chosen_paths and fixed[selector] != chosen_paths[path]:
            logging.warning(
                "Updating optimize.fixed_runtime_overrides[%s] to explicit choice %s",
                selector,
                chosen_paths[path],
            )
            fixed[selector] = chosen_paths[path]
    bounds = flatten_optimize_bounds(
        prepared["optimize"]["bounds"], strategy_kind=prepared["live"]["strategy_kind"]
    )
    for selector in prepared["optimize"].get("fixed_params", []):
        validate_parameter_path(selector, mode)
        if not resolve_bound_selectors(prepared, [selector], bounds):
            raise ValueError(
                f"optimize.fixed_params selector {selector!r} matches no active bounds"
            )
    # Runtime's canonical override stage validates files as well as inline patches.
    # Materializing its result preserves file-then-inline precedence when output moves.
    scenario_base = deepcopy(prepared)
    scenario_base["coin_overrides"] = normalize_coin_override_keys(
        scenario_base["coin_overrides"], verbose=False
    )
    prepared = parse_overrides(prepared, verbose=True)
    # Validate the actual optimizer policy on a copy. Fixed enablement can activate
    # a restart policy that was valid only while the base scope was disabled.
    optimized = _finalize_optimizer_vector_config(deepcopy(prepared))
    optimized = prepare_config(
        optimized, verbose=False, target="canonical", runtime=None
    )
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
        scenarios, reducer_cfg = build_scenarios(prepared["backtest"])
        for raw, scenario in zip(prepared["backtest"]["scenarios"], scenarios):
            try:
                effective = deepcopy(scenario_base)
                # A dotted scenario path may address a leaf provided by a file.
                # Seed only that addressed path from the already validated view;
                # copying all materialized leaves would make the old file's values
                # override a replacement scenario file.
                for selector in scenario.overrides or {}:
                    try:
                        path = require_existing_config_path(prepared, selector)
                    except KeyError:
                        # Authored file paths remain valid in effective. Unknown
                        # paths are diagnosed by the canonical application below.
                        continue
                    if path[0] != "coin_overrides":
                        continue
                    target, source = effective, prepared
                    for key in path[:-1]:
                        source = source[key]
                        target = target.setdefault(key, {})
                    target.setdefault(path[-1], deepcopy(source[path[-1]]))
                for path in scenario.overrides or {}:
                    resolved = resolve_dotted_config_path(effective, path)
                    if resolved and resolved[0] == "optimize":
                        raise ValueError(
                            "scenario optimizer controls are not applied; move them to top-level optimize"
                        )
                apply_scenario_overrides(effective, scenario.overrides)
                effective = parse_overrides(effective, verbose=False)
                # Atomic coin mappings replace the base mapping, including {}.
                # Persist only the patch, not unrelated scenario-effective fields.
                overrides = deepcopy(scenario.overrides or {})
                coin_keys = [
                    key
                    for key in overrides
                    if key == "coin_overrides" or key.startswith("coin_overrides.")
                ]
                if coin_keys:
                    for key in coin_keys:
                        del overrides[key]
                    overrides["coin_overrides"] = effective["coin_overrides"]
                optimized_scenario = deepcopy(optimized)
                apply_scenario_overrides(optimized_scenario, overrides)
                optimized_scenario = parse_overrides(optimized_scenario, verbose=False)
                _validate_optimizer_inputs(
                    optimized_scenario,
                    prepared,
                    label=scenario.label,
                    reducer_cfg=reducer_cfg,
                )
                if scenario.overrides is not None:
                    raw["overrides"] = overrides
            except (ValueError, TypeError, KeyError, OSError) as exc:
                raise ValueError(
                    f"backtest scenario {scenario.label!r}: {exc}"
                ) from exc
    else:
        _validate_optimizer_inputs(optimized, prepared)
    return strip_config_metadata(prepared)


def main(argv=None):
    parser = ArgumentParser(
        prog="passivbot tool migrate-hsl",
        description="Write a separate, canonically validated HSL config. No exchange access.",
    )
    parser.add_argument("input_config", type=Path)
    parser.add_argument("output_config", type=Path)
    parser.add_argument(
        "--restart-policy",
        action="append",
        default=[],
        metavar="SCOPE=POLICY",
        help="Explicit long, short or portfolio choice: always or never. Repeat per scope.",
    )
    parser.add_argument(
        "--portfolio-policy",
        type=Path,
        help="JSON object containing the explicit unified bot.hsl policy",
    )
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
                raise ValueError(
                    "supply each restart scope once as SCOPE=always or SCOPE=never"
                )
            choices[parts[0]] = parts[1]
        source, _, _ = load_input_config(str(args.input_config), log_info=False)
        portfolio = (
            json.loads(args.portfolio_policy.read_text())
            if args.portfolio_policy is not None
            else None
        )
        output = migrate(
            source,
            restart_policies=choices,
            portfolio=portfolio,
            base_config_path=str(args.input_config),
        )
        serialized = json.dumps(output, indent=4, allow_nan=False) + "\n"
        # Exclusive creation also closes the race after the existence check.
        with args.output_config.open("x") as stream:
            stream.write(serialized)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print("Wrote a HSL configuration. No bot was started or changed.")
    print(
        "Re-backtest thresholds: equity-peak drawdown, current-only RED and lookback-bounded "
        "restart semantics are not equivalent to legacy HSL. Do not reuse legacy optimizer fitness.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
