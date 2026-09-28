"""Offline, explicit configuration migration to revised HSL; never deploys a bot."""
from argparse import ArgumentParser
from copy import deepcopy
import json
from pathlib import Path
import sys

from config import prepare_config
from config.load import load_input_config
from config_utils import strip_config_metadata


def migrate(source, *, restart_policies=None, portfolio=None):
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
    bot = result["bot"]
    if portfolio is not None:
        if live.get("hsl_signal_mode", "coin") != "unified":
            raise ValueError("--portfolio-policy applies only to unified signal mode")
        if "hsl" in bot:
            raise ValueError("bot.hsl already exists; edit it explicitly instead of replacing it")
        bot["hsl"] = deepcopy(portfolio)
    for scope, policy in (restart_policies or {}).items():
        if scope not in {"long", "short", "portfolio"} or policy not in {"always", "never"}:
            raise ValueError("restart choices must be long|short|portfolio=always|never")
        if scope == "portfolio":
            if live.get("hsl_signal_mode") != "unified" or "hsl" not in bot:
                raise ValueError("portfolio restart choice requires an explicit unified bot.hsl block")
            block = bot["hsl"]
        else:
            if live.get("hsl_signal_mode") == "unified":
                raise ValueError("side restart choices are inactive in unified mode; choose portfolio")
            block = bot.setdefault(scope, {}).setdefault("hsl", {})
        block["restart_after_red_policy"] = policy
    prepared = prepare_config(result, verbose=True, target="canonical", runtime=None)
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
        output = migrate(source, restart_policies=choices, portfolio=portfolio)
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
