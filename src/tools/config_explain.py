"""Explain configuration values, local sources and optimizer treatment offline."""

import argparse
import json
from pathlib import Path

from config.authoring import explain_config
from config.parse import load_raw_config
from json_utils import json_dumps_streamlined


def main(argv=None):
    parser = argparse.ArgumentParser(prog="passivbot tool config-explain", description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("path", help="Canonical dotted configuration path")
    parser.add_argument("--coin", help="Inspect an effective coin override")
    parser.add_argument("--scenario", help="Apply the scenario with this explicit label")
    parser.add_argument("--optimize", action="store_true", help="Apply fixed runtime policy and optimizer transforms")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = parser.parse_args(argv)
    try:
        report = explain_config(
            load_raw_config(str(args.config), log_errors=False), args.path,
            base_config_path=str(args.config.resolve()), coin=args.coin,
            scenario=args.scenario, optimize=args.optimize,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json_dumps_streamlined(report, allow_nan=False))
    else:
        print(f"{report['path']} = {json.dumps(report['value'], ensure_ascii=False)}")
        print("type: " + (" | ".join(report["types"]) if report["types"] else "unknown"))
        for name in ("scope", "unit", "denominator"):
            print(f"{name}: {report[name] if report[name] is not None else 'unknown'}")
        print("default: " + (json.dumps(report["default"]["value"]) if report["default"]["declared"] else "not declared"))
        for event in report["sources"]:
            print("source: " + json.dumps(event, ensure_ascii=False))
        print("search: " + json.dumps(report["search"], ensure_ascii=False))
        if report["unresolved"]:
            print("unresolved offline: " + ", ".join(report["unresolved"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
