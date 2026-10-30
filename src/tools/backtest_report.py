"""Regenerate views and optional exports from an existing result, offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtest_reports import add_report_arguments, options_from_args, render_reports
from backtest_results import ResultStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--scope", help="One native controller, e.g. BTC:long, long or portfolio."
    )
    add_report_arguments(parser)
    args = parser.parse_args(argv)
    options = options_from_args(args)
    from dataclasses import replace

    options = replace(options, scope=args.scope)
    status = render_reports(ResultStore(args.result), args.output_dir, options=options)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
