from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live.restart_smoke_targets import build_live_restart_target_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool live-restart-target-report",
        description=(
            "Resolve exact local tmux restart targets without signalling or "
            "starting processes."
        ),
    )
    parser.add_argument("supervisor_config", help="Tmuxp-style supervisor config.")
    parser.add_argument(
        "--session-name",
        required=True,
        help="Exact tmux session name to confirm and inspect.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact single-line JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_live_restart_target_report(
            args.supervisor_config,
            session_name=args.session_name,
            config_base_dir=Path.cwd(),
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
