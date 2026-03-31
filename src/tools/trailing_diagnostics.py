from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
try:
    sys.path.remove(str(TOOLS_ROOT))
except ValueError:
    pass
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trailing_diagnostics_tool import build_parser, run_interactive


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_interactive(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if not getattr(args, "wizard", False):
            print("hint: restart the bot to refresh monitor fields, or rerun with --wizard", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
