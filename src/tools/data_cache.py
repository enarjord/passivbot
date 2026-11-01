"""CLI entrypoint for portable offline cache snapshots."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main(argv=None):
    from data_cache import main as run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
