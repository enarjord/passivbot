#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: install_passivbot.sh [--upgrade-packaging] [--link-passivbot PATH]

Creates (or reuses) the risk-management virtual environment without touching
your existing Passivbot installation.  By default the environment is isolated
from Passivbot; provide --link-passivbot /path/to/passivbot/src if you want the
environment to import Passivbot's source tree directly.  Pass
--upgrade-packaging to refresh pip, setuptools, and wheel inside the virtual
environment.

Options:
  --upgrade-packaging      Upgrade pip/setuptools/wheel inside the venv.
  --link-passivbot PATH    Write a .pth file pointing at PATH for imports.
  -h, --help               Show this message and exit.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv_passivbot_risk"

UPGRADE_PACKAGING=false
PASSIVBOT_LINK=""

while (($#)); do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --upgrade-packaging)
            UPGRADE_PACKAGING=true
            shift
            ;;
        --link-passivbot)
            shift
            if [ $# -eq 0 ]; then
                echo "Error: --link-passivbot requires a path argument." >&2
                usage >&2
                exit 1
            fi
            PASSIVBOT_LINK="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

PYTHON_BIN="python3"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python3 is required but not found. Please install Python 3.9 or newer." >&2
    exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
    echo "Reusing existing virtual environment at ${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

if [ "${UPGRADE_PACKAGING}" = true ]; then
    if ! pip install --upgrade pip setuptools wheel; then
        echo "Warning: Unable to upgrade pip/setuptools/wheel; continuing with existing versions." >&2
    fi
else
    echo "Skipped upgrading pip/setuptools/wheel (use --upgrade-packaging to enable)."
fi

if [ -n "${PASSIVBOT_LINK}" ]; then
    if [ ! -d "${PASSIVBOT_LINK}" ]; then
        echo "Error: --link-passivbot path '${PASSIVBOT_LINK}' does not exist or is not a directory." >&2
        exit 1
    fi
    SITE_PACKAGES=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
    PTH_FILE="${SITE_PACKAGES}/passivbot-risk-path.pth"
    python - <<PYTHON
from pathlib import Path
import sys

target = Path(r"${PASSIVBOT_LINK}").expanduser().resolve()
if not target.exists():
    sys.exit("Resolved link path does not exist: {}".format(target))

pth_path = Path(r"${PTH_FILE}")
pth_path.write_text(str(target) + "\n", encoding="utf-8")
print(f"Linked Passivbot source tree via {pth_path}")
PYTHON
else
    echo "No Passivbot path linked. Use --link-passivbot PATH to enable imports."
fi

echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
