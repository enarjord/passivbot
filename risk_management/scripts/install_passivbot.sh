#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: install_passivbot.sh [--upgrade-packaging]

Creates (or reuses) the risk-management virtual environment and ensures the
Passivbot source tree is importable from that environment without installing
Passivbot itself.  Pass `--upgrade-packaging` if you would like the script to
refresh `pip`, `setuptools`, and `wheel` inside the virtual environment.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv_passivbot_risk"

UPGRADE_PACKAGING=false

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

SITE_PACKAGES=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
PTH_FILE="${SITE_PACKAGES}/passivbot-risk-path.pth"
echo "${REPO_ROOT}/src" >"${PTH_FILE}"
echo "Linked Passivbot source tree via ${PTH_FILE}"

echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
