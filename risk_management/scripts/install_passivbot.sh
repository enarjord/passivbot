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


usage() {
    cat <<'EOF'
Usage: install_passivbot.sh [--upgrade-packaging]

Creates (or reuses) the risk-management virtual environment and ensures the
Passivbot source tree is importable from that environment without installing
Passivbot itself.  Pass `--upgrade-packaging` if you would like the script to
refresh `pip`, `setuptools`, and `wheel` inside the virtual environment.
EOF
}



usage() {
    cat <<'EOF'
Usage: install_passivbot.sh [--install-passivbot] [--] [pip flags]

Creates (or reuses) the risk-management virtual environment and ensures the
Passivbot source tree is importable from that environment.  Pass
`--install-passivbot` to also install Passivbot into the environment using pip.

Any arguments following `--` are forwarded to the underlying `pip install`
invocation when `--install-passivbot` is provided.  When no custom pip flags are
supplied, the script defaults to `--use-pep517`.
EOF
}



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv_passivbot_risk"


UPGRADE_PACKAGING=false

while (($#)); do
    case "$1" in


INSTALL_PASSIVBOT=false
PIP_INSTALL_FLAGS=()

while (($#)); do
    case "$1" in
        --install-passivbot)
            INSTALL_PASSIVBOT=true
            shift
            ;;


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


        --upgrade-packaging)
            UPGRADE_PACKAGING=true
            shift

        --)
            shift
            PIP_INSTALL_FLAGS=("$@")
            break


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




if ! pip install --upgrade pip setuptools wheel; then
    echo "Warning: Unable to upgrade pip/setuptools/wheel; continuing with existing versions." >&2

fi

SITE_PACKAGES=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
PTH_FILE="${SITE_PACKAGES}/passivbot-risk-path.pth"
echo "${REPO_ROOT}/src" >"${PTH_FILE}"
echo "Linked Passivbot source tree via ${PTH_FILE}"



if [ "${INSTALL_PASSIVBOT}" = true ]; then
    pip install --upgrade setuptools-rust

    if [ ${#PIP_INSTALL_FLAGS[@]} -eq 0 ]; then
        PIP_INSTALL_FLAGS=("--use-pep517")
    else
        echo "Using custom pip install flags: ${PIP_INSTALL_FLAGS[*]}"
    fi

    pip install "${PIP_INSTALL_FLAGS[@]}" -e "${REPO_ROOT}"
    echo "Passivbot has been installed into ${VENV_DIR}."
else
    echo "Skipped pip installation of Passivbot (use --install-passivbot to enable)."
fi


pip install --upgrade pip setuptools wheel


# Install build prerequisites required by Passivbot's setup.py
pip install --upgrade setuptools-rust

# Install passivbot from repository root in editable mode, defaulting to a PEP 517 build
PIP_INSTALL_FLAGS=("$@")
if [ ${#PIP_INSTALL_FLAGS[@]} -eq 0 ]; then
    PIP_INSTALL_FLAGS=("--use-pep517")
else
    echo "Using custom pip install flags: ${PIP_INSTALL_FLAGS[*]}"
fi

pip install "${PIP_INSTALL_FLAGS[@]}" -e "${REPO_ROOT}"


# Install build prerequisites required by Passivbot's setup.py
pip install --upgrade setuptools-rust

# Install passivbot from repository root in editable mode
pip install -e "${REPO_ROOT}"


echo "Passivbot has been installed into ${VENV_DIR}."



echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
