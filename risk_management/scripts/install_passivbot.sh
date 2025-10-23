#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv_passivbot_risk"

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

pip install --upgrade pip setuptools wheel


# Install build prerequisites required by Passivbot's setup.py
pip install --upgrade setuptools-rust

# Install passivbot from repository root in editable mode
pip install -e "${REPO_ROOT}"

echo "Passivbot has been installed into ${VENV_DIR}."
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
