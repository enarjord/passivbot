#!/usr/bin/env bash
set -euo pipefail

# ---------- helpers ----------
log() { printf "\033[1;34m[install]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*" >&2; }
die() { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: install_passivbot.sh [options] [-- [pip flags]]

Creates (or reuses) a virtual environment for risk management. You can either:
  • link a Passivbot source tree into the venv via a .pth file, or
  • install Passivbot into the venv (editable by default), optionally with custom pip flags.

Options:
  --venv DIR                Virtualenv directory (default: <project>/.venv_passivbot_risk)
  --python PATH             Python interpreter to use (default: python3)
  --upgrade-packaging       Upgrade pip/setuptools/wheel in the venv
  --link-passivbot PATH     Write a .pth pointing at PATH so it’s importable (mutually exclusive with --install-passivbot)
  --install-passivbot[=DIR] Install Passivbot into the venv (editable). If DIR omitted, uses repo root detected from script.
  -h, --help                Show this help

Forwarding flags:
  Anything after `--` is forwarded to `pip install` (only when using --install-passivbot).
  If omitted, defaults to: --use-pep517

Examples:
  Link only:
    ./install_passivbot.sh --link-passivbot /path/to/passivbot/src --upgrade-packaging

  Install with custom flags:
    ./install_passivbot.sh --install-passivbot=../passivbot -- --no-build-isolation -e

Notes:
  • --link-passivbot and --install-passivbot are mutually exclusive.
  • Requires Python >= 3.9.
EOF
}

# ---------- defaults ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# One level above project (often the repo root of Passivbot if placed in a tools/ subdir)
DEFAULT_REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
VENV_DIR_DEFAULT="${PROJECT_ROOT}/.venv_passivbot_risk"

PYTHON_BIN="python3"
VENV_DIR="${VENV_DIR_DEFAULT}"
UPGRADE_PACKAGING=false
LINK_PATH=""
INSTALL_MODE=false
INSTALL_REPO_DIR=""
PIP_INSTALL_FLAGS=()

# ---------- parse args ----------
# Support --install-passivbot[=DIR] form as well as space-separated.
while (($#)); do
  case "$1" in
    --venv)
      shift; [[ $# -gt 0 ]] || die "--venv requires a DIR"
      VENV_DIR="$1"; shift
      ;;
    --python)
      shift; [[ $# -gt 0 ]] || die "--python requires a PATH"
      PYTHON_BIN="$1"; shift
      ;;
    --upgrade-packaging)
      UPGRADE_PACKAGING=true; shift
      ;;
    --link-passivbot)
      shift; [[ $# -gt 0 ]] || die "--link-passivbot requires a PATH"
      LINK_PATH="$1"; shift
      ;;
    --install-passivbot)
      INSTALL_MODE=true; shift
      # Optional repo dir if next token is not a flag or the special `--`
      if [[ $# -gt 0 && "$1" != --* ]]; then
        INSTALL_REPO_DIR="$1"; shift
      fi
      ;;
    --install-passivbot=*)
      INSTALL_MODE=true
      INSTALL_REPO_DIR="${1#*=}"
      shift
      ;;
    --)
      shift
      # Everything else goes to pip (if we are installing)
      PIP_INSTALL_FLAGS=("$@")
      break
      ;;
    -h|--help)
      usage; exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

# ---------- validate mode ----------
if [[ -n "$LINK_PATH" && "$INSTALL_MODE" == true ]]; then
  die "Choose either --link-passivbot or --install-passivbot, not both."
fi

# ---------- python checks ----------
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  die "Python not found at '${PYTHON_BIN}'. Please install Python 3.9+ or pass --python PATH."
fi

PY_VERSION="$("${PYTHON_BIN}" - <<'PY'
import sys
print(".".join(map(str, sys.version_info[:3])))
PY
)"
# rudimentary version check (major.minor)
PY_MAJOR_MINOR="${PY_VERSION%.*}"
PY_MAJOR="${PY_MAJOR_MINOR%%.*}"
PY_MINOR="${PY_MAJOR_MINOR##*.}"
if (( PY_MAJOR < 3 || (PY_MAJOR==3 && PY_MINOR < 9) )); then
  die "Python >= 3.9 required (found ${PY_VERSION})."
fi

# ---------- create or reuse venv ----------
if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  log "Reusing existing virtual environment at ${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# ---------- upgrade packaging (optional) ----------
if [[ "${UPGRADE_PACKAGING}" == true ]]; then
  log "Upgrading pip/setuptools/wheel ..."
  if ! pip install --upgrade pip setuptools wheel; then
    warn "Unable to upgrade pip/setuptools/wheel; continuing with existing versions."
  fi
else
  log "Skipped packaging upgrades (use --upgrade-packaging to enable)."
fi

# ---------- link mode ----------
if [[ -n "${LINK_PATH}" ]]; then
  [[ -d "${LINK_PATH}" ]] || die "Link path does not exist or is not a directory: ${LINK_PATH}"

  log "Linking Passivbot source tree via .pth ..."
  python - <<PY
import sys, sysconfig
from pathlib import Path

target = Path(r"${LINK_PATH}").expanduser().resolve()
if not target.exists() or not target.is_dir():
    sys.exit(f"Target path invalid: {target}")

site_packages = Path(sysconfig.get_path("purelib"))
pth_path = site_packages / "passivbot-risk-path.pth"
pth_path.write_text(str(target) + "\n", encoding="utf-8")
print(f"Created {pth_path} -> {target}")
PY
  log "Done. Passivbot will be importable from ${LINK_PATH}."
fi

# ---------- install mode ----------
if [[ "${INSTALL_MODE}" == true ]]; then
  # Default repo path if not supplied
  REPO_DIR="${INSTALL_REPO_DIR:-${DEFAULT_REPO_ROOT}}"
  [[ -d "${REPO_DIR}" ]] || die "Install repo dir not found: ${REPO_DIR}"

  log "Ensuring build prereqs ..."
  # setuptools-rust is sometimes required by transitive deps in crypto envs
  pip install --upgrade setuptools-rust >/dev/null || warn "setuptools-rust upgrade failed; continuing"

  if [[ ${#PIP_INSTALL_FLAGS[@]} -eq 0 ]]; then
    PIP_INSTALL_FLAGS=(--use-pep517)
  fi

  log "Installing Passivbot from: ${REPO_DIR}"
  log "pip install flags: ${PIP_INSTALL_FLAGS[*]}"
  pip install "${PIP_INSTALL_FLAGS[@]}" -e "${REPO_DIR}"
  log "Passivbot installed into ${VENV_DIR}."
fi

# ---------- summary ----------
log "All set! Activate the environment with:"
echo "  source \"${VENV_DIR}/bin/activate\""
if [[ -n "${LINK_PATH}" ]]; then
  echo "Passivbot is linked from: ${LINK_PATH}"
elif [[ "${INSTALL_MODE}" == true ]]; then
  echo "Passivbot is installed from: ${INSTALL_REPO_DIR:-$DEFAULT_REPO_ROOT}"
else
  echo "No Passivbot link or install was requested."
fi
