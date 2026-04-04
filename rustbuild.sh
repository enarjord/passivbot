#!/usr/bin/env bash

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "This script must be run with bash (e.g. 'bash rustbuild.sh')." >&2
    exit 1
fi

set -euo pipefail

show_help() {
    cat <<'EOF'
Usage: rustbuild.sh [options]

Build and install the Rust extension in editable mode.

Options:
  -h, --help   Show this help message and exit.

The script runs:
  cd passivbot-rust
  cargo fmt
  maturin develop --release
  cd ..
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/passivbot-rust"
cargo fmt
maturin develop --release
cd "$script_dir"
PYTHONPATH="$script_dir/src${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
from rust_utils import source_fingerprint, stamp_compiled_extensions, sync_installed_extension_into_src

sync_installed_extension_into_src()
stamp_compiled_extensions(source_fingerprint())
PY
