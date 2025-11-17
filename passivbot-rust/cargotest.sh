#!/usr/bin/env zsh
set -euo pipefail

script_dir="$(cd -- "$(dirname "$0")" && pwd)"
cd "$script_dir"

py_lib=$(
    python3 - <<'PY'
import sys
import sysconfig
from pathlib import Path

lib_names = []
for key in ("LDLIBRARY", "INSTSONAME"):
    val = sysconfig.get_config_var(key)
    if val:
        lib_names.append(val)
ldversion = sysconfig.get_config_var("LDVERSION") or sysconfig.get_config_var("py_version_short")
if ldversion:
    lib_names.append(f"libpython{ldversion}.dylib")
search_dirs = [
    sysconfig.get_config_var("LIBDIR"),
    sysconfig.get_config_var("LIBPL"),
    Path(sysconfig.get_config_var("LIBDIR") or "").parent,
    Path(sys.base_prefix) / "lib",
    Path(sys.base_prefix) / "lib64",
]
for libdir in search_dirs:
    if not libdir:
        continue
    libdir = Path(libdir)
    for name in lib_names:
        candidate = libdir / name
        if candidate.is_file():
            print(candidate)
            raise SystemExit(0)
raise SystemExit("libpython.dylib not found; ensure Python dev libraries are installed.")
PY
) || {
    printf 'Failed to locate libpython for DYLD insertion. Aborting tests.\n' >&2
    exit 1
}

export RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup"
export DYLD_INSERT_LIBRARIES="$py_lib"

cargo test "$@"
