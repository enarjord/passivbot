"""
Helpers for managing the compiled Rust extension.

This module is intentionally free of imports that would load the extension
itself; it only inspects filesystem state.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import time
from shutil import copy2
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

LOCK_FILE = Path("passivbot-rust/.compile.lock")
LOCK_TIMEOUT = 300  # seconds
LOCK_CHECK_INTERVAL = 2  # seconds
COMPILED_EXTENSION_NAME = "libpassivbot_rust"
PYTHON_MODULE_NAME = "passivbot_rust"


def _extension_suffixes() -> list[str]:
    # Prefer the exact interpreter suffix when available (e.g. `.cpython-312-darwin.so`),
    # but keep broad fallbacks for non-standard environments.
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if suffix:
        return [suffix.lstrip(".")]
    return ["so", "pyd", "dll", "dylib", "bundle", "sl"]


def _local_extension_candidates() -> list[Path]:
    exts = _extension_suffixes()
    out: list[Path] = []
    for ext in exts:
        out.extend(Path("src").glob(f"{PYTHON_MODULE_NAME}*.{ext}"))
    return out


def _installed_extension_candidates() -> list[Path]:
    exts = _extension_suffixes()
    out: list[Path] = []
    # The installed extension module produced by `maturin develop` typically lives in
    # `<site-packages>/passivbot_rust/passivbot_rust.*.so` (platform-specific).
    for key in ("platlib", "purelib"):
        root = sysconfig.get_paths().get(key)
        if not root:
            continue
        pkg_dir = Path(root) / PYTHON_MODULE_NAME
        if not pkg_dir.exists():
            continue
        for ext in exts:
            out.extend(pkg_dir.glob(f"{PYTHON_MODULE_NAME}*.{ext}"))
    # Deduplicate (platlib/purelib often match).
    return list(dict.fromkeys(out))


def _target_extension_candidates() -> list[Path]:
    exts = _extension_suffixes()
    return [
        Path("passivbot-rust/target/release") / f"{COMPILED_EXTENSION_NAME}.{ext}".strip(".")
        for ext in exts
    ]


def compiled_extension_paths() -> List[Path]:
    """
    Return extension candidates in import-precedence order.

    When running `src/*.py` scripts, `src/` is typically first on `sys.path`, so a local
    `src/passivbot_rust*.so` will shadow an installed site-packages build.
    """
    return (
        _local_extension_candidates()
        + _installed_extension_candidates()
        + _target_extension_candidates()
    )


def preferred_compiled_mtime() -> Optional[float]:
    """
    Mtime of the extension artifact that is *most likely* to be imported.

    Priority:
    1) `src/passivbot_rust*.so` (shadows everything when running `src/*.py`)
    2) installed site-packages `passivbot_rust/passivbot_rust*.so`
    3) `passivbot-rust/target/release/libpassivbot_rust.*`
    """
    for group in (
        _local_extension_candidates(),
        _installed_extension_candidates(),
        _target_extension_candidates(),
    ):
        mtimes = [p.stat().st_mtime for p in group if p.exists()]
        if mtimes:
            return max(mtimes)
    return None


def latest_compiled_mtime(paths: Iterable[Path]) -> Optional[float]:
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return max(mtimes) if mtimes else None


def latest_source_mtime(root: Path = Path("passivbot-rust")) -> Optional[float]:
    """
    Return the latest mtime of inputs that should trigger a rebuild.

    Notes:
    - Avoid scanning `target/` since build artifacts may contain generated `.rs` files and
      can cause perpetual "stale" detection.
    """
    tracked_roots = [root / "src"]
    tracked_files = [root / "Cargo.toml", root / "Cargo.lock"]

    mtimes: list[float] = []
    for file_path in tracked_files:
        try:
            if file_path.exists():
                mtimes.append(file_path.stat().st_mtime)
        except OSError:
            continue
    for file_path in root.glob("*.rs"):
        try:
            if file_path.exists():
                mtimes.append(file_path.stat().st_mtime)
        except OSError:
            continue
    for scan_root in tracked_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*.rs"):
            try:
                mtimes.append(path.stat().st_mtime)
            except OSError:
                continue
    return max(mtimes) if mtimes else None


def is_stale(compiled_mtime: Optional[float], source_mtime: Optional[float]) -> bool:
    if compiled_mtime is None:
        return True
    if source_mtime is None:
        return False
    return compiled_mtime < source_mtime


def acquire_lock(lock_file: Path = LOCK_FILE) -> bool:
    import time

    start = time.time()
    while True:
        try:
            if lock_file.exists():
                age = time.time() - lock_file.stat().st_mtime
                if age > LOCK_TIMEOUT:
                    try:
                        lock_file.unlink()
                    except OSError:
                        pass
                else:
                    if time.time() - start > LOCK_TIMEOUT:
                        try:
                            lock_file.unlink()
                        except OSError:
                            pass
                        return True
                    time.sleep(LOCK_CHECK_INTERVAL)
                    continue
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            lock_file.write_text(str(os.getpid()))
            return True
        except OSError:
            return False


def release_lock(lock_file: Path = LOCK_FILE) -> None:
    try:
        if lock_file.exists():
            lock_file.unlink()
    except OSError:
        pass


def recompile_rust() -> bool:
    try:
        start = time.time()
        result = subprocess.run(  # noqa: S603,S607
            ["maturin", "develop", "--release"],
            cwd="passivbot-rust",
            check=True,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start
        print(result.stdout)
        print(f"Rust extension rebuild finished in {elapsed:.2f}s")
        sync_installed_extension_into_src()
        return True
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error during Rust compile: {e}")
        return False


def check_and_maybe_compile(
    *,
    skip: bool = False,
    force: bool = False,
    fail_on_stale: bool = False,
) -> None:
    """
    Ensure the Rust extension exists and is up to date.

    This must be called before importing passivbot_rust.
    """
    if skip:
        return

    if "passivbot_rust" in sys.modules:
        # Already loaded in this process; if caller insists on force/fail, error out.
        if force or fail_on_stale:
            raise RuntimeError("passivbot_rust is already imported; restart required.")
        print("passivbot_rust already imported; using existing binary.")
        return

    # If a newer build already exists in site-packages, sync it into `src/` first so we don't
    # spuriously rebuild just because a stale local `.so` is shadowing the installed one.
    sync_installed_extension_into_src()

    compiled_paths = compiled_extension_paths()
    compiled_mtime = preferred_compiled_mtime()
    source_mtime = latest_source_mtime()
    stale = is_stale(compiled_mtime, source_mtime)

    needs_compile = force or stale
    if fail_on_stale and stale:
        raise RuntimeError("Rust extension is stale; rebuild required (fail-on-stale enabled).")
    if not needs_compile:
        return

    if compiled_mtime is None:
        print("Rust extension missing; compiling...")
    elif stale:
        print("Rust extension is stale; recompiling...")
    elif force:
        print("Rust extension rebuild forced; recompiling...")

    if not acquire_lock():
        raise RuntimeError("Failed to acquire Rust compile lock.")
    try:
        if not recompile_rust():
            raise RuntimeError("Rust compilation failed.")
    finally:
        release_lock()

    # Re-check staleness after compile
    compiled_mtime = preferred_compiled_mtime()
    if is_stale(compiled_mtime, latest_source_mtime()):
        raise RuntimeError("Rust extension appears stale even after recompilation; check build.")


def sync_installed_extension_into_src() -> None:
    """
    Ensure `src/passivbot_rust*.so` matches the installed `maturin develop` build.

    `src/*.py` scripts put `src/` first on `sys.path`, so a stale local `.so` will shadow
    the freshly-installed site-packages build and make changes appear to have no effect.
    """
    installed = [p for p in _installed_extension_candidates() if p.exists()]
    if not installed:
        return
    installed_path = max(installed, key=lambda p: p.stat().st_mtime)
    dst = Path("src") / installed_path.name

    # Remove any shadowing local builds with a different filename.
    for local in _local_extension_candidates():
        try:
            if local.exists() and local.name != dst.name:
                local.unlink()
        except OSError:
            pass

    # Copy only when needed.
    try:
        if dst.exists():
            src_stat = installed_path.stat()
            dst_stat = dst.stat()
            if src_stat.st_size == dst_stat.st_size and src_stat.st_mtime <= dst_stat.st_mtime:
                return
        dst.parent.mkdir(parents=True, exist_ok=True)
        copy2(installed_path, dst)
        print(f"Synced Rust extension into {dst}")
    except OSError:
        # Best-effort; if we can't sync, the build still exists in site-packages.
        return
