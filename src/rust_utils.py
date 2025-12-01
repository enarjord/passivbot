"""
Helpers for managing the compiled Rust extension.

This module is intentionally free of imports that would load the extension
itself; it only inspects filesystem state.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

LOCK_FILE = Path("passivbot-rust/.compile.lock")
LOCK_TIMEOUT = 300  # seconds
LOCK_CHECK_INTERVAL = 2  # seconds
COMPILED_EXTENSION_NAME = "libpassivbot_rust"


def compiled_extension_paths() -> List[Path]:
    exts = ["so", "dylib", "dll", "pyd", "", "bundle", "sl"]
    return [
        Path("passivbot-rust/target/release") / f"{COMPILED_EXTENSION_NAME}.{ext}".strip(".")
        for ext in exts
    ]


def latest_compiled_mtime(paths: Iterable[Path]) -> Optional[float]:
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return max(mtimes) if mtimes else None


def latest_source_mtime(root: Path = Path("passivbot-rust")) -> Optional[float]:
    mtimes: List[float] = []
    for path in root.rglob("*.rs"):
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
        cwd = os.getcwd()
        os.chdir("passivbot-rust")
        result = subprocess.run(
            ["maturin", "develop", "--release"], check=True, capture_output=True, text=True
        )
        os.chdir(cwd)
        print(result.stdout)
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

    compiled_paths = compiled_extension_paths()
    compiled_mtime = latest_compiled_mtime(compiled_paths)
    source_mtime = latest_source_mtime()
    stale = is_stale(compiled_mtime, source_mtime)

    if "passivbot_rust" in sys.modules:
        raise RuntimeError("passivbot_rust is already imported; restart required.")

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
    compiled_mtime = latest_compiled_mtime(compiled_paths)
    if is_stale(compiled_mtime, latest_source_mtime()):
        raise RuntimeError("Rust extension appears stale even after recompilation; check build.")
