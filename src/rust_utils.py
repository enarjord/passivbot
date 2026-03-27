"""
Helpers for managing the compiled Rust extension.

This module is intentionally free of imports that would load the extension
itself; it only inspects filesystem state.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import sysconfig
import time
import hashlib
from shutil import copy2
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

LOCK_FILE = Path("passivbot-rust/.compile.lock")
LOCK_TIMEOUT = 300  # seconds
LOCK_CHECK_INTERVAL = 2  # seconds
COMPILED_EXTENSION_NAME = "libpassivbot_rust"
PYTHON_MODULE_NAME = "passivbot_rust"
SOURCE_STAMP_SUFFIX = ".rust-src-sha256"


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
    # The installed extension produced by `maturin develop` is typically a direct module in
    # `<site-packages>/passivbot_rust.*.so` (platform-specific), though some layouts may place
    # it under a package directory.
    for key in ("platlib", "purelib"):
        root = sysconfig.get_paths().get(key)
        if not root:
            continue
        root_path = Path(root)
        for ext in exts:
            out.extend(root_path.glob(f"{PYTHON_MODULE_NAME}*.{ext}"))
        pkg_dir = root_path / PYTHON_MODULE_NAME
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


def _import_target_compiled_path() -> Optional[Path]:
    """
    Resolve the compiled artifact Python would import in the current process.

    This must work both for direct extension modules and for the package layout produced by
    `maturin develop`, where `find_spec("passivbot_rust")` resolves to `__init__.py` and the
    compiled extension lives alongside it.
    """
    spec = importlib.util.find_spec(PYTHON_MODULE_NAME)
    if spec is None:
        return None

    origin = getattr(spec, "origin", None)
    if origin and origin not in {"built-in", "frozen"}:
        origin_path = Path(origin)
        suffixes = {suffix.lower() for suffix in _extension_suffixes()}
        if any(str(origin_path).lower().endswith(f".{suffix}") for suffix in suffixes):
            return origin_path

    locations = list(getattr(spec, "submodule_search_locations", []) or [])
    if not locations:
        return None

    suffixes = _extension_suffixes()
    for location in locations:
        location_path = Path(location)
        for ext in suffixes:
            matches = [p for p in location_path.glob(f"{PYTHON_MODULE_NAME}*.{ext}") if p.exists()]
            if matches:
                return max(matches, key=lambda p: p.stat().st_mtime)
    return None


def preferred_compiled_mtime() -> Optional[float]:
    """
    Mtime of the extension artifact that is *most likely* to be imported.

    Priority:
    1) `src/passivbot_rust*.so` (shadows everything when running `src/*.py`)
    2) installed site-packages `passivbot_rust/passivbot_rust*.so`
    3) `passivbot-rust/target/release/libpassivbot_rust.*`
    """
    import_target = _import_target_compiled_path()
    if import_target is not None and import_target.exists():
        return import_target.stat().st_mtime
    for group in (
        _local_extension_candidates(),
        _installed_extension_candidates(),
        _target_extension_candidates(),
    ):
        mtimes = [p.stat().st_mtime for p in group if p.exists()]
        if mtimes:
            return max(mtimes)
    return None

def preferred_compiled_path() -> Optional[Path]:
    """
    Path of the extension artifact that is *most likely* to be imported.

    Priority matches `preferred_compiled_mtime()`.
    """
    import_target = _import_target_compiled_path()
    if import_target is not None and import_target.exists():
        return import_target
    for group in (
        _local_extension_candidates(),
        _installed_extension_candidates(),
        _target_extension_candidates(),
    ):
        existing = [p for p in group if p.exists()]
        if existing:
            return max(existing, key=lambda p: p.stat().st_mtime)
    return None


def sha256_file(path: str | Path | None) -> Optional[str]:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_runtime_provenance() -> dict:
    preferred_path = preferred_compiled_path()
    preferred_str = str(preferred_path) if preferred_path is not None else None
    preferred_hash = sha256_file(preferred_str)
    runtime_path = None
    runtime_hash = None
    runtime_mtime = None
    module_loaded = False
    module_name = PYTHON_MODULE_NAME
    module = sys.modules.get(module_name)
    if module is not None:
        module_loaded = True
        runtime_path = getattr(module, "__file__", None)
        runtime_hash = sha256_file(runtime_path)
        try:
            runtime_mtime = Path(runtime_path).stat().st_mtime if runtime_path else None
        except OSError:
            runtime_mtime = None
    preferred_mtime = None
    try:
        preferred_mtime = Path(preferred_str).stat().st_mtime if preferred_str else None
    except OSError:
        preferred_mtime = None
    return {
        "module_name": module_name,
        "module_loaded": module_loaded,
        "runtime_module_path": runtime_path,
        "runtime_module_sha256": runtime_hash,
        "runtime_module_mtime": runtime_mtime,
        "preferred_compiled_path": preferred_str,
        "preferred_compiled_sha256": preferred_hash,
        "preferred_compiled_mtime": preferred_mtime,
        "runtime_matches_preferred": (
            runtime_hash is not None and preferred_hash is not None and runtime_hash == preferred_hash
        ),
        "pid": os.getpid(),
    }


def latest_compiled_mtime(paths: Iterable[Path]) -> Optional[float]:
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return max(mtimes) if mtimes else None


def _tracked_source_files(root: Path = Path("passivbot-rust")) -> list[Path]:
    tracked_files: list[Path] = []
    for file_path in (root / "Cargo.toml", root / "Cargo.lock"):
        if file_path.exists():
            tracked_files.append(file_path)
    for file_path in root.glob("*.rs"):
        if file_path.exists():
            tracked_files.append(file_path)
    src_root = root / "src"
    if src_root.exists():
        tracked_files.extend(path for path in src_root.rglob("*.rs") if path.exists())
    return sorted(set(tracked_files))


def latest_source_mtime(root: Path = Path("passivbot-rust")) -> Optional[float]:
    """
    Return the latest mtime of inputs that should trigger a rebuild.

    Notes:
    - Avoid scanning `target/` since build artifacts may contain generated `.rs` files and
      can cause perpetual "stale" detection.
    """
    mtimes: list[float] = []
    for file_path in _tracked_source_files(root):
        try:
            mtimes.append(file_path.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else None


def source_fingerprint(root: Path = Path("passivbot-rust")) -> Optional[str]:
    tracked_files = _tracked_source_files(root)
    if not tracked_files:
        return None
    digest = hashlib.sha256()
    for path in tracked_files:
        rel = path.relative_to(root)
        digest.update(str(rel).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def is_stale(compiled_mtime: Optional[float], source_mtime: Optional[float]) -> bool:
    if compiled_mtime is None:
        return True
    if source_mtime is None:
        return False
    return compiled_mtime < source_mtime


def source_stamp_path(compiled_path: Path) -> Path:
    return compiled_path.with_name(f"{compiled_path.name}{SOURCE_STAMP_SUFFIX}")


def read_source_stamp(compiled_path: Path) -> Optional[str]:
    stamp_path = source_stamp_path(compiled_path)
    try:
        if stamp_path.exists():
            return stamp_path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None
    return None


def write_source_stamp(compiled_path: Path, fingerprint: str) -> None:
    stamp_path = source_stamp_path(compiled_path)
    stamp_path.write_text(f"{fingerprint}\n", encoding="utf-8")


def extension_needs_rebuild(
    compiled_path: Optional[Path],
    source_mtime: Optional[float],
    fingerprint: Optional[str],
) -> bool:
    if compiled_path is None or not compiled_path.exists():
        return True
    if is_stale(compiled_path.stat().st_mtime, source_mtime):
        return True
    if fingerprint is None:
        return False
    return read_source_stamp(compiled_path) != fingerprint


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


def stamp_compiled_extensions(fingerprint: Optional[str]) -> None:
    if fingerprint is None:
        return
    for compiled_path in compiled_extension_paths():
        if not compiled_path.exists():
            continue
        try:
            write_source_stamp(compiled_path, fingerprint)
        except OSError:
            continue


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
        stamp_compiled_extensions(source_fingerprint())
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

    source_mtime = latest_source_mtime()
    fingerprint = source_fingerprint()
    compiled_path = preferred_compiled_path()
    stale = extension_needs_rebuild(compiled_path, source_mtime, fingerprint)

    needs_compile = force or stale
    if fail_on_stale and stale:
        raise RuntimeError("Rust extension is stale; rebuild required (fail-on-stale enabled).")
    if not needs_compile:
        return

    if compiled_path is None:
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
    compiled_path = preferred_compiled_path()
    if extension_needs_rebuild(compiled_path, latest_source_mtime(), source_fingerprint()):
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
    installed_stamp = source_stamp_path(installed_path)
    dst_stamp = source_stamp_path(dst)

    # Remove any shadowing local builds with a different filename.
    for local in _local_extension_candidates():
        try:
            if local.exists() and local.name != dst.name:
                local.unlink()
                source_stamp_path(local).unlink(missing_ok=True)
        except OSError:
            pass

    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    # Copy only when needed.
    try:
        if dst.exists():
            src_stat = installed_path.stat()
            dst_stat = dst.stat()
            if src_stat.st_size == dst_stat.st_size:
                # Avoid rewriting an identical dylib. Rewriting in-place can race with imports
                # from concurrent processes on macOS and trigger "Code Signature Invalid" kills.
                if _sha256(installed_path) == _sha256(dst):
                    # Content is identical; update mtime so staleness checks pass.
                    try:
                        inst_mtime = installed_path.stat().st_mtime
                        os.utime(dst, (inst_mtime, inst_mtime))
                    except OSError:
                        pass
                    try:
                        if installed_stamp.exists():
                            copy2(installed_stamp, dst_stamp)
                        elif dst_stamp.exists():
                            dst_stamp.unlink()
                    except OSError:
                        pass
                    return
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}")
        copy2(installed_path, tmp)
        if sys.platform == "darwin":
            # Best-effort ad-hoc sign; keeps macOS happy when loading copied extension pages.
            subprocess.run(  # noqa: S603,S607
                ["codesign", "--force", "--sign", "-", str(tmp)],
                check=False,
                capture_output=True,
                text=True,
            )
        os.replace(tmp, dst)
        try:
            if installed_stamp.exists():
                copy2(installed_stamp, dst_stamp)
            elif dst_stamp.exists():
                dst_stamp.unlink()
        except OSError:
            pass
        print(f"Synced Rust extension into {dst}")
    except OSError:
        # Best-effort; if we can't sync, the build still exists in site-packages.
        return
