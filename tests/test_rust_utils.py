import sys
import tempfile
import time
import types
from types import SimpleNamespace
from pathlib import Path

import pytest

from rust_utils import (
    _installed_extension_candidates,
    acquire_lock,
    compiled_extension_paths,
    is_stale,
    latest_compiled_mtime,
    latest_source_mtime,
    check_and_maybe_compile,
    extension_needs_rebuild,
    maturin_env,
    prune_shadowing_local_extensions,
    preferred_compiled_path,
    release_lock,
    source_fingerprint,
    source_stamp_path,
    stamp_compiled_extensions,
    verify_loaded_runtime_extension,
    write_source_stamp,
)


def test_is_stale_logic():
    now = time.time()
    assert is_stale(None, now) is True  # missing compiled
    assert is_stale(now, None) is False  # no source info
    assert is_stale(now - 10, now) is True
    assert is_stale(now, now - 10) is False


def test_latest_mtime_helpers(tmp_path: Path):
    src_dir = tmp_path / "passivbot-rust"
    src_dir.mkdir()
    (src_dir / "main.rs").write_text("// rust")
    compiled_dir = src_dir / "target" / "release"
    compiled_dir.mkdir(parents=True)
    (compiled_dir / "libpassivbot_rust.so").write_text("bin")
    src_mtime = latest_source_mtime(src_dir)
    comp_mtime = latest_compiled_mtime([compiled_dir / "libpassivbot_rust.so"])
    assert src_mtime is not None
    assert comp_mtime is not None
    assert isinstance(src_mtime, float)
    assert isinstance(comp_mtime, float)


def test_check_and_maybe_compile_errors_on_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "passivbot_rust", object())
    with pytest.raises(RuntimeError):
        check_and_maybe_compile(force=True)


def test_check_and_maybe_compile_rechecks_after_lock_and_skips_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delitem(sys.modules, "passivbot_rust", raising=False)
    compiled = tmp_path / "passivbot_rust.cpython-312-darwin.so"
    compiled.write_text("binary")
    stale_checks = iter([True, False])
    releases: list[bool] = []

    monkeypatch.setattr("rust_utils.prune_shadowing_local_extensions", lambda: None)
    monkeypatch.setattr("rust_utils.latest_source_mtime", lambda: 2.0)
    monkeypatch.setattr("rust_utils.source_fingerprint", lambda: "abc123")
    monkeypatch.setattr("rust_utils.preferred_compiled_path", lambda: compiled)
    monkeypatch.setattr(
        "rust_utils.extension_needs_rebuild",
        lambda compiled_path, source_mtime, fingerprint: next(stale_checks),
    )
    monkeypatch.setattr("rust_utils.acquire_lock", lambda **_kwargs: True)
    monkeypatch.setattr("rust_utils.release_lock", lambda: releases.append(True))
    monkeypatch.setattr(
        "rust_utils.recompile_rust",
        lambda: pytest.fail("waiter should skip rebuild after lock recheck"),
    )

    check_and_maybe_compile()

    assert releases == [True]
    out = capsys.readouterr().out
    assert "Rust extension is stale; recompiling" not in out
    assert "Rust extension was rebuilt by another process; continuing." in out


def test_check_and_maybe_compile_logs_rebuild_only_after_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delitem(sys.modules, "passivbot_rust", raising=False)
    compiled = tmp_path / "passivbot_rust.cpython-312-darwin.so"
    compiled.write_text("binary")
    stale_checks = iter([True, True, False])
    events: list[str] = []

    monkeypatch.setattr("rust_utils.prune_shadowing_local_extensions", lambda: None)
    monkeypatch.setattr("rust_utils.latest_source_mtime", lambda: 2.0)
    monkeypatch.setattr("rust_utils.source_fingerprint", lambda: "abc123")
    monkeypatch.setattr("rust_utils.preferred_compiled_path", lambda: compiled)
    monkeypatch.setattr(
        "rust_utils.extension_needs_rebuild",
        lambda compiled_path, source_mtime, fingerprint: next(stale_checks),
    )

    def _acquire_lock(**_kwargs):
        events.append("lock")
        return True

    def _recompile_rust():
        events.append("compile")
        return True

    monkeypatch.setattr("rust_utils.acquire_lock", _acquire_lock)
    monkeypatch.setattr("rust_utils.release_lock", lambda: events.append("release"))
    monkeypatch.setattr("rust_utils.recompile_rust", _recompile_rust)

    check_and_maybe_compile()

    out = capsys.readouterr().out
    assert events == ["lock", "compile", "release"]
    assert "Rust extension is stale; acquired compile lock; recompiling..." in out


def test_acquire_lock_logs_waiter_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    attempts = iter([False, True])
    sleeps: list[float] = []

    def _lock_then_succeed(_handle):
        if not next(attempts):
            raise BlockingIOError()

    monkeypatch.setattr("rust_utils._lock_file_nonblocking", _lock_then_succeed)
    monkeypatch.setattr("rust_utils.time.sleep", lambda seconds: sleeps.append(seconds))

    try:
        assert (
            acquire_lock(
                tmp_path / "compile.lock",
                timeout=10.0,
                wait_message="Rust extension is stale; waiting for compile lock...",
            )
            is True
        )
    finally:
        release_lock()

    assert len(sleeps) == 1
    assert sleeps[0] > 0.0
    out = capsys.readouterr().out
    assert out.count("Rust extension is stale; waiting for compile lock...") == 1


def test_acquire_lock_timeout_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    times = iter([0.0, 2.0])

    def _raise_blocked(_handle):
        raise BlockingIOError()

    monkeypatch.setattr("rust_utils._lock_file_nonblocking", _raise_blocked)
    monkeypatch.setattr("rust_utils.time.time", lambda: next(times))
    monkeypatch.setattr("rust_utils.time.sleep", lambda _seconds: None)

    assert acquire_lock(tmp_path / "compile.lock", timeout=1.0) is False


def test_stamp_compiled_extensions_only_stamps_selected_paths(tmp_path: Path):
    stale = tmp_path / "stale" / "passivbot_rust.cpython-312-darwin.so"
    fresh = tmp_path / "fresh" / "passivbot_rust.cpython-312-darwin.so"
    stale.parent.mkdir()
    fresh.parent.mkdir()
    stale.write_text("stale")
    fresh.write_text("fresh")

    stamp_compiled_extensions("abc123", paths=[fresh])

    read_stamp = source_stamp_path(fresh).read_text(encoding="utf-8").strip()
    assert read_stamp
    assert read_stamp == "abc123"
    assert not source_stamp_path(stale).exists()


def test_maturin_env_prefers_current_virtualenv(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("rust_utils.sys.prefix", "/tmp/current-venv")
    monkeypatch.setattr("rust_utils.sys.base_prefix", "/usr")
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/wrong-venv")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = maturin_env()

    assert env["VIRTUAL_ENV"] == "/tmp/current-venv"
    assert env["PATH"].startswith("/tmp/current-venv/bin")


def test_latest_source_mtime_includes_root_level_rs_files(tmp_path: Path):
    """Test that latest_source_mtime detects root-level .rs files (PR #537 fix)."""
    src_dir = tmp_path / "passivbot-rust"
    src_dir.mkdir()

    # Create a root-level .rs file
    root_rs = src_dir / "lib.rs"
    root_rs.write_text("// root level rust file")

    # Create a subdirectory .rs file
    sub_dir = src_dir / "src"
    sub_dir.mkdir()
    nested_rs = sub_dir / "main.rs"
    nested_rs.write_text("// nested rust file")

    # Get mtime
    src_mtime = latest_source_mtime(src_dir)

    assert src_mtime is not None
    assert isinstance(src_mtime, float)

    # The mtime should be the max of both files
    root_mtime = root_rs.stat().st_mtime
    nested_mtime = nested_rs.stat().st_mtime
    expected_max = max(root_mtime, nested_mtime)

    assert abs(src_mtime - expected_max) < 0.001  # Allow small floating point difference


def test_latest_source_mtime_only_root_level_rs(tmp_path: Path):
    """Test that latest_source_mtime works with only root-level .rs files."""
    src_dir = tmp_path / "passivbot-rust"
    src_dir.mkdir()

    # Only create root-level .rs files
    (src_dir / "lib.rs").write_text("// lib")
    (src_dir / "build.rs").write_text("// build")

    src_mtime = latest_source_mtime(src_dir)

    assert src_mtime is not None
    assert isinstance(src_mtime, float)


def test_latest_source_mtime_no_root_level_rs(tmp_path: Path):
    """Test that latest_source_mtime still works without root-level .rs files."""
    src_dir = tmp_path / "passivbot-rust"
    src_dir.mkdir()

    # Only create nested .rs files
    sub_dir = src_dir / "src"
    sub_dir.mkdir()
    (sub_dir / "main.rs").write_text("// nested")

    src_mtime = latest_source_mtime(src_dir)

    assert src_mtime is not None
    assert isinstance(src_mtime, float)


def test_source_fingerprint_changes_when_tracked_source_changes(tmp_path: Path):
    src_dir = tmp_path / "passivbot-rust"
    nested_dir = src_dir / "src"
    nested_dir.mkdir(parents=True)
    cargo = src_dir / "Cargo.toml"
    rust_file = nested_dir / "lib.rs"

    cargo.write_text("[package]\nname = 'x'\nversion = '0.1.0'\n")
    rust_file.write_text("// first\n")
    first = source_fingerprint(src_dir)

    rust_file.write_text("// second\n")
    second = source_fingerprint(src_dir)

    assert first is not None
    assert second is not None
    assert first != second


def test_extension_needs_rebuild_when_source_stamp_missing_or_mismatched(tmp_path: Path):
    compiled = tmp_path / "passivbot_rust.cpython-312-darwin.so"
    compiled.write_text("binary")
    source_mtime = compiled.stat().st_mtime - 10

    assert extension_needs_rebuild(compiled, source_mtime, "abc123") is True

    write_source_stamp(compiled, "wrong")
    assert extension_needs_rebuild(compiled, source_mtime, "abc123") is True

    write_source_stamp(compiled, "abc123")
    assert extension_needs_rebuild(compiled, source_mtime, "abc123") is False
    assert source_stamp_path(compiled).exists()


def test_extension_needs_rebuild_prefers_matching_stamp_over_newer_source_mtime(tmp_path: Path):
    compiled = tmp_path / "passivbot_rust.cpython-312-darwin.so"
    compiled.write_text("binary")
    write_source_stamp(compiled, "abc123")

    newer_source_mtime = compiled.stat().st_mtime + 60.0

    assert extension_needs_rebuild(compiled, newer_source_mtime, "abc123") is False


def test_installed_extension_candidates_find_root_level_maturin_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    installed = site_packages / "passivbot_rust.cpython-312-darwin.so"
    installed.write_text("binary")

    monkeypatch.setattr(
        "rust_utils._extension_suffixes",
        lambda: ["cpython-312-darwin.so"],
    )
    monkeypatch.setattr(
        "rust_utils.sysconfig.get_paths",
        lambda: {"platlib": str(site_packages), "purelib": str(site_packages)},
    )

    assert _installed_extension_candidates() == [installed]


def test_preferred_compiled_path_uses_actual_import_target_for_package_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    stale_local = src_dir / "passivbot_rust.cpython-312-darwin.so"
    stale_local.write_text("stale-local")

    site_packages = tmp_path / "site-packages"
    package_dir = site_packages / "passivbot_rust"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("from .passivbot_rust import *\n")
    installed = package_dir / "passivbot_rust.cpython-312-darwin.so"
    installed.write_text("fresh-installed")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "rust_utils._extension_suffixes",
        lambda: ["cpython-312-darwin.so"],
    )
    monkeypatch.setattr(
        "rust_utils.sysconfig.get_paths",
        lambda: {"platlib": str(site_packages), "purelib": str(site_packages)},
    )
    monkeypatch.setattr(
        "rust_utils.importlib.util.find_spec",
        lambda name: SimpleNamespace(
            origin=str(package_dir / "__init__.py"),
            submodule_search_locations=[str(package_dir)],
        ),
    )

    assert preferred_compiled_path() == installed


def test_prune_shadowing_local_extensions_removes_src_copy_when_installed_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    local = src_dir / "passivbot_rust.cpython-312-darwin.so"
    local.write_text("stale-local")
    write_source_stamp(local, "old")

    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    installed = site_packages / "passivbot_rust.cpython-312-darwin.so"
    installed.write_text("fresh-installed")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rust_utils._extension_suffixes", lambda: ["cpython-312-darwin.so"])
    monkeypatch.setattr(
        "rust_utils.sysconfig.get_paths",
        lambda: {"platlib": str(site_packages), "purelib": str(site_packages)},
    )

    prune_shadowing_local_extensions()

    assert installed.exists()
    assert not local.exists()
    assert not source_stamp_path(local).exists()


def test_verify_loaded_runtime_extension_uses_loaded_package_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    site_packages = tmp_path / "site-packages"
    package_dir = site_packages / "passivbot_rust"
    package_dir.mkdir(parents=True)
    init_py = package_dir / "__init__.py"
    init_py.write_text("from .passivbot_rust import *\n")
    installed = package_dir / "passivbot_rust.cpython-312-darwin.so"
    installed.write_text("binary")
    write_source_stamp(installed, "abc123")

    module = types.ModuleType("passivbot_rust")
    module.__file__ = str(init_py)
    module.__path__ = [str(package_dir)]

    monkeypatch.setitem(sys.modules, "passivbot_rust", module)
    monkeypatch.setattr("rust_utils._extension_suffixes", lambda: ["cpython-312-darwin.so"])

    info = verify_loaded_runtime_extension(fingerprint="abc123")

    assert info["runtime_compiled_path"] == str(installed)
    assert info["runtime_compiled_source_stamp"] == "abc123"


def test_verify_loaded_runtime_extension_skips_stub_modules(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "passivbot_rust", types.ModuleType("passivbot_rust"))

    info = verify_loaded_runtime_extension(fingerprint="abc123")

    assert info["skipped"] == "stub_module"
