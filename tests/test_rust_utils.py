import sys
import tempfile
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from rust_utils import (
    _installed_extension_candidates,
    compiled_extension_paths,
    is_stale,
    latest_compiled_mtime,
    latest_source_mtime,
    check_and_maybe_compile,
    extension_needs_rebuild,
    preferred_compiled_path,
    source_fingerprint,
    source_stamp_path,
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
