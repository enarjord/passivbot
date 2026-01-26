import sys
import tempfile
import time
from pathlib import Path

import pytest

from rust_utils import (
    compiled_extension_paths,
    is_stale,
    latest_compiled_mtime,
    latest_source_mtime,
    check_and_maybe_compile,
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
