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
