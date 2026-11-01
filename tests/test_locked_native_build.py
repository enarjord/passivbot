"""A stale Rust lock must fail rather than silently select new dependencies."""

import shutil
import subprocess

import pytest


def test_locked_cargo_rejects_stale_lock_without_rewriting_it(tmp_path):
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("Cargo is required for the native build integration check")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/lib.rs").write_text("pub fn value() -> u8 { 1 }\n")
    manifest = tmp_path / "Cargo.toml"
    manifest.write_text('[package]\nname="locked_fixture"\nversion="0.1.0"\nedition="2021"\n')
    subprocess.run([cargo, "generate-lockfile", "--offline"], cwd=tmp_path, check=True, capture_output=True)
    lock = tmp_path / "Cargo.lock"
    original = lock.read_bytes()
    manifest.write_text(manifest.read_text().replace('version="0.1.0"', 'version="0.2.0"'))

    result = subprocess.run([cargo, "check", "--locked", "--offline"], cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode != 0
    assert "--locked" in result.stderr
    assert lock.read_bytes() == original
