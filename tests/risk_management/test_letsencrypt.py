from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from risk_management.letsencrypt import LetsEncryptError, ensure_certificate


class Recorder:
    def __init__(self) -> None:
        self.command: Sequence[str] | None = None


def test_ensure_certificate_invokes_certbot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorder = Recorder()

    def fake_run(command: Sequence[str], **kwargs) -> None:  # type: ignore[no-untyped-def]
        recorder.command = command
        assert kwargs.get("check") is True

    monkeypatch.setattr("risk_management.letsencrypt.shutil.which", lambda exe: "/usr/bin/certbot")
    monkeypatch.setattr("risk_management.letsencrypt.subprocess.run", fake_run)

    live_dir = tmp_path / "live" / "example.com"
    live_dir.mkdir(parents=True)
    cert_path = live_dir / "fullchain.pem"
    key_path = live_dir / "privkey.pem"
    cert_path.write_text("dummy-cert")
    key_path.write_text("dummy-key")

    returned_cert, returned_key = ensure_certificate(
        executable="certbot",
        domains=["example.com"],
        email="ops@example.com",
        staging=True,
        http_port=5002,
        config_dir=tmp_path,
    )

    assert recorder.command is not None
    assert "--staging" in recorder.command
    assert "-d" in recorder.command
    assert "example.com" in recorder.command
    assert returned_cert == cert_path
    assert returned_key == key_path


def test_ensure_certificate_missing_certbot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("risk_management.letsencrypt.shutil.which", lambda exe: None)
    with pytest.raises(LetsEncryptError):
        ensure_certificate(domains=("example.com",))


def test_ensure_certificate_handles_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("risk_management.letsencrypt.shutil.which", lambda exe: "/usr/bin/certbot")

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.CalledProcessError(returncode=1, cmd=command)

    import subprocess

    monkeypatch.setattr("risk_management.letsencrypt.subprocess.run", fake_run)

    with pytest.raises(LetsEncryptError):
        ensure_certificate(domains=("example.com",), config_dir=tmp_path)
