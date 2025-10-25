from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

    assert recorder.command[0] == "/usr/bin/certbot"
    assert "--staging" in recorder.command
    assert "-d" in recorder.command
    assert "example.com" in recorder.command
    assert returned_cert == cert_path
    assert returned_key == key_path


def test_ensure_certificate_missing_certbot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("risk_management.letsencrypt.shutil.which", lambda exe: None)
    real_import = builtins.__import__

    def raising_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name.split(".")[0] == "certbot":
            raise ModuleNotFoundError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raising_import)

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


def test_ensure_certificate_falls_back_to_python_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("risk_management.letsencrypt.shutil.which", lambda exe: None)

    class ModuleRecorder:
        def __init__(self) -> None:
            self.args: Sequence[str] | None = None

        def __call__(self, args: Sequence[str]) -> int:  # type: ignore[no-untyped-def]
            self.args = list(args)
            return 0

    module_recorder = ModuleRecorder()

    fake_certbot = types.ModuleType("certbot")
    fake_certbot_main = types.ModuleType("certbot.main")
    fake_certbot_main.main = module_recorder  # type: ignore[attr-defined]
    fake_certbot.main = fake_certbot_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "certbot", fake_certbot)
    monkeypatch.setitem(sys.modules, "certbot.main", fake_certbot_main)

    live_dir = tmp_path / "live" / "fallback.example"
    live_dir.mkdir(parents=True)
    (live_dir / "fullchain.pem").write_text("dummy")
    (live_dir / "privkey.pem").write_text("dummy")

    ensure_certificate(domains=("fallback.example",), config_dir=tmp_path)

    assert module_recorder.args is not None
    assert module_recorder.args[0] == "certonly"
    assert "fallback.example" in module_recorder.args
