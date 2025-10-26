"""Tests for risk management web server helpers."""

from __future__ import annotations

import importlib
import sys
import types

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "uvicorn" not in sys.modules:
    uvicorn_stub = types.ModuleType("uvicorn")

    def _noop_run(*_args, **_kwargs) -> None:  # pragma: no cover - helper for import
        return None

    uvicorn_stub.run = _noop_run
    sys.modules["uvicorn"] = uvicorn_stub

from risk_management.configuration import AccountConfig, RealtimeConfig  # noqa: E402
from risk_management.web_server import _determine_uvicorn_logging  # noqa: E402


def _make_config(global_debug: bool = False, account_debug: bool = False) -> RealtimeConfig:
    account = AccountConfig(
        name="Example",
        exchange="binance",
        credentials={},
        debug_api_payloads=account_debug,
    )
    return RealtimeConfig(accounts=[account], debug_api_payloads=global_debug)


def test_determine_uvicorn_logging_defaults_when_disabled() -> None:
    config = _make_config()

    log_config, log_level = _determine_uvicorn_logging(config)

    assert log_config is None
    assert log_level == "info"


def test_determine_uvicorn_logging_uses_uvicorn_config(monkeypatch) -> None:
    dummy_logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {},
        "handlers": {},
        "loggers": {"": {"handlers": ["default"], "level": "INFO"}},
    }
    uvicorn_module = types.ModuleType("uvicorn")
    uvicorn_config_module = types.ModuleType("uvicorn.config")
    uvicorn_config_module.LOGGING_CONFIG = dummy_logging_config

    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_module)
    monkeypatch.setitem(sys.modules, "uvicorn.config", uvicorn_config_module)

    config = _make_config(global_debug=True)

    log_config, log_level = _determine_uvicorn_logging(config)

    assert log_level == "debug"
    assert log_config["loggers"][""]["level"] == "DEBUG"
    assert log_config["loggers"]["risk_management"]["level"] == "DEBUG"


def test_determine_uvicorn_logging_handles_missing_uvicorn(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "uvicorn", raising=False)
    monkeypatch.delitem(sys.modules, "uvicorn.config", raising=False)

    original_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "uvicorn.config":
            raise ModuleNotFoundError("uvicorn unavailable")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    config = _make_config(account_debug=True)

    log_config, log_level = _determine_uvicorn_logging(config)

    assert log_config is None
    assert log_level == "debug"
