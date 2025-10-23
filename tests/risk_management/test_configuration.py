"""Tests for realtime risk management configuration helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_management.configuration import load_realtime_config


def _write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "configs" / "risk" / "realtime.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _base_payload() -> dict:
    return {
        "accounts": [
            {
                "name": "Example",
                "exchange": "binanceusdm",
                "credentials": {"key": "abc", "secret": "def"},
            }
        ],
    }


def test_custom_endpoint_path_resolves_relative(tmp_path) -> None:
    payload = _base_payload()
    payload["custom_endpoints"] = {"path": "../custom_endpoints.json", "autodiscover": False}
    config_path = _write_config(tmp_path, payload)

    config = load_realtime_config(config_path)

    expected_path = (config_path.parent / Path("../custom_endpoints.json")).resolve()
    assert config.custom_endpoints is not None
    assert config.custom_endpoints.path == str(expected_path)


def test_custom_endpoint_path_keeps_absolute(tmp_path) -> None:
    payload = _base_payload()
    absolute_path = tmp_path / "custom" / "endpoints.json"
    payload["custom_endpoints"] = {"path": str(absolute_path), "autodiscover": True}
    config_path = _write_config(tmp_path, payload)

    config = load_realtime_config(config_path)

    assert config.custom_endpoints is not None
    assert config.custom_endpoints.path == str(absolute_path.resolve())
    assert config.custom_endpoints.autodiscover is True
