import asyncio
import subprocess
import sys

import pytest

from monitor_web import build_dashboard_url, run_monitor_web


def test_build_dashboard_url_defaults_to_dashboard_path():
    url = build_dashboard_url("http://127.0.0.1:8765")
    assert url == "http://127.0.0.1:8765/dashboard"


def test_build_dashboard_url_keeps_initial_focus_query():
    url = build_dashboard_url(
        "http://127.0.0.1:8765",
        exchange="bybit",
        user="ebybitsub03",
        focus_symbol="XMR/USDT:USDT",
    )
    assert "exchange=bybit" in url
    assert "user=ebybitsub03" in url
    assert "symbol=XMR%2FUSDT%3AUSDT" in url


@pytest.mark.asyncio
async def test_run_monitor_web_uses_existing_relay_and_prints_dashboard_url(monkeypatch, capsys):
    monkeypatch.setattr("monitor_web.relay_healthcheck", lambda relay_url: True)
    monkeypatch.setattr("monitor_web.launch_relay_subprocess", lambda **kwargs: None)
    monkeypatch.setattr("monitor_web.wait_for_relay", lambda *args, **kwargs: None)
    monkeypatch.setattr("monitor_web.stop_relay_subprocess", lambda process: None)

    async def fake_sleep(_seconds):
        raise KeyboardInterrupt()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(KeyboardInterrupt):
        await run_monitor_web(relay_url="http://127.0.0.1:8765")

    out = capsys.readouterr().out
    assert "http://127.0.0.1:8765/dashboard" in out


def test_monitor_web_tool_help_runs_without_import_errors():
    result = subprocess.run(
        [sys.executable, "src/tools/monitor_web.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--relay-url" in result.stdout
    assert "--open-browser" in result.stdout
