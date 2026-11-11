import sys

import pytest


@pytest.mark.parametrize(
    ("host_args", "expected_host"),
    [([], "127.0.0.1"), (["--host", "0.0.0.0"], "0.0.0.0")],
)
def test_fill_events_dashboard_passes_bind_address_to_server(monkeypatch, host_args, expected_host):
    from tools import fill_events_dash

    accounts = {"example": {}}
    monkeypatch.setattr(
        sys, "argv", ["fill-events-dash", "--users", "example", "--port", "8051", *host_args]
    )
    monkeypatch.setattr(fill_events_dash, "_build_managers", lambda *args: accounts)
    monkeypatch.setattr(fill_events_dash, "configure_logging", lambda *args: None)
    calls = []
    monkeypatch.setattr(
        fill_events_dash, "serve_dash", lambda accounts, **kwargs: calls.append((accounts, kwargs))
    )

    fill_events_dash.main()

    assert calls == [(accounts, {"default_days": 30, "port": 8051, "host": expected_host})]
