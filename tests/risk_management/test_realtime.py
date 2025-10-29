import asyncio
import json
import logging
from pathlib import Path
from typing import Iterable, List, Mapping

import pytest

from custom_endpoint_overrides import (
    configure_custom_endpoint_loader,
    resolve_custom_endpoint_override,
)

from risk_management.configuration import AccountConfig, RealtimeConfig
from risk_management.realtime import (
    AuthenticationError,
    RealtimeDataFetcher,
    _configure_custom_endpoints,
)


class StubAccountClient:
    """Test double for ``AccountClientProtocol`` returning predefined outcomes."""

    def __init__(self, outcomes: Iterable[object]):
        self._outcomes: List[object] = list(outcomes)

    async def fetch(self):  # pragma: no cover - exercised through tests
        if not self._outcomes:
            raise RuntimeError("No more outcomes configured for StubAccountClient")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def close(self) -> None:  # pragma: no cover - included for interface completeness
        return None


class RecordingNotifications:
    def __init__(self) -> None:
        self.daily: List[tuple[Mapping[str, object], float]] = []
        self.alerts: List[Mapping[str, object]] = []

    def send_daily_snapshot(self, snapshot: Mapping[str, object], portfolio_balance: float) -> None:
        self.daily.append((snapshot, portfolio_balance))

    def dispatch_alerts(self, snapshot: Mapping[str, object]) -> None:
        self.alerts.append(snapshot)


def _make_config() -> RealtimeConfig:
    return RealtimeConfig(accounts=[AccountConfig(name="Test", exchange="test")])


def test_authentication_warning_logged_once(caplog) -> None:
    client = StubAccountClient(
        [AuthenticationError("invalid"), AuthenticationError("invalid")]
    )
    fetcher = RealtimeDataFetcher(_make_config(), account_clients=[client])

    caplog.set_level(logging.WARNING)

    snapshot_first = asyncio.run(fetcher.fetch_snapshot())
    snapshot_second = asyncio.run(fetcher.fetch_snapshot())

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert snapshot_first["account_messages"]["Test"].startswith(
        "Test: authentication failed"
    )
    assert snapshot_second["account_messages"]["Test"].startswith(
        "Test: authentication failed"
    )
    asyncio.run(fetcher.close())


def test_authentication_warning_resets_after_success(caplog) -> None:
    client = StubAccountClient(
        [
            AuthenticationError("invalid"),
            {"name": "Test", "balance": 123.0, "positions": []},
            AuthenticationError("invalid"),
        ]
    )
    fetcher = RealtimeDataFetcher(_make_config(), account_clients=[client])

    caplog.set_level(logging.INFO)

    asyncio.run(fetcher.fetch_snapshot())
    snapshot_success = asyncio.run(fetcher.fetch_snapshot())
    asyncio.run(fetcher.fetch_snapshot())

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    infos = [record for record in caplog.records if record.levelno == logging.INFO]

    assert len(warnings) == 2
    assert any("Authentication for Test restored" in record.message for record in infos)
    assert snapshot_success["accounts"][0]["balance"] == 123.0
    asyncio.run(fetcher.close())


def test_snapshot_includes_configured_account_messages() -> None:
    config = RealtimeConfig(
        accounts=[AccountConfig(name="Test", exchange="test")],
        account_messages={"Test": "Maintenance window"},
    )
    client = StubAccountClient([
        {"name": "Test", "balance": 10.0, "positions": []},
    ])
    fetcher = RealtimeDataFetcher(config, account_clients=[client])

    snapshot = asyncio.run(fetcher.fetch_snapshot())

    assert snapshot["account_messages"]["Test"] == "Maintenance window"
    asyncio.run(fetcher.close())


def test_runtime_account_messages_override_configured_messages() -> None:
    config = RealtimeConfig(
        accounts=[AccountConfig(name="Test", exchange="test")],
        account_messages={"Test": "Maintenance window"},
    )
    client = StubAccountClient([RuntimeError("boom")])
    fetcher = RealtimeDataFetcher(config, account_clients=[client])

    snapshot = asyncio.run(fetcher.fetch_snapshot())

    assert snapshot["account_messages"]["Test"].startswith("Test: boom")
    asyncio.run(fetcher.close())


def test_configure_custom_endpoints_prefers_config_directory(tmp_path: Path) -> None:
    overrides_dir = tmp_path / "configs"
    overrides_dir.mkdir(parents=True)
    overrides_path = overrides_dir / "custom_endpoints.json"
    overrides_path.write_text(
        json.dumps(
            {
                "exchanges": {
                    "binanceusdm": {
                        "rest": {
                            "rewrite_domains": {
                                "https://fapi.binance.com": "https://mltech.example"
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        _configure_custom_endpoints(None, overrides_dir)
        override = resolve_custom_endpoint_override("binanceusdm")
        assert override is not None
        assert (
            override.rest_domain_rewrites["https://fapi.binance.com"] == "https://mltech.example"
        )
    finally:
        configure_custom_endpoint_loader(None, autodiscover=True)


def test_portfolio_stop_loss_triggers_notifications() -> None:
    config = RealtimeConfig(
        accounts=[
            AccountConfig(name="Alpha", exchange="test"),
            AccountConfig(name="Beta", exchange="test"),
        ],
        notification_channels=["email:risk@example.com"],
    )
    client_alpha = StubAccountClient(
        [
            {"name": "Alpha", "balance": 1_000.0, "positions": []},
            {"name": "Alpha", "balance": 950.0, "positions": []},
            {"name": "Alpha", "balance": 800.0, "positions": []},
        ]
    )
    client_beta = StubAccountClient(
        [
            {"name": "Beta", "balance": 1_000.0, "positions": []},
            {"name": "Beta", "balance": 980.0, "positions": []},
            {"name": "Beta", "balance": 900.0, "positions": []},
        ]
    )
    fetcher = RealtimeDataFetcher(config, account_clients=[client_alpha, client_beta])
    recorder = RecordingNotifications()
    fetcher._notifications = recorder  # type: ignore[attr-defined]

    initial_snapshot = asyncio.run(fetcher.fetch_snapshot())
    assert "portfolio_stop_loss" not in initial_snapshot

    asyncio.run(fetcher.set_portfolio_stop_loss(10.0))
    pre_trigger_snapshot = asyncio.run(fetcher.fetch_snapshot())
    state = pre_trigger_snapshot["portfolio_stop_loss"]
    assert state["threshold_pct"] == 10.0
    assert state["triggered"] is False
    assert state["baseline_balance"] == pytest.approx(2_000.0, rel=1e-6)
    assert state["current_balance"] == pytest.approx(1_930.0, rel=1e-6)
    assert state["current_drawdown_pct"] == pytest.approx((2_000.0 - 1_930.0) / 2_000.0, rel=1e-6)

    triggered_snapshot = asyncio.run(fetcher.fetch_snapshot())
    triggered_state = triggered_snapshot["portfolio_stop_loss"]
    assert triggered_state["triggered"] is True
    assert triggered_state["triggered_at"] is not None
    assert triggered_state["current_balance"] == pytest.approx(1_700.0, rel=1e-6)
    assert triggered_state["current_drawdown_pct"] == pytest.approx((2_000.0 - 1_700.0) / 2_000.0, rel=1e-6)

    assert recorder.daily, "daily snapshot notifications should be recorded"
    assert recorder.daily[-1][0]["portfolio_stop_loss"]["triggered"] is True
    assert recorder.alerts, "alert dispatches should be recorded"
    assert recorder.alerts[-1]["portfolio_stop_loss"]["triggered"] is True

    asyncio.run(fetcher.close())
