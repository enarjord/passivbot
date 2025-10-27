"""Notification helpers for realtime risk monitoring."""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from typing import Any, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from .configuration import RealtimeConfig
from .dashboard import evaluate_alerts, parse_snapshot
from .email_notifications import EmailAlertSender
from .telegram_notifications import TelegramNotifier

__all__ = [
    "NotificationCoordinator",
]

logger = logging.getLogger(__name__)


class NotificationCoordinator:
    """Coordinate email and telegram notifications for realtime snapshots."""

    def __init__(self, config: RealtimeConfig) -> None:
        self._email_sender = EmailAlertSender(config.email) if config.email else None
        self._email_recipients = self._extract_email_recipients(config.notification_channels)
        self._telegram_targets = self._extract_telegram_targets(config.notification_channels)
        self._telegram_notifier = TelegramNotifier() if self._telegram_targets else None
        self._active_alerts: set[str] = set()
        self._daily_snapshot_tz = ZoneInfo("America/New_York")
        self._daily_snapshot_sent_date: Optional[date] = None

    @staticmethod
    def _extract_email_recipients(channels: Sequence[Any]) -> list[str]:
        recipients: list[str] = []
        for channel in channels:
            if not isinstance(channel, str):
                continue
            if channel.lower().startswith("email:"):
                address = channel.split(":", 1)[1].strip()
                if address:
                    recipients.append(address)
        return recipients

    @staticmethod
    def _extract_telegram_targets(channels: Sequence[Any]) -> list[tuple[str, str]]:
        targets: list[tuple[str, str]] = []
        for channel in channels:
            if not isinstance(channel, str):
                continue
            if not channel.lower().startswith("telegram:"):
                continue
            payload = channel.split(":", 1)[1]
            token = ""
            chat_id = ""
            if "@" in payload:
                token, _, chat_id = payload.partition("@")
            elif "/" in payload:
                token, _, chat_id = payload.partition("/")
            else:
                parts = payload.split(":", 1)
                if len(parts) == 2:
                    token, chat_id = parts
            token = token.strip()
            chat_id = chat_id.strip()
            if token and chat_id:
                targets.append((token, chat_id))
        return targets

    def send_daily_snapshot(self, snapshot: Mapping[str, Any], portfolio_balance: float) -> None:
        if not self._email_sender or not self._email_recipients:
            return
        now_ny = datetime.now(self._daily_snapshot_tz)
        current_date = now_ny.date()
        if self._daily_snapshot_sent_date and current_date > self._daily_snapshot_sent_date:
            self._daily_snapshot_sent_date = None
        if now_ny.time() < time(16, 0):
            return
        if self._daily_snapshot_sent_date == current_date:
            return
        accounts = snapshot.get("accounts", [])
        lines = [
            f"Daily portfolio snapshot ({now_ny.strftime('%Y-%m-%d')} 16:00 ET)",
            f"Total balance: ${portfolio_balance:,.2f}",
            "",
            "Accounts:",
        ]
        for account in accounts or []:
            if not isinstance(account, Mapping):
                continue
            name = str(account.get("name", "unknown"))
            balance = float(account.get("balance", 0.0))
            realised = float(account.get("daily_realized_pnl", 0.0))
            lines.append(
                f"- {name}: balance ${balance:,.2f}, daily realised PnL ${realised:,.2f}"
            )
        body = "\n".join(lines)
        subject = "Daily portfolio balance snapshot"
        self._email_sender.send(subject, body, self._email_recipients)
        self._daily_snapshot_sent_date = current_date

    def dispatch_alerts(self, snapshot: Mapping[str, Any]) -> None:
        if not (self._email_sender or self._telegram_notifier):
            return
        try:
            _, accounts, thresholds, _ = parse_snapshot(dict(snapshot))
            alerts = evaluate_alerts(accounts, thresholds)
        except Exception as exc:  # pragma: no cover - snapshot parsing errors are logged for diagnostics
            logger.debug("Skipping email alert dispatch due to parsing error: %s", exc, exc_info=True)
            return
        alerts_set = set(alerts)
        new_alerts = [alert for alert in alerts if alert not in self._active_alerts]
        self._active_alerts = alerts_set
        if not new_alerts:
            return
        generated_at = snapshot.get("generated_at")
        timestamp = (
            generated_at
            if isinstance(generated_at, str)
            else datetime.now(timezone.utc).isoformat()
        )
        lines = [f"Exposure thresholds were exceeded at {timestamp}.", "", "Alerts:"]
        lines.extend(f"- {alert}" for alert in new_alerts)
        body = "\n".join(lines)
        subject = "Risk alert: exposure threshold breached"
        if self._email_sender and self._email_recipients:
            self._email_sender.send(subject, body, self._email_recipients)
        if self._telegram_notifier and self._telegram_targets:
            message = f"Exposure alert at {timestamp}\n" + "\n".join(new_alerts)
            for token, chat_id in self._telegram_targets:
                self._telegram_notifier.send(token, chat_id, message)

    @property
    def email_sender(self) -> Optional[EmailAlertSender]:
        return self._email_sender

    @property
    def email_recipients(self) -> Sequence[str]:
        return tuple(self._email_recipients)

    @property
    def telegram_targets(self) -> Sequence[tuple[str, str]]:
        return tuple(self._telegram_targets)

