"""Utilities for dispatching email alerts from the risk dashboard."""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from typing import Iterable, Sequence

from .configuration import EmailSettings

logger = logging.getLogger(__name__)


class EmailAlertSender:
    """Send exposure breach notifications via SMTP."""

    def __init__(self, settings: EmailSettings) -> None:
        if not settings.host:
            raise ValueError("Email settings require a host to be configured.")
        self._settings = settings

    def send(self, subject: str, body: str, recipients: Sequence[str]) -> None:
        """Send a plaintext message to ``recipients``.

        The method is synchronous because it executes within background tasks that are
        already awaited by the caller. Errors are logged and suppressed to avoid
        interrupting the polling loop.
        """

        if not recipients:
            return

        message = EmailMessage()
        message["Subject"] = subject
        sender = self._determine_sender()
        message["From"] = sender
        message["To"] = ", ".join(recipient for recipient in recipients if recipient)
        message.set_content(body)

        try:
            if self._settings.use_ssl:
                with smtplib.SMTP_SSL(self._settings.host, self._settings.port, timeout=10) as smtp:
                    self._authenticate_and_send(smtp, message, recipients)
            else:
                with smtplib.SMTP(self._settings.host, self._settings.port, timeout=10) as smtp:
                    if self._settings.use_tls:
                        smtp.starttls()
                    self._authenticate_and_send(smtp, message, recipients)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to send alert email: %s", exc, exc_info=True)

    def _determine_sender(self) -> str:
        if self._settings.sender:
            return self._settings.sender
        if self._settings.username:
            return self._settings.username
        return "alerts@localhost"

    def _authenticate_and_send(
        self, smtp: smtplib.SMTP, message: EmailMessage, recipients: Iterable[str]
    ) -> None:
        username = self._settings.username
        password = self._settings.password
        if username and password:
            smtp.login(username, password)
        smtp.send_message(message, from_addr=message["From"], to_addrs=list(recipients))


__all__ = ["EmailAlertSender"]
