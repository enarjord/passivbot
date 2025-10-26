"""Utilities for sending Telegram notifications."""

from __future__ import annotations

import logging

try:  # pragma: no cover - httpx is optional in some environments
    import httpx
except ModuleNotFoundError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Dispatch notifications to Telegram chats via the Bot API."""

    def __init__(self, *, timeout: float = 10.0) -> None:
        self._timeout = timeout

    def send(self, token: str, chat_id: str, message: str) -> None:
        """Send ``message`` to ``chat_id`` using ``token``.

        Errors are logged but otherwise suppressed so they don't interrupt the
        realtime polling loop.
        """

        if not token or not chat_id or not message:
            return
        if httpx is None:  # pragma: no cover - optional dependency safeguard
            logger.debug("Telegram notifier skipped: httpx is unavailable")
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, json=payload)
                if response.status_code >= 400:
                    logger.error(
                        "Telegram notification failed with status %s: %s",
                        response.status_code,
                        response.text,
                    )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to send Telegram notification: %s", exc, exc_info=True)


__all__ = ["TelegramNotifier"]
