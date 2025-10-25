"""Utilities for provisioning TLS certificates via Let's Encrypt."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

LOGGER = logging.getLogger(__name__)


class LetsEncryptError(RuntimeError):
    """Raised when Let's Encrypt automation fails."""


def _normalize_domains(domains: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for domain in domains:
        candidate = domain.strip()
        if not candidate:
            continue
        normalized.append(candidate)
    return normalized


def ensure_certificate(
    *,
    executable: str = "certbot",
    domains: Sequence[str],
    email: str | None = None,
    staging: bool = False,
    http_port: int = 80,
    cert_name: str | None = None,
    config_dir: Path | None = None,
    work_dir: Path | None = None,
    logs_dir: Path | None = None,
    dry_run: bool = False,
) -> tuple[Path, Path]:
    """Ensure a Let's Encrypt certificate exists and return its paths.

    The helper invokes the ``certbot`` CLI in standalone mode so the http-01
    challenge is handled automatically. Existing certificates are reused until
    renewal is required thanks to the ``--keep-until-expiring`` flag.
    """

    normalized_domains = _normalize_domains(domains)
    if not normalized_domains:
        raise LetsEncryptError("At least one domain must be supplied for Let's Encrypt provisioning.")

    certbot_path = shutil.which(executable)
    if certbot_path is None:
        raise LetsEncryptError(
            f"Unable to locate the '{executable}' executable required for Let's Encrypt automation."
        )

    storage_dir = Path(config_dir) if config_dir else Path("/etc/letsencrypt")
    lineage = cert_name or normalized_domains[0]

    command: list[str] = [
        certbot_path,
        "certonly",
        "--non-interactive",
        "--agree-tos",
        "--keep-until-expiring",
        "--standalone",
        "--preferred-challenges",
        "http",
        "--http-01-port",
        str(http_port),
    ]

    if email:
        command.extend(["--email", email])
    else:
        command.append("--register-unsafely-without-email")

    if staging:
        command.append("--staging")

    if cert_name:
        command.extend(["--cert-name", cert_name])

    if config_dir:
        command.extend(["--config-dir", str(config_dir)])
    if work_dir:
        command.extend(["--work-dir", str(work_dir)])
    if logs_dir:
        command.extend(["--logs-dir", str(logs_dir)])
    if dry_run:
        command.append("--dry-run")

    for domain in normalized_domains:
        command.extend(["-d", domain])

    LOGGER.info(
        "Requesting/renewing Let's Encrypt certificate for %s via %s",
        ", ".join(normalized_domains),
        certbot_path,
    )

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - exercised via unit tests
        raise LetsEncryptError(
            "Let's Encrypt provisioning failed; inspect certbot output for details."
        ) from exc

    certfile = storage_dir / "live" / lineage / "fullchain.pem"
    keyfile = storage_dir / "live" / lineage / "privkey.pem"

    if not certfile.exists() or not keyfile.exists():
        raise LetsEncryptError(
            "Certbot completed but expected certificate files were not found at "
            f"{certfile} and {keyfile}."
        )

    return certfile, keyfile

