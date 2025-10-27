"""Command line entry point for the risk management web dashboard."""

from __future__ import annotations

import argparse
import copy
import importlib
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .configuration import CustomEndpointSettings, load_realtime_config
from .letsencrypt import LetsEncryptError, ensure_certificate

if TYPE_CHECKING:  # pragma: no cover - used only for type hints
    import uvicorn


def _import_uvicorn() -> "uvicorn":
    """Import :mod:`uvicorn` with a helpful error message when missing."""

    try:
        import uvicorn  # type: ignore[import]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime environment
        raise ModuleNotFoundError(
            "The 'uvicorn' package is required to run the risk management web server. "
            "Install it via 'pip install standalone-risk[dashboard]' or add uvicorn to your environment."
        ) from exc
    return uvicorn


def _determine_uvicorn_logging(config) -> tuple[Optional[dict], str]:
    """Return logging configuration overrides for uvicorn."""

    debug_requested = config.debug_api_payloads or any(
        account.debug_api_payloads for account in getattr(config, "accounts", [])
    )
    if not debug_requested:
        return None, "info"
    try:
        uvicorn_config = importlib.import_module("uvicorn.config")
    except ModuleNotFoundError:  # pragma: no cover - uvicorn not importable in tests
        return None, "debug"
    LOGGING_CONFIG = getattr(uvicorn_config, "LOGGING_CONFIG", None)
    if LOGGING_CONFIG is None:  # pragma: no cover - unexpected configuration shape
        return None, "debug"

    log_config = copy.deepcopy(LOGGING_CONFIG)
    loggers = log_config.setdefault("loggers", {})
    root_logger = loggers.setdefault("", {"handlers": ["default"], "level": "INFO"})
    root_logger["level"] = "DEBUG"

    risk_logger = loggers.setdefault(
        "risk_management", {"handlers": ["default"], "level": "INFO", "propagate": False}
    )
    if not risk_logger.get("handlers"):
        risk_logger["handlers"] = ["default"]
    risk_logger["level"] = "DEBUG"
    risk_logger.setdefault("propagate", False)

    # Make sure the namespace used by our modules inherits the debug level as well.
    risk_root = loggers.setdefault(
        "risk_management.realtime", {"handlers": ["default"], "level": "INFO", "propagate": False}
    )
    risk_root["level"] = "DEBUG"
    risk_root.setdefault("propagate", False)

    return log_config, "debug"


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the risk dashboard web UI")
    parser.add_argument("--config", type=Path, required=True, help="Path to the realtime configuration file")
    parser.add_argument("--host", default="0.0.0.0", help="Host address for the web server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the web server")
    parser.add_argument(
        "--custom-endpoints",
        help=(
            "Override custom endpoint behaviour. Provide a JSON file path to reuse the same "
            "proxy configuration as the trading system, 'auto' to enable auto-discovery, or "
            "'none' to disable overrides."
        ),
    )
    parser.add_argument("--reload", action="store_true", help="Enable autoreload (development only)")
    parser.add_argument("--ssl-certfile", type=Path, help="Path to the TLS certificate file")
    parser.add_argument("--ssl-keyfile", type=Path, help="Path to the TLS private key file")
    parser.add_argument(
        "--ssl-keyfile-password",
        help="Password used to decrypt the TLS private key, if required",
    )
    parser.add_argument(
        "--letsencrypt-webroot",
        type=Path,
        help=(
            "Serve ACME http-01 challenges from this directory. Useful when certbot "
            "is executed separately with the --webroot plugin."
        ),
    )
    parser.add_argument(
        "--letsencrypt-domain",
        action="append",
        dest="letsencrypt_domains",
        help=(
            "Provision certificates automatically using certbot in standalone mode. "
            "Specify one or more domains by repeating this flag."
        ),
    )
    parser.add_argument(
        "--letsencrypt-email",
        help="Contact email used when registering with Let's Encrypt",
    )
    parser.add_argument(
        "--letsencrypt-staging",
        action="store_true",
        help="Use the Let's Encrypt staging environment for testing",
    )
    parser.add_argument(
        "--letsencrypt-http-port",
        type=int,
        default=80,
        help="HTTP port used by certbot's standalone challenge server",
    )
    parser.add_argument(
        "--letsencrypt-cert-name",
        help="Override the certificate lineage name stored by certbot",
    )
    parser.add_argument(
        "--letsencrypt-config-dir",
        type=Path,
        help="Custom directory for certbot configuration data",
    )
    parser.add_argument(
        "--letsencrypt-work-dir",
        type=Path,
        help="Custom directory for certbot working files",
    )
    parser.add_argument(
        "--letsencrypt-logs-dir",
        type=Path,
        help="Custom directory for certbot logs",
    )
    parser.add_argument(
        "--letsencrypt-executable",
        default="certbot",
        help="Path to the certbot executable",
    )
    parser.add_argument(
        "--letsencrypt-dry-run",
        action="store_true",
        help="Perform a dry-run against the ACME staging environment",
    )
    args = parser.parse_args(argv)

    config = load_realtime_config(args.config)
    log_config, log_level = _determine_uvicorn_logging(config)
    from .web import create_app  # imported lazily to avoid heavy dependencies at import time
    override = args.custom_endpoints
    if override is not None:
        override_normalized = override.strip()
        if not override_normalized:
            config.custom_endpoints = None
        else:
            lowered = override_normalized.lower()
            if lowered in {"none", "off", "disable"}:
                config.custom_endpoints = CustomEndpointSettings(path=None, autodiscover=False)
            elif lowered in {"auto", "autodiscover", "default"}:
                config.custom_endpoints = CustomEndpointSettings(path=None, autodiscover=True)
            else:
                config.custom_endpoints = CustomEndpointSettings(
                    path=override_normalized,
                    autodiscover=False,
                )
    if bool(args.ssl_certfile) ^ bool(args.ssl_keyfile):
        parser.error("Both --ssl-certfile and --ssl-keyfile must be provided to enable HTTPS.")

    letsencrypt_domains = args.letsencrypt_domains or []
    ssl_certfile = str(args.ssl_certfile) if args.ssl_certfile else None
    ssl_keyfile = str(args.ssl_keyfile) if args.ssl_keyfile else None

    if letsencrypt_domains:
        if ssl_certfile or ssl_keyfile:
            parser.error(
                "Manual TLS parameters cannot be combined with automatic Let's Encrypt provisioning."
            )
        try:
            cert_path, key_path = ensure_certificate(
                executable=args.letsencrypt_executable,
                domains=letsencrypt_domains,
                email=args.letsencrypt_email,
                staging=args.letsencrypt_staging or args.letsencrypt_dry_run,
                http_port=args.letsencrypt_http_port,
                cert_name=args.letsencrypt_cert_name,
                config_dir=args.letsencrypt_config_dir,
                work_dir=args.letsencrypt_work_dir,
                logs_dir=args.letsencrypt_logs_dir,
                dry_run=args.letsencrypt_dry_run,
            )
        except LetsEncryptError as exc:
            parser.error(str(exc))
        ssl_certfile, ssl_keyfile = str(cert_path), str(key_path)

    app = create_app(config, letsencrypt_challenge_dir=args.letsencrypt_webroot)

    if (
        getattr(config, "auth", None)
        and getattr(config.auth, "https_only", False)
        and not ssl_certfile
        and not ssl_keyfile
    ):
        logging.getLogger("risk_management.web_server").warning(
            "Authentication is configured for HTTPS-only sessions but no TLS certificate/key "
            "were supplied. Either launch the server with --ssl-certfile/--ssl-keyfile or set "
            "'auth.https_only' to false in the realtime configuration for non-TLS development environments."
        )

    uvicorn = _import_uvicorn()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=log_config,
        log_level=log_level,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_keyfile_password=args.ssl_keyfile_password,
    )


if __name__ == "__main__":
    main()
