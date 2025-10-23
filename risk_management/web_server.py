"""Command line entry point for the risk management web dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .configuration import CustomEndpointSettings, load_realtime_config
from .web import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the Passivbot risk dashboard web UI")
    parser.add_argument("--config", type=Path, required=True, help="Path to the realtime configuration file")
    parser.add_argument("--host", default="0.0.0.0", help="Host address for the web server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the web server")
    parser.add_argument(
        "--custom-endpoints",
        help=(
            "Override custom endpoint behaviour. Provide a JSON file path to reuse the same "
            "proxy configuration as Passivbot, 'auto' to enable auto-discovery, or 'none' to "
            "disable overrides."
        ),
    )
    parser.add_argument("--reload", action="store_true", help="Enable autoreload (development only)")
    args = parser.parse_args(argv)

    config = load_realtime_config(args.config)
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
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
