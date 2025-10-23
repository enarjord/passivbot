"""Command line entry point for the risk management web dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .configuration import load_realtime_config
from .web import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the Passivbot risk dashboard web UI")
    parser.add_argument("--config", type=Path, required=True, help="Path to the realtime configuration file")
    parser.add_argument("--host", default="0.0.0.0", help="Host address for the web server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the web server")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload (development only)")
    args = parser.parse_args(argv)

    config = load_realtime_config(args.config)
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
