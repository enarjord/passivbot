"""Read-only, localhost-only browser inspection of saved backtest results."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from backtest_reports import equity_view, hsl_view, scope_name
from backtest_results import ResultStore

STATIC = Path(__file__).with_name("backtest_dashboard_static")


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class BacktestAPI:
    def __init__(self, directory):
        self.store = ResultStore(directory)

    def request(self, path):
        url = urlsplit(path)
        query = parse_qs(url.query, keep_blank_values=True)

        def text(name, default=None):
            values = query.get(name)
            if values is None:
                return default
            if len(values) != 1:
                raise ValueError("Query fields must occur once")
            return values[0]

        def integer(name, default=None):
            value = text(name)
            if value is None or value == "":
                return default
            if not value.isascii() or not value.isdecimal():
                raise ValueError("Query bounds must be nonnegative integers")
            result = int(value)
            if result > 2**53 - 1:
                raise ValueError("Query integer exceeds the supported clock domain")
            return result

        start, end = integer("start"), integer("end")
        if start is not None and end is not None and start > end:
            raise ValueError("Start must precede end")
        points = integer("points", 2000)
        if not 100 <= points <= 10_000:
            raise ValueError("Display points must be between 100 and 10000")
        if url.path == "/api/summary":
            report = self.store.hsl()
            return {
                "analysis": self.store.json("analysis.json"),
                "dataset": {"coins": self.store.json("dataset.json")["coins"]},
                "hsl": (
                    None
                    if report is None
                    else {
                        "summary": report["summary"],
                        "scopes": [
                            {
                                "name": scope_name(report, scope),
                                "policy": scope["policy"],
                            }
                            for scope in report["scopes"]
                        ],
                        "execution_episodes": report.get("execution_episodes"),
                        "events": report["events"],
                        "coins": report["coins"],
                        "trace_available": "trace" in report
                        or bool(report.get("samples")),
                    }
                ),
                "timings": self.store.json("timings.json", optional=True),
            }
        if url.path == "/api/equity":
            return equity_view(
                self.store,
                max_points=points,
                start=start,
                end=end,
                btc=text("btc", "0") == "1",
            )
        if url.path == "/api/hsl":
            view = hsl_view(
                self.store,
                scope=text("scope") or None,
                max_points=points,
                start=start,
                end=end,
            )
            if view is None:
                return None
            return {
                "scopes": view["scopes"],
                "trace_available": view["trace_available"],
            }
        if url.path == "/api/fills":
            frame = self.store.fills()
            coin, side = text("coin"), text("side", "all")
            if side not in {"all", "long", "short"}:
                raise ValueError("Unknown position side")
            if coin:
                if coin not in self.store.json("dataset.json")["coins"]:
                    raise ValueError("Unknown coin")
                frame = frame.loc[frame.coin == coin]
            if side != "all":
                frame = frame.loc[frame.type.str.endswith("_" + side)]
            stamps = frame.timestamp.dt.as_unit("ms").astype("int64")
            if start is not None:
                frame = frame.loc[stamps >= start]
            if end is not None:
                frame = frame.loc[stamps.loc[frame.index] <= end]
            limit, offset = integer("limit", 200), integer("offset", 0)
            if not 1 <= limit <= 1000:
                raise ValueError("Fill page size must be between 1 and 1000")
            count = len(frame)
            frame = frame.iloc[offset : offset + limit].copy()
            frame["timestamp"] = frame.timestamp.dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            return {
                "total": count,
                "offset": offset,
                "limit": limit,
                "rows": frame.to_dict(orient="records"),
            }
        raise KeyError("Unknown API path")


def handler_for(api):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, content, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            port = self.server.server_port
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            host = self.headers.get("Host", "").lower()
            origin = self.headers.get("Origin")
            if (
                host not in allowed
                or origin is not None
                and origin not in {f"http://{h}" for h in allowed}
            ):
                self.send(
                    403, b"Local same-origin requests only", "text/plain; charset=utf-8"
                )
                return
            path = urlsplit(self.path).path
            static = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "application/javascript"),
                "/style.css": ("style.css", "text/css"),
            }
            if path in static:
                name, content_type = static[path]
                self.send(
                    200, (STATIC / name).read_bytes(), content_type + "; charset=utf-8"
                )
                return
            try:
                result = json_safe(api.request(self.path))
                content = json.dumps(
                    result, allow_nan=False, separators=(",", ":")
                ).encode()
            except KeyError:
                self.send(
                    404,
                    b'{"error":"Result endpoint or field unavailable"}',
                    "application/json",
                )
            except ValueError:
                self.send(
                    400,
                    b'{"error":"Invalid result request or unsupported result schema"}',
                    "application/json",
                )
            except Exception:
                # File paths and raw exception payloads do not belong in HTTP
                # responses; the result stays readable by local CLI diagnostics.
                self.send(
                    500,
                    b'{"error":"Unable to read result artifact"}',
                    "application/json",
                )
            else:
                self.send(200, content, "application/json; charset=utf-8")

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument(
        "--port", type=int, default=0, help="Local port; 0 selects an available port."
    )
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), handler_for(BacktestAPI(args.result))
    )
    server.daemon_threads = True
    print(f"Backtest analyzer: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
