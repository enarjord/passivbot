"""Loopback access to the CLI selector and its staged candidate export."""

from __future__ import annotations

import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import shlex
import sys
from threading import Lock
from typing import Callable
from uuid import uuid4

from pareto_explorer import (
    evaluate_selection,
    export_selection_seeds,
    selection_backtest_command,
    selection_cli_args,
)

MAX_REQUEST_BYTES = 65536


class PlotSelection:
    def __init__(self, pareto_dir, candidates, specs, args, initial, metrics, output):
        self.pareto_dir = pareto_dir
        self.candidates = candidates
        self.specs = specs
        self.args = args
        self.initial = initial
        self.output = output
        self.rows = {candidate.filename: index for index, candidate in enumerate(candidates)}
        self.limit_basis = {}
        for item in metrics:
            key = item["key"]
            if key.startswith("stats."):
                _, metric, reducer = key.split(".")
                self.limit_basis[key] = {"metric": metric, "reducer": reducer}
            else:
                self.limit_basis[key] = {"metric": key}
        if args.path and Path(args.path).expanduser().is_file():
            self.source = Path(args.path).expanduser().absolute()
        else:
            self.source = pareto_dir
        self.evaluate = lru_cache(maxsize=4)(self._evaluate)

    def limit_keys(self, limits):
        if not isinstance(limits, list) or len(limits) > len(self.limit_basis):
            raise ValueError("Expected a list of plot limits.")
        keys = []
        used = set()
        for limit in limits:
            if not isinstance(limit, dict):
                raise ValueError("Expected an object for each plot limit.")
            key, mode, value = (limit.get(field) for field in ("key", "mode", "value"))
            if (
                not isinstance(key, str)
                or key not in self.limit_basis
                or key in used
                or mode not in ("floor", "ceiling")
                or type(value) not in (int, float)
                or abs(value) > sys.float_info.max
                or not math.isfinite(value)
            ):
                raise ValueError("Plot limits require a distinct saved metric and a finite floor or ceiling.")
            used.add(key)
            keys.append((key, mode, value))
        return tuple(keys)

    def _evaluate(self, limits):
        entries = [
            json.dumps({
                **self.limit_basis[key],
                "penalize_if": "less_than" if mode == "floor" else "greater_than",
                "value": value,
            })
            for key, mode, value in limits
        ]
        args = argparse.Namespace(**vars(self.args))
        args.limit_entries = [*(args.limit_entries or []), *entries]
        evaluation = evaluate_selection(self.candidates, self.specs, args) if entries else self.initial
        return args, evaluation

    def result(self, limits=(), *, export=False):
        args, evaluation = self.evaluate(limits)
        candidate = evaluation.result.candidate
        result = {
            "selected": candidate.filename,
            "retained": [self.rows[item.filename] for item in evaluation.filtered_candidates],
            "method": evaluation.result.method,
            "objective_count": len(evaluation.active_specs),
            "selection_command": shlex.join([
                "passivbot", "tool", "pareto", str(self.source), *selection_cli_args(args)
            ]),
            "backtest_command": selection_backtest_command(candidate, evaluation.scenario),
        }
        if export:
            output = self.output.parent / f"{self.output.stem}-seeds-{uuid4().hex[:12]}"
            directory = export_selection_seeds(output, self.pareto_dir, evaluation)
            limits_for_optimizer = []
            for entry in evaluation.active_limits:
                limit = dict(entry)
                if evaluation.scenario is not None and "scenario" not in limit:
                    # A projected scenario has one observation. Its mean is the
                    # scenario value, which the optimizer accepts without a reducer.
                    limit.pop("reducer", None)
                    limit["scenario"] = evaluation.scenario
                limits_for_optimizer.append(limit)
            result.update(
                seed_directory=str(directory),
                optimize_command=shlex.join([
                    "passivbot", "optimize", str(candidate.path),
                    "--start", str(directory), "--limits", json.dumps(limits_for_optimizer),
                ]),
            )
        return result


def selection_server(selection: PlotSelection, render: Callable[[dict], str]):
    route = "/" + secrets.token_urlsafe(24)
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format, *args):
            pass

        def respond(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # The browser can leave the page while a request is running.
                return

        def do_GET(self):
            if self.headers.get("Host") != host or self.path != route:
                self.send_error(404)
                return
            self.respond(200, page, "text/html; charset=utf-8")

        def do_POST(self):
            if self.headers.get("Host") != host or self.path != route + "/selection":
                self.send_error(404)
                return
            if self.headers.get("Origin") != "http://" + host:
                self.send_error(403)
                return
            if self.headers.get_content_type() != "application/json":
                self.send_error(415)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REQUEST_BYTES:
                    raise ValueError("Invalid plot selection request size.")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict) or type(data.get("export")) is not bool:
                    raise ValueError("Invalid plot selection request.")
                keys = selection.limit_keys(data.get("limits"))
                with lock:
                    result = selection.result(keys, export=data["export"])
                status = 200
            except (ValueError, OSError, RecursionError) as error:
                result = {"error": str(error)}
                status = 400
            self.respond(status, json.dumps(result, allow_nan=False).encode("utf-8"), "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    host = f"127.0.0.1:{server.server_port}"
    try:
        page = render({"endpoint": route + "/selection", "initial": selection.result()}).encode("utf-8")
    except Exception:
        server.server_close()
        raise
    return server, "http://" + host + route
