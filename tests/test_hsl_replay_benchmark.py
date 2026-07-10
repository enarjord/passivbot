from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

from passivbot_cli import main as cli_main
from tools import hsl_replay_benchmark


pbr = pytest.importorskip("passivbot_rust", reason="passivbot_rust extension not available")
if bool(getattr(pbr, "__is_stub__", False)):
    pytest.skip("passivbot_rust extension not available", allow_module_level=True)


def _without_elapsed_ns(value):
    if isinstance(value, dict):
        return {
            key: _without_elapsed_ns(item)
            for key, item in value.items()
            if key not in {"elapsed_ns", "throughput"}
        }
    if isinstance(value, list):
        return [_without_elapsed_ns(item) for item in value]
    return value


def test_hsl_replay_benchmark_is_deterministic_and_has_no_side_effects():
    first = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(minutes=4, symbols=2, iterations=2)
    )
    second = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(minutes=4, symbols=2, iterations=2)
    )

    assert first["offline"] is True
    assert first["fixture"]["timeline_rows"] == 4
    assert first["fixture"]["fill_events"] == 2
    assert first["counters"] == {
        "iterations": 2,
        "active_pairs": 4,
        "expected_replay_samples": 16,
        "expected_current_samples": 4,
        "coin_metrics_sample_calls": 20,
        "replay_samples_applied": 16,
    }
    assert first["timings"]["history_load"]["calls"] == 2
    assert first["timings"]["coin_metrics_sample"]["calls"] == 20
    assert first["timings"]["current_upnl"]["calls"] == 4
    assert first["timings"]["cache_reuse_skipped"]["calls"] == 2
    assert first["timings"]["cache_persist_skipped"]["calls"] == 2
    assert first["throughput"]["timeline_rows_per_second"] > 0.0
    assert first["throughput"]["pair_rows_per_second"] == pytest.approx(
        first["throughput"]["timeline_rows_per_second"] * 2
    )
    assert first["side_effects"] == {
        "cache_reads": 0,
        "cache_writes": 0,
        "latch_removals": 0,
        "latch_writes": 0,
        "monitor_events": 0,
        "network_calls": 0,
    }
    assert _without_elapsed_ns(first) == _without_elapsed_ns(second)


def test_hsl_replay_benchmark_main_emits_json(capsys):
    assert hsl_replay_benchmark.main(["--minutes", "2", "--symbols", "1", "--compact"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["kind"] == "hsl_replay_benchmark"
    assert report["fixture"]["timeline_rows"] == 2


def test_hsl_replay_benchmark_tool_dispatch_forwards_module_and_prog(monkeypatch):
    captured = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["tool", "hsl-replay-benchmark", "--minutes", "10"]) == 0

    assert captured == {
        "module_name": "tools.hsl_replay_benchmark",
        "argv": ["passivbot tool hsl-replay-benchmark", "--minutes", "10"],
        "prog_env": "passivbot tool hsl-replay-benchmark",
    }
