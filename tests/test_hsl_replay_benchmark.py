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
        "held_pairs": 4,
        "background_pairs": 0,
        "expected_replay_samples": 16,
        "expected_current_samples": 4,
        "expected_held_samples": 16,
        "expected_background_samples": 0,
        "expected_background_yields": 0,
        "coin_metrics_sample_calls": 20,
        "replay_samples_applied": 16,
        "held_replay_samples": 16,
        "background_replay_samples": 0,
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


def test_hsl_replay_benchmark_separates_held_and_background_work():
    report = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(
            minutes=205, symbols=4, held_symbols=2, iterations=2
        )
    )

    assert report["fixture"]["held_symbols"] == 2
    assert report["fixture"]["background_symbols"] == 2
    assert report["fixture"]["fill_events"] == 2
    assert report["counters"] == {
        "iterations": 2,
        "active_pairs": 8,
        "held_pairs": 4,
        "background_pairs": 4,
        "expected_replay_samples": 1_640,
        "expected_current_samples": 8,
        "expected_held_samples": 820,
        "expected_background_samples": 820,
        "expected_background_yields": 8,
        "coin_metrics_sample_calls": 1_648,
        "replay_samples_applied": 1_640,
        "held_replay_samples": 820,
        "background_replay_samples": 820,
    }


def test_hsl_replay_benchmark_memory_profile_is_opt_in():
    report = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(
            minutes=2,
            symbols=2,
            held_symbols=1,
            profile_memory=True,
            history_format="compact",
        )
    )

    assert report["fixture"]["history_format"] == "compact"
    assert report["fixture"]["timeline_rows"] == 0
    assert report["fixture"]["compact_rows"] == 2
    assert report["memory"]["tracemalloc"] is True
    assert report["memory"]["current_bytes"] >= 0
    assert report["memory"]["peak_bytes"] >= report["memory"]["current_bytes"]


def test_hsl_replay_benchmark_local_scale_limits_are_opt_in():
    with pytest.raises(ValueError, match="between 1 and 1440"):
        hsl_replay_benchmark.build_coin_hsl_history_fixture(minutes=1_441, symbols=1)

    history = hsl_replay_benchmark.build_coin_hsl_history_fixture(
        minutes=1, symbols=30, held_symbols=1, local_scale=True
    )
    assert len(history["timeline"]) == 1
    assert len(history["timeline"][0]["realized_pnl_by_coin_pside"]) == 30
    assert len(history["fill_events"]) == 1
    assert hsl_replay_benchmark._validate_fixture_shape(
        43_201, 30, 1, local_scale=True
    ) == (43_201, 30, 1)
    with pytest.raises(ValueError, match="between 1 and 43201"):
        hsl_replay_benchmark._validate_fixture_shape(
            43_202, 30, 1, local_scale=True
        )
    with pytest.raises(ValueError, match="between 1 and 30"):
        hsl_replay_benchmark._validate_fixture_shape(1, 31, 1, local_scale=True)


def test_hsl_replay_benchmark_compact_and_timeline_reach_identical_state():
    timeline = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(
            minutes=8, symbols=3, held_symbols=2, history_format="timeline"
        )
    )
    compact = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(
            minutes=8, symbols=3, held_symbols=2, history_format="compact"
        )
    )

    assert timeline["determinism"] == compact["determinism"]
    assert timeline["counters"] == compact["counters"]
    assert timeline["side_effects"] == compact["side_effects"]
    assert timeline["fixture"]["timeline_rows"] == 8
    assert compact["fixture"]["compact_rows"] == 8


def test_hsl_replay_benchmark_main_emits_json(capsys):
    assert hsl_replay_benchmark.main(["--minutes", "2", "--symbols", "1", "--compact"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["kind"] == "hsl_replay_benchmark"
    assert report["fixture"]["timeline_rows"] == 2

    assert (
        hsl_replay_benchmark.main(
            [
                "--minutes",
                "2",
                "--symbols",
                "1",
                "--history-format",
                "compact",
                "--compact",
            ]
        )
        == 0
    )
    compact_report = json.loads(capsys.readouterr().out)
    assert compact_report["fixture"]["history_format"] == "compact"

    assert (
        hsl_replay_benchmark.main(
            [
                "--local-scale",
                "--minutes",
                "1",
                "--symbols",
                "1",
                "--compact",
            ]
        )
        == 0
    )
    local_report = json.loads(capsys.readouterr().out)
    assert local_report["fixture"]["minutes"] == 1


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
