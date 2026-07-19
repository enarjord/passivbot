from __future__ import annotations

import asyncio
import copy
import json
import os
import sys

import pytest

from passivbot_cli import main as cli_main
from tools import hsl_replay_benchmark


pbr = pytest.importorskip("passivbot_rust", reason="passivbot_rust extension not available")
if bool(getattr(pbr, "__is_stub__", False)):
    pytest.skip("passivbot_rust extension not available", allow_module_level=True)


def _without_timing(value):
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if key not in {"timings", "throughput", "stage_profile", "pipeline_profile"}
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def test_hsl_replay_benchmark_is_deterministic_and_has_no_side_effects():
    first = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(minutes=4, symbols=2, iterations=2)
    )
    second = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(minutes=4, symbols=2, iterations=2)
    )

    assert first["offline"] is True
    assert first["schema_version"] == 3
    assert first["fixture"]["compact_rows"] == 4
    assert first["fixture"]["fill_events"] == 1
    assert first["counters"] == {
        "iterations": 2,
        "active_pairs": 2,
        "held_pairs": 2,
        "background_pairs": 0,
        "expected_replay_samples": 8,
        "expected_current_samples": 2,
        "expected_held_samples": 8,
        "expected_background_samples": 0,
        "expected_background_yields": 0,
        "coin_metrics_sample_calls": 10,
        "replay_samples_applied": 8,
        "held_replay_samples": 8,
        "background_replay_samples": 0,
    }
    assert {stage: first["timings"][stage]["calls"] for stage in first["timings"]} == {
        "fixture_construction": 1,
        "history_load": 2,
        "cache_reuse_skipped": 2,
        "coin_metrics_sample": 10,
        "held_coin_metrics_sample": 10,
        "background_coin_metrics_sample": 0,
        "current_upnl": 2,
        "cache_persist_skipped": 2,
        "final_state_projection": 2,
        "full_replay": 2,
    }
    assert all(values["elapsed_ns"] >= 0 for values in first["timings"].values())
    assert first["throughput"]["timeline_rows_per_second"] > 0.0
    assert first["throughput"]["pair_rows_per_second"] == pytest.approx(
        first["throughput"]["timeline_rows_per_second"]
    )
    assert first["throughput"]["applied_pair_samples_per_second"] > 0.0
    assert first["side_effects"] == {
        "cache_reads": 0,
        "cache_writes": 0,
        "latch_removals": 0,
        "latch_writes": 0,
        "monitor_events": 0,
        "network_calls": 0,
    }
    assert first["equivalence"]["matches"] is True
    assert first["equivalence"]["sample_counts"]["matches"] is True
    assert first["equivalence"]["output_state"]["matches"] is True
    assert _without_timing(first) == _without_timing(second)


def test_hsl_replay_benchmark_separates_held_and_background_work():
    report = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(
            minutes=700, symbols=4, held_symbols=1, iterations=2
        )
    )

    assert report["fixture"]["held_symbols"] == 1
    assert report["fixture"]["background_symbols"] == 1
    assert report["fixture"]["fill_events"] == 4
    assert report["counters"] == {
        "iterations": 2,
        "active_pairs": 4,
        "held_pairs": 2,
        "background_pairs": 2,
        "expected_replay_samples": 2_800,
        "expected_current_samples": 4,
        "expected_held_samples": 1_400,
        "expected_background_samples": 1_400,
        "expected_background_yields": 14,
        "coin_metrics_sample_calls": 1_438,
        "replay_samples_applied": 1_434,
        "held_replay_samples": 1_400,
        "background_replay_samples": 34,
    }
    timings = report["timings"]
    assert timings["held_coin_metrics_sample"]["calls"] == 1_402
    assert timings["background_coin_metrics_sample"]["calls"] == 36
    assert (
        timings["held_coin_metrics_sample"]["calls"]
        + timings["background_coin_metrics_sample"]["calls"]
        == timings["coin_metrics_sample"]["calls"]
    )
    assert (
        timings["held_coin_metrics_sample"]["elapsed_ns"]
        + timings["background_coin_metrics_sample"]["elapsed_ns"]
        == timings["coin_metrics_sample"]["elapsed_ns"]
    )


def test_hsl_replay_benchmark_stage_profile_is_exclusive_and_bounded():
    report = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(
            minutes=700, symbols=4, held_symbols=1, iterations=2
        )
    )

    profile = report["stage_profile"]
    full_replay = profile["full_replay"]
    assert profile["taxonomy"] == "exclusive_full_replay_leaf_stages_v1"
    assert list(full_replay["stages"]) == list(
        hsl_replay_benchmark.FULL_REPLAY_LEAF_STAGE_NAMES
    )
    assert full_replay["accounted_elapsed_ns"] == sum(
        stage["elapsed_ns"] for stage in full_replay["stages"].values()
    )
    assert full_replay["elapsed_ns"] == (
        full_replay["accounted_elapsed_ns"]
        + full_replay["residual_orchestration"]["elapsed_ns"]
    )
    for stage in list(full_replay["stages"].values()) + [
        full_replay["residual_orchestration"]
    ]:
        assert stage["elapsed_ns"] >= 0
        assert 0.0 <= stage["percent_of_full_replay"] <= 100.0
    assert profile["outside_full_replay"]["fixture_construction"]["calls"] == 1
    assert profile["outside_full_replay"]["final_state_projection"]["calls"] == 2


def _assert_exact_profile_accounting(profile, percent_key):
    assert profile["accounted_elapsed_ns"] == sum(
        stage["elapsed_ns"] for stage in profile["stages"].values()
    )
    assert profile["elapsed_ns"] == (
        profile["accounted_elapsed_ns"]
        + profile["residual_orchestration"]["elapsed_ns"]
    )
    for stage in list(profile["stages"].values()) + [
        profile["residual_orchestration"]
    ]:
        assert stage["elapsed_ns"] >= 0
        assert 0.0 <= stage[percent_key] <= 100.0


def test_hsl_replay_benchmark_pipeline_profiles_distinct_runs_without_overlap():
    report = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(
            minutes=700, symbols=4, held_symbols=1, iterations=2
        )
    )

    reference = report["dense_reference"]
    assert reference["same_run_as_candidate"] is False
    assert reference["fixture"]["history_format"] == "timeline"
    assert reference["stage_profile"]["run"]["calls"] == 1
    assert reference["stage_profile"]["full_replay"]["calls"] == 2
    assert reference["stage_profile"]["outside_full_replay"][
        "final_state_projection"
    ]["calls"] == 2
    assert reference["throughput"]["timeline_rows_per_second"] > 0.0
    assert list(report["pipeline_profile"]["stages"]) == [
        "candidate_run",
        "dense_reference_run",
        "equivalence_comparison",
    ]

    _assert_exact_profile_accounting(
        report["stage_profile"]["full_replay"], "percent_of_full_replay"
    )
    _assert_exact_profile_accounting(report["stage_profile"]["run"], "percent_of_run")
    _assert_exact_profile_accounting(
        reference["stage_profile"]["full_replay"], "percent_of_full_replay"
    )
    _assert_exact_profile_accounting(
        reference["stage_profile"]["run"], "percent_of_run"
    )
    _assert_exact_profile_accounting(report["pipeline_profile"], "percent_of_pipeline")


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
    assert len(history["timeline"][0]["realized_pnl_by_coin_pside"]) == 1
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
    assert hsl_replay_benchmark._validate_fixture_shape(
        1, 1, 0, local_scale=True
    ) == (1, 1, 0)


def test_hsl_replay_benchmark_full_scale_fixture_has_sparse_replay_cases():
    fixture = hsl_replay_benchmark.build_coin_hsl_compact_fixture(
        minutes=43_201, symbols=4, held_symbols=1, local_scale=True
    )
    contract = fixture["fixture_contract"]
    compact = fixture["hsl_coin_compact_replay"]

    assert len(compact["timestamps"]) == 43_201
    assert contract["held_symbols"] == ["HSLBENCH00/USDT:USDT"]
    assert contract["ema_span_minutes"] > 1.0
    assert contract["pair_values_are_dense"] is True
    assert contract["account_balance_driver_symbol"] == "HSLBENCH02/USDT:USDT"
    assert [episode["name"] for episode in contract["episodes"]] == [
        "historical_a",
        "historical_b",
        "historical_c",
        "balance_driver",
        "panic_flatten",
    ]
    historical_episodes = [
        episode for episode in contract["episodes"] if episode["symbol_index"] == 1
    ]
    assert len(historical_episodes) == 3
    assert historical_episodes[1]["start_minute"] - historical_episodes[0]["end_minute"] > 1_000
    assert historical_episodes[2]["start_minute"] - historical_episodes[1]["end_minute"] > 1_000
    fill_ids = [event["id"] for event in fixture["fill_events"]]
    assert [fill_ids.index(event_id) for event_id in contract["same_timestamp_fill_order"]] == sorted(
        fill_ids.index(event_id) for event_id in contract["same_timestamp_fill_order"]
    )
    ordered_fills = [
        event
        for event in fixture["fill_events"]
        if event["id"] in contract["same_timestamp_fill_order"]
    ]
    assert len({event["timestamp"] for event in ordered_fills}) == 1
    assert compact["balances"][18_004] == pytest.approx(1_000.0)
    assert compact["balances"][18_005] == pytest.approx(1_024.0)
    assert fixture["panic_flatten_events"] == [
        {
            "timestamp": 2_160_900_200,
            "minute_timestamp": 2_160_900_000,
            "pside": "long",
            "symbol": "HSLBENCH03/USDT:USDT",
        }
    ]


def test_hsl_replay_benchmark_compact_and_timeline_reach_identical_state():
    timeline = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(
            minutes=700, symbols=2, held_symbols=1, history_format="timeline"
        )
    )
    compact = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(
            minutes=700, symbols=2, held_symbols=1, history_format="compact"
        )
    )

    assert timeline["determinism"] == compact["determinism"]
    assert timeline["side_effects"] == compact["side_effects"]
    assert timeline["fixture"]["scenario_sha256"] == compact["fixture"]["scenario_sha256"]
    assert compact["dense_reference"]["history_format"] == "timeline"
    assert compact["dense_reference"]["same_run_as_candidate"] is False
    assert timeline["dense_reference"]["same_run_as_candidate"] is True
    assert "stage_profile" not in timeline["dense_reference"]
    assert "throughput" not in timeline["dense_reference"]
    assert list(timeline["pipeline_profile"]["stages"]) == [
        "candidate_run",
        "equivalence_comparison",
    ]
    assert compact["equivalence"]["matches"] is True
    assert compact["equivalence"]["output_state"]["matches"] is True
    assert compact["equivalence"]["sample_counts"]["matches"] is True
    assert compact["equivalence"]["sample_counts"]["reduction"]["replay_samples_applied"] > 0
    assert timeline["fixture"]["timeline_rows"] == 700
    assert compact["fixture"]["compact_rows"] == 700


def test_hsl_replay_benchmark_dense_reference_comparison_detects_mismatch():
    reference = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(minutes=8, symbols=3, held_symbols=1)
    )
    candidate = copy.deepcopy(reference)
    candidate["counters"]["replay_samples_applied"] = (
        reference["dense_reference"]["sample_counts"]["replay_samples_applied"] + 1
    )

    comparison = hsl_replay_benchmark.compare_dense_reference_output(reference, candidate)

    assert comparison["fixture_scenario_matches"] is True
    assert comparison["sample_counts"]["matches"] is False
    assert comparison["output_state"]["matches"] is True
    assert comparison["matches"] is False


def test_hsl_replay_benchmark_dense_reference_comparison_excludes_timing():
    reference = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(minutes=8, symbols=3, held_symbols=1)
    )
    candidate = copy.deepcopy(reference)
    candidate["timings"]["full_replay"]["elapsed_ns"] += 1
    candidate["stage_profile"]["full_replay"]["residual_orchestration"]["elapsed_ns"] += 1
    candidate["stage_profile"]["run"]["residual_orchestration"]["elapsed_ns"] += 1
    candidate["stage_profile"]["full_replay"]["accounted_percent_of_full_replay"] = 0.0
    candidate["pipeline_profile"]["elapsed_ns"] += 1
    candidate["dense_reference"]["timings"]["full_replay"]["elapsed_ns"] += 1

    comparison = hsl_replay_benchmark.compare_dense_reference_output(reference, candidate)

    assert comparison["sample_counts"]["matches"] is True
    assert comparison["output_state"]["matches"] is True
    assert comparison["matches"] is True


@pytest.mark.parametrize(
    "path",
    [
        ("pnl_reset_timestamp_ms",),
        ("cooldown_intervention_active",),
        ("cooldown_repanic_reset_pending",),
        ("cooldown_unresolved_residue",),
        ("runtime", "red_seen_in_episode"),
        ("last_metrics", "red_seen_in_episode"),
    ],
)
def test_hsl_replay_benchmark_equivalence_detects_runtime_contract_state(path):
    reference = asyncio.run(
        hsl_replay_benchmark.run_hsl_replay_benchmark(
            minutes=8, symbols=3, held_symbols=1
        )
    )
    candidate = copy.deepcopy(reference)
    value = candidate["output_state"]["pairs"][0]
    for key in path[:-1]:
        value = value[key]
    current = value[path[-1]]
    value[path[-1]] = not current if isinstance(current, bool) else 123_456
    candidate["output_state"]["sha256"] = (
        hsl_replay_benchmark._state_digest_from_projection(
            candidate["output_state"]["pairs"]
        )
    )

    comparison = hsl_replay_benchmark.compare_dense_reference_output(
        reference, candidate
    )

    assert comparison["sample_counts"]["matches"] is True
    assert comparison["output_state"]["matches"] is False
    assert comparison["matches"] is False


def test_hsl_replay_benchmark_main_emits_json(capsys):
    assert hsl_replay_benchmark.main(["--minutes", "2", "--symbols", "1", "--compact"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["kind"] == "hsl_replay_benchmark"
    assert report["fixture"]["compact_rows"] == 2

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
