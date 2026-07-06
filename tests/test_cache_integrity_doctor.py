import json
import os
import sys

import numpy as np

import passivbot_cli.main as cli_main
from tools.cache_integrity_doctor import build_cache_integrity_report, main


def test_cache_integrity_report_counts_files_and_flags_corrupt_artifacts(tmp_path):
    root = tmp_path / "caches"
    root.mkdir()
    (root / "markets.json").write_text(json.dumps({"BTC": {}}), encoding="utf-8")
    (root / "bad.json").write_text("{bad json", encoding="utf-8")
    (root / "events.ndjson").write_text(
        json.dumps({"ok": True}) + "\n{bad ndjson\n",
        encoding="utf-8",
    )
    (root / "empty.ndjson").write_text("", encoding="utf-8")
    np.save(root / "prices.npy", np.array([1.0, 2.0]))
    (root / "corrupt.npy").write_bytes(b"not a numpy file")

    report = build_cache_integrity_report([root])

    assert report["ok"] is False
    assert report["summary"]["file_count"] == 6
    assert report["roots"][0]["by_extension"] == {
        ".json": 2,
        ".ndjson": 2,
        ".npy": 2,
    }
    issues = {(issue["code"], issue["path"]) for issue in report["issues"]}
    assert ("json_decode_failed", str(root / "bad.json")) in issues
    assert ("ndjson_decode_failed", str(root / "events.ndjson")) in issues
    assert ("empty_file", str(root / "empty.ndjson")) in issues
    assert ("npy_load_failed", str(root / "corrupt.npy")) in issues


def test_cache_integrity_report_summarizes_cache_families(tmp_path):
    root = tmp_path / "caches"
    (root / "okx" / "ohlcvs_1m").mkdir(parents=True)
    np.save(root / "okx" / "ohlcvs_1m" / "BTC.npy", np.array([1.0, 2.0]))
    (root / "okx" / "ohlcvs_1m" / "ETH.npy").write_bytes(b"not numpy")
    (root / "fills" / "binance").mkdir(parents=True)
    (root / "fills" / "binance" / "events.ndjson").write_text(
        json.dumps({"id": 1}) + "\n",
        encoding="utf-8",
    )
    (root / "markets.json").write_text(json.dumps({"BTC": {}}), encoding="utf-8")
    (root / "tmp.npy.lock").write_text("locked", encoding="utf-8")

    report = build_cache_integrity_report([root])

    families = report["summary"]["by_family"]
    assert families["candles"]["file_count"] == 2
    assert families["candles"]["by_extension"] == {".npy": 2}
    assert families["candles"]["issue_count"] == 1
    assert families["fills"]["file_count"] == 1
    assert families["fills"]["by_extension"] == {".ndjson": 1}
    assert families["markets"]["file_count"] == 1
    assert families["locks"]["file_count"] == 1
    root_families = report["roots"][0]["by_family"]
    assert root_families == families
    issue = next(item for item in report["issues"] if item["code"] == "npy_load_failed")
    assert issue["family"] == "candles"


def _hsl_replay_cache_rows():
    return [
        {"ts": 60_000, "price": 100.0, "psize": 0.0, "pprice": 0.0, "pnl": 0.0, "upnl": 0.0},
        {"ts": 120_000, "price": 101.0, "psize": 1.0, "pprice": 100.0, "pnl": 0.5, "upnl": 1.0},
        {"ts": 180_000, "price": 102.0, "psize": 1.0, "pprice": 100.0, "pnl": 0.0, "upnl": 2.0},
    ]


def _hsl_replay_cache_metadata():
    return {
        "exchange": "binance",
        "market_type": "swap",
        "user": "user_01",
        "config_digest": "cfg_digest",
        "signal_mode": "coin",
        "pside": "long",
        "symbol": "BTC/USDT:USDT",
        "fill_covered_start_ms": 60_000,
        "fill_covered_end_ms": 180_000,
        "fill_history_scope": "window",
        "fill_coverage_proven": True,
        "candle_covered_start_ms": 60_000,
        "candle_covered_end_ms": 180_000,
    }


def test_cache_integrity_report_validates_hsl_replay_cache(tmp_path):
    import passivbot_hsl as hsl

    cache_dir = tmp_path / "caches" / "equity_hard_stop" / "binance" / "user_01" / "BTC"
    hsl._write_hsl_replay_matrix_cache(
        cache_dir,
        _hsl_replay_cache_rows(),
        _hsl_replay_cache_metadata(),
    )

    report = build_cache_integrity_report([tmp_path / "caches"])

    risk = report["summary"]["by_family"]["risk"]
    metadata = risk["metadata"]
    assert metadata["hsl_replay_cache_count"] == 1
    assert metadata["hsl_replay_cache_valid_count"] == 1
    assert metadata["hsl_replay_cache_invalid_count"] == 0
    assert metadata["hsl_replay_cache_reason_counts"] == {}
    assert metadata["hsl_compatibility"] == "hsl_replay_cache_valid"
    assert report["issues"] == []


def test_cache_integrity_report_flags_invalid_hsl_replay_cache(tmp_path):
    import passivbot_hsl as hsl

    cache_dir = tmp_path / "caches" / "equity_hard_stop" / "binance" / "user_01" / "BTC"
    hsl._write_hsl_replay_matrix_cache(
        cache_dir,
        _hsl_replay_cache_rows(),
        _hsl_replay_cache_metadata(),
    )
    (cache_dir / hsl._HSL_REPLAY_CACHE_MATRIX_FILENAME).unlink()

    report = build_cache_integrity_report([tmp_path / "caches"])

    risk = report["summary"]["by_family"]["risk"]
    metadata = risk["metadata"]
    assert metadata["hsl_replay_cache_count"] == 1
    assert metadata["hsl_replay_cache_valid_count"] == 0
    assert metadata["hsl_replay_cache_invalid_count"] == 1
    assert metadata["hsl_replay_cache_reason_counts"] == {"matrix_missing": 1}
    assert metadata["hsl_compatibility"] == "hsl_replay_cache_invalid"
    assert report["ok"] is True
    issue = next(item for item in report["issues"] if item["code"] == "hsl_replay_cache_invalid")
    assert issue["severity"] == "warning"
    assert issue["family"] == "risk"
    assert "matrix_missing" in issue["message"]


def test_cache_integrity_report_summarizes_candle_coverage_windows_and_gaps(tmp_path):
    root = tmp_path / "caches"
    month_dir = root / "ohlcv" / "data" / "binance" / "1m" / "BTC_USDT" / "2026"
    month_dir.mkdir(parents=True)
    expected_rows = 44_640
    valid = np.zeros(expected_rows, dtype=bool)
    valid[[1, 2, 5, 6]] = True
    np.save(month_dir / "01.npy", np.full((expected_rows, 4), 1.0, dtype=np.float32))
    np.save(month_dir / "01.valid.npy", valid)

    report = build_cache_integrity_report([root])

    coverage = report["summary"]["by_family"]["candles"]["coverage"]
    assert coverage["warm_cache_evidence"] == "coverage_with_gaps"
    assert coverage["artifact_count"] == 1
    assert coverage["covered_artifact_count"] == 1
    assert coverage["length_mismatch_count"] == 0
    assert coverage["row_count"] == expected_rows
    assert coverage["expected_row_count"] == expected_rows
    assert coverage["valid_row_count"] == 4
    assert coverage["gap_count"] == 2
    assert coverage["interior_gap_count"] == 1
    assert coverage["boundary_gap_count"] == 1
    assert coverage["leading_missing_artifact_count"] == 1
    assert coverage["leading_missing_rows"] == 1
    assert coverage["trailing_shortfall_rows"] == 0
    assert coverage["max_gap_rows"] == 2
    assert coverage["first_valid_date"] == "2026-01-01T00:01:00+00:00"
    assert coverage["last_valid_date"] == "2026-01-01T00:06:00+00:00"
    assert coverage["artifact_samples"][0]["symbol"] == "BTC_USDT"
    assert coverage["artifact_samples"][0]["gap_count"] == 2
    assert coverage["artifact_samples"][0]["interior_gap_count"] == 1
    assert coverage["artifact_samples"][0]["leading_missing_rows"] == 1
    assert coverage["gap_samples"] == [
        {
            "path": str(month_dir / "01.valid.npy"),
            "exchange": "binance",
            "timeframe": "1m",
            "symbol": "BTC_USDT",
            "start_ms": 1767225780000,
            "end_ms": 1767225840000,
            "start_date": "2026-01-01T00:03:00+00:00",
            "end_date": "2026-01-01T00:04:00+00:00",
            "rows": 2,
            "duration_ms": 120000,
        },
        {
            "path": str(month_dir / "01.valid.npy"),
            "exchange": "binance",
            "timeframe": "1m",
            "symbol": "BTC_USDT",
            "start_ms": 1767225600000,
            "end_ms": 1767225600000,
            "start_date": "2026-01-01T00:00:00+00:00",
            "end_date": "2026-01-01T00:00:00+00:00",
            "rows": 1,
            "duration_ms": 60000,
            "boundary": "leading_missing",
        },
    ]


def test_cache_integrity_report_summarizes_candle_no_trade_gap_metadata(tmp_path):
    root = tmp_path / "caches"
    month_dir = root / "ohlcv" / "data" / "kucoin" / "1m" / "ETH_USDT" / "2026"
    month_dir.mkdir(parents=True)
    expected_rows = 44_640
    valid = np.ones(expected_rows, dtype=bool)
    valid[3:5] = False
    np.save(month_dir / "01.valid.npy", valid)
    (month_dir.parent / "index.json").write_text(
        json.dumps(
            {
                "meta": {
                    "known_gaps": [
                        {
                            "start_ts": 1767225780000,
                            "end_ts": 1767225840000,
                            "reason": "no_trades",
                            "retry_count": 0,
                        },
                        [1767226200000, 1767226260000],
                    ],
                    "last_refresh_ms": 1767312000000,
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_cache_integrity_report([root])

    candles = report["summary"]["by_family"]["candles"]
    metadata = candles["metadata"]
    assert metadata["compatibility"] == "known_gaps_unclassified_or_non_no_trade"
    assert metadata["known_gap_count"] == 2
    assert metadata["known_gap_reason_counts"] == {
        "legacy_unclassified": 1,
        "no_trades": 1,
    }
    assert metadata["no_trade_known_gap_count"] == 1
    assert metadata["unclassified_known_gap_count"] == 1
    coverage = candles["coverage"]
    assert coverage["warm_cache_evidence"] == "coverage_with_gaps"
    readiness = report["summary"]["warm_cache_readiness"]["families"]["candles"]
    assert readiness["no_trade_gap_evidence"] == "partial_no_trade_known_gap_evidence"
    assert readiness["known_gap_count"] == 2
    assert readiness["known_gap_reason_counts"] == {
        "legacy_unclassified": 1,
        "no_trades": 1,
    }
    assert "candle_partial_no_trade_known_gap_evidence" in readiness["reasons"]
    assert "candle_unclassified_known_gaps_present" in readiness["reasons"]
    assert "candle_synthetic_no_trade_evidence_unproven" in readiness["reasons"]


def test_cache_integrity_report_marks_candle_gaps_without_no_trade_evidence(tmp_path):
    root = tmp_path / "caches"
    month_dir = root / "ohlcv" / "data" / "binance" / "1m" / "BTC_USDT" / "2026"
    month_dir.mkdir(parents=True)
    expected_rows = 44_640
    valid = np.ones(expected_rows, dtype=bool)
    valid[10:12] = False
    np.save(month_dir / "01.valid.npy", valid)

    report = build_cache_integrity_report([root])

    readiness = report["summary"]["warm_cache_readiness"]["families"]["candles"]
    assert readiness["readiness"] == "attention"
    assert readiness["no_trade_gap_evidence"] == "no_local_no_trade_gap_evidence"
    assert "candle_synthetic_no_trade_evidence_unproven" in readiness["reasons"]


def test_cache_integrity_report_marks_leading_only_candle_gaps(tmp_path):
    root = tmp_path / "caches"
    month_dir = root / "ohlcv" / "data" / "binance" / "1m" / "BTC_USDT" / "2026"
    month_dir.mkdir(parents=True)
    expected_rows = 44_640
    valid = np.ones(expected_rows, dtype=bool)
    valid[:10] = False
    np.save(month_dir / "01.valid.npy", valid)

    report = build_cache_integrity_report([root])

    coverage = report["summary"]["by_family"]["candles"]["coverage"]
    assert coverage["warm_cache_evidence"] == "coverage_with_gaps"
    assert coverage["gap_count"] == 1
    assert coverage["interior_gap_count"] == 0
    assert coverage["boundary_gap_count"] == 1
    assert coverage["leading_missing_artifact_count"] == 1
    assert coverage["leading_missing_rows"] == 10
    assert coverage["max_gap_rows"] == 10
    assert coverage["gap_samples"] == [
        {
            "path": str(month_dir / "01.valid.npy"),
            "exchange": "binance",
            "timeframe": "1m",
            "symbol": "BTC_USDT",
            "start_ms": 1767225600000,
            "end_ms": 1767226140000,
            "start_date": "2026-01-01T00:00:00+00:00",
            "end_date": "2026-01-01T00:09:00+00:00",
            "rows": 10,
            "duration_ms": 600000,
            "boundary": "leading_missing",
        }
    ]
    readiness = report["summary"]["warm_cache_readiness"]["families"]["candles"]
    assert readiness["readiness"] == "attention"
    assert readiness["evidence"] == "coverage_with_gaps"
    assert readiness["boundary_gap_count"] == 1
    assert readiness["leading_missing_rows"] == 10
    assert "candle_boundary_gaps_present" in readiness["reasons"]
    assert "candle_leading_missing_rows_present" in readiness["reasons"]
    assert "candles" in report["summary"]["warm_cache_readiness"]["attention_families"]


def test_cache_integrity_report_marks_empty_candle_coverage_masks(tmp_path):
    root = tmp_path / "caches"
    month_dir = root / "ohlcv" / "data" / "okx" / "1h" / "ETH_USDT" / "2026"
    month_dir.mkdir(parents=True)
    expected_rows = 672
    np.save(month_dir / "02.valid.npy", np.zeros(expected_rows, dtype=bool))

    report = build_cache_integrity_report([root])

    coverage = report["summary"]["by_family"]["candles"]["coverage"]
    assert coverage["warm_cache_evidence"] == "no_valid_rows"
    assert coverage["artifact_count"] == 1
    assert coverage["covered_artifact_count"] == 0
    assert coverage["length_mismatch_count"] == 0
    assert coverage["row_count"] == expected_rows
    assert coverage["expected_row_count"] == expected_rows
    assert coverage["first_valid_date"] is None
    assert coverage["last_valid_date"] is None


def test_cache_integrity_report_flags_truncated_candle_coverage_masks(tmp_path):
    root = tmp_path / "caches"
    month_dir = root / "ohlcv" / "data" / "binance" / "1m" / "BTC_USDT" / "2026"
    month_dir.mkdir(parents=True)
    np.save(month_dir / "01.valid.npy", np.ones(8, dtype=bool))

    report = build_cache_integrity_report([root])

    coverage = report["summary"]["by_family"]["candles"]["coverage"]
    assert report["ok"] is True
    assert coverage["warm_cache_evidence"] == "coverage_length_mismatch"
    assert coverage["length_mismatch_count"] == 1
    assert coverage["row_count"] == 8
    assert coverage["expected_row_count"] == 44_640
    assert coverage["gap_count"] == 1
    assert coverage["interior_gap_count"] == 0
    assert coverage["boundary_gap_count"] == 1
    assert coverage["trailing_shortfall_gap_count"] == 1
    assert coverage["leading_missing_rows"] == 0
    assert coverage["trailing_shortfall_rows"] == 44_632
    assert coverage["gap_samples"][0]["boundary"] == "trailing_shortfall"
    assert coverage["gap_samples"][0]["rows"] == 44_632
    assert coverage["artifact_samples"][0]["length_mismatch"] is True
    issue = report["issues"][0]
    assert issue["severity"] == "warning"
    assert issue["code"] == "coverage_length_mismatch"
    assert issue["family"] == "candles"
    readiness = report["summary"]["warm_cache_readiness"]["families"]["candles"]
    assert readiness["boundary_gap_count"] == 1
    assert readiness["trailing_shortfall_rows"] == 44_632
    assert "candle_boundary_gaps_present" in readiness["reasons"]


def test_cache_integrity_report_summarizes_fill_cache_metadata_contract_and_coverage(tmp_path):
    root = tmp_path / "caches"
    fill_dir = root / "fill_events" / "binance" / "user_01"
    fill_dir.mkdir(parents=True)
    current_contract = "gross_pnl_quote_fee_best_effort_v2"
    (fill_dir / "metadata.json").write_text(
        json.dumps(
            {
                "pnl_contract": current_contract,
                "history_scope": "all",
                "covered_start_ms": 1767225600000,
                "oldest_event_ts": 1767312000000,
                "newest_event_ts": 1767398400000,
                "last_refresh_ms": 1767484800000,
                "known_gaps": [{"start_ts": 1767355200000, "end_ts": 1767358800000}],
            }
        ),
        encoding="utf-8",
    )
    (fill_dir / "2026-01-02.ndjson").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "id": "a",
                    "timestamp": 1767312000000,
                    "pnl_contract": current_contract,
                },
                {"id": "b", "timestamp": 1767315600000},
                {
                    "id": "c",
                    "timestamp": 1767319200000,
                    "pnl_contract": "legacy_contract",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_cache_integrity_report([root])

    metadata = report["summary"]["by_family"]["fills"]["metadata"]
    assert metadata["compatibility"] == "legacy_or_missing_pnl_contract"
    assert metadata["artifact_count"] == 2
    assert metadata["metadata_file_count"] == 1
    assert metadata["record_count"] == 3
    assert metadata["current_pnl_contract_count"] == 2
    assert metadata["legacy_pnl_contract_count"] == 1
    assert metadata["missing_pnl_contract_count"] == 1
    assert metadata["history_scope_counts"] == {"all": 1}
    assert metadata["known_gap_count"] == 1
    assert metadata["covered_start_date"] == "2026-01-01T00:00:00+00:00"
    assert metadata["first_event_date"] == "2026-01-02T00:00:00+00:00"
    assert metadata["last_event_date"] == "2026-01-03T00:00:00+00:00"
    assert metadata["newest_event_date"] == "2026-01-03T00:00:00+00:00"
    assert metadata["last_refresh_date"] == "2026-01-04T00:00:00+00:00"


def test_cache_integrity_report_marks_current_fill_contract_with_unproven_coverage(tmp_path):
    root = tmp_path / "caches"
    fill_dir = root / "fill_events" / "binance" / "user_01"
    fill_dir.mkdir(parents=True)
    current_contract = "gross_pnl_quote_fee_best_effort_v2"
    (fill_dir / "metadata.json").write_text(
        json.dumps(
            {
                "pnl_contract": current_contract,
                "history_scope": "partial",
                "newest_event_ts": 1767398400000,
                "last_refresh_ms": 1767484800000,
                "known_gaps": [],
            }
        ),
        encoding="utf-8",
    )

    report = build_cache_integrity_report([root])

    metadata = report["summary"]["by_family"]["fills"]["metadata"]
    assert metadata["compatibility"] == "current_pnl_contract_unproven_coverage"
    readiness = report["summary"]["warm_cache_readiness"]["families"]["fills"]
    assert readiness["readiness"] == "attention"
    assert readiness["history_scope_all_count"] == 0
    assert "fill_history_scope_all_missing" in readiness["reasons"]
    assert "fill_covered_start_missing" in readiness["reasons"]
    assert "fill_records_missing" in readiness["reasons"]


def test_cache_integrity_report_summarizes_hsl_state_metadata(tmp_path):
    root = tmp_path / "caches"
    hsl_dir = root / "equity_hard_stop" / "binance"
    hsl_dir.mkdir(parents=True)
    (hsl_dir / "user_01.json").write_text(
        json.dumps(
            {
                "pside": "long",
                "symbol": "BTCUSDT",
                "tier": "red",
                "last_red_ts": 1767312000000,
                "cooldown_until_ms": 1767315600000,
            }
        ),
        encoding="utf-8",
    )

    report = build_cache_integrity_report([root])

    metadata = report["summary"]["by_family"]["risk"]["metadata"]
    assert metadata["compatibility"] == "local_state_with_timestamps"
    assert metadata["hsl_compatibility"] == "hsl_state_with_timestamps"
    assert metadata["hsl_artifact_count"] == 1
    assert metadata["artifact_count"] == 1
    assert metadata["timestamp_field_count"] == 2
    assert metadata["first_event_date"] == "2026-01-02T00:00:00+00:00"
    assert metadata["last_event_date"] == "2026-01-02T01:00:00+00:00"
    sample = metadata["artifact_samples"][0]
    assert sample["hsl_related"] is True
    assert sample["top_level_keys"] == [
        "cooldown_until_ms",
        "last_red_ts",
        "pside",
        "symbol",
        "tier",
    ]


def test_cache_integrity_report_derives_warm_cache_readiness(tmp_path):
    root = tmp_path / "caches"
    month_dir = root / "ohlcv" / "data" / "binance" / "1h" / "BTC_USDT" / "2026"
    month_dir.mkdir(parents=True)
    np.save(month_dir / "01.valid.npy", np.ones(744, dtype=bool))
    fill_dir = root / "fill_events" / "binance" / "user_01"
    fill_dir.mkdir(parents=True)
    current_contract = "gross_pnl_quote_fee_best_effort_v2"
    (fill_dir / "metadata.json").write_text(
        json.dumps(
            {
                "pnl_contract": current_contract,
                "history_scope": "all",
                "covered_start_ms": 1767225600000,
                "oldest_event_ts": 1767312000000,
                "newest_event_ts": 1767312000000,
                "last_refresh_ms": 1767315600000,
                "known_gaps": [],
            }
        ),
        encoding="utf-8",
    )
    (fill_dir / "2026-01-02.ndjson").write_text(
        json.dumps(
            {
                "id": "a",
                "timestamp": 1767312000000,
                "pnl_contract": current_contract,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_cache_integrity_report([root])

    readiness = report["summary"]["warm_cache_readiness"]
    assert readiness["mode"] == "report_only_non_enforcing"
    assert readiness["readiness"] == "core_evidence_observed"
    assert readiness["missing_families"] == ["risk"]
    assert readiness["attention_families"] == []
    assert readiness["suspicious_gap_count"] == 0
    assert readiness["families"]["candles"]["readiness"] == "observed"
    assert readiness["families"]["candles"]["interior_gap_count"] == 0
    assert readiness["families"]["candles"]["boundary_gap_count"] == 0
    assert readiness["families"]["candles"]["leading_missing_rows"] == 0
    assert readiness["families"]["candles"]["trailing_shortfall_rows"] == 0
    assert readiness["families"]["fills"]["readiness"] == "observed"
    assert readiness["families"]["fills"]["covered_start_date"] == "2026-01-01T00:00:00+00:00"
    assert readiness["families"]["risk"]["readiness"] == "missing_optional"
    assert report["roots"][0]["warm_cache_readiness"] == readiness


def test_cache_integrity_report_marks_missing_root_as_warning(tmp_path):
    missing = tmp_path / "missing"

    report = build_cache_integrity_report([missing])

    assert report["ok"] is True
    assert report["summary"]["by_severity"] == {"warning": 1}
    assert report["summary"]["warm_cache_readiness"]["readiness"] == "no_cache_evidence"
    assert report["summary"]["warm_cache_readiness"]["missing_families"] == [
        "candles",
        "fills",
        "risk",
    ]
    assert report["issues"][0]["family"] == "root"
    assert report["issues"][0]["code"] == "root_missing"


def test_cache_integrity_doctor_cli_emits_json(tmp_path, capsys):
    root = tmp_path / "caches"
    root.mkdir()
    (root / "cache_meta.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    rc = main([str(root), "--compact"])

    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["summary"]["file_count"] == 1


def test_cache_integrity_doctor_tool_dispatch_forwards_module_and_prog(monkeypatch):
    captured = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ["PASSIVBOT_CLI_PROG"]
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert cli_main.main(["tool", "cache-integrity-doctor", "caches", "--compact"]) == 0

    assert captured["module_name"] == "tools.cache_integrity_doctor"
    assert captured["argv"] == [
        "passivbot tool cache-integrity-doctor",
        "caches",
        "--compact",
    ]
    assert captured["prog_env"] == "passivbot tool cache-integrity-doctor"
