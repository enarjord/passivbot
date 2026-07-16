from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from live.log_secret_inventory import (
    SECRET_CLASSES,
    build_log_secret_inventory,
    classify_secret_like_text,
    summarize_log_secret_inventory,
)
from passivbot_cli import main as cli_main
from tools import log_secret_inventory


def test_classifies_all_required_secret_classes_and_ignores_benign_lookalikes():
    text = "\n".join(
        [
            "wss://stream.example/private?token=private-ws-token",
            "Authorization: ApiKey header-secret",
            "Cookie: session=cookie-secret",
            "X-MBX-APIKEY: exchange-header-secret",
            "Authorization: [redacted]",
            "Authorization: ApiKey [redacted]",
            "Cookie: session=[redacted]; other=[redacted]",
            "api_key=key-secret",
            '{"apiKey":"json-key-secret","token":"json-token-secret"}',
            "api_key_count=2",
            "token_count: 7",
            "password: [redacted]",
            "Bearer bearer-secret",
            "Bearer [redacted]",
            "Basic dXNlcjpwYXNz",
            "-----BEGIN PRIVATE KEY-----",
            "https://example.invalid/path?signature=query-secret",
            "https://example.invalid/path?token=[redacted]",
            "response body: <html><body>private response</body></html>",
            "response body:",
        ]
    )

    counts = classify_secret_like_text(text)

    assert set(counts) == set(SECRET_CLASSES)
    assert all(counts[name] > 0 for name in SECRET_CLASSES)
    assert counts["labeled_secret_values"] == 3
    assert counts["authorization_cookie_headers"] == 3
    assert classify_secret_like_text("response body:")["raw_http_body_html_markers"] == 0


def test_classifies_scheme_less_secret_query_params_without_retaining_values(tmp_path: Path):
    secrets = ("path-api-key-secret", "signature-secret", "token-secret", "full-url-secret")
    text = "\n".join(
        [
            f"GET /private/orders?apiKey={secrets[0]}",
            f"GET /private/orders?limit=10&signature={secrets[1]}&token={secrets[2]}",
            "GET /private/orders?token=[redacted]&signature=[redacted]",
            'response={"url":"/private/orders?token=[redacted]"}',
            "query=/private/orders?signature=[redacted], status=failed",
            "GET /private/orders?api_key_count=2&token_present=true&signature_enabled=true",
            f"https://example.invalid/path?password={secrets[3]}",
        ]
    )

    counts = classify_secret_like_text(text)

    assert counts["secret_url_query_params"] == 4
    assert counts["labeled_secret_values"] == 0

    (tmp_path / "service.log").write_text(text)
    serialized = json.dumps(build_log_secret_inventory(tmp_path), sort_keys=True)

    assert all(secret not in serialized for secret in secrets)


def test_report_is_value_free_and_root_relative(tmp_path: Path):
    secrets = (
        "private-ws-token",
        "header-secret",
        "cookie-secret",
        "key-secret",
        "bearer-secret",
        "query-secret",
        "private-response-marker",
    )
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "service.log").write_text(
        "\n".join(
            [
                f"wss://stream.example/private?token={secrets[0]}",
                f"Authorization: ApiKey {secrets[1]}",
                f"Cookie: session={secrets[2]}",
                f"api_key={secrets[3]}",
                f"Bearer {secrets[4]}",
                "-----BEGIN PRIVATE KEY-----",
                f"https://example.invalid/path?signature={secrets[5]}",
                f"response body: <html>{secrets[6]}</html>",
            ]
        )
    )

    report = build_log_secret_inventory(tmp_path, now_s=2_000_000_000)
    serialized = json.dumps(report, sort_keys=True)

    assert all(secret not in serialized for secret in secrets)
    assert "Authorization:" not in serialized
    assert report["root"] == "."
    assert report["files"][0]["path"] == "nested/service.log"
    assert report["files"][0]["class_counts"]["authorization_cookie_headers"] == 2
    assert report["read_only"] is True
    assert all(report["class_counts"][name] > 0 for name in SECRET_CLASSES)


def test_order_and_prefix_hashes_are_deterministic(tmp_path: Path):
    (tmp_path / "z.log").write_bytes(b"second")
    (tmp_path / "a.log").write_bytes(b"first")

    first = build_log_secret_inventory(tmp_path, now_s=2_000_000_000)
    second = build_log_secret_inventory(tmp_path, now_s=2_000_000_000)

    assert first == second
    assert [item["path"] for item in first["files"]] == ["a.log", "z.log"]
    assert len(first["files"][0]["sha256"]) == 64
    assert first["files"][0]["sha256_scope"] == "full_content"


def test_file_and_byte_bounds_are_explicit(tmp_path: Path):
    (tmp_path / "a.log").write_bytes(b"Authorization: Bearer too-long")
    (tmp_path / "b.log").write_text("token=another-secret")

    report = build_log_secret_inventory(tmp_path, max_files=1, max_bytes_per_file=8, now_s=0)

    assert report["summary"]["files_discovered"] == 2
    assert report["summary"]["files_scanned"] == 1
    assert report["summary"]["files_skipped_max_files"] == 1
    assert report["files"][0]["bytes_scanned"] == 8
    assert report["files"][0]["truncated"] is True
    assert report["files"][0]["sha256_scope"] == "content_prefix"


def test_scans_gzip_rotated_logs_with_uncompressed_byte_cap(tmp_path: Path):
    path = tmp_path / "service.log.1.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(b"Bearer gzip-secret\n" + b"x" * 100)

    report = build_log_secret_inventory(tmp_path, max_bytes_per_file=32, now_s=0)

    summary = report["files"][0]
    assert summary["compressed"] is True
    assert summary["truncated"] is True
    assert summary["sha256_scope"] == "decompressed_content_prefix"
    assert summary["class_counts"]["bearer_basic_schemes"] == 1

    full_summary = build_log_secret_inventory(
        tmp_path, max_bytes_per_file=1_000, now_s=0
    )["files"][0]
    assert full_summary["truncated"] is False
    assert full_summary["sha256_scope"] == "decompressed_content"


def test_unreadable_gzip_is_bounded_to_error_type_without_exception_text(tmp_path: Path):
    path = tmp_path / "broken.log.gz"
    path.write_bytes(b"not a gzip stream")

    report = build_log_secret_inventory(tmp_path, now_s=0)
    summary = report["files"][0]

    assert summary["status"] == "unreadable"
    assert summary["error_type"] == "BadGzipFile"
    assert "not a gzip" not in json.dumps(report)


def test_symlinked_files_are_not_followed_outside_root(tmp_path: Path):
    outside = tmp_path.parent / "outside-secret.log"
    outside.write_text("token=outside-secret")
    (tmp_path / "linked.log").symlink_to(outside)

    report = build_log_secret_inventory(tmp_path, now_s=0)

    assert report["summary"]["files_discovered"] == 0
    assert report["summary"]["symlinks_skipped"] == 1
    assert "outside-secret" not in json.dumps(report)


def test_invalid_bounds_and_root_raise_or_return_nonzero(tmp_path: Path, capsys):
    with pytest.raises(ValueError, match="max_files"):
        build_log_secret_inventory(tmp_path, max_files=-1)
    with pytest.raises(ValueError, match="existing directory"):
        build_log_secret_inventory(tmp_path / "missing")

    assert log_secret_inventory.main([str(tmp_path / "missing"), "--compact"]) == 2
    assert "existing directory" in capsys.readouterr().err


def test_tool_compact_json_output(tmp_path: Path, capsys):
    (tmp_path / "current.log").write_text("token=tool-secret")

    assert log_secret_inventory.main([str(tmp_path), "--compact"]) == 0
    output = capsys.readouterr().out

    assert "\n" not in output.rstrip("\n")
    assert "tool-secret" not in output
    assert json.loads(output)["class_counts"]["labeled_secret_values"] == 1


def test_summary_projection_aggregates_scan_evidence_without_file_details(tmp_path: Path):
    (tmp_path / "a.log").write_text("token=summary-secret")
    (tmp_path / "b.log").write_bytes(b"x" * 20)

    report = build_log_secret_inventory(
        tmp_path, max_bytes_per_file=8, now_s=0
    )
    summary = summarize_log_secret_inventory(report)
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["projection"] == "summary"
    assert summary["summary"]["bytes_scanned"] == 16
    assert summary["summary"]["files_positive"] == 1
    assert summary["summary"]["files_truncated"] == 2
    assert summary["class_counts"]["labeled_secret_values"] == 1
    assert "files" not in summary
    assert "a.log" not in serialized
    assert "sha256" not in serialized
    assert "summary-secret" not in serialized


def test_tool_summary_compact_json_output_omits_file_details(tmp_path: Path, capsys):
    (tmp_path / "account-name.log").write_text("token=tool-summary-secret")

    assert (
        log_secret_inventory.main(
            [str(tmp_path), "--summary", "--compact"]
        )
        == 0
    )
    output = capsys.readouterr().out
    report = json.loads(output)

    assert "\n" not in output.rstrip("\n")
    assert report["projection"] == "summary"
    assert report["summary"]["files_positive"] == 1
    assert report["class_counts"]["labeled_secret_values"] == 1
    assert "files" not in report
    assert "account-name.log" not in output
    assert "tool-summary-secret" not in output


def test_unified_cli_dispatches_log_secret_inventory(tmp_path: Path, capsys):
    (tmp_path / "current.log").write_text("token=cli-secret")

    assert (
        cli_main.main(
            ["tool", "log-secret-inventory", str(tmp_path), "--compact"]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "cli-secret" not in output
    assert json.loads(output)["class_counts"]["labeled_secret_values"] == 1
