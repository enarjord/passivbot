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


def test_cache_integrity_report_marks_missing_root_as_warning(tmp_path):
    missing = tmp_path / "missing"

    report = build_cache_integrity_report([missing])

    assert report["ok"] is True
    assert report["summary"]["by_severity"] == {"warning": 1}
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
