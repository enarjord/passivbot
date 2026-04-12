import json

from ccxt_contracts import diff_snapshots, dump_snapshot, sanitize_for_json


def test_sanitize_for_json_handles_non_json_types(tmp_path):
    dumped = sanitize_for_json(
        {
            "path": tmp_path / "a.json",
            "payload": {1, 2},
            "raw": b"abc",
            "nan": float("nan"),
        }
    )

    assert dumped["path"].endswith("a.json")
    assert sorted(dumped["payload"]) == [1, 2]
    assert dumped["raw"] == "abc"
    assert dumped["nan"] == "nan"


def test_diff_snapshots_reports_added_removed_and_changed_fields():
    old = {
        "meta": {"captured_at": "old", "exchange": "binance"},
        "markets": {"summary": {"contracts": {"BTC/USDT:USDT": {"min_qty": 0.001}}}},
    }
    new = {
        "meta": {"captured_at": "new", "exchange": "binance"},
        "markets": {
            "summary": {
                "contracts": {
                    "BTC/USDT:USDT": {"min_qty": 0.01},
                    "ETH/USDT:USDT": {"min_qty": 0.1},
                }
            }
        },
    }

    diff = diff_snapshots(old, new)

    assert diff["summary"] == {"added": 1, "removed": 0, "changed": 1}
    assert diff["added"][0]["path"].endswith("ETH/USDT:USDT.min_qty")
    assert diff["changed"][0]["path"].endswith("BTC/USDT:USDT.min_qty")
    assert diff["changed"][0]["old"] == 0.001
    assert diff["changed"][0]["new"] == 0.01


def test_dump_snapshot_writes_json_file(tmp_path):
    path = dump_snapshot({"hello": "world"}, tmp_path / "snapshot.json")

    assert json.loads(path.read_text()) == {"hello": "world"}
