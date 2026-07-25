from __future__ import annotations

import json
from pathlib import Path

from outcome.adapters import hyperliquid
from outcome.archive import OutcomeTradeArchive
from outcome.models import OutcomeVenue
from tools.backfill_hyperliquid_outcome_trades import main


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def test_plain_node_files_are_ingested_in_explicit_cross_file_order(
    tmp_path,
    monkeypatch,
    capsys,
):
    market_payload = json.loads(
        (FIXTURES / "hyperliquid_price_binary.json").read_text()
    )
    market = hyperliquid.normalize_market(market_payload)
    metadata_path = tmp_path / "market.json"
    metadata_path.write_text(json.dumps(market_payload))
    first_path = tmp_path / "01.ndjson"
    second_path = tmp_path / "02.ndjson"
    first_path.write_text(
        json.dumps(
            {
                "block_number": 1,
                "block_time": "1970-01-01T00:00:01Z",
                "events": [],
            }
        )
        + "\n"
    )
    second_path.write_text(
        json.dumps(
            {
                "block_number": 2,
                "block_time": "1970-01-01T00:00:02Z",
                "events": [],
            }
        )
        + "\n"
    )
    archive_path = tmp_path / "outcomes.sqlite"
    monkeypatch.setattr(
        "sys.argv",
        [
            "backfill_hyperliquid_outcome_trades.py",
            "--market-json",
            str(metadata_path),
            "--input-files",
            str(first_path),
            str(second_path),
            "--archive",
            str(archive_path),
            "--collector-session",
            "historical-test",
            "--received-time-ms",
            "3000",
        ],
    )

    assert main() == 0

    report = json.loads(capsys.readouterr().out)
    assert report["trades_inserted"] == 0
    assert report["coverage_by_asset"][market.yes_asset.asset_id] == [
        {"start_ms": 1_000, "end_ms": 2_000}
    ]
    archive = OutcomeTradeArchive(archive_path)
    assert archive.load_market_metadata(OutcomeVenue.HYPERLIQUID, "913") == [
        market
    ]
