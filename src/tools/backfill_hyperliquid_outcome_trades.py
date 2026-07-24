#!/usr/bin/env python3
"""Archive ordered HIP-4 fills from downloaded node_fills_by_block files."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable, Iterator
import uuid

from outcome.adapters import hyperliquid
from outcome.archive import OutcomeTradeArchive
from outcome.historical import (
    archive_historical_batch,
    parse_hyperliquid_node_fills_by_block,
)


def _select_market_payload(payload: object, market_id: str | None) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("HIP-4 market metadata must be an object")
    if isinstance(payload.get("outcomes"), list):
        if market_id is None:
            raise ValueError("--market-id is required for a complete outcomeMeta payload")
        matches = [
            row
            for row in payload["outcomes"]
            if isinstance(row, dict) and str(row.get("outcome", "")) == market_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"outcomeMeta contains {len(matches)} entries for market {market_id}"
            )
        return matches[0]
    if market_id is not None and str(payload.get("outcome", "")) != market_id:
        raise ValueError("HIP-4 market metadata does not match --market-id")
    return payload


@contextmanager
def _decoded_lines(path: Path) -> Iterator[Iterable[str]]:
    if path.suffix.casefold() != ".lz4":
        with path.open(encoding="utf-8") as stream:
            yield stream
        return
    process = subprocess.Popen(
        ["lz4", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        yield process.stdout
    finally:
        process.stdout.close()
        stderr = process.stderr.read()
        return_code = process.wait()
        process.stderr.close()
        if return_code != 0:
            raise RuntimeError(
                f"lz4 failed for {path} with exit {return_code}: {stderr.strip()}"
            )


def _ordered_lines(paths: list[Path]) -> Iterator[str]:
    for path in paths:
        with _decoded_lines(path) as lines:
            yield from lines


def _source_cursor(paths: list[Path]) -> str:
    manifest = "\n".join(str(path.resolve()) for path in paths)
    return (
        "node_fills_by_block:file_manifest:"
        + hashlib.sha256(manifest.encode()).hexdigest()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market-json",
        required=True,
        type=Path,
        help="Retained raw HIP-4 market row or complete outcomeMeta JSON",
    )
    parser.add_argument(
        "--market-id",
        help="Required when --market-json contains a complete outcomeMeta response",
    )
    parser.add_argument(
        "--input-files",
        required=True,
        nargs="+",
        type=Path,
        help="Chronologically ordered plain NDJSON or .lz4 node_fills_by_block files",
    )
    parser.add_argument("--archive", type=Path, default=Path("caches/outcome_markets.sqlite"))
    parser.add_argument(
        "--collector-session",
        default=None,
        help="Stable import identity; defaults to a generated local session",
    )
    parser.add_argument("--received-time-ms", type=int, default=None)
    args = parser.parse_args()

    if not args.market_json.is_file():
        parser.error(f"--market-json does not exist: {args.market_json}")
    input_paths = list(args.input_files)
    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        parser.error(f"input file does not exist: {missing[0]}")
    if args.received_time_ms is not None and args.received_time_ms < 0:
        parser.error("--received-time-ms must be non-negative")

    metadata = json.loads(args.market_json.read_text())
    market_payload = _select_market_payload(metadata, args.market_id)
    market = hyperliquid.normalize_market(market_payload)
    cursor = _source_cursor(input_paths)
    collector_session = (
        args.collector_session
        or f"hip4-historical-{market.market_id}-{uuid.uuid4().hex}"
    )
    batch = parse_hyperliquid_node_fills_by_block(
        _ordered_lines(input_paths),
        market,
        source_cursor=cursor,
        received_time_ms=args.received_time_ms,
    )
    archive = OutcomeTradeArchive(args.archive)
    try:
        inserted, ignored = archive_historical_batch(
            archive,
            batch,
            collector_session=collector_session,
        )
    finally:
        archive.close()

    print(
        json.dumps(
            {
                "authenticated": False,
                "mutations_performed": False,
                "market": {
                    "market_id": market.market_id,
                    "title": market.title,
                    "scheduled_event_time_ms": (
                        market.lifecycle.scheduled_event_time_ms
                    ),
                },
                "input_files": [str(path) for path in input_paths],
                "source_cursor": cursor,
                "collector_session": collector_session,
                "trades_inserted": inserted,
                "trades_ignored_as_duplicates": ignored,
                "settlements": [
                    asdict(settlement) for settlement in batch.settlements
                ],
                "coverage_by_asset": {
                    asset_id: [asdict(interval) for interval in intervals]
                    for asset_id, intervals in batch.coverage_by_asset.items()
                },
                "archive": str(args.archive),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
