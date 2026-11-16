import json
import os
import sqlite3
from dataclasses import asdict, replace
from decimal import Decimal
from multiprocessing import get_context

import pytest

from live import bitunix_create_journal as journal_module
from live.bitunix_create_journal import (
    BitunixCreateAlreadyPending,
    BitunixCreateAttempt,
    BitunixCreateJournal,
    BitunixCreateJournalError,
    BitunixCreateObservation,
)


def _wire_close():
    return {
        "clientId": "0x01-test/long:close",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "reduceOnly": True,
        "tradeSide": "CLOSE",
        "positionId": "12345",
        "orderType": "LIMIT",
        "qty": "0.0040",
        "price": "10001.10",
        "effect": "POST_ONLY",
    }


@pytest.fixture
def attempt():
    return BitunixCreateAttempt.from_wire(_wire_close(), submitted_at_ms=1763312000000)


def test_preserves_exact_attempt_across_instances_and_resolves_exact_id(
    tmp_path, attempt
):
    path = tmp_path / "account" / "create_attempts.sqlite3"
    before = BitunixCreateJournal(path)
    assert before.load() == ()
    assert not path.exists()

    before.record(attempt)
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600

    restarted = BitunixCreateJournal(path)
    assert restarted.load() == (attempt,)
    assert attempt.position_side == "long"
    assert attempt.side == "sell"
    assert (attempt.wire_side, attempt.trade_side) == ("BUY", "CLOSE")
    assert (attempt.qty, attempt.price) == ("0.0040", "10001.10")
    assert attempt.position_id == "12345"
    assert attempt.effect == "POST_ONLY"

    original_mtime = path.stat().st_mtime_ns
    restarted.resolve("unrelated-id")
    assert path.stat().st_mtime_ns == original_mtime
    assert restarted.load() == (attempt,)

    restarted.resolve(attempt.client_id)
    assert path.exists()
    assert before.load() == ()
    next_attempt = replace(attempt, client_id="new-attempt")
    before.record(next_attempt)
    restarted.resolve(attempt.client_id)
    assert restarted.load() == (next_attempt,)


def test_older_venue_observations_cannot_lower_durable_executed_quantity(
    tmp_path, attempt
):
    journal = BitunixCreateJournal(tmp_path / "create_attempts.sqlite3")
    journal.record(attempt)
    assert journal.load_pending() == (attempt, None)
    assert not journal.resolve(
        attempt.client_id, order_id="venue-1", filled_qty=Decimal(0)
    )
    assert journal.observe(
        attempt.client_id, "venue-1", Decimal("0.000000001")
    ) == BitunixCreateObservation("venue-1", Decimal("0.000000001"))
    advanced = BitunixCreateObservation("venue-1", Decimal("0.000000002"))
    assert (
        journal.observe(attempt.client_id, "venue-1", advanced.filled_qty) == advanced
    )

    restarted = BitunixCreateJournal(journal.path)
    original_mtime = journal.path.stat().st_mtime_ns
    assert restarted.observe(attempt.client_id, "venue-1", Decimal(0)) == advanced
    assert (
        restarted.observe(attempt.client_id, "venue-1", Decimal("0.0000000020"))
        == advanced
    )
    assert restarted.observe("unrelated-id", "venue-2", Decimal(0)) is None
    assert journal.path.stat().st_mtime_ns == original_mtime
    assert restarted.load_pending() == (attempt, advanced)

    assert not restarted.resolve(attempt.client_id)
    assert not restarted.resolve(
        attempt.client_id, order_id="venue-1", filled_qty=Decimal("0.000000001")
    )
    assert not restarted.resolve(
        attempt.client_id, order_id="venue-2", filled_qty=advanced.filled_qty
    )
    assert restarted.load_pending() == (attempt, advanced)
    assert restarted.resolve(
        attempt.client_id, order_id=advanced.order_id, filled_qty=advanced.filled_qty
    )
    assert restarted.load_pending() is None


def test_conflicting_venue_identity_blocks_resolution_after_reopen(tmp_path, attempt):
    journal = BitunixCreateJournal(tmp_path / "create_attempts.sqlite3")
    journal.record(attempt)
    first_fill = Decimal("0.001")
    journal.observe(attempt.client_id, "venue-1", first_fill)
    conflict = BitunixCreateObservation("venue-1", first_fill, "venue-2")
    assert journal.observe(attempt.client_id, "venue-2", Decimal(0)) == conflict

    restarted = BitunixCreateJournal(journal.path)
    assert restarted.load_pending() == (attempt, conflict)
    original_mtime = journal.path.stat().st_mtime_ns
    assert restarted.observe(attempt.client_id, "venue-3", Decimal(0)) == conflict
    assert journal.path.stat().st_mtime_ns == original_mtime
    assert not restarted.resolve(attempt.client_id)
    assert not restarted.resolve(
        attempt.client_id, order_id="venue-1", filled_qty=first_fill
    )
    assert not restarted.resolve(
        attempt.client_id, order_id="venue-2", filled_qty=Decimal(0)
    )
    assert restarted.load_pending() == (attempt, conflict)


@pytest.mark.parametrize(
    "change",
    [
        {"clientId": ""},
        {"clientId": "bad space"},
        {"clientId": "x" * 37},
        {"symbol": "BTC USDT"},
        {"side": "SHORT"},
        {"reduceOnly": "true"},
        {"tradeSide": "OPEN"},
        {"positionId": None},
        {"orderType": "TRAILING"},
        {"qty": "0"},
        {"qty": -1},
        {"price": "NaN"},
        {"price": "1e3"},
        {"effect": "UNKNOWN"},
    ],
)
def test_rejects_invalid_wire_facts(change):
    with pytest.raises(ValueError):
        BitunixCreateAttempt.from_wire(
            {**_wire_close(), **change}, submitted_at_ms=1763312000000
        )


def test_accepts_market_entry_without_price_or_position():
    wire = {
        "clientId": "market-attempt",
        "symbol": "BTCUSDT",
        "side": "SELL",
        "reduceOnly": False,
        "tradeSide": "OPEN",
        "orderType": "MARKET",
        "qty": "0.12",
    }
    attempt = BitunixCreateAttempt.from_wire(wire, submitted_at_ms=1763312000000)
    assert attempt.position_side == "short"
    assert attempt.side == "sell"
    assert attempt.position_id is None
    assert attempt.price is None
    assert attempt.effect is None


def _record_and_exit(path, candidate, barrier):
    barrier.wait(timeout=5)
    try:
        BitunixCreateJournal(path).record(candidate)
    except BitunixCreateAlreadyPending:
        os._exit(17)
    os._exit(0)


@pytest.mark.parametrize("pending_file_exists", [False, True])
def test_concurrent_writers_cannot_replace_a_pending_attempt(
    tmp_path, attempt, pending_file_exists
):
    path = tmp_path / "nested" / "account" / "create_attempts.sqlite3"
    if pending_file_exists:
        journal = BitunixCreateJournal(path)
        journal.record(attempt)
        journal.resolve(attempt.client_id)

    ctx = get_context("spawn")
    barrier = ctx.Barrier(2)
    attempts = [replace(attempt, client_id=client_id) for client_id in ("id1", "id2")]
    processes = [
        ctx.Process(target=_record_and_exit, args=(path, candidate, barrier))
        for candidate in attempts
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
        assert sorted(process.exitcode for process in processes) == [0, 17]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()
    winner = next(
        candidate
        for process, candidate in zip(processes, attempts)
        if process.exitcode == 0
    )
    reopened = BitunixCreateJournal(path)
    assert reopened.load() == (winner,)
    with pytest.raises(BitunixCreateAlreadyPending):
        reopened.record(winner)


@pytest.mark.parametrize("contents", [b"", b"broken", b"SQLite format 3\x00"])
def test_corrupt_file_cannot_become_empty_or_get_replaced(tmp_path, attempt, contents):
    path = tmp_path / "create_attempts.sqlite3"
    path.write_bytes(contents)
    path.chmod(0o600)
    journal = BitunixCreateJournal(path)
    with pytest.raises(BitunixCreateJournalError):
        journal.load()
    with pytest.raises(BitunixCreateJournalError):
        journal.record(attempt)
    with pytest.raises(BitunixCreateJournalError):
        journal.resolve(attempt.client_id)
    assert path.read_bytes() == contents


@pytest.mark.parametrize(
    "tamper", ["version", "schema", "attempt", "repeat_field", "observation"]
)
def test_incompatible_or_malformed_durable_state_fails_closed(
    tmp_path, attempt, tamper
):
    path = tmp_path / "create_attempts.sqlite3"
    journal = BitunixCreateJournal(path)
    journal.record(attempt)
    conn = sqlite3.connect(path)
    try:
        if tamper == "version":
            conn.execute("PRAGMA user_version=9")
        elif tamper == "schema":
            conn.execute("CREATE TABLE extra (extra INTEGER)")
        elif tamper == "observation":
            conn.execute("UPDATE pending_create SET order_id = 'venue-1'")
        else:
            if tamper == "attempt":
                payload = json.dumps({**asdict(attempt), "qty": 0})
            else:
                payload = json.dumps(asdict(attempt))[:-1] + ', "client_id": "second"}'
            conn.execute("UPDATE pending_create SET attempt = ?", (payload,))
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(BitunixCreateJournalError):
        journal.load()
    with pytest.raises(BitunixCreateJournalError):
        journal.record(attempt)


def test_failed_first_publish_does_not_replace_existing_or_leave_empty_db(
    tmp_path, attempt, monkeypatch
):
    path = tmp_path / "account" / "create_attempts.sqlite3"
    journal = BitunixCreateJournal(path)

    def fail_link(*args):
        raise OSError("simulated file publication failure")

    monkeypatch.setattr(journal_module.os, "link", fail_link)
    with pytest.raises(BitunixCreateJournalError):
        journal.record(attempt)
    assert not list(path.parent.iterdir())
    assert journal.load() == ()


def test_failed_directory_sync_after_publish_keeps_conservative_pending_state(
    tmp_path, attempt, monkeypatch
):
    path = tmp_path / "account" / "create_attempts.sqlite3"
    journal = BitunixCreateJournal(path)

    def fail_sync(*args):
        raise OSError("simulated directory sync failure")

    monkeypatch.setattr(journal_module, "_sync_directories", fail_sync)
    with pytest.raises(BitunixCreateJournalError):
        journal.record(attempt)
    assert BitunixCreateJournal(path).load() == (attempt,)
