import asyncio
from types import SimpleNamespace

import pytest
from ccxt.base.errors import NetworkError, OrderNotFound

from live import bitunix_create_recovery as recovery_module
from live.bitunix_create_journal import BitunixCreateJournal
from live.bitunix_create_recovery import BitunixCreateRecovery
from live.freshness import FreshnessLedger
from live.position_fill_sync import WriteDeferred


BODY = {
    "clientId": "original",
    "symbol": "BTCUSDT",
    "side": "BUY",
    "tradeSide": "OPEN",
    "qty": "1",
    "price": "99.25",
    "effect": "POST_ONLY",
    "orderType": "LIMIT",
    "reduceOnly": False,
}
ORDER_ID = "9007199254740999"


class Account:
    def __init__(self):
        self.ledger = FreshnessLedger()
        self.fetched_open_orders = []
        self._authoritative_pending_confirmations = {}
        self.events = []
        self._pnls_manager = SimpleNamespace(get_events=lambda **_: self.events)

    def _ensure_freshness_ledger(self):
        return self.ledger

    def capture(self, rows=()):
        self.ledger.begin_epoch()
        self.fetched_open_orders = [
            {"id": row["orderId"], "clientOrderId": row["clientId"], "info": row}
            for row in rows
        ]
        for kind in ("balance", "positions", "open_orders"):
            self.ledger.stamp(kind, now_ms=recovery_module._utc_ms())

    def fill_capture(self, recovery, start_ms, *quantities):
        fills = [
            {
                "source": "fetch_my_trades",
                "data": {
                    "order": ORDER_ID,
                    "id": str(index),
                    "clientOrderId": BODY["clientId"],
                    "info": {"symbol": BODY["symbol"], "qty": qty},
                },
            }
            for index, qty in enumerate(quantities)
        ]
        self.events = [SimpleNamespace(position_side="long", raw=fills)]
        end_ms = start_ms + 1
        self.ledger.stamp("fills", now_ms=end_ms)
        recovery.record_fill_refresh(self._pnls_manager, (start_ms, end_ms))


def order_row(attempt, status="NEW", filled="0", **overrides):
    return {
        **BODY,
        "orderId": ORDER_ID,
        "positionMode": "HEDGE",
        "ctime": attempt.submitted_at_ms,
        "status": status,
        "tradeQty": filled,
        **overrides,
    }


@pytest.fixture
def journal(tmp_path):
    return BitunixCreateJournal(tmp_path / "create_attempts.sqlite3")


async def unexpected_lookup(*_args, **_kwargs):
    pytest.fail("the committed account snapshot was sufficient")


@pytest.mark.asyncio
async def test_restart_adopts_working_order_after_a_new_cohort(journal):
    first = BitunixCreateRecovery(Account(), journal)
    attempt = first.record(BODY, None)
    account = Account()
    recovery = BitunixCreateRecovery(account, BitunixCreateJournal(journal.path))
    account.capture([order_row(attempt)])

    recovery.poll(unexpected_lookup)
    assert recovery.pending
    with pytest.raises(WriteDeferred):
        recovery.record({**BODY, "clientId": "new"}, None)

    account.capture([order_row(attempt)])
    account._authoritative_pending_confirmations["open_orders"] = account.ledger.epoch
    recovery.poll(unexpected_lookup)
    assert not recovery.pending
    assert journal.load() == ()
    assert recovery.task is None


@pytest.mark.asyncio
async def test_exact_detail_links_a_pending_order_without_client_id(journal):
    account = Account()
    recovery = BitunixCreateRecovery(account, journal)
    attempt = recovery.record(BODY, None)
    account.capture([order_row(attempt, clientId="")])

    async def lookup(*_args, **_kwargs):
        return order_row(attempt)

    recovery.poll(lookup)
    await recovery.task
    account.capture([order_row(attempt, clientId="another-client-id")])
    recovery.poll(lookup)
    assert (
        recovery.pending
    )  # a conflicting explicit ID cannot borrow the lookup's proof
    account.capture([order_row(attempt, clientId="")])
    recovery.poll(lookup)
    assert journal.load() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("contradictory_id", [False, True])
async def test_pending_view_cannot_override_exact_working_detail(
    journal, contradictory_id
):
    account = Account()
    recovery = BitunixCreateRecovery(account, journal)
    attempt = recovery.record(BODY, None)
    account.capture([order_row(attempt, clientId="")])

    async def lookup(*_args, **_kwargs):
        return order_row(attempt, "PART_FILLED", "0.3")

    recovery.poll(lookup)
    await recovery.task
    recovery.poll(lookup)
    assert recovery.pending  # no new account cohort after learning the partial fill

    old_row = order_row(attempt, clientId="")
    if contradictory_id:
        old_row = order_row(attempt, orderId="unrelated-venue-id")
    account.capture([old_row])
    recovery.poll(lookup)
    assert journal.load() == (attempt,)

    account.capture([order_row(attempt, "PART_FILLED", "0.3", clientId="")])
    account.fill_capture(recovery, recovery.fill_after_ms + 1, "0.1", "0.2")
    recovery.poll(lookup)
    assert recovery.pending is contradictory_id


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_lookup_failure_or_miss_never_releases_restart_reservation(
    journal, timeout
):
    first = BitunixCreateRecovery(Account(), journal)
    attempt = first.record(BODY, None)
    account = Account()
    recovery = BitunixCreateRecovery(account, journal)
    calls = []

    async def lookup(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if timeout:
            raise NetworkError("offline timeout")
        if "get_order_detail" in path:
            raise OrderNotFound("20007")
        return {"orderList": [], "total": 0}

    recovery.poll(lookup)
    account.capture([order_row(attempt, clientId="same-shape-other-id")])
    recovery.poll(lookup)
    await recovery.task
    recovery.poll(lookup)
    assert recovery.pending and journal.load() == (attempt,)
    assert len(calls) == (1 if timeout else 3)
    assert all(
        row[0] == "GET" and row[2]["params"]["clientId"] == "original" for row in calls
    )
    assert recovery.task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("filled", ["0", "0.3"])
async def test_terminal_requires_later_account_and_agreeing_fill_receipt(
    journal, filled
):
    account = Account()
    recovery = BitunixCreateRecovery(account, journal)
    attempt = recovery.record(BODY, None)
    account.capture()

    async def lookup(*args, **kwargs):
        return order_row(attempt, "CANCELED", filled)

    recovery.poll(lookup)
    await recovery.task
    recovery.poll(lookup)
    assert recovery.terminal is not None
    account.fill_capture(recovery, recovery.fill_after_ms - 10)
    account.capture()
    recovery.poll(lookup)
    assert recovery.pending  # fill receipt started before the terminal evidence

    conflict = ("0.4",) if filled == "0" else ("0.1",)
    account.fill_capture(recovery, recovery.fill_after_ms + 1, *conflict)
    recovery.poll(lookup)
    assert (
        recovery.pending
    )  # includes a contradictory fill for a zero-fill cancellation

    coherent = () if filled == "0" else ("0.1", "0.2")
    account.fill_capture(recovery, recovery.fill_after_ms + 2, *coherent)
    account.capture([order_row(attempt)])
    recovery.poll(lookup)
    assert recovery.pending  # account still shows a conflicting pending order
    account.capture()
    recovery.poll(lookup)
    assert not recovery.pending
    assert journal.load() == ()


@pytest.mark.asyncio
async def test_partial_working_order_waits_for_exact_fills(journal):
    account = Account()
    recovery = BitunixCreateRecovery(account, journal)
    attempt = recovery.record(BODY, None)
    account.capture([order_row(attempt, "PART_FILLED", "0.3")])
    recovery.next_lookup = float("inf")  # account and fills are being checked locally
    recovery.poll(unexpected_lookup)
    assert recovery.pending
    account.fill_capture(recovery, recovery.fill_after_ms + 1, "0.1", "0.2")
    recovery.poll(unexpected_lookup)
    assert journal.load() == ()


@pytest.mark.asyncio
async def test_close_cancels_lookup_and_preserves_attempt(journal):
    account = Account()
    recovery = BitunixCreateRecovery(account, journal)
    attempt = recovery.record(BODY, None)
    account.capture()
    entered = asyncio.Event()

    async def lookup(*_args, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    recovery.poll(lookup)
    await asyncio.wait_for(entered.wait(), 1)
    await recovery.close()
    assert recovery.task.cancelled()
    assert journal.load() == (attempt,)
