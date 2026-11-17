import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest
from ccxt.base.errors import NetworkError, OrderNotFound

from live import bitunix_create_recovery as recovery_module
from live.bitunix_create_journal import BitunixCreateJournal, BitunixCreateJournalError
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
        "clientId": attempt.client_id,
        "symbol": attempt.symbol,
        "side": attempt.side.upper(),
        "tradeSide": attempt.trade_side,
        "qty": attempt.qty,
        "price": attempt.price,
        "positionId": attempt.position_id,
        "effect": attempt.effect,
        "orderType": attempt.order_type.upper(),
        "reduceOnly": attempt.reduce_only,
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
    account.fill_capture(
        recovery, recovery.attempts[attempt.client_id].fill_after_ms + 1, "0.1", "0.2"
    )
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
@pytest.mark.parametrize("later_source", ["lookup", "pending"])
async def test_later_partial_survives_restart_with_older_terminal_detail(
    journal, later_source
):
    account = Account()
    recovery = BitunixCreateRecovery(account, journal)
    attempt = recovery.record(BODY, None)
    account.capture()
    response = order_row(attempt, "CANCELED", "0")

    async def lookup(*_args, **_kwargs):
        return response

    recovery.poll(lookup)
    await recovery.task
    recovery.poll(lookup)
    assert recovery.attempts[attempt.client_id].terminal is not None

    partial = order_row(attempt, "PART_FILLED", "0.3")
    if later_source == "lookup":
        account.capture()
        response = partial
        recovery.attempts[attempt.client_id].next_lookup = 0
        recovery.poll(lookup)
        await recovery.task
        recovery.poll(lookup)
    else:
        account.capture([partial])
        recovery.poll(lookup)
    account.capture()
    account.fill_capture(
        recovery, recovery.attempts[attempt.client_id].fill_after_ms + 1
    )
    recovery.poll(lookup)
    assert recovery.pending

    account = Account()
    recovery = BitunixCreateRecovery(account, BitunixCreateJournal(journal.path))
    response = order_row(attempt, "CANCELED", "0")
    recovery.poll(lookup)
    account.capture()
    recovery.poll(lookup)
    await recovery.task
    recovery.poll(lookup)
    account.capture()
    account.fill_capture(
        recovery, recovery.attempts[attempt.client_id].fill_after_ms + 1
    )
    recovery.poll(lookup)
    assert recovery.pending

    response = order_row(attempt, "PART_FILLED_CANCELED", "0.3")
    recovery.attempts[attempt.client_id].next_lookup = 0
    recovery.poll(lookup)
    await recovery.task
    recovery.poll(lookup)
    account.capture()
    account.fill_capture(
        recovery, recovery.attempts[attempt.client_id].fill_after_ms + 1, "0.1", "0.2"
    )
    recovery.poll(lookup)
    assert journal.load() == ()


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
    assert recovery.attempts[attempt.client_id].terminal is not None
    account.fill_capture(
        recovery, recovery.attempts[attempt.client_id].fill_after_ms - 10
    )
    account.capture()
    recovery.poll(lookup)
    assert recovery.pending  # fill receipt started before the terminal evidence

    conflict = ("0.4",) if filled == "0" else ("0.1",)
    account.fill_capture(
        recovery, recovery.attempts[attempt.client_id].fill_after_ms + 1, *conflict
    )
    recovery.poll(lookup)
    assert (
        recovery.pending
    )  # includes a contradictory fill for a zero-fill cancellation

    coherent = () if filled == "0" else ("0.1", "0.2")
    account.fill_capture(
        recovery, recovery.attempts[attempt.client_id].fill_after_ms + 2, *coherent
    )
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
    recovery.attempts[attempt.client_id].next_lookup = float(
        "inf"
    )  # account and fills are being checked locally
    recovery.poll(unexpected_lookup)
    assert recovery.pending
    account.fill_capture(
        recovery, recovery.attempts[attempt.client_id].fill_after_ms + 1, "0.1", "0.2"
    )
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


def _wave_contexts(account):
    contexts = [
        SimpleNamespace(bot=account, order={"custom_id": client_id}, action="create")
        for client_id in ("first", "second", "other")
    ]
    account._hsl_live = SimpleNamespace(
        _ordinary_create_wave=SimpleNamespace(
            wave_id="active-wave",
            permits=lambda bot, order, action: (
                bot is account
                and action == "create"
                and any(c.order is order for c in contexts)
            ),
        )
    )
    bodies = [
        {
            **BODY,
            "clientId": ctx.order["custom_id"],
            "symbol": symbol,
            "positionId": f"position-{symbol}",
            "tradeSide": "CLOSE",
            "reduceOnly": True,
        }
        for ctx, symbol in zip(contexts, ("BTCUSDT", "ETHUSDT", "SOLUSDT"))
    ]
    return contexts, bodies


def test_only_bound_in_memory_wave_members_can_join(journal):
    account = Account()
    recovery = BitunixCreateRecovery(account, journal)
    (first, second, other), bodies = _wave_contexts(account)
    first_attempt = recovery.record(bodies[0], first)

    with pytest.raises(WriteDeferred):
        recovery.record(bodies[1], first)  # exact client ID comes from the bound order
    with pytest.raises(WriteDeferred):
        recovery.record(
            bodies[1], SimpleNamespace(**{**vars(second), "order": dict(second.order)})
        )
    with pytest.raises(WriteDeferred):
        recovery.record(bodies[1], None)
    with pytest.raises(WriteDeferred):
        BitunixCreateRecovery(account, journal).record(bodies[1], second)

    second_attempt = recovery.record(bodies[1], second)
    assert journal.load() == (first_attempt, second_attempt)
    recovery.resolve(second_attempt, "not_sent")
    assert journal.load() == (first_attempt,)
    with pytest.raises(WriteDeferred):
        recovery.record(bodies[2], other)  # freeing a slot does not add a wave member
    _wave_contexts(account)  # new object, even with the same token
    with pytest.raises(WriteDeferred):
        recovery.record(bodies[1], second)


def test_wave_cannot_continue_if_earlier_durable_attempt_vanishes(journal):
    account = Account()
    recovery = BitunixCreateRecovery(account, journal)
    contexts, bodies = _wave_contexts(account)
    first = recovery.record(bodies[0], contexts[0])
    assert journal.resolve(first.client_id)
    with pytest.raises(WriteDeferred):
        recovery.record(bodies[1], contexts[1])
    assert recovery.pending and journal.load() == ()


@pytest.mark.parametrize("persist_fails", [False, True])
def test_acknowledgment_commits_identity_and_retains_attempt(
    journal, monkeypatch, persist_fails
):
    account = Account()
    recovery = BitunixCreateRecovery(account, journal)
    contexts, bodies = _wave_contexts(account)
    first = recovery.record(bodies[0], contexts[0])
    if persist_fails:

        def fail_observation(*_args):
            raise BitunixCreateJournalError("simulated ACK journal failure")

        monkeypatch.setattr(journal, "observe", fail_observation)
    assert recovery.record_acknowledgment(first, ORDER_ID) is (not persist_fails)
    ((durable, observation),) = BitunixCreateJournal(journal.path).load_pending_all()
    assert durable == first and recovery.pending
    if persist_fails:
        assert observation is None
        with pytest.raises(WriteDeferred):
            recovery.record(bodies[1], contexts[1])
    else:
        assert observation.order_id == ORDER_ID
        assert observation.filled_qty == Decimal(0)
        assert not recovery.resolve(first, "not_sent")


@pytest.mark.asyncio
async def test_restart_recovers_sibling_despite_other_conflicting_identity(journal):
    account = Account()
    recovery = BitunixCreateRecovery(account, journal)
    contexts, bodies = _wave_contexts(account)
    first, second = [
        recovery.record(body, ctx) for body, ctx in zip(bodies[:2], contexts[:2])
    ]
    journal.observe(first.client_id, ORDER_ID, Decimal(0))
    journal.observe(first.client_id, "conflicting-id", Decimal(0))

    account = Account()
    recovery = BitunixCreateRecovery(account, BitunixCreateJournal(journal.path))
    recovery.poll(unexpected_lookup)
    account.capture([order_row(second, orderId="sibling-open-order")])
    recovery.poll(unexpected_lookup)
    assert journal.load() == (first,)
    assert recovery.pending
    assert recovery.task is None


@pytest.mark.asyncio
async def test_unfinished_lookup_does_not_block_sibling_account_evidence(journal):
    account = Account()
    recovery = BitunixCreateRecovery(account, journal)
    contexts, bodies = _wave_contexts(account)
    first, second = [
        recovery.record(body, ctx) for body, ctx in zip(bodies[:2], contexts[:2])
    ]
    account.capture()
    entered = asyncio.Event()

    async def lookup(*_args, **kwargs):
        assert kwargs["params"]["clientId"] == first.client_id
        entered.set()
        await asyncio.Event().wait()

    recovery.poll(lookup)
    await asyncio.wait_for(entered.wait(), 1.0)
    task = recovery.task
    account.capture([order_row(second, orderId="sibling-open-order")])
    recovery.poll(lookup)
    assert recovery.task is task and not task.done()
    assert journal.load() == (first,)
    await recovery.close()


@pytest.mark.asyncio
async def test_lookup_rotates_between_unresolved_attempts_and_honors_backoff(journal):
    account = Account()
    recovery = BitunixCreateRecovery(account, journal)
    contexts, bodies = _wave_contexts(account)
    first, second = [
        recovery.record(body, ctx) for body, ctx in zip(bodies[:2], contexts[:2])
    ]
    account.capture()
    queried = []

    async def lookup(*_args, **kwargs):
        client_id = kwargs["params"]["clientId"]
        queried.append(client_id)
        exc = NetworkError("offline error")
        if client_id == first.client_id:
            exc.retry_after_seconds = 120.0
        raise exc

    recovery.poll(lookup)
    await recovery.task
    recovery.poll(lookup)
    assert queried == [first.client_id] and recovery.task is None
    recovery._retry_after = 0.0
    recovery.poll(lookup)
    await recovery.task
    recovery.poll(lookup)
    assert queried == [first.client_id, second.client_id]
    assert journal.load() == (first, second)
