"""No network: execute the real CCXT queue or fake exchange transport boundary."""

import asyncio
from types import SimpleNamespace
import pytest
from live import position_fill_sync as syncmod, hsl_revised_live, executor
from ccxt.async_support.base.exchange import Exchange
from exchanges.fake import FakeCCXTClient
from test_fake_exchange import _scenario


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["create", "cancel"])
@pytest.mark.parametrize("transport", ["ccxt", "fake", "bitunix"])
@pytest.mark.parametrize("engine", ["legacy", "revised"])
async def test_change_during_connector_queue_defers_at_transport(
    action, transport, engine, monkeypatch
):
    key = ("BTC/USDT:USDT", "long")
    sync = syncmod.PositionFillSync(lambda: 0.0)
    sync.observe({key: (0.0, 0.0)})
    entered, release = asyncio.Event(), asyncio.Event()
    sent = []

    async def wait(*args, **kwargs):
        entered.set()
        await release.wait()

    if transport == "ccxt":
        client = Exchange({"enableRateLimit": True})
        client.throttle = wait
        client.sign = lambda *a, **kw: dict(
            url="https://offline.invalid", method="POST", headers={}, body=None
        )

        async def fetch(*args, **kwargs):
            sent.append("transport")
            return {}

        client.fetch = fetch

        async def submit():
            return await client.fetch2("orders")

    elif transport == "bitunix":
        from exchanges.bitunix import BitunixClient

        client = BitunixClient()
        client._throttle = wait

        class Session:
            def request(self, *args, **kwargs):
                sent.append("transport")
                raise AssertionError("must defer before HTTP submission")

        async def session():
            return Session()

        client._get_session = session

        async def submit():
            return await client._request("POST", "/offline-test")

    else:
        client = FakeCCXTClient(_scenario(), quote="USDT")
        existing = await client.create_order(
            key[0], "limit", "buy", 1.0, 50.0, {"positionSide": "LONG"}
        )
        before = client.export_state()

        async def submit():
            await wait()
            if action == "create":
                return await client.create_order(
                    key[0], "market", "buy", 1.0, 100.0, {"positionSide": "LONG"}
                )
            return await client.cancel_order(existing["id"], key[0])

    bot = SimpleNamespace(
        cca=client, _position_fill_sync=sync, config={"live": {"hsl_engine": engine}}
    )

    if engine == "revised":
        owner = SimpleNamespace(_write_lock=asyncio.Lock(), admit=lambda order: True)
        monkeypatch.setattr(hsl_revised_live, "owner", lambda bot: owner)
        bot._request_authoritative_confirmation = lambda scopes: None
    ownership, cancellations, events = [], [], []
    bot.add_to_recent_order_cancellations = cancellations.append
    bot.log_order_action = lambda *a, **kw: None
    bot._log_order_action_summary = lambda *a, **kw: None
    callbacks = SimpleNamespace(
        _record_emitted_order_custom_id=lambda bot, order, **kw: ownership.append(
            order
        ),
        _record_order_churn_allowance_attempts=lambda *a, **kw: None,
        _emit_execution_order_event=lambda *a, **kw: events.append(kw),
    )
    monkeypatch.setattr(executor, "_pb_attr", lambda name: callbacks)

    @hsl_revised_live.connector_write(action)
    async def write(bot, order):
        if action == "create":
            executor.record_create_connector_admission(bot, order)
        return await submit()

    task = asyncio.create_task(write(bot, dict(symbol=key[0], position_side=key[1])))
    await asyncio.wait_for(entered.wait(), 1.0)
    sync.observe({key: (1.0, 100.0)})
    if transport == "ccxt":
        # The ContextVar must not gate unrelated reads on the same client.
        await client.fetch("offline read")
        assert sent == ["transport"]
        sent.clear()
    release.set()
    result = await asyncio.wait_for(task, 1.0)
    assert isinstance(
        result,
        (
            executor.DeferredOrderCreation
            if action == "create"
            else executor.DeferredOrderCancellation
        ),
    )
    assert not sent
    assert not ownership and not cancellations and not events
    assert syncmod._write_context.get() is None
    if transport == "fake":
        after = client.export_state()
        assert after["positions"] == before["positions"]
        assert after["fills"] == before["fills"]
        assert after["open_orders"] == before["open_orders"]
    if transport == "ccxt":
        await client.close()


@pytest.mark.asyncio
async def test_fake_runner_settles_without_waiting_on_frozen_market_clock():
    from tools.run_fake_live import _install_runtime_overrides

    calls = []

    class Bot:
        config = {"live": {"hsl_engine": "legacy"}}
        cca = FakeCCXTClient(_scenario(), quote="USDT")

        async def update_pnls(self, **kwargs):
            calls.append(syncmod.state(self).clock())
            return True

    bot = Bot()
    _install_runtime_overrides(bot, {})
    gate = syncmod.state(bot)
    key = ("BTC/USDT:USDT", "long")
    gate.observe({key: (0.0, 0.0)})
    gate.observe({key: (1.0, 100.0)})
    changed = gate.clock()
    market_time = bot.cca.now_ms
    assert await asyncio.wait_for(bot.update_pnls(), 1.0)
    assert calls == [changed + 5.0]
    assert bot.cca.now_ms == market_time


def test_submission_bookkeeping_waits_for_write_and_survives_ambiguous_retry():
    from ccxt.base.errors import NetworkError

    gate = syncmod.PositionFillSync(lambda: 0.0)
    key = ("BTC/USDT:USDT", "long")
    gate.observe({key: (0.0, 0.0)})
    bot = SimpleNamespace(
        cca=SimpleNamespace(_position_fill_transport_guard=True),
        _position_fill_sync=gate,
        config={},
    )
    calls = []
    with syncmod.connector_context(bot, dict(symbol=key[0], position_side=key[1])):
        assert syncmod.defer_submission(lambda: calls.append("submitted"))
        syncmod.check_transport_admission(is_write=False)
        assert calls == []  # Connector preflight reads do not establish ownership.
        syncmod.check_transport_admission()
        syncmod.check_transport_admission()  # Retries do not duplicate provenance.
        assert calls == ["submitted"]
        gate.observe({key: (1.0, 100.0)})
        with pytest.raises(NetworkError, match="submitted request retry"):
            syncmod.check_transport_admission()
        assert calls == ["submitted"]
    assert syncmod._write_context.get() is None
