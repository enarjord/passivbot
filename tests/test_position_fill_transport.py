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
async def test_legacy_change_during_connector_queue_defers_at_transport(
    action, transport
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
        cca=client, _position_fill_sync=sync, config={"live": {"hsl_engine": "legacy"}}
    )

    @hsl_revised_live.connector_write(action)
    async def write(bot, order):
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
