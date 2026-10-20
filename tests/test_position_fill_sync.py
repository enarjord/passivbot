"""Five-second capture ordering and finite release under adversarial scheduling."""

from types import SimpleNamespace
import pytest
from live.position_fill_sync import PositionFillSync, permits

A = ("A/USDT:USDT", "long")
B = ("A/USDT:USDT", "short")
C = ("B/USDT:USDT", "long")


def settling_config(scope="coin", *, enabled=True):
    """Only canonical facts consumed by the position/fill admission gate."""
    policy = {"scope": scope}
    if scope == "portfolio":
        policy["enabled"] = enabled
    return {
        "config_version": "v9.0.0",
        "bot": {'global': {'hsl': policy}, 'long': {'hsl': {'enabled': enabled}}, 'short': {'hsl': {'enabled': enabled}}},
    }


def gate():
    now = [0.0]
    sync = PositionFillSync(lambda: now[0])
    sync.observe({A: (0, 0), B: (0, 0), C: (0, 0)})
    sync.observe({A: (1, 100), B: (0, 0), C: (0, 0)})
    return sync, now


def test_wait_requires_capture_started_after_settle_not_only_late_completion():
    sync, now = gate()
    old = sync.begin_fetch()
    for t in (0, 1, 4.999):
        now[0] = t
        assert not sync.fetch_ready()
        assert sync.blocked(A)
        assert not sync.blocked(B)
    now[0] = 5
    sync.finish_fetch(old)
    assert sync.blocked(A)
    receipt = sync.begin_fetch()
    sync.finish_fetch(receipt)  # success need not discover the missing fill
    assert not sync.blocked(A)


@pytest.mark.parametrize(
    "failure", ["never_returns", "raises", "no_fetch", "continuous_changes"]
)
def test_wait_cannot_be_extended_indefinitely(failure, caplog):
    sync, now = gate()
    for t in range(100):
        now[0] = t
        if failure == "continuous_changes":
            sync.observe({A: (t + 1, 100), B: (0, 0), C: (0, 0)})
        else:
            sync.observe(sync.positions)  # repeated polling cannot renew deadline
        if sync.fetch_ready():
            sync.begin_fetch()  # hung/failed/absent result never finishes
        assert sync.blocked(A) == (t < 15)
        if t >= 15:
            assert (
                sync.fetch_ready()
            )  # repeated changes cannot starve acquisition either
    assert caplog.text.count("synchronization expired") == 1


def test_change_during_fetch_does_not_accept_stale_receipt_or_reset_hard_deadline():
    sync, now = gate()
    now[0] = 5
    old = sync.begin_fetch()
    now[0] = 6
    sync.observe({A: (2, 100), B: (0, 0), C: (0, 0)})
    sync.finish_fetch(old)
    assert sync.blocked(A)
    assert not sync.fetch_ready()
    now[0] = 15
    assert not sync.blocked(A)
    receipt = sync.begin_fetch()
    sync.finish_fetch(receipt)
    sync.observe({A: (3, 100), B: (0, 0), C: (0, 0)})
    assert sync.blocked(A)  # confirmed recovery allows a new bounded burst


@pytest.mark.parametrize(
    "mode,blocked",
    [
        ("coin", [True, False, False]),
        ("pside", [True, False, True]),
        ("portfolio", [True, True, True]),
    ],
)
def test_scope_dependency_for_both_create_and_cancel(mode, blocked):
    sync, now = gate()
    bot = SimpleNamespace(
        _position_fill_sync=sync,
        config=settling_config(mode),
    )
    for key, expected in zip((A, B, C), blocked):
        assert permits(bot, dict(symbol=key[0], position_side=key[1])) is not expected
    now[0] = 15
    assert all(permits(bot, dict(symbol=k[0], position_side=k[1])) for k in (A, B, C))


def test_restart_does_not_restore_expired_local_gate():
    sync, now = gate()
    now[0] = 20
    assert not sync.blocked(A)
    restarted = PositionFillSync(lambda: now[0])
    restarted.observe(sync.positions)
    assert not restarted.blocked(A)  # startup follows ordinary initial fetch checks


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["create", "cancel"])
async def test_connector_boundary_defers_all_actions_before_any_io(
    action, monkeypatch
):
    from live.hsl_live import connector_write
    from live.executor import DeferredOrderCreation, DeferredOrderCancellation

    sync, now = gate()
    called, recorded = [], []
    from live import executor

    monkeypatch.setattr(
        executor,
        "record_cancel_connector_admission",
        lambda bot, order: recorded.append(order),
    )
    bot = SimpleNamespace(
        _position_fill_sync=sync, config=settling_config()
    )

    @connector_write(action)
    async def write(bot, order):
        called.append(order)
        return "sent"

    order = dict(symbol=A[0], position_side=A[1])
    result = await write(bot, order)
    assert isinstance(
        result,
        DeferredOrderCreation if action == "create" else DeferredOrderCancellation,
    )
    assert not called and not recorded


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["create", "cancel"])
async def test_batch_gate_does_not_require_available_history_to_release(action):
    from live import executor

    sync, now = gate()
    bot = SimpleNamespace(
        _position_fill_sync=sync, live_value=lambda key: 10, config=settling_config(),
    )
    method = (
        executor.execute_orders_parent
        if action == "create"
        else executor.execute_cancellations_parent
    )
    assert await method(bot, [dict(symbol=A[0], position_side=A[1])]) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["create", "cancel"])
async def test_new_change_while_queued_is_checked_again_at_connector(
    monkeypatch, action
):
    import asyncio
    import utils
    from live import hsl_live, executor
    from test_hsl_runtime import bot as make_bot, quotes, NOW, SYMBOL

    monkeypatch.setattr(utils, "utc_ms", lambda: NOW)
    bot = make_bot()
    bot.get_exchange_time = lambda: NOW
    bot.approved_coins_minus_ignored_coins = {"long": {SYMBOL}, "short": set()}
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot._ensure_freshness_ledger().stamp("open_orders", now_ms=NOW)
    sync = bot._position_fill_sync = PositionFillSync(lambda: 0.0)
    sync.observe({(SYMBOL, "long"): (9.0, 100.0)})
    owner = hsl_live.owner(bot)
    wave = owner.capture(quotes())
    order = dict(symbol=SYMBOL, position_side="long")
    owner.bind(wave, (), (order,))
    assert owner.admit(order)
    calls = []

    @hsl_live.connector_write(action)
    async def write(bot, order):
        calls.append(order)

    await owner._write_lock.acquire()
    task = asyncio.create_task(write(bot, order))
    await asyncio.sleep(0)
    assert not task.done()
    sync.observe({(SYMBOL, "long"): (10.0, 100.0)})
    owner._write_lock.release()
    result = await task
    expected = (
        executor.DeferredOrderCreation
        if action == "create"
        else executor.DeferredOrderCancellation
    )
    assert isinstance(result, expected)
    assert not calls


def test_successful_cap_authorized_fetch_rearms_a_new_burst():
    sync, now = gate()
    now[0] = 14.9
    sync.observe({A: (2, 100), B: (0, 0), C: (0, 0)})
    now[0] = 15
    assert sync.fetch_ready() and not sync.blocked(A)
    sync.finish_fetch(sync.begin_fetch())
    assert A not in sync.pending
    now[0] = 16
    sync.observe({A: (3, 100), B: (0, 0), C: (0, 0)})
    assert sync.blocked(A) and not sync.fetch_ready()
    now[0] = 21
    assert sync.fetch_ready()


@pytest.mark.parametrize("mode", ["pside", "portfolio"])
def test_disabled_aggregate_policy_keeps_only_own_coin_side_dependency(mode):
    sync, now = gate()
    bot = SimpleNamespace(
        _position_fill_sync=sync,
        config=settling_config(mode, enabled=False),
    )
    assert not permits(bot, dict(symbol=A[0], position_side=A[1]))
    assert permits(bot, dict(symbol=B[0], position_side=B[1]))
    assert permits(bot, dict(symbol=C[0], position_side=C[1]))


@pytest.mark.parametrize("pending,scope,blocked", [
    (A, "coin", {A}), (A, "pside", {A, C}), (A, "portfolio", {A, B, C}),
    (B, "coin", {B}), (B, "pside", {B}), (B, "portfolio", {A, B, C}),
    (C, "coin", {C}), (C, "pside", {A, C}), (C, "portfolio", {A, B, C}),
])
def test_canonical_aggregate_scope_defers_every_contributing_pending_position(pending, scope, blocked):
    now = [0.0]
    sync = PositionFillSync(lambda: now[0])
    positions = {key: (0, 0) for key in (A, B, C)}
    sync.observe(positions)
    sync.observe({**positions, pending: (1, 100)})
    bot = SimpleNamespace(_position_fill_sync=sync, config=settling_config(scope))
    for key in (A, B, C):
        assert permits(bot, dict(symbol=key[0], pside=key[1])) is (key not in blocked)
    now[0] = 15
    assert all(permits(bot, dict(symbol=key[0], position_side=key[1])) for key in (A, B, C))


def test_pside_gate_uses_only_the_canonical_order_side_enabled_policy():
    sync, _ = gate()
    config = settling_config("pside")
    config["bot"]["long"]["hsl"]["enabled"] = False
    bot = SimpleNamespace(_position_fill_sync=sync, config=config)
    assert not permits(bot, dict(symbol=A[0], position_side="long"))
    assert permits(bot, dict(symbol=C[0], position_side="long"))
    assert permits(bot, dict(symbol=B[0], position_side="short"))
    config["bot"]["long"]["hsl"]["enabled"] = True
    assert not permits(bot, dict(symbol=C[0], position_side="long"))


@pytest.mark.parametrize("scope", ["pside", "portfolio"])
def test_legacy_policy_fields_cannot_override_canonical_scope(scope):
    sync, _ = gate()
    config = settling_config(scope)
    config["live"] = {"hsl_signal_mode": "coin"}
    config["bot"]["hsl"] = {"enabled": False}
    bot = SimpleNamespace(_position_fill_sync=sync, config=config)
    target = C if scope == "pside" else B
    assert not permits(bot, dict(symbol=target[0], position_side=target[1]))


@pytest.mark.parametrize("scope", [None, True, 1, [], {}, "", "unified", "invalid"])
def test_invalid_canonical_scope_fails_explicitly_when_gate_is_consumed(scope):
    sync, _ = gate()
    config = settling_config()
    config["bot"]["global"]["hsl"]["scope"] = scope
    config["live"] = {"hsl_signal_mode": "coin"}
    with pytest.raises(ValueError, match="invalid HSL scope"):
        permits(SimpleNamespace(_position_fill_sync=sync, config=config),
                dict(symbol=C[0], position_side=C[1]))


@pytest.mark.parametrize("config", [{}, {"bot": {}}, {"bot": {"risk": {}}},
                                    {"bot": {'global': {'hsl': {}}}}])
def test_missing_canonical_scope_has_no_legacy_or_coin_fallback(config):
    sync, _ = gate()
    with pytest.raises(KeyError):
        permits(SimpleNamespace(_position_fill_sync=sync, config=config),
                dict(symbol=C[0], position_side=C[1]))


@pytest.mark.parametrize("scope", ["pside", "portfolio"])
def test_missing_enabled_fact_for_consumed_aggregate_policy_fails(scope):
    sync, _ = gate()
    config = settling_config(scope)
    policy = config["bot"]["global"]["hsl"] if scope == "portfolio" else config["bot"]["long"]["hsl"]
    del policy["enabled"]
    with pytest.raises(KeyError, match="enabled"):
        permits(SimpleNamespace(_position_fill_sync=sync, config=config),
                dict(symbol=C[0], position_side=C[1]))


def test_absent_gate_does_not_consume_uninitialized_configuration():
    assert permits(SimpleNamespace(), dict(symbol=A[0], position_side=A[1]))
