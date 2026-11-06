"""A final local trailing read cannot mix immutable acquisition cohorts."""

import asyncio
from copy import deepcopy
from dataclasses import replace

import pytest

from live import hsl_live, hsl_runtime
from live.hsl_candles import Sources
from test_hsl_ordinary_freshness import planning_bot
from test_hsl_runtime import NOW, SYMBOL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change,expected_reason",
    [
        ("unchanged", None),
        ("equal_source", "hsl_candle_receipt_replaced"),
        ("equal_fill", "hsl_fill_receipt_replaced"),
        ("position", "account_facts_changed_during_preparation"),
        ("required_fill", "account_facts_changed_during_preparation"),
        ("expired_account", "current_account_inputs_unavailable"),
    ],
)
async def test_final_local_derivation_keeps_one_current_cohort(
    monkeypatch, change, expected_reason
):
    fixture = planning_bot(monkeypatch)
    bot, owner = fixture.bot, fixture.owner
    owner.sources[SYMBOL] = Sources((), (), 0)
    bot._hsl_fill_observation = hsl_runtime.observe_fills(bot, (NOW, NOW))
    assert bot._hsl_fill_observation is not None
    initial = owner.ordinary_receipts()
    entered, release = asyncio.Event(), asyncio.Event()
    remote_permissions = []

    async def preload(*, allow_remote_fetch):
        assert allow_remote_fetch is False
        return {}

    async def derive(*, allow_remote_fetch, prepared_candles):
        remote_permissions.append(allow_remote_fetch)
        assert prepared_candles == {}
        entered.set()
        await release.wait()

    bot._preload_trailing_candles = preload
    bot.update_trailing_data = derive
    prepared = owner.account_facts()
    task = asyncio.create_task(owner.prepare_ordinary_consumption(prepared))
    try:
        await asyncio.wait_for(entered.wait(), 1.0)
        if change == "equal_source":
            old = owner.sources[SYMBOL]
            owner.sources[SYMBOL] = replace(old)
            assert owner.sources[SYMBOL] == old and owner.sources[SYMBOL] is not old
        elif change == "equal_fill":
            old = bot._hsl_fill_observation
            bot._hsl_fill_observation = replace(old)
            assert bot._hsl_fill_observation == old and bot._hsl_fill_observation is not old
        elif change == "position":
            bot.positions[SYMBOL]["long"]["size"] += 1.0
        elif change == "required_fill":
            bot._ensure_freshness_ledger().stamp("fills", now_ms=NOW, signature=("new",))
        elif change == "expired_account":
            fixture.clock[0] += 15_000
        release.set()
        if expected_reason is None:
            receipts = await asyncio.wait_for(task, 1.0)
            assert receipts.matches(initial)
            assert receipts.matches(owner.ordinary_receipts())
        else:
            with pytest.raises(hsl_live.OrdinaryPlanDeferred) as exc:
                await asyncio.wait_for(task, 1.0)
            assert exc.value.reason == expected_reason
        assert remote_permissions == [False]
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_fill_renewal_during_local_read_cannot_clear_pending_confirmation(
    tmp_path, monkeypatch
):
    from test_hsl_trailing_consumption import NOW as TRAILING_NOW
    from test_hsl_trailing_consumption import fill_event, pending_position_bot

    bot, symbol, exchange = await pending_position_bot(tmp_path, monkeypatch)
    key = (symbol, "long")
    pending = deepcopy(bot._trailing_pending_fill_confirmations)
    minimums = deepcopy(bot._trailing_pending_fill_min_generations)
    positions = deepcopy(bot._trailing_pending_position_states)
    assert key in pending
    old_receipt = hsl_runtime.observe_fills(bot, (TRAILING_NOW - 200, TRAILING_NOW - 100))
    bot._hsl_fill_observation = old_receipt
    old_anchor = bot._get_last_position_change_anchors()[symbol]["long"]
    old_signature = bot._fill_events_signature(bot._pnls_manager.get_events())
    owner = hsl_live.owner(bot)
    original = bot.cm.get_candles_with_resolution_ladder
    reads, renewed = [], []

    async def renew_during_acquisition(*args, **kwargs):
        reads.append(kwargs)
        assert kwargs["allow_remote_fetch"] is False
        if not renewed:
            # A canonical refresh can correct derived after-state while the
            # latest fill identity and coarse ledger signature stay equal.
            # Before acquisition completes, the old matching after-state must
            # not have irreversibly confirmed the exchange position.
            bot._pnls_manager._events[-1] = fill_event(
                symbol=symbol, position_side="long", timestamp=240_000,
                id="new-fill", psize=2.5, pprice=130.0, c_mult=1.0,
            )
            latest = hsl_runtime.observe_fills(bot, (TRAILING_NOW - 50, TRAILING_NOW))
            assert latest is not old_receipt and latest.tape == old_receipt.tape
            bot._hsl_fill_observation = latest
            bot._trailing_fill_fetch_generation = 3
            renewed.append(latest)
            await asyncio.sleep(0)
        return await original(*args, **kwargs)

    bot.cm.get_candles_with_resolution_ladder = renew_during_acquisition
    accepted = []
    for _ in range(2):
        try:
            receipt = await owner.prepare_ordinary_consumption(owner.account_facts())
        except hsl_live.OrdinaryPlanDeferred as exc:
            # A conservative first-attempt discard is safe only if it does not
            # lose the safety receipt for the next attempt.
            assert exc.reason == "hsl_fill_receipt_replaced"
        else:
            accepted.append(receipt)
            assert receipt.fills is renewed[0]
        assert bot._trailing_pending_fill_confirmations == pending
        assert bot._trailing_pending_fill_min_generations == minimums
        assert bot._trailing_pending_position_states == positions
        assert bot._fill_events_signature(bot._pnls_manager.get_events()) == old_signature
        anchor = bot._get_last_position_change_anchors()[symbol]["long"]
        assert anchor["epoch"] == old_anchor["epoch"]
        assert anchor["timestamp"] == old_anchor["timestamp"]
        assert anchor["psize"] != old_anchor["psize"]
        assert bot._orchestrator_trailing_unavailable_psides == {symbol: ["long"]}
        assert bot._trailing_fill_confirmation_diagnostics[key]["failed_predicates"] == [
            "fill_after_state_mismatch"
        ]
    assert accepted, "the current cohort remains usable with the affected branch unavailable"
    assert reads and exchange.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("hsl_enabled", [True, False])
async def test_confirmed_anchor_correction_revalidates_and_same_id_repair_recovers(
    tmp_path, monkeypatch, hsl_enabled
):
    from passivbot import _orchestrator_trailing_input
    from test_hsl_trailing_consumption import NOW as TRAILING_NOW
    from test_hsl_trailing_consumption import fill_event, pending_position_bot

    bot, symbol, exchange = await pending_position_bot(tmp_path, monkeypatch)
    for side in ("long", "short"):
        bot.config["bot"][side]["hsl"]["enabled"] = hsl_enabled
    owner = hsl_live.owner(bot)
    original_anchor = bot._get_last_position_change_anchors()[symbol]["long"]
    signature = bot._fill_events_signature(bot._pnls_manager.get_events())
    tape = None
    for index, (size, price, available) in enumerate(
        [(1.5, 101.0, True), (2.5, 130.0, False), (1.5, 101.0, True)],
        start=1,
    ):
        bot._pnls_manager._events[-1] = fill_event(
            symbol=symbol, position_side="long", timestamp=240_000,
            id="new-fill", psize=size, pprice=price, c_mult=1.0,
        )
        latest = hsl_runtime.observe_fills(bot, (TRAILING_NOW - 50, TRAILING_NOW))
        assert (latest is not None) is hsl_enabled
        if latest is not None:
            if tape is not None:
                assert latest.tape == tape
            tape = latest.tape
        bot._hsl_fill_observation = latest
        bot._trailing_fill_fetch_generation = index + 1
        receipt = await owner.prepare_ordinary_consumption(owner.account_facts())
        assert receipt.fills is latest
        assert bot._fill_events_signature(bot._pnls_manager.get_events()) == signature
        anchor = bot._get_last_position_change_anchors()[symbol]["long"]
        assert anchor["epoch"] == original_anchor["epoch"]
        assert anchor["timestamp"] == original_anchor["timestamp"]
        assert _orchestrator_trailing_input(bot, symbol, "long")[1] is available
        if available:
            assert bot._trailing_pending_fill_confirmations == {}
            assert bot._orchestrator_trailing_unavailable_symbols == set()
        else:
            assert (symbol, "long") in bot._trailing_pending_fill_confirmations
            assert bot._orchestrator_trailing_unavailable_psides == {symbol: ["long"]}
            assert bot._trailing_fill_confirmation_diagnostics[(symbol, "long")][
                "failed_predicates"
            ] == ["fill_after_state_mismatch"]
    assert exchange.calls == []


@pytest.mark.asyncio
async def test_disabled_owner_rebuilds_trailing_after_correction_during_ema(
    tmp_path, monkeypatch
):
    import argparse
    import json

    import hjson
    import passivbot
    import passivbot_rust as pbr
    import tools.run_fake_live as runner
    from config.hsl import generated_template
    from hsl_fixture import load_fake_hsl_config, mutable_runtime_fixture
    from test_run_fake_live import REPO_ROOT, _cleanup_fake_user_state

    cfg = generated_template(mutable_runtime_fixture(load_fake_hsl_config()), "coin")
    for side in ("long", "short"):
        cfg["bot"][side]["hsl"]["enabled"] = False
        cfg["bot"][side]["unstuck"]["enabled"] = False
        cfg["bot"][side]["entry_cooldown"]["base_duration_minutes"] = 0.0
    cfg["bot"]["long"]["universe"]["approved_coins"] = ["BTC"]
    cfg["bot"]["long"]["strategy"]["trailing_martingale"]["close"][
        "retracement_base_fraction"
    ] = 0.01
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}))
    scenario = hjson.loads(
        (REPO_ROOT / "scenarios/fake_live/hsl_long_red_restart.hjson").read_text()
    )
    scenario.pop("assertions", None)
    scenario["boot_index"] = 5
    scenario["run_initial_cycle"] = True
    candles = scenario["replay"]["symbols"]["BTC/USDT:USDT"]["candles"]
    for row in candles[2:]:
        row[1:5] = [110.0, 112.0, 107.0, 110.0]
    scenario_path = tmp_path / "scenario.hjson"
    scenario_path.write_text(hjson.dumps(scenario))
    user = f"fake_hsl_consumption_{tmp_path.name}"
    _cleanup_fake_user_state(user)
    completed = []

    async def exercise(bot):
        symbol = "BTC/USDT:USDT"
        owner = hsl_live.owner(bot)
        await bot.refresh_protective_authoritative_state(require_balance=True)
        # Genuine startup fill confirmation establishes the ordinary trailing
        # proof before the awaited EMA phase. No owner/helper is substituted.
        await bot.update_trailing_data(allow_remote_fetch=False)
        assert not bot._trailing_pending_fill_confirmations
        assert passivbot._orchestrator_trailing_input(bot, symbol, "long")[1]
        old_trailing = deepcopy(bot.trailing_prices[symbol]["long"])
        assert not hsl_runtime.enabled(bot)
        original_ema = bot._load_orchestrator_ema_bundle
        entered, release = asyncio.Event(), asyncio.Event()
        native_calls = []
        native_compute = pbr.compute_ideal_orders_json

        def compute(payload):
            result = native_compute(payload)
            native_calls.append((json.loads(payload), json.loads(result)))
            return result

        async def slow_ema(*args, **kwargs):
            result = await original_ema(*args, **kwargs)
            entered.set()
            await release.wait()
            return result

        monkeypatch.setattr(pbr, "compute_ideal_orders_json", compute)
        monkeypatch.setattr(bot, "_load_orchestrator_ema_bundle", slow_ema)
        before = bot._get_last_position_change_anchors()[symbol]["long"]
        signature = bot._fill_events_signature(bot._pnls_manager.get_events())
        task = asyncio.create_task(owner._ordinary_plan())
        try:
            await asyncio.wait_for(entered.wait(), 5.0)
            events = bot._pnls_manager._events
            index = next(i for i, event in enumerate(events) if event.id == "12")
            old = events[index]
            events[index] = replace(old, psize=old.psize + 1.0, pprice=old.pprice + 30.0)
            bot._trailing_fill_fetch_generation += 1
            assert bot._fill_events_signature(bot._pnls_manager.get_events()) == signature
            after = bot._get_last_position_change_anchors()[symbol]["long"]
            assert after["epoch"] == before["epoch"]
            assert after["timestamp"] == before["timestamp"]
            release.set()
            plan = await asyncio.wait_for(task, 5.0)
            assert plan is not None
            assert native_calls and not getattr(pbr, "__is_stub__", False)
            native_input, result = native_calls[-1]
            row = next(row for row in native_input["symbols"] if row["long"]["position"]["size"])
            assert row["long"]["trailing_available"] is False
            assert not any(order["order_type"] == "close_trailing_long" for order in result["orders"])
            control = deepcopy(native_input)
            control_row = next(candidate for candidate in control["symbols"]
                               if candidate["symbol_idx"] == row["symbol_idx"])
            control_row["long"].update(trailing=old_trailing, trailing_available=True)
            control_result = json.loads(native_compute(json.dumps(control)))
            assert any(order["order_type"] == "close_trailing_long" for order in control_result["orders"])
            assert (symbol, "long") in bot._trailing_pending_fill_confirmations
            completed.append(True)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        return {"ordinary_disabled_trailing_current": True}

    monkeypatch.setattr(runner, "_run_fake_cycle", exercise)
    try:
        args = argparse.Namespace(
            config=str(config_path), scenario=str(scenario_path), user=user,
            max_steps=1, output_dir=str(tmp_path / "run"), log_level=1,
            snapshot_each_step=False,
        )
        assert await runner._async_main(args) == 0
        assert completed
    finally:
        _cleanup_fake_user_state(user)
