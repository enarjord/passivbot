"""Ordinary preparation must not consume the current-account freshness budget."""
import asyncio
from types import SimpleNamespace

import pytest

from live import hsl_live
from live.market_snapshot import MarketSnapshot
from passivbot import Passivbot
from test_hsl_runtime import NOW, SYMBOL, bot as make_bot


class CaptureReached(Exception):
    pass


def planning_bot(monkeypatch, *, mode="coin", stage="ema", change=None, refresh_results=None):
    import passivbot
    import utils

    clock = [NOW]
    monkeypatch.setattr(utils, "utc_ms", lambda: clock[0])
    monkeypatch.setattr(passivbot, "utc_ms", lambda: clock[0])
    bot = make_bot(mode)
    ledger = bot._ensure_freshness_ledger()
    for surface in ("balance", "positions", "open_orders", "fills"):
        ledger.stamp(surface, now_ms=NOW)
    bot._staged_planner_required_surfaces = lambda **kwargs: {"fills"}
    bot.get_exchange_time = lambda: clock[0]
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot.stop_signal_received = False
    bot.active_symbols = [SYMBOL]
    bot.approved_coins_minus_ignored_coins = {"long": {SYMBOL}, "short": set()}
    bot.effective_min_cost = {SYMBOL: 1.0}
    bot._build_orchestrator_mode_overrides = lambda symbols: {"long": {}, "short": {}}
    bot._apply_exchange_symbol_unavailable_planning_policy = lambda *args, **kwargs: set()
    bot._staged_execution_ready_state = lambda **kwargs: (True, {})
    entered, release = asyncio.Event(), asyncio.Event()
    calls, defers, captures = [], [], []
    bot._log_staged_execution_defer = defers.append
    instance = hsl_live.owner(bot)

    async def preparation(name):
        if name == stage:
            entered.set()
            await release.wait()
        await asyncio.sleep(0)

    async def noop(**kwargs):
        return True

    async def warmup(symbols):
        return set()

    async def ema(symbols, modes):
        await preparation("ema")
        return ({}, {}, {}, {}, {}, {})

    async def quotes(symbols):
        await preparation("ticker")
        return {SYMBOL: MarketSnapshot(SYMBOL, 100.0, 100.0, 100.0, clock[0], "fixture")}

    async def refresh(**kwargs):
        calls.append(kwargs)
        # A real native capture before the confirming cohort reproduces the
        # availability failure, independently of the new orchestration helper.
        quote = MarketSnapshot(SYMBOL, 100.0, 100.0, 100.0, clock[0], "fixture")
        assert instance.capture({SYMBOL: quote}).permission(SYMBOL, "long")[0] == "unavailable"
        await asyncio.sleep(0)
        if refresh_results and not refresh_results.pop(0):
            return False
        ledger.begin_epoch(now_ms=clock[0])
        for surface in ("balance", "positions", "open_orders"):
            ledger.stamp(surface, now_ms=clock[0])
        if change == "position":
            bot.positions[SYMBOL]["long"]["size"] += 1.0
        elif change == "order":
            bot.open_orders[SYMBOL] = [dict(id="new", qty=1.0, price=100.0)]
        elif change == "strategy_balance":
            bot.get_hysteresis_snapped_balance = lambda: 900.0
        elif change == "generation":
            bot._account_invalidation_generation = 1
        elif change == "fill":
            ledger.stamp("fills", now_ms=clock[0], signature=("changed",))
        elif change == "raw_balance":
            bot.get_raw_balance = lambda: 1000.001
        return True

    def preconditions(*, include_market_snapshot, **kwargs):
        if include_market_snapshot:
            captures.append(bot._hsl_planning_wave)
            bot._current_planning_snapshot = object()
            raise CaptureReached

    async def calculate():
        try:
            await Passivbot.calc_ideal_orders_orchestrator(bot)
        except CaptureReached:
            return [], []
        raise AssertionError("test boundary was not reached")

    bot.prepare_planning_universe = noop
    bot.update_trailing_data = noop

    async def cached_candles(**kwargs):
        return {}

    bot._preload_trailing_candles = cached_candles
    bot.refresh_market_state_if_needed = noop
    bot._warmup_new_forager_normal_symbols = warmup
    bot._load_orchestrator_ema_bundle = ema
    bot._get_orchestrator_market_snapshots = quotes
    bot._assert_staged_planner_preconditions = preconditions
    bot.refresh_protective_authoritative_state = refresh
    bot.calc_orders_to_cancel_and_create = calculate
    return SimpleNamespace(bot=bot, owner=instance, clock=clock, entered=entered, release=release,
                           calls=calls, defers=defers, captures=captures)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
@pytest.mark.parametrize("stage", ["ema", "ticker"])
@pytest.mark.parametrize("restart", [False, True])
async def test_slow_preparation_confirms_account_before_real_hsl_capture(monkeypatch, mode, stage, restart):
    fixture = planning_bot(monkeypatch, mode=mode, stage=stage)
    if restart:
        fixture.owner = hsl_live.Owner(fixture.bot)
        fixture.bot._hsl_live = fixture.owner
    task = asyncio.create_task(fixture.owner._ordinary_plan())
    await asyncio.wait_for(fixture.entered.wait(), 1.0)
    fixture.clock[0] += 15_000
    fixture.release.set()
    plan = await task
    assert plan is not None and len(fixture.calls) == 1
    assert fixture.captures[-1].permission(SYMBOL, "long")[0] == "normal"
    assert fixture.owner._account_matches(plan[3], fixture.clock[0])
    assert not fixture.defers


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["position", "order", "strategy_balance", "generation", "fill"])
async def test_changed_account_confirmation_discards_prepared_universe(monkeypatch, change):
    fixture = planning_bot(monkeypatch, change=change)
    task = asyncio.create_task(fixture.owner._ordinary_plan())
    await asyncio.wait_for(fixture.entered.wait(), 1.0)
    fixture.clock[0] += 15_000
    fixture.release.set()
    assert await task is None
    assert not fixture.captures
    assert fixture.defers[-1]["defer_reason"] == "account_facts_changed_during_preparation"


@pytest.mark.asyncio
async def test_raw_balance_confirmation_uses_actual_current_value(monkeypatch):
    fixture = planning_bot(monkeypatch, change="raw_balance")
    task = asyncio.create_task(fixture.owner._ordinary_plan())
    await asyncio.wait_for(fixture.entered.wait(), 1.0)
    fixture.clock[0] += 15_000
    fixture.release.set()
    plan = await task
    assert plan is not None and plan[3].balance == 1000.001
    assert fixture.owner._account_matches(plan[3], fixture.clock[0])


@pytest.mark.asyncio
async def test_unavailable_account_refresh_defers_then_recovers(monkeypatch):
    fixture = planning_bot(monkeypatch, refresh_results=[False, True])
    task = asyncio.create_task(fixture.owner._ordinary_plan())
    await asyncio.wait_for(fixture.entered.wait(), 1.0)
    fixture.clock[0] += 15_000
    fixture.release.set()
    assert await task is None
    assert not fixture.captures
    assert fixture.defers[-1]["defer_reason"] == "current_account_refresh_unavailable"
    plan = await fixture.owner._ordinary_plan()
    assert plan is not None and fixture.owner._account_matches(plan[3], fixture.clock[0])


@pytest.mark.asyncio
async def test_existing_unchanged_refresh_cohort_is_reused_without_extra_fetch(monkeypatch):
    fixture = planning_bot(monkeypatch)
    account = fixture.owner.account_facts()
    fixture.clock[0] += 15_000
    await fixture.owner._refresh_lock.acquire()
    task = asyncio.create_task(fixture.owner.refresh_ordinary_account(account))
    await asyncio.sleep(0)
    assert not task.done()
    ledger = fixture.bot._ensure_freshness_ledger()
    ledger.begin_epoch(now_ms=fixture.clock[0])
    for surface in ("balance", "positions", "open_orders"):
        ledger.stamp(surface, now_ms=fixture.clock[0])
    fixture.owner._refresh_lock.release()
    await task
    assert not fixture.calls


def _record_owner_pass(fixture, actions):
    async def refresh(*, require_balance):
        assert require_balance
        actions.append("account")
        return True

    async def protect(**kwargs):
        actions.append("protection")
        return False

    async def execute(cancels, creates):
        assert not cancels and not creates
        actions.append("ordinary")

    fixture.bot._begin_live_event_cycle = lambda **kwargs: None
    fixture.bot.refresh_protective_authoritative_state = refresh
    fixture.bot.execute_order_plan_to_exchange = execute
    fixture.owner.protect = protect
    fixture.owner.schedule_history = fixture.owner.schedule_sources = lambda: None
    fixture.owner.schedule_account_reporting = lambda: None


@pytest.mark.asyncio
async def test_pending_fill_refresh_is_consumed_before_quotes_and_owner_handoff(monkeypatch):
    from dataclasses import replace
    from live.hsl_runtime import observe_fills

    fixture = planning_bot(monkeypatch)
    bot, owner = fixture.bot, fixture.owner
    old_receipt = observe_fills(bot, (NOW, NOW))
    new_receipt = replace(old_receipt)
    bot._hsl_fill_observation = old_receipt
    fill_release = asyncio.Event()
    quote_receipts = []
    actions = []
    _record_owner_pass(fixture, actions)

    async def refresh_fills():
        await fill_release.wait()
        bot._hsl_fill_observation = new_receipt
        return True

    async def quotes(symbols):
        quote_receipts.append(bot._hsl_fill_observation)
        return {SYMBOL: MarketSnapshot(SYMBOL, 100.0, 100.0, 100.0, NOW, "fixture")}

    bot._get_orchestrator_market_snapshots = quotes
    fill_task = owner._fill_task = asyncio.create_task(refresh_fills())
    plan_task = owner._ordinary = asyncio.create_task(owner._ordinary_plan())
    try:
        await asyncio.wait_for(fixture.entered.wait(), 1.0)
        fixture.release.set()
        # The fixture EMA returns after two loop turns. Without a join the
        # in-memory quote and planning path can finish in that same turn.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        fill_release.set()
        await fill_task
        plan = await plan_task
        assert plan is not None
        result = await owner.cycle()
        assert result["ordinary_completed"] and result["ordinary_executed"]
        assert actions == ["protection", "ordinary", "account", "protection"]
        assert len(quote_receipts) == 1 and quote_receipts[0] is new_receipt
        assert plan[3].ordinary_receipts.fills is new_receipt
    finally:
        fixture.release.set()
        fill_release.set()
        tasks = [task for task in (owner._ordinary, plan_task, fill_task) if task is not None]
        owner.cancel_inputs()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "cancel"])
async def test_slow_fill_join_leaves_protection_and_fill_task_independent(monkeypatch, outcome):
    fixture = planning_bot(monkeypatch)
    owner = fixture.owner
    fill_release = asyncio.Event()
    actions = []
    _record_owner_pass(fixture, actions)
    fill_task = owner._fill_task = asyncio.create_task(fill_release.wait())
    plan_task = owner._ordinary = asyncio.create_task(owner._ordinary_plan())
    try:
        await asyncio.wait_for(fixture.entered.wait(), 1.0)
        fixture.release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not plan_task.done()
        result = await asyncio.wait_for(owner.cycle(), 1.0)
        assert result["updated"] and not result["ordinary_completed"]
        assert actions == ["account", "protection"]
        assert owner._ordinary is plan_task and not plan_task.done()
        if outcome == "cancel":
            plan_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await plan_task
        else:
            plan = await asyncio.wait_for(plan_task, 4.0)
            assert plan is not None and owner._account_matches(plan[3], NOW)
        assert owner._fill_task is fill_task and not fill_task.done()
    finally:
        fixture.release.set()
        fill_release.set()
        owner.cancel_inputs()
        await asyncio.gather(plan_task, fill_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "handoff"),
    [("ema", "early"), ("ticker", "early"), ("ema", "tail"), ("ema", "expired_account")],
    ids=["ema-before-cycle", "ticker-before-cycle", "ema-during-protection", "ema-expired-account"],
)
async def test_real_owner_accepts_green_ordinary_plan_after_slow_preparation(
    tmp_path, monkeypatch, stage, handoff
):
    import argparse
    import json
    import hjson
    import passivbot
    import utils
    from config import prepare_config
    from config.hsl import generated_template
    from hsl_fixture import load_fake_hsl_config, mutable_runtime_fixture, set_risk_memory_spans
    from live import market_snapshot
    from test_run_fake_live import REPO_ROOT, _cleanup_fake_user_state
    import tools.run_fake_live as runner

    user = f"fake_hsl_freshness_{tmp_path.name}"
    _cleanup_fake_user_state(user)
    legacy = prepare_config(load_fake_hsl_config(), target="canonical", runtime=None, verbose=False)
    cfg = generated_template(mutable_runtime_fixture(legacy), "coin")
    for side in ("long", "short"):
        cfg["bot"][side]["unstuck"]["enabled"] = False
        cfg["bot"][side]["entry_cooldown"]["base_duration_minutes"] = 0.0
    cfg["bot"]["portfolio"]["loss_gate"]["max_drawdown_fraction"] = 1.0
    cfg["bot"]["long"]["universe"]["approved_coins"] = ["BTC"]
    cfg["bot"]["long"]["hsl"].update(enabled=True, drawdown_fraction_ema_span_minutes=1.0,
                                          drawdown_threshold_fraction=0.5)
    set_risk_memory_spans(cfg, 288.0)
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps({k:v for k,v in cfg.items() if not k.startswith("_")}))
    scenario = hjson.loads((REPO_ROOT/"scenarios/fake_live/hsl_long_red_restart.hjson").read_text())
    scenario.pop("assertions", None)
    scenario["run_initial_cycle"] = True
    scenario_path = tmp_path/"scenario.hjson"
    scenario_path.write_text(hjson.dumps(scenario))
    completed = []

    async def exercise(bot):
        instance = hsl_live.owner(bot)
        instance.poll_inputs()
        clock = [int(utils.utc_ms())]
        monkeypatch.setattr(utils, "utc_ms", lambda: clock[0])
        monkeypatch.setattr(passivbot, "utc_ms", lambda: clock[0])
        monkeypatch.setattr(market_snapshot, "utc_ms", lambda: clock[0])
        await bot.refresh_protective_authoritative_state(require_balance=True)
        if handoff == "expired_account":
            # Isolate account receipt expiry from independent fill/candle renewals.
            for refresh in (instance._fill_task, instance._source_task):
                if refresh is not None:
                    await asyncio.wait_for(refresh, 5.0)
            instance._next_history = instance._next_sources = float("inf")
        entered, release = asyncio.Event(), asyncio.Event()
        original = getattr(bot, "_load_orchestrator_ema_bundle" if stage=="ema" else "_get_orchestrator_market_snapshots")

        async def slow(*args, **kwargs):
            if stage == "ema":
                prepared = await original(*args, **kwargs)
            entered.set()
            await release.wait()
            return prepared if stage == "ema" else await original(*args, **kwargs)

        monkeypatch.setattr(bot, "_load_orchestrator_ema_bundle" if stage=="ema" else "_get_orchestrator_market_snapshots", slow)
        task = asyncio.create_task(instance._ordinary_plan())
        instance._ordinary = task
        await asyncio.wait_for(entered.wait(), 5.0)
        clock[0] += 15_000

        async def finish_preparation():
            release.set()
            plan = await asyncio.wait_for(task, 5.0)
            assert plan is not None
            _, creates, _, wave = plan
            symbol = "BTC/USDT:USDT"
            assert wave.permission(symbol, "long")[0] == "normal"
            assert bot.PB_modes["long"][symbol] == "normal"
            assert creates, "real Rust planning should produce ordinary orders for the held position"
            assert instance._account_matches(wave, clock[0])

        def create_count():
            return sum(call["method"] == "create_order" for call in bot.cca.export_request_log())

        creates_before = create_count()
        if handoff == "tail":
            protect = instance.protect

            async def protect_and_finish(**kwargs):
                assert not task.done()
                protective_work = await protect(**kwargs)
                assert create_count() == creates_before
                # Let the real background refresh publish before preparation
                # consumes the cohort it will need at the connector boundary.
                for refresh in (instance._fill_task, instance._source_task):
                    if refresh is not None:
                        await asyncio.wait_for(refresh, 5.0)
                await finish_preparation()
                return protective_work

            with monkeypatch.context() as patch:
                patch.setattr(instance, "protect", protect_and_finish)
                result = await instance.cycle()
        else:
            await finish_preparation()
            if handoff == "expired_account":
                clock[0] += bot._live_market_snapshot_max_age_ms() + 1
                assert instance._account_mismatch_reason(task.result()[3], clock[0]) == (
                    "current_account_inputs_unavailable"
                )
            result = await instance.cycle()
        assert result["ordinary_completed"] and result["ordinary_executed"]
        assert create_count() > creates_before
        completed.append(True)
        return {"ordinary_recovered": True}

    monkeypatch.setattr(runner, "_run_fake_cycle", exercise)
    try:
        args = argparse.Namespace(config=str(config_path), scenario=str(scenario_path), user=user,
            max_steps=1, output_dir=str(tmp_path/"run"), log_level=1, snapshot_each_step=False)
        assert await runner._async_main(args)==0
        assert completed
    finally:
        _cleanup_fake_user_state(user)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["source", "fill"])
@pytest.mark.parametrize("stage", ["before_consumption", "during_trailing"])
async def test_equal_receipt_replacement_tracks_actual_consumption(monkeypatch, kind, stage):
    from dataclasses import replace
    from live.hsl_candles import Sources
    from live.hsl_runtime import observe_fills

    fixture = planning_bot(monkeypatch)
    fixture.owner.sources[SYMBOL] = Sources((), (), 0)
    fixture.bot._hsl_fill_observation = observe_fills(fixture.bot, (NOW, NOW))
    attribute = (
        "refresh_protective_authoritative_state"
        if stage == "before_consumption" else "update_trailing_data"
    )
    original = getattr(fixture.bot, attribute)

    async def replace_receipt(**kwargs):
        result = await original(**kwargs)
        if kind == "source":
            fixture.owner.sources[SYMBOL] = replace(fixture.owner.sources[SYMBOL])
        else:
            fixture.bot._hsl_fill_observation = replace(fixture.bot._hsl_fill_observation)
        return result

    setattr(fixture.bot, attribute, replace_receipt)
    task = asyncio.create_task(fixture.owner._ordinary_plan())
    await asyncio.wait_for(fixture.entered.wait(), 1.0)
    fixture.clock[0] += 15_000
    fixture.release.set()
    plan = await task
    assert (plan is not None) == (stage == "before_consumption")
    if stage == "before_consumption":
        assert fixture.captures[-1].permission(SYMBOL, "long")[0] == "normal"
    else:
        assert not fixture.captures
        assert fixture.defers[-1]["defer_reason"] == f"hsl_{'candle' if kind == 'source' else 'fill'}_receipt_replaced"


@pytest.mark.asyncio
async def test_slow_quote_reacquisition_defers_without_an_account_refresh_loop(monkeypatch):
    fixture = planning_bot(monkeypatch)
    quotes = fixture.bot._get_orchestrator_market_snapshots
    quote_calls = []

    async def slow_second_quote(symbols):
        quote_calls.append(True)
        if len(quote_calls) == 2:
            await asyncio.sleep(0)
            fixture.clock[0] += 11_000
        return await quotes(symbols)

    fixture.bot._get_orchestrator_market_snapshots = slow_second_quote
    task = asyncio.create_task(fixture.owner._ordinary_plan())
    await asyncio.wait_for(fixture.entered.wait(), 1.0)
    fixture.clock[0] += 15_000
    fixture.release.set()
    assert await task is None
    assert len(fixture.calls) == 1 and len(quote_calls) == 2
    assert not fixture.captures
    assert fixture.defers[-1]["defer_reason"] == "current_account_inputs_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("error_kind", ["network", "surface", "invalid_nonce", "malformed"])
async def test_refresh_failure_classification_does_not_weaken_fatal_boundaries(monkeypatch, error_kind):
    from ccxt.base.errors import InvalidNonce, NetworkError
    from live.state_refresh import AuthoritativeSurfaceUnavailable

    fixture = planning_bot(monkeypatch)
    failures = {"network": NetworkError("transport unavailable"),
                "surface": AuthoritativeSurfaceUnavailable("balance", "unavailable"),
                "invalid_nonce": InvalidNonce("time mismatch"),
                "malformed": ValueError("malformed account")}
    failure = failures[error_kind]

    async def fail(**kwargs):
        await asyncio.sleep(0)
        raise failure

    fixture.bot.refresh_protective_authoritative_state = fail
    task = asyncio.create_task(fixture.owner._ordinary_plan())
    await asyncio.wait_for(fixture.entered.wait(), 1.0)
    fixture.clock[0] += 15_000
    fixture.release.set()
    if error_kind in ("network", "surface"):
        assert await task is None
        assert fixture.defers[-1]["defer_reason"] == "current_account_refresh_unavailable"
    else:
        with pytest.raises(type(failure)):
            await task
        assert not fixture.defers
    assert not fixture.captures


@pytest.mark.asyncio
async def test_cancelling_ordinary_lock_wait_does_not_cancel_or_unlock_refresh_holder(monkeypatch):
    fixture = planning_bot(monkeypatch)
    account = fixture.owner.account_facts()
    await fixture.owner._refresh_lock.acquire()
    task = asyncio.create_task(fixture.owner.refresh_ordinary_account(account))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fixture.owner._refresh_lock.locked() and not fixture.calls
    fixture.owner._refresh_lock.release()


@pytest.mark.asyncio
async def test_fresh_unchanged_account_needs_no_additional_fetch(monkeypatch):
    fixture = planning_bot(monkeypatch)
    fixture.release.set()
    plan = await fixture.owner._ordinary_plan()
    assert plan is not None and not fixture.calls
    assert fixture.owner._account_matches(plan[3], fixture.clock[0])


@pytest.mark.asyncio
async def test_ordinary_account_io_failure_does_not_skip_next_protective_pass(monkeypatch):
    from ccxt.base.errors import NetworkError

    fixture = planning_bot(monkeypatch)
    refresh = fixture.bot.refresh_protective_authoritative_state

    async def fail(**kwargs):
        raise NetworkError("ordinary transport unavailable")

    fixture.bot.refresh_protective_authoritative_state = fail
    task = asyncio.create_task(fixture.owner._ordinary_plan())
    await asyncio.wait_for(fixture.entered.wait(), 1.0)
    fixture.clock[0] += 15_000
    fixture.release.set()
    assert await task is None
    fixture.owner._ordinary = task
    fixture.bot.refresh_protective_authoritative_state = refresh
    fixture.bot._begin_live_event_cycle = lambda **kwargs: None
    fixture.owner.poll_inputs = lambda: None
    fixture.owner.schedule_history = fixture.owner.schedule_sources = lambda: None
    protective_passes = []

    async def protect(**kwargs):
        protective_passes.append(True)
        return False

    fixture.owner.protect = protect
    try:
        result = await fixture.owner.cycle()
        assert result["ordinary_completed"] and result["updated"]
        assert protective_passes
    finally:
        fixture.owner.cancel_inputs()
        if fixture.owner._ordinary is not None:
            await asyncio.gather(fixture.owner._ordinary, return_exceptions=True)
