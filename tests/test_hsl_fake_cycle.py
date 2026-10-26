"""The standard offline fake runner exercises the production hsl owner."""

from hsl_fixture import set_risk_memory_spans, mutable_runtime_fixture

import argparse
import json
from copy import deepcopy

import hjson
import pytest

from config import prepare_config
from config.hsl import generated_template
from config_utils import load_config
from test_run_fake_live import REPO_ROOT, _cleanup_fake_user_state
import tools.run_fake_live as runner


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["in_process", "cli"])
@pytest.mark.parametrize("case", ["panic", "entry"])
@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
@pytest.mark.parametrize("side", ["long", "short"])
async def test_standard_fake_runner_hsl_execution_and_trace(
    tmp_path, monkeypatch, mode, side, case, entrypoint
):
    await _exercise_standard_fake_runner(tmp_path, monkeypatch, mode, side, case, entrypoint)


async def _exercise_standard_fake_runner(
    tmp_path, monkeypatch, mode, side, case, entrypoint, *, start_exposed=False
):
    user = f"fake_hsl_cli_{tmp_path.name}"
    _cleanup_fake_user_state(user)
    legacy = prepare_config(
        load_fake_hsl_config(), target="canonical", runtime=None, verbose=False
    )
    cfg = generated_template(mutable_runtime_fixture(legacy), mode)
    if side == "short":
        cfg["bot"]["short"] = deepcopy(cfg["bot"]["long"])
        cfg["bot"]["long"]["hsl"]["enabled"] = False
        cfg["bot"]["long"]["risk"].update(
            max_positions=0, total_wallet_exposure_limit_ratio=0.0
        )
    block = cfg["bot"]["portfolio"]["hsl"] if mode == "portfolio" else cfg["bot"][side]["hsl"]
    block.update(
        enabled=True,
        drawdown_threshold_fraction=0.06,
        drawdown_fraction_ema_span_minutes=1.0,
        panic_close_order_type="market",
    )
    set_risk_memory_spans(cfg, 288.0)
    if case == "entry":
        block["drawdown_threshold_fraction"] = 0.5
        for pside in ("long", "short"):
            cfg["bot"][pside]["unstuck"]["enabled"] = False
            cfg["bot"][pside]["entry_cooldown"]["base_duration_minutes"] = 0.0
            cfg['bot'][pside]['universe']['approved_coins'] = ["BTC"] if pside == side else []
        cfg["bot"]["portfolio"]["loss_gate"]["max_drawdown_fraction"] = 1.0
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")})
    )
    scenario = hjson.loads(
        (REPO_ROOT / "scenarios/fake_live/hsl_long_red_restart.hjson").read_text()
    )
    scenario.pop("assertions", None)
    scenario["account"]["fills"] = []
    scenario["account"]["positions"][0]["position_side"] = side
    if side == "short":
        for candle in scenario["replay"]["symbols"]["BTC/USDT:USDT"]["candles"]:
            o, h, l, c = candle[1:5]
            candle[1:5] = [200 - o, 200 - l, 200 - h, 200 - c]
    if case == "entry":
        scenario["account"]["positions"] = (
            [dict(scenario["account"]["positions"][0], qty=1.0)] if start_exposed else []
        )
        scenario["run_initial_cycle"] = True
    scenario_path = tmp_path / "scenario.hjson"
    scenario_path.write_text(hjson.dumps(scenario))
    output = tmp_path / "output"
    try:
        args = argparse.Namespace(
            config=str(config_path),
            scenario=str(scenario_path),
            user=user,
            max_steps=4,
            output_dir=str(output),
            log_level=1,
            snapshot_each_step=True,
        )
        # No cycle, evaluator, reconciler, or executor replacement.
        if entrypoint == "in_process":
            assert await runner._async_main(args) == 0
        else:
            import os
            import subprocess
            import sys

            # Exercise the unmodified public CLI with the local fake exchange.
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.run_fake_live",
                    str(config_path),
                    str(scenario_path),
                    "--user",
                    user,
                    "--max-steps",
                    "4",
                    "--output-dir",
                    str(output),
                    "--log-level",
                    "1",
                    "--snapshot-each-step",
                ],
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
                capture_output=True,
                text=True,
                timeout=60.0,
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr
        paths = list(output.rglob("hsl_trace.json"))
        assert len(paths) == 1
        artifacts = paths[0].parent
        trace = json.loads(paths[0].read_text())
        assert set(trace) == {"hsl"}
        assert trace["hsl"]["signal_mode"] == mode
        assert trace["hsl"]["scope_count"] >= 1
        fills = json.loads((artifacts / "fills.json").read_text())
        if case == "panic":
            assert any(fill.get("reduceOnly") for fill in fills)
            positions = json.loads((artifacts / "positions.json").read_text())
            assert not positions
        else:
            calls = json.loads((artifacts / "remote_calls.json").read_text())
            assert any(call["method"] == "create_order" for call in calls)
        summaries = json.loads((artifacts / "step_summaries.json").read_text())
        assert len(summaries) == 4
        assert all("'engine': 'hsl'" in step["result"] for step in summaries)
        assert all("passes" in step["result"] for step in summaries)
        if entrypoint == "in_process" and side == "long":
            first = runner._load_run_artifacts(artifacts)
            _cleanup_fake_user_state(user)
            args.output_dir = str(tmp_path / "repeat")
            assert await runner._async_main(args) == 0
            (repeated_trace,) = (tmp_path / "repeat").rglob("hsl_trace.json")
            second = runner._load_run_artifacts(repeated_trace.parent)
            comparison = runner._compare_run_artifacts(first, second)
            assert comparison["match"], json.dumps(comparison["diffs"], indent=2)
    finally:
        _cleanup_fake_user_state(user)


@pytest.mark.asyncio
@pytest.mark.parametrize("start_exposed", [False, True], ids=["market_entry", "typed_debt"])
async def test_fake_ordinary_execution_settles_first_bar(tmp_path, monkeypatch, start_exposed):
    from live.executor import DeferredOrderCreation

    real_cycle = runner._run_fake_cycle
    checked = set()

    async def check_settled_cycle(bot):
        if bot in checked:
            return await real_cycle(bot)
        checked.add(bot)
        actual_results = []
        real_orders = bot.execute_orders

        async def observe_orders(orders):
            result = await real_orders(orders)
            actual_results.extend(result)
            return result

        monkeypatch.setattr(bot, "execute_orders", observe_orders)
        result = await real_cycle(bot)
        assert not result.get("preparation_pending"), result
        if start_exposed:
            assert any(isinstance(item, DeferredOrderCreation) for item in actual_results)
        before = runner._canonicalize_artifact(
            "fake_exchange_state", bot.cca.export_state()
        )
        timestamp = bot.cca.now_ms
        # A fresh normal drain cannot add residual creations after either an
        # initial market fill or the held-position typed-deferral wave.
        await real_cycle(bot)
        assert bot.cca.now_ms == timestamp
        assert runner._canonicalize_artifact(
            "fake_exchange_state", bot.cca.export_state()
        ) == before
        return result

    monkeypatch.setattr(runner, "_run_fake_cycle", check_settled_cycle)
    await _exercise_standard_fake_runner(
        tmp_path, monkeypatch, "portfolio", "long", "entry", "in_process",
        start_exposed=start_exposed,
    )


@pytest.mark.asyncio
async def test_fake_cycle_returns_bounded_pending_while_protection_keeps_running():
    import asyncio
    from types import SimpleNamespace
    from live.hsl_live import owner

    release = asyncio.Event()

    async def refresh(**kwargs):
        return True

    bot = SimpleNamespace(
        config={"live": {}},
        refresh_protective_authoritative_state=refresh,
        _begin_live_event_cycle=lambda **kwargs: None,
        _staged_planner_required_surfaces=lambda **kwargs: set(),
    )
    instance = owner(bot)
    instance.remember_position = lambda: None
    instance.schedule_history = instance.schedule_sources = lambda: None
    instance._ordinary_plan = release.wait
    waves = []

    async def protect(*, deferred_reports):
        assert deferred_reports == []
        waves.append(True)
        return True

    instance.protect = protect
    try:
        result = await asyncio.wait_for(runner._run_fake_cycle_ready(bot), 4.0)
        assert result["preparation_pending"]
        assert result["passes"] == len(waves) == 8
        assert result["protective_work"]
        assert not release.is_set()
        assert not instance._ordinary.done()
    finally:
        release.set()
        instance.cancel_inputs()


def _admission_receipt_bot():
    from types import SimpleNamespace
    from passivbot import Passivbot
    from types import MethodType

    bot = SimpleNamespace(
        config={"live": {}}, create_result=[], cancel_result=[], plan_error=None,
        invalidate_during_plan=False,
        stop_signal_received=False,
        _account_invalidation_generation=0,
        _current_planning_snapshot=SimpleNamespace(account_invalidation_generation=0),
        _authoritative_pending_confirmations={},
        _begin_live_event_cycle=lambda **kwargs: None,
        _staged_planner_required_surfaces=lambda **kwargs: set(),
    )

    async def create(orders):
        return bot.create_result

    async def cancel(orders):
        return bot.cancel_result

    async def execute(cancels, creates, *, configure_creations=True):
        if cancels:
            await bot.execute_cancellations(cancels)
        if creates:
            await bot.execute_orders(creates)
        if bot.plan_error is not None:
            raise bot.plan_error
        if bot.invalidate_during_plan:
            bot._account_invalidation_generation += 1
        return "unchanged plan result"

    bot.execute_orders, bot.execute_cancellations = create, cancel
    bot.execute_order_plan_to_exchange = execute
    bot._shutdown_requested = lambda: bot.stop_signal_received
    ledger = SimpleNamespace(epoch=1, surfaces={surface: SimpleNamespace(epoch=1)
                             for surface in ("balance", "positions", "open_orders", "fills")})
    bot._ensure_freshness_ledger = lambda: ledger
    bot._request_authoritative_confirmation = MethodType(
        Passivbot._request_authoritative_confirmation, bot
    )
    runner._install_fake_ordinary_execution_receipts(bot)
    return bot


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["create", "cancel"])
async def test_fake_admission_receipts_scope_typed_results_and_fresh_empty_plans(kind):
    from live.executor import DeferredOrderCreation, DeferredOrderCancellation

    bot = _admission_receipt_bot()
    result = [DeferredOrderCreation() if kind == "create" else DeferredOrderCancellation()]
    setattr(bot, f"{kind}_result", result)
    cancels, creates = ([{}], []) if kind == "cancel" else ([], [{}])
    assert await bot.execute_order_plan_to_exchange(cancels, creates) == "unchanged plan result"
    assert runner._fake_ordinary_execution_pending(bot)
    # Both successful and deferred protective plans leave ordinary debt alone.
    await bot.execute_order_plan_to_exchange(cancels, creates, configure_creations=False)
    setattr(bot, f"{kind}_result", [{}])
    await bot.execute_order_plan_to_exchange(cancels, creates, configure_creations=False)
    assert runner._fake_ordinary_execution_pending(bot)
    await bot.execute_order_plan_to_exchange([], [])
    assert not runner._fake_ordinary_execution_pending(bot)
    # A result outside an ordinary plan cannot create debt or change transport.
    setattr(bot, f"{kind}_result", result)
    operation = bot.execute_orders if kind == "create" else bot.execute_cancellations
    assert await operation([{}]) is result
    assert not runner._fake_ordinary_execution_pending(bot)
    # Counts do not infer debt from intentional filtering or terminal results.
    setattr(bot, f"{kind}_result", [])
    await bot.execute_order_plan_to_exchange(cancels, creates)
    assert not runner._fake_ordinary_execution_pending(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_fake_admission_receipts_preserve_errors_and_do_not_clear_debt(cancelled, monkeypatch):
    import asyncio
    import contextvars
    from live.executor import DeferredOrderCreation

    make_context = contextvars.ContextVar
    context = make_context("test_ordinary_receipt", default=None)
    monkeypatch.setattr(contextvars, "ContextVar", lambda name, **kwargs:
                        context if name == "fake_ordinary_execution_receipt"
                        else make_context(name, **kwargs))
    bot = _admission_receipt_bot()
    bot.create_result = [DeferredOrderCreation()]
    await bot.execute_order_plan_to_exchange([], [{}])
    bot.create_result = [{}]
    error = asyncio.CancelledError() if cancelled else RuntimeError("plan failure")
    bot.plan_error = error
    with pytest.raises(type(error)) as caught:
        await bot.execute_order_plan_to_exchange([], [{}])
    assert caught.value is error
    assert context.get() is None
    assert runner._fake_ordinary_execution_pending(bot)
    bot.plan_error = None
    await bot.execute_order_plan_to_exchange([], [])
    assert not runner._fake_ordinary_execution_pending(bot)


@pytest.mark.asyncio
async def test_fake_admission_receipts_preserve_debt_on_shutdown():
    from live.executor import DeferredOrderCreation

    bot = _admission_receipt_bot()
    bot.create_result = [DeferredOrderCreation()]
    await bot.execute_order_plan_to_exchange([], [{}])
    bot.stop_signal_received = True
    await bot.execute_order_plan_to_exchange([], [])
    assert runner._fake_ordinary_execution_pending(bot)
    bot.stop_signal_received = False
    await bot.execute_order_plan_to_exchange([], [])
    assert not runner._fake_ordinary_execution_pending(bot)


@pytest.mark.asyncio
async def test_fake_successful_write_requires_replan_after_confirmation_is_consumed():
    bot = _admission_receipt_bot()
    result = [{}]

    async def write(orders):
        bot._request_authoritative_confirmation({"balance", "positions", "open_orders"})
        ledger = bot._ensure_freshness_ledger()
        for surface, minimum in bot._authoritative_pending_confirmations.items():
            ledger.surfaces[surface].epoch = minimum
        return result

    bot.execute_orders = write
    await bot.execute_order_plan_to_exchange([], [{}])
    assert runner._fake_ordinary_execution_pending(bot)
    assert bot._fake_ordinary_execution_receipt.progress
    await bot.execute_order_plan_to_exchange([], [])
    assert not runner._fake_ordinary_execution_pending(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("surfaces,minimum", [(set(), None), ({"fills"}, None), ({"open_orders"}, 1)])
async def test_fake_ineffective_account_confirmation_does_not_create_execution_debt(surfaces, minimum):
    bot = _admission_receipt_bot()

    async def no_write(orders):
        bot._request_authoritative_confirmation(surfaces, min_epoch=minimum)
        return []

    bot.execute_orders = no_write
    await bot.execute_order_plan_to_exchange([], [{}])
    assert not runner._fake_ordinary_execution_pending(bot)
    assert not bot._fake_ordinary_execution_receipt.progress


@pytest.mark.asyncio
async def test_fake_protective_confirmation_cannot_set_or_clear_ordinary_execution_debt():
    bot = _admission_receipt_bot()

    async def write(orders):
        bot._request_authoritative_confirmation({"open_orders"})
        return [{}]

    bot.execute_orders = write
    await bot.execute_order_plan_to_exchange([], [{}], configure_creations=False)
    assert bot._fake_ordinary_execution_receipt is None
    await bot.execute_order_plan_to_exchange([], [{}])
    previous = bot._fake_ordinary_execution_receipt
    await bot.execute_order_plan_to_exchange([], [], configure_creations=False)
    assert bot._fake_ordinary_execution_receipt is previous
    assert runner._fake_ordinary_execution_pending(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["create", "cancel", "confirmation"])
async def test_fake_failed_plan_retains_new_execution_debt(kind):
    from ccxt.base.errors import NetworkError
    from live.executor import DeferredOrderCreation, DeferredOrderCancellation

    bot = _admission_receipt_bot()
    assert not runner._fake_ordinary_execution_pending(bot)
    if kind == "confirmation":
        async def write(orders):
            bot._request_authoritative_confirmation({"open_orders"})
            return [{}]

        bot.execute_orders = write
    else:
        setattr(bot, f"{kind}_result", [
            DeferredOrderCreation() if kind == "create" else DeferredOrderCancellation()
        ])
    error = bot.plan_error = NetworkError("ordinary account unavailable")
    cancels, creates = ([{}], []) if kind == "cancel" else ([], [{}])
    with pytest.raises(NetworkError) as caught:
        await bot.execute_order_plan_to_exchange(cancels, creates)
    assert caught.value is error
    assert runner._fake_ordinary_execution_pending(bot)
    assert bot._fake_ordinary_execution_receipt.progress


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_fake_stale_or_missing_snapshot_cannot_clear_execution_debt(missing):
    from live.executor import DeferredOrderCreation

    bot = _admission_receipt_bot()
    bot.create_result = [DeferredOrderCreation()]
    await bot.execute_order_plan_to_exchange([], [{}])
    if missing:
        bot._current_planning_snapshot = None
        with pytest.raises(AttributeError):
            await bot.execute_order_plan_to_exchange([], [])
    else:
        bot._account_invalidation_generation += 1
        await bot.execute_order_plan_to_exchange([], [])
    assert runner._fake_ordinary_execution_pending(bot)
    assert bot._fake_ordinary_execution_receipt.progress


@pytest.mark.asyncio
async def test_fake_generation_changes_keep_pending_without_earning_execution_budget(monkeypatch):
    from types import SimpleNamespace
    from live import hsl_live

    bot = _admission_receipt_bot()
    bot.invalidate_during_plan = True
    await bot.execute_order_plan_to_exchange([], [{}])
    assert runner._fake_ordinary_execution_pending(bot)
    assert not bot._fake_ordinary_execution_receipt.progress
    bot.invalidate_during_plan = False
    bot._current_planning_snapshot.account_invalidation_generation = 1
    await bot.execute_order_plan_to_exchange([], [])
    assert not runner._fake_ordinary_execution_pending(bot)
    # A later protective/account invalidation supersedes that quiet receipt.
    bot._account_invalidation_generation += 1
    calls = []

    async def cycle():
        calls.append(True)
        return dict(updated=False, ordinary_executed=False)

    instance = SimpleNamespace(cycle=cycle, _ordinary=None, _quote_tasks={},
                               _discard_quote_tasks=set(), required_fill_facts=lambda: None)
    monkeypatch.setattr(hsl_live, "owner", lambda bot: instance)
    result = await runner._run_fake_cycle_ready(bot)
    assert result["ordinary_execution_pending"]
    assert result["passes"] == len(calls) == 8


@pytest.mark.asyncio
async def test_fake_sealed_plan_ignores_late_inherited_confirmation_context():
    import asyncio

    bot = _admission_receipt_bot()
    release = asyncio.Event()
    tasks = []

    async def write(orders):
        async def background():
            await release.wait()
            bot._request_authoritative_confirmation({"open_orders"})

        tasks.append(asyncio.create_task(background()))
        return []

    bot.execute_orders = write
    await bot.execute_order_plan_to_exchange([], [{}])
    completed = bot._fake_ordinary_execution_receipt
    release.set()
    await asyncio.gather(*tasks)
    assert bot._fake_ordinary_execution_receipt is completed
    assert not runner._fake_ordinary_execution_pending(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["create", "cancel", "write"])
async def test_fake_ordinary_execution_has_one_finite_extra_budget(kind, tmp_path, monkeypatch):
    import asyncio
    from live.hsl_live import owner
    from live.executor import DeferredOrderCreation, DeferredOrderCancellation

    bot = _admission_receipt_bot()
    if kind == "write":
        async def write(orders):
            ledger = bot._ensure_freshness_ledger()
            ledger.epoch += 1
            bot._request_authoritative_confirmation({"open_orders"})
            ledger.surfaces["open_orders"].epoch = bot._authoritative_pending_confirmations["open_orders"]
            return [{}]

        bot.execute_orders = write
    else:
        setattr(bot, f"{kind}_result", [
            DeferredOrderCreation() if kind == "create" else DeferredOrderCancellation()
        ])

    async def refresh(**kwargs):
        return True

    bot.refresh_protective_authoritative_state = refresh
    instance = owner(bot)
    instance.remember_position = lambda: None
    instance.schedule_history = instance.schedule_sources = lambda: None
    instance._account_matches = lambda *args: True
    waves = []

    async def protect(**kwargs):
        waves.append(True)
        # Protection's successful empty plan must not erase ordinary debt.
        await bot.execute_order_plan_to_exchange([], [], configure_creations=False)
        return True

    async def ordinary():
        cancels, creates = ([{}], []) if kind == "cancel" else ([], [{}])
        return cancels, creates, bot._current_planning_snapshot, object()

    instance.protect, instance._ordinary_plan = protect, ordinary
    try:
        result = await asyncio.wait_for(runner._run_fake_cycle_ready(bot), 4.0)
        assert result["passes"] == 16
        assert result["ordinary_execution_pending"]
        assert result["preparation_pending"]
        assert len(waves) >= 16
        with pytest.raises(RuntimeError, match="ordinary execution pending"):
            runner._refuse_unsettled_fake_bar(result, [], tmp_path / "snapshots")
        incomplete = json.loads((tmp_path / "incomplete.json").read_text())
        assert incomplete["reason"] == "ordinary_execution_pending"
        assert incomplete["market_clock_advanced_after_pending"] is False
        from exchanges.fake import FakeCCXTClient
        from test_run_fake_live import _scenario

        client = FakeCCXTClient(_scenario(), quote="USDT")
        timestamp = client.now_ms

        async def still_pending(_bot):
            return result

        monkeypatch.setattr(runner, "_run_fake_cycle", still_pending)
        with pytest.raises(RuntimeError, match="ordinary execution pending"):
            await runner._run_fake_bot(bot, client, max_steps=None)
        assert client.current_index == 0 and client.now_ms == timestamp
        # A subsequent drain retains the debt; new deferred receipts cannot
        # reset its finite budget, and a fresh accepted empty plan can clear it.
        async def unaccepted():
            return None

        instance._ordinary_plan = unaccepted
        again = await runner._run_fake_cycle_ready(bot)
        assert again["passes"] == 16 and again["ordinary_execution_pending"]
        assert runner._fake_ordinary_execution_pending(bot)
        await bot.execute_order_plan_to_exchange([], [])
        assert not runner._fake_ordinary_execution_pending(bot)
    finally:
        instance.cancel_inputs()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_fake_admission_debt_blocks_unavailable_account_early_return(monkeypatch):
    from types import SimpleNamespace
    from live import hsl_live
    from live.executor import DeferredOrderCreation

    bot = _admission_receipt_bot()
    bot.create_result = [DeferredOrderCreation()]
    await bot.execute_order_plan_to_exchange([], [{}])
    calls = []

    async def cycle():
        calls.append(True)
        return dict(updated=False, ordinary_executed=False)

    instance = SimpleNamespace(cycle=cycle, _ordinary=None, _quote_tasks={},
                               _discard_quote_tasks=set(), required_fill_facts=lambda: None)
    monkeypatch.setattr(hsl_live, "owner", lambda bot: instance)
    result = await runner._run_fake_cycle_ready(bot)
    assert result["ordinary_execution_pending"]
    assert result["passes"] == len(calls) == 16


@pytest.mark.asyncio
async def test_fake_protective_only_deferrals_keep_original_preparation_budget():
    import asyncio
    from live.hsl_live import owner
    from live.executor import DeferredOrderCreation

    bot = _admission_receipt_bot()
    bot.create_result = [DeferredOrderCreation()]

    async def refresh(**kwargs):
        return True

    bot.refresh_protective_authoritative_state = refresh
    instance = owner(bot)
    instance.remember_position = lambda: None
    instance.schedule_history = instance.schedule_sources = lambda: None
    release = asyncio.Event()
    instance._ordinary_plan = release.wait
    waves = []

    async def protect(**kwargs):
        waves.append(True)
        await bot.execute_order_plan_to_exchange([], [{}], configure_creations=False)
        return True

    instance.protect = protect
    try:
        result = await asyncio.wait_for(runner._run_fake_cycle_ready(bot), 4.0)
        assert result["passes"] == len(waves) == 8
        assert not result["ordinary_execution_pending"]
    finally:
        release.set()
        instance.cancel_inputs()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_fake_protective_write_confirmation_survives_failed_account_refresh():
    import asyncio
    from live.hsl_live import owner

    bot = _admission_receipt_bot()
    refreshed, protected = [], []

    async def refresh(**kwargs):
        refreshed.append(True)
        return False

    async def write(orders):
        bot._request_authoritative_confirmation({"balance", "positions", "open_orders"})
        return [{}]

    async def protect(**kwargs):
        protected.append(True)
        await bot.execute_order_plan_to_exchange([], [{}], configure_creations=False)
        return True

    async def prepared():
        return [], [], bot._current_planning_snapshot, object()

    bot.execute_orders = write
    bot.refresh_protective_authoritative_state = refresh
    instance = owner(bot)
    instance.protect = protect
    instance._account_matches = lambda *args: False
    instance._ordinary = asyncio.create_task(prepared())
    await instance._ordinary
    try:
        result = await runner._run_fake_cycle_ready(bot)
        assert result["passes"] == len(refreshed) == 8
        assert len(protected) == 1  # No fresh account facts permit another write.
        assert result["confirmation_pending"]
        assert not result["ordinary_execution_pending"]
        assert bot._fake_ordinary_execution_receipt is None
        with pytest.raises(RuntimeError, match="confirmation pending"):
            runner._refuse_unsettled_fake_bar(result, [], None)
    finally:
        instance.cancel_inputs()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_production_cycle_rejects_overlapping_owner_calls_and_releases_after_failure():
    import asyncio
    from types import SimpleNamespace
    from live.hsl_live import Owner

    entered, release = asyncio.Event(), asyncio.Event()

    async def refresh(**kwargs):
        entered.set()
        await release.wait()
        raise ValueError("malformed account producer")

    instance = Owner(
        SimpleNamespace(
            refresh_protective_authoritative_state=refresh,
            _begin_live_event_cycle=lambda **kwargs: None,
        )
    )
    task = asyncio.create_task(instance.cycle())
    await entered.wait()
    with pytest.raises(RuntimeError, match="cycle is already running"):
        await instance.cycle()
    release.set()
    with pytest.raises(ValueError, match="malformed account producer"):
        await task
    assert not instance._running and not instance._cycle_running


@pytest.mark.asyncio
async def test_each_owner_pass_recovers_prior_cycle_state_before_account_refresh():
    from types import SimpleNamespace
    from live.hsl_live import Owner

    cycles, refreshed = [], []
    bot = SimpleNamespace(
        execution_scheduled=True,
        state_change_detected_by_symbol={"TEST/USDT:USDT"},
        _live_event_current_cycle_id="previous",
    )

    def begin(**kwargs):
        assert kwargs["loop_start_ms"] > 0
        cycles.append(len(cycles) + 1)
        bot._live_event_current_cycle_id = cycles[-1]

    async def refresh(**kwargs):
        assert bot.state_change_detected_by_symbol == set()
        assert bot.execution_scheduled is False
        assert bot._live_event_current_cycle_id == len(cycles)
        refreshed.append(True)
        # Simulate the preceding pass's failed/ambiguous cancellation and
        # scheduled work. The next pass must not inherit its execution embargo.
        bot.state_change_detected_by_symbol.add("TEST/USDT:USDT")
        bot.execution_scheduled = True
        return False

    bot._begin_live_event_cycle = begin
    bot.refresh_protective_authoritative_state = refresh
    owner = Owner(bot)
    for _ in range(2):
        assert (await owner.cycle())["updated"] is False
    assert len(cycles) == len(refreshed) == 2


def test_hsl_trace_comparison_preserves_trading_evidence():
    first = dict(
        hsl=dict(
            engine="hsl",
            captured_at_ms=100,
            input_expires_at_ms=110,
            age_ms=1,
            observation_status="current",
            scopes=[dict(action="panic", red_at=500, flat_at=None)],
            counts={"red": 1},
        )
    )
    second = deepcopy(first)
    second["hsl"].update(captured_at_ms=200, input_expires_at_ms=210, age_ms=2)
    assert runner._canonical_hsl_trace(first) == runner._canonical_hsl_trace(second)
    assert "captured_at_ms" in first["hsl"]  # Comparison does not rewrite artifacts.
    for key, changed in [("action", "normal"), ("red_at", 600)]:
        altered = deepcopy(second)
        altered["hsl"]["scopes"][0][key] = changed
        assert runner._canonical_hsl_trace(first) != runner._canonical_hsl_trace(
            altered
        )


@pytest.mark.asyncio
async def test_ready_ordinary_plan_is_serviced_before_balance_refresh_with_protection_first():
    import asyncio
    from types import SimpleNamespace
    from live.hsl_live import Owner

    events = []
    bot = SimpleNamespace(_begin_live_event_cycle=lambda **kwargs: None, raw=1000.0)

    async def refresh(**kwargs):
        events.append("refresh")
        bot.raw += 0.01
        return True

    bot.refresh_protective_authoritative_state = refresh
    instance = Owner(bot)
    instance._account_matches = lambda *args: True
    instance.remember_position = lambda: None
    instance.schedule_history = instance.schedule_sources = lambda: None

    async def protect(**kwargs):
        events.append("protect")
        return False

    instance.protect = protect

    async def prepared():
        return [], [dict(raw_at_rust=bot.raw)], object(), object()

    async def execute(cancels, creates):
        events.append("write")
        assert creates[0]["raw_at_rust"] == bot.raw

    bot.execute_order_plan_to_exchange = execute
    instance._ordinary_plan = prepared
    instance._ordinary = asyncio.create_task(prepared())
    await instance._ordinary
    try:
        result = await instance.cycle()
        assert result["ordinary_executed"]
        assert events == ["protect", "write", "refresh", "protect"]
    finally:
        instance.cancel_inputs()
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("finishes_during_cycle", [False, True])
async def test_fake_cycle_settles_current_source_before_advancing_scenario(
    monkeypatch, finishes_during_cycle
):
    import asyncio
    from types import SimpleNamespace
    from live import hsl_live

    reads = asyncio.create_task(asyncio.sleep(0.01))
    seen = []

    async def cycle():
        seen.append(reads.done())
        if finishes_during_cycle:
            await reads
        return dict(updated=True, ordinary_executed=True)

    instance = SimpleNamespace(cycle=cycle, _ordinary=None, _source_task=reads,
                               _quote_tasks={}, _discard_quote_tasks=set(),
                               required_fill_facts=lambda: None)
    monkeypatch.setattr(hsl_live, "owner", lambda bot: instance)
    bot = SimpleNamespace(config={"live": {}})
    result = await runner._run_fake_cycle_ready(bot)
    assert seen == [False, True]
    assert result["ordinary_executed"] and result["passes"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("updated", [False, True])
async def test_fake_cycle_raises_read_failure_that_finishes_during_final_pass(
    monkeypatch, updated
):
    import asyncio
    from types import SimpleNamespace
    from live import hsl_live

    async def read():
        await asyncio.sleep(0)
        raise RuntimeError("unexpected history failure")

    reads = asyncio.create_task(read())

    async def cycle():
        await asyncio.sleep(0.01)
        return dict(updated=updated, ordinary_executed=True)

    instance = SimpleNamespace(cycle=cycle, _ordinary=None, _fill_task=reads,
                               _quote_tasks={}, _discard_quote_tasks=set())
    monkeypatch.setattr(hsl_live, "owner", lambda bot: instance)
    bot = SimpleNamespace(config={"live": {}})
    with pytest.raises(RuntimeError, match="unexpected history failure"):
        await runner._run_fake_cycle_ready(bot)


@pytest.mark.asyncio
async def test_fake_cycle_raises_read_failure_in_last_bounded_wait(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from live import hsl_live

    calls = 0

    async def read():
        raise RuntimeError("last wait history failure")

    async def cycle():
        nonlocal calls
        calls += 1
        if calls == 8:
            instance._fill_task = asyncio.create_task(read())
        return dict(updated=True, ordinary_executed=False)

    instance = SimpleNamespace(cycle=cycle, _ordinary=None,
                               _quote_tasks={}, _discard_quote_tasks=set(),
                               required_fill_facts=lambda: None)
    monkeypatch.setattr(hsl_live, "owner", lambda bot: instance)
    bot = SimpleNamespace(config={"live": {}})
    with pytest.raises(RuntimeError, match="last wait history failure"):
        await runner._run_fake_cycle_ready(bot)
    assert calls == 8


from hsl_fixture import load_fake_hsl_config


@pytest.mark.asyncio
async def test_account_reporting_after_protection_cannot_block_following_passes(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from live import event_emitters
    from live.hsl_live import Owner
    from live.state_refresh import queue_protective_account_report

    entered, release = asyncio.Event(), asyncio.Event()
    events = []
    bot = SimpleNamespace(
        stop_signal_received=False, _begin_live_event_cycle=lambda **kwargs: None
    )

    async def positions(old, new):
        events.append("position-report")
        entered.set()
        await release.wait()

    def balance(current_bot):
        assert current_bot is bot
        events.append("balance-report")

    publisher = Mock(side_effect=balance)
    monkeypatch.setattr(event_emitters, "publish_committed_balance_observation", publisher)
    bot.log_position_changes = positions
    bot.handle_balance_update = AsyncMock(side_effect=AssertionError("legacy balance callback forbidden"))

    async def refresh(**kwargs):
        queue_protective_account_report(bot, [], [], require_balance=True)
        return True

    bot.refresh_protective_authoritative_state = refresh
    instance = Owner(bot)
    instance.remember_position = instance.schedule_history = (
        instance.schedule_sources
    ) = lambda: None

    async def protect(**kwargs):
        events.append("protect")
        return False

    instance.protect = protect

    async def ordinary():
        await release.wait()

    instance._ordinary_plan = ordinary
    try:
        assert (await instance.cycle())["updated"]
        await asyncio.wait_for(entered.wait(), 1.0)
        assert events[0] == "protect"
        assert (await asyncio.wait_for(instance.cycle(), 1.0))["updated"]
        assert events.count("protect") == 2
        assert events.count("position-report") == 1
        release.set()
        await instance._account_report
        assert events[-1] == "balance-report"
        publisher.assert_called_once_with(bot)
        bot.handle_balance_update.assert_not_awaited()
    finally:
        release.set()
        instance.cancel_inputs()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_protective_reporting_coalesces_immutable_receipts_and_survives_hook_error(
    caplog, monkeypatch,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from live import event_emitters
    from live.state_refresh import (
        queue_protective_account_report,
        publish_protective_account_report,
    )

    bot = SimpleNamespace(stop_signal_received=False)
    bot.log_position_changes = AsyncMock(
        side_effect=RuntimeError("private report details")
    )
    bot.handle_balance_update = AsyncMock(side_effect=AssertionError("legacy balance callback forbidden"))
    publisher = Mock()
    monkeypatch.setattr(event_emitters, "publish_committed_balance_observation", publisher)
    initial, middle, latest = [{"size": 0}], [{"size": 1}], [{"size": 2}]
    queue_protective_account_report(bot, initial, middle, require_balance=True)
    queue_protective_account_report(bot, middle, latest, require_balance=False)
    initial[0]["size"] = middle[0]["size"] = latest[0]["size"] = 99
    await publish_protective_account_report(bot)
    bot.log_position_changes.assert_awaited_once_with([{"size": 0}], [{"size": 2}])
    publisher.assert_called_once_with(bot)
    bot.handle_balance_update.assert_not_awaited()
    assert bot._protective_account_report is None
    assert "RuntimeError" in caplog.text and "private report details" not in caplog.text


@pytest.mark.asyncio
async def test_shutdown_cancels_owned_account_reporting():
    import asyncio
    from types import SimpleNamespace
    from live.hsl_live import Owner
    from live.state_refresh import queue_protective_account_report

    entered = asyncio.Event()

    async def stalled(*args):
        entered.set()
        await asyncio.Event().wait()

    bot = SimpleNamespace(stop_signal_received=False, log_position_changes=stalled)
    queue_protective_account_report(bot, [], [], require_balance=False)
    instance = Owner(bot)
    instance.schedule_account_reporting()
    await asyncio.wait_for(entered.wait(), 1.0)
    bot.stop_signal_received = True
    instance.cancel_inputs()
    await asyncio.gather(instance._account_report, return_exceptions=True)
    assert instance._account_report.cancelled()
