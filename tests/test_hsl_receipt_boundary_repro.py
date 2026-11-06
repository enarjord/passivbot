"""Genuine fill renewal before consumption must not starve ordinary planning."""
import asyncio
import pytest
from live import hsl_live

@pytest.mark.asyncio
@pytest.mark.parametrize("renew_fill_receipt", [True, False])
async def test_renewal_before_consumption_does_not_starve_ordinary_plans(tmp_path, monkeypatch, renew_fill_receipt):
    stage = "ema"
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
        entered, release = asyncio.Event(), asyncio.Event()
        original = getattr(bot, "_load_orchestrator_ema_bundle" if stage=="ema" else "_get_orchestrator_market_snapshots")

        async def slow(*args, **kwargs):
            if stage == "ema":
                prepared = await original(*args, **kwargs)
            entered.set()
            await release.wait()
            return prepared if stage == "ema" else await original(*args, **kwargs)

        monkeypatch.setattr(bot, "_load_orchestrator_ema_bundle" if stage=="ema" else "_get_orchestrator_market_snapshots", slow)
        # Keep background ownership out of this experiment; the genuine fill
        # acquisition is explicitly scheduled at its real five-second cadence.
        instance.schedule_history = instance.schedule_sources = lambda: None
        preparations = []
        actual_calculate = bot.calc_orders_to_cancel_and_create

        async def calculate():
            result = await actual_calculate()
            preparations.append(bot._hsl_planning_wave)
            return result

        monkeypatch.setattr(bot, "calc_orders_to_cancel_and_create", calculate)
        outcomes = []
        factual_fetches = []
        original_fetch = bot.cca.get_fill_events

        def fetch(*args, **kwargs):
            factual_fetches.append((args, kwargs))
            return original_fetch(*args, **kwargs)

        monkeypatch.setattr(bot.cca, "get_fill_events", fetch)
        for attempt in range(3):
            entered.clear()
            release.clear()
            # Start each preparation from current factual account state.
            await bot.refresh_protective_authoritative_state(require_balance=True)
            before = getattr(bot, "_hsl_fill_observation", None)
            assert before is not None
            before_required = instance.required_fill_facts()
            task = asyncio.create_task(instance._ordinary_plan())
            await asyncio.wait_for(entered.wait(), 5.0)
            clock[0] += 5_000
            request_count = len(factual_fetches)
            if renew_fill_receipt:
                await instance._refresh_history({})
            after = bot._hsl_fill_observation
            assert (after is not before) == renew_fill_receipt
            assert after.tape == before.tape
            assert instance.required_fill_facts() == before_required
            assert (len(factual_fetches) > request_count) == renew_fill_receipt
            # Account remains legitimately fresh for this bounded eight-second
            # preparation; no timestamp or identity guard is modified.
            clock[0] += 3_000
            release.set()
            plan = await asyncio.wait_for(task, 5.0)
            assert plan is not None
            if attempt == 0:
                assert plan[1], "the first real native plan has ordinary orders"
            assert preparations[-1].permission("BTC/USDT:USDT", "long")[0] == "normal"
            assert bot.PB_modes["long"]["BTC/USDT:USDT"] == "normal"
            instance._ordinary = task
            writes_before = len(bot.cca.export_request_log())
            result = await instance.cycle()
            if attempt == 0:
                assert any(call["method"] == "create_order" for call in
                           bot.cca.export_request_log()[writes_before:])
            assert result["ordinary_completed"]
            assert result["ordinary_executed"]
            outcomes.append(dict(attempt=attempt + 1, native_permission="normal",
                                 new_factual_fill_receipt=renew_fill_receipt, equal_tape=True,
                                 same_required_fill_facts=True,
                                 ordinary_executed=result["ordinary_executed"]))
            # The owner schedules its next independent task at cycle end.
            pending, instance._ordinary = instance._ordinary, None
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        completed.append(True)
        return {"ordinary_receipt_renewal_progress": True}

    monkeypatch.setattr(runner, "_run_fake_cycle", exercise)
    try:
        args = argparse.Namespace(config=str(config_path), scenario=str(scenario_path), user=user,
            max_steps=1, output_dir=str(tmp_path/"run"), log_level=1, snapshot_each_step=False)
        assert await runner._async_main(args)==0
        assert completed
    finally:
        _cleanup_fake_user_state(user)
