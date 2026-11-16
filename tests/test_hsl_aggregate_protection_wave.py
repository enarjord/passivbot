"""Exercise aggregate RED through the real Rust planner and offline fake connector."""

import argparse
import asyncio
from copy import deepcopy
import json

import hjson
import passivbot_rust
import pytest

from config import prepare_config
from config.hsl import generated_template
from hsl_fixture import load_fake_hsl_config, mutable_runtime_fixture, set_risk_memory_spans
from live import hsl_live, position_fill_sync
from test_run_fake_live import REPO_ROOT, _cleanup_fake_user_state
from utils import utc_ms
import tools.run_fake_live as runner


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["portfolio", "position_side"])
@pytest.mark.parametrize(
    "interrupt", [None, "fill_hint", "ambiguous_cancel", "cancel_capacity"]
)
async def test_aggregate_red_flattens_before_next_account_refresh(
    tmp_path, monkeypatch, mode, interrupt
):
    assert not getattr(passivbot_rust, "__is_stub__", False)
    user = f"fake_hsl_aggregate_{tmp_path.name}"
    _cleanup_fake_user_state(user)
    canonical = prepare_config(
        load_fake_hsl_config(), target="canonical", runtime=None, verbose=False
    )
    config = generated_template(mutable_runtime_fixture(canonical), mode)
    policy_group = "portfolio" if mode == "portfolio" else "long"
    config["bot"][policy_group]["hsl"].update(
        enabled=True,
        drawdown_threshold_fraction=0.95,
        drawdown_fraction_ema_span_minutes=1.0,
        panic_close_order_type="market",
    )
    set_risk_memory_spans(config, 288.0)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({k: v for k, v in config.items() if not k.startswith("_")})
    )

    first, second = "BTC/USDT:USDT", "ETH/USDT:USDT"
    symbols = (first, second)
    scenario = hjson.loads(
        (REPO_ROOT / "scenarios/fake_live/hsl_long_red_restart.hjson").read_text()
    )
    scenario.pop("assertions", None)
    scenario["run_initial_cycle"] = True
    scenario["account"]["fills"] = []
    scenario["account"]["balance"] = 1000.0
    scenario["symbols"][second] = deepcopy(scenario["symbols"][first])
    scenario["replay"]["symbols"][second] = deepcopy(scenario["replay"]["symbols"][first])
    scenario["account"]["positions"] = [
        dict(symbol=symbol, position_side="long", qty=5.0, price=100.0)
        for symbol in symbols
    ]
    scenario["account"]["open_orders"] = [
        dict(
            id=f"entry-{i}",
            symbol=symbol,
            position_side="long",
            side="buy",
            amount=1.0,
            price=65.0,
        )
        for i, symbol in enumerate(symbols)
    ]
    scenario_path = tmp_path / "scenario.hjson"
    scenario_path.write_text(hjson.dumps(scenario))
    completed = []

    async def exercise(bot):
        old_owner = hsl_live.owner(bot)
        old_owner.cancel_inputs()
        await asyncio.sleep(0)
        owner = bot._hsl_live = hsl_live.Owner(bot)
        # Neither an opening-fill history nor an in-flight history task should
        # become a prerequisite while current account facts still support RED.
        bot._pnls_manager = None
        first_history_read = asyncio.Event()
        release_history = asyncio.Event()

        async def stalled_history():
            first_history_read.set()
            await release_history.wait()

        owner._fill_task = asyncio.create_task(stalled_history())
        try:
            await first_history_read.wait()
            clock = [0.0]
            sync = position_fill_sync.state(bot)
            sync.clock = lambda: clock[0]
            await bot.refresh_protective_authoritative_state(require_balance=True)
            bot.cca.now_ms += 60_000
            for index, symbol in enumerate(symbols):
                clock[0] = float(index * 4)
                bot.cca.now_ms += index * 4_000
                bot.cca._apply_manual_fill(
                    dict(
                        symbol=symbol, position_side="long", side="buy",
                        qty=1.0, price=100.0,
                    )
                )
                await bot.refresh_protective_authoritative_state(require_balance=True)

            # Discovery of the second external fill invalidates the account
            # read that observed it. Confirm the same complete facts again.
            assert await bot.refresh_protective_authoritative_state(require_balance=True)
            bot.config["bot"][policy_group]["hsl"]["drawdown_threshold_fraction"] = 0.06
            bot.cca.get_current_step()["prices"].update(
                {symbol: 80.0 for symbol in symbols}
            )
            bot.market_snapshot_provider._cache.clear()
            assert all(sync.blocked((symbol, "long")) for symbol in symbols)

            before = len(bot.cca.export_request_log())
            assert not owner._current_account_unavailable(int(utc_ms()))
            original_create = bot.cca.create_order
            original_cancel = bot.cca.cancel_order
            create_attempts = 0

            async def create_with_fill_hint(*args, **kwargs):
                nonlocal create_attempts
                create_attempts += 1
                if create_attempts == 2:
                    # A private event reached the owner while the next request
                    # was queued at the connector, after initial admission.
                    await asyncio.sleep(0)
                    bot._account_invalidation_generation += 1
                return await original_create(*args, **kwargs)

            async def cancel_with_uncertain_ack(*args, **kwargs):
                await original_cancel(*args, **kwargs)
                raise RuntimeError("order already filled")

            if interrupt == "fill_hint":
                monkeypatch.setattr(bot.cca, "create_order", create_with_fill_hint)
            elif interrupt == "ambiguous_cancel":
                monkeypatch.setattr(bot.cca, "cancel_order", cancel_with_uncertain_ack)
            elif interrupt == "cancel_capacity":
                live_value = bot.live_value
                monkeypatch.setattr(
                    bot, "live_value",
                    lambda key: 1 if key == "max_n_cancellations_per_batch" else live_value(key),
                )
            reports = []
            assert await asyncio.wait_for(
                owner.protect(deferred_reports=reports), 3.0
            )
            assert reports[0].permission(first, "long") == ("panic", "market")
            wave_requests = bot.cca.export_request_log()[before:]
            assert not any(
                request["method"] in {"fetch_balance", "fetch_positions", "fetch_open_orders"}
                for request in wave_requests
            )
            writes = [
                request
                for request in wave_requests
                if request["method"] in {"create_order", "cancel_order"}
            ]
            assert set(bot._authoritative_pending_confirmations) >= {
                "balance", "positions", "open_orders"
            }
            if interrupt in {"ambiguous_cancel", "cancel_capacity"}:
                assert [write["method"] for write in writes] == ["cancel_order"]
                assert len(bot.cca.export_positions()) == 2
                assert len(bot.cca.open_orders) == 1
                completed.append(True)
                if interrupt == "ambiguous_cancel":
                    return {"ambiguous_cancel_requires_confirmation": True}
                assert await bot.refresh_protective_authoritative_state(require_balance=True)
                assert await owner.protect(deferred_reports=reports)
                assert not bot.cca.open_orders
                assert bot.cca.export_positions() == []
                return {"panic_waited_for_resting_entry_cancel": True}
            if interrupt == "fill_hint":
                assert create_attempts == 2
                assert [write["method"] for write in writes] == [
                    "cancel_order", "cancel_order", "create_order"
                ]
                assert len(bot.cca.export_positions()) == 1
                assert not bot.cca.open_orders
                completed.append(True)
                return {"queued_write_rechecks_account": True}
            assert [write["method"] for write in writes] == [
                "cancel_order", "cancel_order", "create_order", "create_order"
            ]
            creates = writes[2:]
            assert {create["symbol"] for create in creates} == set(symbols)
            assert all(
                create["order_type"] == "market" and create["reduce_only"]
                for create in creates
            )
            assert {write["order_id"] for write in writes[:2]} == {"entry-0", "entry-1"}
            assert bot.cca.export_positions() == []
            assert not bot.cca.open_orders
            assert not owner._fill_task.done()
            assert all(sync.blocked((symbol, "long")) for symbol in symbols)
            completed.append(True)
            return {"aggregate_protection_wave_completed": True}
        finally:
            release_history.set()
            await owner._fill_task
            owner.cancel_inputs()

    monkeypatch.setattr(runner, "_run_fake_cycle", exercise)
    try:
        args = argparse.Namespace(
            config=str(config_path),
            scenario=str(scenario_path),
            user=user,
            max_steps=1,
            output_dir=str(tmp_path / "output"),
            log_level=1,
            snapshot_each_step=False,
        )
        assert await runner._async_main(args) == 0
        assert completed
    finally:
        _cleanup_fake_user_state(user)
