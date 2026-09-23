"""Adversarial observation timing through the offline fake connector and Rust HSL."""

import argparse
import asyncio
from copy import deepcopy
import json
import hjson
import pytest
from config import prepare_config
from config.hsl_revised import generated_template
from config_utils import load_config
from live import hsl_revised_live, position_fill_sync
from test_run_fake_live import REPO_ROOT, _cleanup_fake_user_state
import tools.run_fake_live as runner


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize(
    "failure",
    [
        "never_returns",
        "network_error",
        "missing_forever",
        "continuous_changes",
        "change_during_fetch",
        "fetch_skipped",
    ],
)
async def test_pending_history_cannot_starve_real_protective_execution(
    tmp_path, monkeypatch, side, failure
):
    user = f"fake_sync_{tmp_path.name}"
    _cleanup_fake_user_state(user)
    cfg = generated_template(
        prepare_config(
            load_config(
                str(REPO_ROOT / "configs/fake_live_hsl_btc.hjson"), verbose=False
            ),
            target="canonical",
            runtime=None,
            verbose=False,
        ),
        "coin",
    )
    if side == "short":
        cfg["bot"]["short"] = deepcopy(cfg["bot"]["long"])
        cfg["bot"]["long"]["hsl"]["enabled"] = False
        cfg["bot"]["long"]["risk"].update(
            n_positions=0, total_wallet_exposure_limit=0.0
        )
    cfg["bot"][side]["hsl"].update(
        enabled=True,
        red_threshold=0.9,
        ema_span_minutes=356.0,
        panic_close_order_type="market",
        restart_after_red_policy="always",
    )
    cfg["live"]["pnls_max_lookback_days"] = 1.0
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")})
    )
    symbol = "BTC/USDT:USDT"
    scenario = hjson.loads(
        (REPO_ROOT / "scenarios/fake_live/hsl_long_red_restart.hjson").read_text()
    )
    scenario.pop("assertions", None)
    scenario["account"]["balance"] = 1000.0
    scenario["account"]["positions"] = [
        dict(symbol=symbol, position_side=side, qty=1.0, price=100.0)
    ]
    scenario["account"]["fills"] = []  # permanently unavailable opening execution
    scenario["run_initial_cycle"] = True
    for candle in scenario["replay"]["symbols"][symbol]["candles"]:
        candle[1:5] = [100.0, 100.0, 100.0, 100.0]
    scenario_path = tmp_path / "scenario.hjson"
    scenario_path.write_text(hjson.dumps(scenario))
    completed = []

    async def exercise(bot):
        old = hsl_revised_live.owner(bot)
        old.cancel_inputs()
        await asyncio.sleep(0.01)
        clock = [0.0]
        sync = position_fill_sync.state(bot)
        sync.clock = lambda: clock[0]
        # Real fake exchange changes position before any corresponding fill exists.
        bot.cca.positions[(symbol, side)]["size"] = 2.0
        await bot.refresh_protective_authoritative_state(require_balance=True)
        assert sync.blocked((symbol, side))
        bot.config["bot"][side]["hsl"]["red_threshold"] = 0.03
        bot.cca.get_current_step()["prices"][symbol] = 10.0 if side == "long" else 190.0
        bot.market_snapshot_provider._cache.clear()
        owner = bot._hsl_revised_live = hsl_revised_live.Owner(bot)
        await bot.refresh_protective_authoritative_state(require_balance=True)
        before = len(
            [c for c in bot.cca.export_request_log() if c["method"] == "create_order"]
        )
        release = asyncio.Event()
        original = bot._pnls_manager.refresh
        attempts = []

        async def fault(**kwargs):
            attempts.append(clock[0])
            if failure == "never_returns":
                await release.wait()
            elif failure in ("network_error", "continuous_changes"):
                from ccxt.base.errors import NetworkError

                raise NetworkError("synthetic history outage")
            elif failure == "fetch_skipped":
                return False
            elif failure == "change_during_fetch":
                clock[0] = 6.0
                bot.cca.positions[(symbol, side)]["size"] += 0.1
                await bot.refresh_protective_authoritative_state(require_balance=True)
            return await original(**kwargs)

        monkeypatch.setattr(bot._pnls_manager, "refresh", fault)
        task = None
        try:
            # A pre-settle call must not even initiate connector history I/O.
            for t in (0.0, 2.0, 4.999):
                clock[0] = t
                assert await bot.update_pnls(since_ms=0) is False
                assert not attempts
                # Ordinary batch creates and cancels obey the same gate, even
                # before a revised wave has been prepared for connector writes.
                order = dict(
                    symbol=symbol,
                    position_side=side,
                    qty=1.0,
                    price=100.0,
                    side="sell" if side == "long" else "buy",
                    reduce_only=True,
                )
                assert await bot.execute_orders_parent([order]) == []
                assert await bot.execute_cancellations_parent([order]) == []
                await owner.protect()
                assert (
                    len(
                        [
                            c
                            for c in bot.cca.export_request_log()
                            if c["method"] == "create_order"
                        ]
                    )
                    == before
                )
            clock[0] = 5.0
            task = asyncio.create_task(owner._refresh_history({"since_ms": 0}))
            if failure == "never_returns":
                for _ in range(100):
                    if attempts:
                        break
                    await asyncio.sleep(0.001)
                assert attempts == [5.0]
            else:
                await asyncio.wait_for(task, 3.0)
            if failure == "fetch_skipped":
                assert bot._hsl_revised_fill_capture_interval is None
                assert bot._last_fill_refresh_block_reason == "fill_refresh_skipped"
            if failure == "missing_forever":
                # Successful late fetch with no opening fill releases immediately;
                # observed current loss is still protected by Rust.
                assert not sync.blocked((symbol, side))
            else:
                for t in (6.0, 10.0, 14.999):
                    clock[0] = t
                    if failure == "continuous_changes":
                        bot.cca.positions[(symbol, side)]["size"] += 0.1
                    await bot.refresh_protective_authoritative_state(
                        require_balance=True
                    )
                    await owner.protect()
                    assert (
                        len(
                            [
                                c
                                for c in bot.cca.export_request_log()
                                if c["method"] == "create_order"
                            ]
                        )
                        == before
                    )
                clock[0] = 15.0
            await bot.refresh_protective_authoritative_state(require_balance=True)
            await asyncio.wait_for(owner.protect(), 3.0)
            assert (
                len(
                    [
                        c
                        for c in bot.cca.export_request_log()
                        if c["method"] == "create_order"
                    ]
                )
                > before
            )
            await bot.refresh_protective_authoritative_state(require_balance=True)
            assert bot.positions[symbol][side]["size"] == 0.0
            assert any(f.get("reduceOnly") for f in bot.cca.fills)
            completed.append(True)
        finally:
            release.set()
            if task is not None:
                await asyncio.wait_for(task, 3.0)
            owner.cancel_inputs()
        return {"bounded_sync_released": True}

    monkeypatch.setattr(runner, "_run_fake_cycle", exercise)
    try:
        args = argparse.Namespace(
            config=str(config),
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
