"""Offline production-owner regression for historical flat-boundary read skew."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import hjson
import pytest

from config import prepare_config
from config.hsl_revised import generated_template
from config_utils import load_config
from live import hsl_revised_live
from test_run_fake_live import REPO_ROOT, _cleanup_fake_user_state
import tools.run_fake_live as runner


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
@pytest.mark.parametrize('side', ['long', 'short'])
async def test_completed_loss_does_not_repanic_when_account_refresh_follows_fills(tmp_path, monkeypatch, mode, side):
    user = f'fake_history_timing_{tmp_path.name}'
    _cleanup_fake_user_state(user)
    legacy = prepare_config(load_config(str(REPO_ROOT/'configs/fake_live_hsl_btc.hjson'), verbose=False),
                            target='canonical', runtime=None, verbose=False)
    cfg = generated_template(legacy, mode)
    if side == 'short':
        cfg['bot']['short'] = deepcopy(cfg['bot']['long'])
        cfg['bot']['long']['hsl']['enabled'] = False
        cfg['bot']['long']['risk'].update(n_positions=0, total_wallet_exposure_limit=0.)
    policy = cfg['bot']['hsl'] if mode == 'unified' else cfg['bot'][side]['hsl']
    policy.update(enabled=True, red_threshold=.9, ema_span_minutes=1.,
                  cooldown_minutes_after_red=1., restart_after_red_policy='always',
                  panic_close_order_type='market')
    cfg['live']['pnls_max_lookback_days'] = 1.
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps({k:v for k,v in cfg.items() if not k.startswith('_')}))
    symbol = 'BTC/USDT:USDT'
    scenario = hjson.loads((REPO_ROOT/'scenarios/fake_live/hsl_long_red_restart.hjson').read_text())
    scenario.pop('assertions', None)
    base = datetime(2026, 1, 1, 1, tzinfo=timezone.utc)
    stamp = lambda minutes: (base+timedelta(minutes=minutes)).isoformat().replace('+00:00','Z')
    for i, candle in enumerate(scenario['replay']['symbols'][symbol]['candles']):
        candle[:] = [stamp(i), 100., 100., 100., 100., 10.]
    scenario['account']['balance'] = 1000.
    scenario['account']['positions'] = [dict(symbol=symbol, position_side=side, qty=1., price=100.)]
    entry, close = ('buy','sell') if side == 'long' else ('sell','buy')
    scenario['account']['fills'] = [dict(id=str(i), order=str(i), timestamp=stamp(t), symbol=symbol,
        position_side=side, side=order_side, amount=qty, price=price, pnl=pnl,
        reduceOnly=order_side==close) for i,t,order_side,qty,price,pnl in [
            (10,-9,entry,10.,100.,0.), (11,-7,close,10.,80. if side=='long' else 120.,-200.),
            (12,0,entry,1.,100.,0.)]]
    scenario['run_initial_cycle'] = True
    scenario_path = tmp_path/'scenario.hjson'
    scenario_path.write_text(hjson.dumps(scenario))
    completed = []

    async def exercise(bot):
        import asyncio
        hsl_revised_live.owner(bot).cancel_inputs()
        await asyncio.sleep(.01)
        policy = bot.config['bot']['hsl'] if mode == 'unified' else bot.config['bot'][side]['hsl']
        policy['red_threshold'] = .05
        await bot.update_pnls()
        # Real account reads happen after the same fill receipt. Each fresh owner
        # must still reconstruct the completed cooldown, without a saved RED flag.
        for _ in range(2):
            await bot.refresh_protective_authoritative_state()
            await bot.refresh_protective_authoritative_state()
            instance = hsl_revised_live.Owner(bot)
            bot._hsl_revised_live = instance
            result = await instance.protect()
            assert result is False
            assert not any(c['method']=='create_order' for c in bot.cca.export_request_log())
            assert abs(bot.positions[symbol][side]['size']) == 1.
        # A genuine new loss still reaches the real reduce-only connector.
        bot.cca.get_current_step()['prices'][symbol] = 10. if side=='long' else 190.
        bot.market_snapshot_provider._cache.clear()
        await bot.refresh_protective_authoritative_state()
        bot._hsl_revised_live = hsl_revised_live.Owner(bot)
        await bot._hsl_revised_live.protect()
        await bot.refresh_protective_authoritative_state()
        assert bot.positions[symbol][side]['size'] == 0.
        assert any(f.get('reduceOnly') and str(f['id']) not in {'10','11','12'} for f in bot.cca.fills)
        completed.append(True)
        return {'historical_cooldown_preserved': True}

    monkeypatch.setattr(runner, '_run_fake_cycle', exercise)
    try:
        args = argparse.Namespace(config=str(config_path), scenario=str(scenario_path), user=user,
            max_steps=1, output_dir=str(tmp_path/'output'), log_level=1, snapshot_each_step=False)
        assert await runner._async_main(args) == 0
        assert completed
    finally:
        _cleanup_fake_user_state(user)
