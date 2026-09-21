"""The standard offline fake runner exercises the production revised owner."""
import argparse
import json
from copy import deepcopy

import hjson
import pytest

from config import prepare_config
from config.hsl_revised import generated_template
from config_utils import load_config
from test_run_fake_live import REPO_ROOT, _cleanup_fake_user_state
import tools.run_fake_live as runner


@pytest.mark.asyncio
@pytest.mark.parametrize('entrypoint', ['in_process', 'cli'])
@pytest.mark.parametrize('case', ['panic', 'entry'])
@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
@pytest.mark.parametrize('side', ['long', 'short'])
async def test_standard_fake_runner_revised_execution_and_trace(tmp_path, monkeypatch, mode, side, case, entrypoint):
    import config.hsl_revised as config_hsl
    monkeypatch.setattr(config_hsl, 'require_runtime_support', lambda *args, **kwargs: None)
    user = f'fake_revised_cli_{tmp_path.name}'
    _cleanup_fake_user_state(user)
    legacy = prepare_config(load_config(str(REPO_ROOT / 'configs/fake_live_hsl_btc.hjson'), verbose=False),
                            target='canonical', runtime=None, verbose=False)
    cfg = generated_template(legacy, mode)
    if side == 'short':
        cfg['bot']['short'] = deepcopy(cfg['bot']['long'])
        cfg['bot']['long']['hsl']['enabled'] = False
        cfg['bot']['long']['risk'].update(n_positions=0, total_wallet_exposure_limit=0.)
    block = cfg['bot']['hsl'] if mode == 'unified' else cfg['bot'][side]['hsl']
    block.update(enabled=True, red_threshold=.06, ema_span_minutes=1.,
                 panic_close_order_type='market', restart_after_red_policy='never')
    cfg['live']['pnls_max_lookback_days'] = 1.
    if case == 'entry':
        block['red_threshold'] = .5
        for pside in ('long', 'short'):
            cfg['bot'][pside]['unstuck']['enabled'] = False
            cfg['bot'][pside]['risk']['entry_cooldown_minutes'] = 0.
            cfg['live']['approved_coins'][pside] = ['BTC'] if pside == side else []
        cfg['live']['max_realized_loss_pct'] = 1.
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps({k: v for k, v in cfg.items() if not k.startswith('_')}))
    scenario = hjson.loads((REPO_ROOT / 'scenarios/fake_live/hsl_long_red_restart.hjson').read_text())
    scenario.pop('assertions', None)
    scenario['account']['fills'] = []
    scenario['account']['positions'][0]['position_side'] = side
    if side == 'short':
        for candle in scenario['replay']['symbols']['BTC/USDT:USDT']['candles']:
            o, h, l, c = candle[1:5]
            candle[1:5] = [200-o, 200-l, 200-h, 200-c]
    if case == 'entry':
        scenario['account']['positions'] = []
        scenario['run_initial_cycle'] = True
    scenario_path = tmp_path / 'scenario.hjson'
    scenario_path.write_text(hjson.dumps(scenario))
    output = tmp_path / 'output'
    try:
        args = argparse.Namespace(config=str(config_path), scenario=str(scenario_path), user=user,
            max_steps=4, output_dir=str(output), log_level=1, snapshot_each_step=True)
        # No cycle, evaluator, reconciler, or executor replacement.
        if entrypoint == 'in_process':
            assert await runner._async_main(args) == 0
        else:
            import os
            import subprocess
            import sys
            # Test-only guard bypass, then the actual public CLI parser/main.
            # The supplied fake scenario is the only client; no credentials.
            code = ("import config.hsl_revised as h; "
                    "h.require_runtime_support = lambda *a, **k: None; "
                    "import tools.run_fake_live as r; raise SystemExit(r.main())")
            completed = subprocess.run([sys.executable, '-c', code, str(config_path),
                str(scenario_path), '--user', user, '--max-steps', '4',
                '--output-dir', str(output), '--log-level', '1', '--snapshot-each-step'],
                cwd=REPO_ROOT, env={**os.environ, 'PYTHONPATH': str(REPO_ROOT / 'src')},
                capture_output=True, text=True, timeout=60.)
            assert completed.returncode == 0, completed.stdout + completed.stderr
        paths = list(output.rglob('hsl_trace.json'))
        assert len(paths) == 1
        artifacts = paths[0].parent
        trace = json.loads(paths[0].read_text())
        assert set(trace) == {'revised'}
        assert trace['revised']['signal_mode'] == mode
        assert trace['revised']['scope_count'] >= 1
        fills = json.loads((artifacts / 'fills.json').read_text())
        if case == 'panic':
            assert any(fill.get('reduceOnly') for fill in fills)
            positions = json.loads((artifacts / 'positions.json').read_text())
            assert not positions
        else:
            calls = json.loads((artifacts / 'remote_calls.json').read_text())
            assert any(call['method'] == 'create_order' for call in calls)
        summaries = json.loads((artifacts / 'step_summaries.json').read_text())
        assert len(summaries) == 4
        assert all("'engine': 'revised'" in step['result'] for step in summaries)
        assert all('passes' in step['result'] for step in summaries)
    finally:
        _cleanup_fake_user_state(user)


@pytest.mark.asyncio
async def test_fake_cycle_returns_bounded_pending_while_protection_keeps_running():
    import asyncio
    from types import SimpleNamespace
    from live.hsl_revised_live import owner
    release = asyncio.Event()
    async def refresh(**kwargs):
        return True
    bot = SimpleNamespace(config={'live': {'hsl_engine': 'revised'}},
        refresh_protective_authoritative_state=refresh)
    instance = owner(bot)
    instance.remember_position = lambda: None
    instance.schedule_history = instance.schedule_sources = lambda: None
    instance._ordinary_plan = release.wait
    waves = []
    async def protect():
        waves.append(True)
        return True
    instance.protect = protect
    try:
        result = await asyncio.wait_for(runner._run_fake_cycle_ready(bot), 4.)
        assert result['preparation_pending']
        assert result['passes'] == len(waves) == 8
        assert result['protective_work']
        assert not release.is_set()
        assert not instance._ordinary.done()
    finally:
        release.set()
        instance.cancel_inputs()


@pytest.mark.asyncio
async def test_production_cycle_rejects_overlapping_owner_calls_and_releases_after_failure():
    import asyncio
    from types import SimpleNamespace
    from live.hsl_revised_live import Owner
    entered, release = asyncio.Event(), asyncio.Event()
    async def refresh(**kwargs):
        entered.set()
        await release.wait()
        raise ValueError('malformed account producer')
    instance = Owner(SimpleNamespace(refresh_protective_authoritative_state=refresh))
    task = asyncio.create_task(instance.cycle())
    await entered.wait()
    with pytest.raises(RuntimeError, match='cycle is already running'):
        await instance.cycle()
    release.set()
    with pytest.raises(ValueError, match='malformed account producer'):
        await task
    assert not instance._running and not instance._cycle_running
