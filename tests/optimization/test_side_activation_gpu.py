"""Explicit side permission uses private GPU slots without masking risk dispatch."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from test_side_minimum_cost_gpu import cpu_data


@pytest.mark.parametrize('kind', ['EmaAnchor', 'TrailingMartingale'])
@pytest.mark.parametrize('long,short', product((False, True), repeat=2))
def test_activation_settings_keep_factual_cost_and_age_layouts(monkeypatch, kind, long, short):
    kernel, run, data = cpu_data(monkeypatch)
    activation = {'long': long, 'short': short}
    filters = {'long': True, 'short': False}
    for side in ('long', 'short'):
        runner = getattr(kernel, f'Mps{kind}MulticoinRunner')(
            run, data, side=side, filter_by_min_effective_cost=filters, side_activation=activation)
        assert len(runner.settings) == 16
        assert runner.settings[15].item() == activation[side]
        assert runner.settings[11].item() == filters[side]
        assert runner.settings[12:14].tolist() == [0., 0.]
    runner = getattr(kernel, f'Mps{kind}MulticoinFusedRunner')(
        run, data, filter_by_min_effective_cost=filters, side_activation=activation)
    assert len(runner.settings) == 20
    assert runner.settings[18:].tolist() == [long, short]
    assert runner.settings[13].item() == 1
    assert runner.settings[17].item() == 0
    runner.settings[14], runner.settings[15] = 7, 1
    assert runner.settings[18:].tolist() == [long, short]
    assert data['coin_settings'].shape[1] == 15


@pytest.mark.parametrize('invalid', [None, True, {}, {'long': True}, {'long': 1, 'short': False}])
def test_runner_activation_is_required_complete_boolean_policy(monkeypatch, invalid):
    kernel, run, data = cpu_data(monkeypatch)
    with pytest.raises(TypeError):
        kernel.MpsEmaAnchorMulticoinFusedRunner(run, data,
            filter_by_min_effective_cost={'long': False, 'short': False}, side_activation=invalid)
    with pytest.raises(TypeError, match='side_activation'):
        kernel.MpsEmaAnchorMulticoinFusedRunner(run, data,
            filter_by_min_effective_cost={'long': False, 'short': False})


def test_shader_and_cuda_lowering_keep_entry_gate_separate_from_management():
    import passivbot_rust as pbr
    from optimization.gpu.cuda_kernel import cuda_source
    from rust_utils import verify_loaded_runtime_extension
    verify_loaded_runtime_extension()
    for source in (pbr.mps_ema_anchor_multicoin_source_py(), pbr.mps_trailing_martingale_multicoin_source_py()):
        for code in (source, cuda_source(source)):
            assert 'side_activation = run_settings[15]' in code
            assert 'long_activation = run_settings[18]' in code
            assert 'short_activation = run_settings[19]' in code
            assert 'long_can_generate && long_entries_enabled, short_can_generate && short_entries_enabled' in code
            assert 'bool entries_enabled' in code
            assert '(entry_capacity || ' in code


def activation_inputs(kind, topology, active_side):
    from tools.gpu_parity import build_parser, fixture_inputs
    args = build_parser().parse_args(['--fixture', kind, '--hsl', 'disabled', '--sides', 'both',
                                     '--coins', '2', '--bars', '64', '--seed', '2026'])
    config, candles, markets, btc, timestamps = fixture_inputs(args)
    config['bot']['portfolio']['hedge_mode'] = topology != 'oneway'
    config['bot']['portfolio']['warmup']['max_minutes'] = 1.0
    for side in ('long', 'short'):
        bot = config['bot'][side]
        bot['enabled'] = active_side == 'both' or active_side == side
        bot['risk']['max_positions'] = 1
        bot['entry_cooldown']['base_duration_minutes'] = 1000.0
        strategy = bot['strategy'][kind]
        spans = strategy if kind == 'ema_anchor' else strategy['entry']
        spans.update(ema_span_0_minutes=2.0, ema_span_1_minutes=3.0)
        if kind == 'ema_anchor':
            strategy.update(base_qty_fraction=0.01, offset_fraction=0.0, entry_double_down_factor=0.0)
        else:
            strategy['entry'].update(initial_qty_fraction=0.01, initial_ema_distance_fraction=0.0,
                                     double_down_factor=0.0)
            for group in ('entry', 'close'):
                strategy[group].update(threshold_volatility_1h_weight=0.0,
                                       retracement_volatility_1h_weight=0.0)
    for coin, market in markets.items():
        if coin != '__meta__':
            market['warmup_minutes'] = 1
    return args, config, candles, markets, btc, timestamps


@pytest.mark.parametrize('kind', ['trailing_martingale', 'ema_anchor'])
@pytest.mark.parametrize('topology', ['directional', 'fused', 'oneway'])
@pytest.mark.parametrize('active_side', ['long', 'short'])
def test_actual_gpu_activation_matches_native_both_strategies_and_side_flips(monkeypatch, kind, topology, active_side):
    torch = pytest.importorskip('torch')
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip('local Metal or CUDA device unavailable')
    from backtest import build_backtest_payload, execute_backtest
    from optimization.gpu.service import SharedAccountReplay

    args, config, candles, markets, btc, timestamps = activation_inputs(kind, topology, active_side)
    payload = build_backtest_payload(candles, markets, config, args.exchange, btc, timestamps)
    fills, _, native = execute_backtest(payload, config)
    if topology == 'directional':
        proxy = SharedAccountReplay(config=config, hlcvs=candles, mss=markets, btc=btc,
            timestamps=timestamps, exchange=args.exchange, batch_size=1, needed_metrics={'fills_per_day'})
        runner = proxy.runners[active_side]
        matrix = proxy._parameter_matrix([{}], active_side)
    else:
        # Exercise both fused private slots with asymmetric policy. Normal
        # service construction may specialize a single active side instead.
        baseline = deepcopy(config)
        for side in ('long', 'short'):
            baseline['bot'][side]['enabled'] = True
        proxy = SharedAccountReplay(config=baseline, hlcvs=candles, mss=markets, btc=btc,
            timestamps=timestamps, exchange=args.exchange, batch_size=1, needed_metrics={'fills_per_day'})
        cls, options = proxy._runner_specs['fused']
        options = {**options, 'side_activation': payload.backtest_params['side_activation']}
        runner = cls(proxy.run, proxy.data, **options)
        matrix = np.concatenate([proxy._parameter_matrix([{}], side) for side in ('long', 'short')], axis=1)
    raw = runner.run(matrix)
    assert raw['fill_count_long'].item() == native['fills_count_long']
    assert raw['fill_count'].item() - raw['fill_count_long'].item() == native['fills_count_short']
    entries = [row for row in fills if str(row[13]).startswith('entry_')]
    assert entries and all(str(row[13]).endswith('_' + active_side) for row in entries)
    assert raw['first_fill_ts'].item() == int(fills[0][1]) - int(timestamps[0])


@pytest.mark.parametrize('kind', ['trailing_martingale', 'ema_anchor'])
def test_actual_offline_cli_backtest_gpu_smoke(tmp_path, kind):
    torch = pytest.importorskip('torch')
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip('local Metal or CUDA device unavailable')
    from passivbot_cli.main import main
    assert main(['tool', 'gpu-parity', '--fixture', kind, '--sides', 'both', '--coins', '2',
                 '--bars', '64', '--seed', '2026', '--metrics', 'fills_per_day',
                 '--report', str(tmp_path / 'parity.json'), '--compact']) == 0
