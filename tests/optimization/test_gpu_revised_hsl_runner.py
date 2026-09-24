"""Offline hardware checks for the internal revised single-coin runners."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason='Apple MPS and NVIDIA CUDA unavailable',
)
from optimization.gpu.mps_kernel import MpsEmaAnchorRunner, MpsTrailingMartingaleRunner
from optimization.gpu.model import EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS, TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS
from optimization.gpu.service import build_mps_data
from tools.gpu_proxy_benchmark import _synthetic_hlcvs, _market_and_run, _parameter_matrix


def fixture(strategy, *, enabled=True, mode=2, bars=1600):
    x, t = _synthetic_hlcvs(bars, 1, 43)
    m, r = _market_and_run(t, bars)
    d = build_mps_data(x[:,0,0], x[:,0,1], x[:,0,2], t, r, m)
    cls, keys = ((MpsTrailingMartingaleRunner, TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS)
                 if strategy == 'tm' else (MpsEmaAnchorRunner, EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS))
    a = _parameter_matrix(keys, 3, 43, value_overrides={
        'hsl_enabled': float(enabled), 'hsl_restart_policy': 0., 'hsl_signal_mode': float(mode),
        'hsl_red_threshold': .0005, 'hsl_ema_span_minutes': 1.,
        'hsl_cooldown_minutes_after_red': 10., 'total_wallet_exposure_limit': 5.,
        'entry_double_down_factor': 2.,
    })
    return cls, (m,r,d), np.concatenate([a,a], axis=1)


def compare(a, b):
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_allclose(a[key].cpu().numpy(), b[key].cpu().numpy(),
                                   rtol=3e-5, atol=2e-5, equal_nan=True, err_msg=key)


@pytest.mark.parametrize('strategy', ['tm', 'ema'])
@pytest.mark.parametrize('sides', [(True,False), (False,True), (True,True)])
def test_disabled_revised_engine_preserves_strategy(strategy, sides):
    cls, args, matrix = fixture(strategy, enabled=False)
    common = dict(long_enabled=sides[0], short_enabled=sides[1], pnl_lookback_bars=1440,
                  hsl_enabled=False)
    legacy = cls(*args, hsl_engine='legacy', **common).run(matrix)
    revised = cls(*args, hsl_engine='revised', **common).run(matrix)
    compare(legacy, revised)


@pytest.mark.parametrize('mode', [0,1,2])
def test_temporal_chunks_and_reused_buffers_preserve_revised_results(mode):
    cls, args, matrix = fixture('tm', mode=mode)
    common = dict(long_enabled=True, short_enabled=True, pnl_lookback_bars=1440,
                  hsl_engine='revised')
    full_runner = cls(*args, **common)
    full = {k:v.clone() for k,v in full_runner.run(matrix).items()}
    chunks = cls(*args, max_dispatch_candidate_bars=600, **common)
    first = {k:v.clone() for k,v in chunks.run(matrix).items()}
    compare(full, first)
    compare(first, chunks.run(matrix))
    assert float(full['fill_count'].sum()) > 0
    assert float((full['hsl_triggers_long']+full['hsl_triggers_short']).sum()) > 0
    np.testing.assert_array_equal(full['hsl_tier_samples_yellow'].cpu(), 0.)
    np.testing.assert_array_equal(full['hsl_tier_samples_orange'].cpu(), 0.)


@pytest.mark.parametrize('strategy', ['tm','ema'])
def test_candidate_permutation_preserves_revised_outcomes(strategy):
    cls, args, matrix = fixture(strategy)
    runner = cls(*args, long_enabled=True, short_enabled=True, pnl_lookback_bars=1440,
                 hsl_engine='revised')
    a = {k:v.clone() for k,v in runner.run(matrix).items()}
    b = runner.run(matrix[[2,0,1]])
    compare({k:v[[2,0,1]] for k,v in a.items()}, b)


@pytest.mark.parametrize('strategy', ['tm', 'ema'])
def test_scratch_bounded_candidate_batches_match_unsplit_run(strategy):
    cls, args, matrix = fixture(strategy)
    runner = cls(*args, long_enabled=True, short_enabled=True, pnl_lookback_bars=1440,
                 hsl_engine='revised')
    expected = {k:v.clone() for k,v in runner.run(matrix).items()}
    runner.revised_scratch_budget_bytes = runner._revised_bytes_per_candidate() * 2
    actual = runner.run(matrix, profile=True)
    compare(expected, actual)
    assert runner.last_profile['candidate_batch_count'] == 2
    assert runner.last_profile['batch_size'] == len(matrix)
    assert sum(t.numel() * t.element_size() for group in runner._revised_buffers.values()
               for t in group) <= runner.revised_scratch_budget_bytes


@pytest.mark.parametrize('strategy', ['tm', 'ema'])
@pytest.mark.parametrize('key,value', [('hsl_red_threshold', float('nan')),
    ('hsl_ema_span_minutes', .5), ('hsl_restart_policy', 1), ('hsl_slot_count', 0)])
def test_revised_policies_are_not_silently_clamped(strategy, key, value):
    cls, args, matrix = fixture(strategy)
    keys = (TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS if strategy == 'tm'
            else EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)
    matrix[:, keys.index(key)] = value
    runner = cls(*args, long_enabled=True, short_enabled=True, pnl_lookback_bars=1440,
                 hsl_engine='revised')
    with pytest.raises(ValueError, match='(Revised|revised)'):
        runner.run(matrix)


@pytest.mark.parametrize('strategy', ['tm', 'ema'])
def test_zero_cooldown_still_counts_terminal_red(strategy):
    cls, args, matrix = fixture(strategy)
    keys = (TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS if strategy == 'tm'
            else EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)
    for side in range(2):
        matrix[:, side * len(keys) + keys.index('hsl_cooldown_minutes_after_red')] = 0.
    runner = cls(*args, long_enabled=True, short_enabled=True, pnl_lookback_bars=1440,
                 hsl_engine='revised')
    result = runner.run(matrix)
    assert float((result['hsl_triggers_long'] + result['hsl_triggers_short']).sum()) > 0
