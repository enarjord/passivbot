"""Inactive candidates must not arbitrate against the enabled position side."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason='Metal or CUDA required',
)

@pytest.mark.parametrize('inactive', ['long', 'short'])
@pytest.mark.parametrize('disable', ['exposure', 'slots'])
@pytest.mark.parametrize('coins', [1, 2])
def test_zero_side_cannot_block_active_initial_entries(inactive, disable, coins):
    from test_gpu_hsl_multicoin import make_proxy, raw
    active = 'short' if inactive == 'long' else 'long'
    candidate = {
        f'{inactive}_entry_initial_ema_dist': -0.05,
        f'{active}_entry_initial_ema_dist': -0.02,
        f'{inactive}_total_wallet_exposure_limit': 0.0 if disable == 'exposure' else 2.0,
        f'{inactive}_n_positions': 0.0 if disable == 'slots' else 1.0,
        f'{active}_hsl_red_threshold': 0.0001,
        f'{active}_hsl_drawdown_ratio_ema_span_minutes': 1.0,
    }
    outputs = []
    for hedge_mode in (False, True):
        proxy = make_proxy('coin', minutes=2000, coin_count=coins, factual_hsl=True)
        runner_cls, kwargs = proxy._runner_specs['fused']
        proxy.fused_runner = proxy._construct_runner(
            'fused', runner_cls, dict(kwargs, hedge_mode=hedge_mode)
        )
        _, result = raw(proxy, [candidate])
        outputs.append(result)
    assert outputs[0]['fill_count'].item() > 0
    inactive_count = (outputs[0]['fill_count_long'] if inactive == 'long' else
                      outputs[0]['fill_count'] - outputs[0]['fill_count_long'])
    assert inactive_count.item() == 0
    assert outputs[0][f'hsl_triggers_{active}'].item() > 0
    for key in outputs[0]:
        if isinstance(outputs[0][key], torch.Tensor):
            np.testing.assert_array_equal(outputs[0][key].cpu().numpy(),
                                          outputs[1][key].cpu().numpy(), err_msg=key)
