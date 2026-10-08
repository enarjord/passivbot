"""Recursive TM sizing follows Rust's advancing simulated order-book touch."""
from functools import lru_cache

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from optimization.gpu.model import (TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
                                    TRAILING_MARTINGALE_COIN_OVERRIDE_COLS)
from optimization.gpu.runtime import gpu_device

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="GPU required"
)

_PROBE = r"""
kernel void recursive_sizing_probe(
    constant float* settings, constant float* overrides, constant float* params,
    constant float* initial, device float* output,
    constant int& short_raw, uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    bool short_side = short_raw != 0;
    TrailingMartingaleMulticoinSideConfig config =
        load_trailing_martingale_multicoin_side_config(params, 0);
    TrailingMartingaleMulticoinSideState state;
    init_trailing_martingale_multicoin_side_state(state, config, settings, overrides, 1);
    state.ema0[0] = state.ema1[0] = state.ema2[0] = initial[1];
    state.volatility_1h[0] = state.volatility_1m[0] = 0.0f;
    float size = initial[0], basis = initial[1];
    int touch = int(round(initial[1] / settings[1]));
    output[0] = size; output[1] = basis;
    for (int i = 1; i < 16; ++i) {
        RecursiveEntryCandidate next = next_recursive_grid_entry(
            state, config, overrides, 0, short_side, size, basis,
            1000.0f, 0.5f, initial[1], touch, initial[1],
            settings[0], settings[1], settings[2], settings[3], settings[4], false, 0.001f
        );
        output[i*2] = next.strategy_qty; output[i*2+1] = next.price;
        float after = round_step(size + next.strategy_qty, settings[0]);
        basis = basis * (size/after) + next.price * (next.strategy_qty/after);
        size = after;
        touch = short_side ? max(touch, next.ticks) : min(touch, next.ticks);
    }
}
"""

@lru_cache(maxsize=1)
def _library():
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension
    from test_gpu_mps import compile_shader
    verify_loaded_runtime_extension()
    return compile_shader(passivbot_rust.mps_trailing_martingale_multicoin_source_py() + _PROBE)

@pytest.mark.parametrize("side,anchor,quantity_pct,spacing", [
    ("short", 116.5, 0.012, 0.001),
    ("long", 100.0, 0.01, 0.01),
])
def test_recursive_initial_quantity_reprices_with_simulated_touch(side, anchor, quantity_pct, spacing):
    import passivbot_rust
    from test_gpu_mps import _multicoin_exposure_fixture
    _, row = _multicoin_exposure_fixture("trailing_martingale", side, count=3)
    values = dict(zip(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, row))
    values.update(entry_double_down_factor=0.2, entry_initial_qty_pct=quantity_pct,
                  entry_initial_ema_dist=0., entry_threshold_base_pct=spacing,
                  entry_threshold_we_weight=0., entry_threshold_volatility_1h_weight=0.,
                  entry_threshold_volatility_1m_weight=0., gate_initial=1., gate_reentry=0.)
    kwargs = dict(qty_step=.001, price_step=.01, min_qty=.001, min_cost=1., c_mult=1.,
        entry_grid_double_down_factor=.2, entry_grid_spacing_pct=spacing,
        entry_initial_ema_dist=0., entry_initial_qty_pct=quantity_pct,
        entry_trailing_double_down_factor=.2, entry_trailing_retracement_pct=0.,
        entry_trailing_threshold_pct=spacing, entry_weight_volatility_1h=0.,
        entry_weight_volatility_1m=0., entry_we_weight=0., wallet_exposure_limit=.5,
        risk_we_excess_allowance_pct=0., balance=1000., position_size=0., position_price=0.,
        min_since_open=0., max_since_min=0., max_since_open=0., min_since_max=0.,
        volatility_ema_1h=0., volatility_ema_1m=0.)
    kwargs['ema_bands_upper' if side=='short' else 'ema_bands_lower'] = anchor
    kwargs['order_book_ask' if side=='short' else 'order_book_bid'] = anchor
    expected = np.asarray(getattr(passivbot_rust, f'calc_entries_{side}_py')(**kwargs)[:16])
    assert len(expected) == 16
    # This fixture must cross a quantity boundary rather than merely vary prices.
    assert np.any(np.abs(expected[2:,0]) != abs(expected[0,0]))
    device = gpu_device()
    settings = torch.zeros((1,13), dtype=torch.float32, device=device)
    settings[0,:5] = torch.tensor([.001,.01,.001,1.,1.],device=device)
    settings[0,7] = 3.
    overrides = torch.full((1,TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
                           float('nan'), dtype=torch.float32, device=device)
    params = torch.tensor([values[k] for k in TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS],dtype=torch.float32,device=device)
    initial = torch.tensor([abs(expected[0,0]),expected[0,1]],dtype=torch.float32,device=device)
    output = torch.empty((16,2),dtype=torch.float32,device=device)
    _library().recursive_sizing_probe(settings,overrides,params,initial,output,int(side=='short'),threads=1)
    actual = output.cpu().numpy()
    np.testing.assert_array_equal(np.rint(actual[:,0]/.001), np.rint(np.abs(expected[:,0])/.001))
    np.testing.assert_array_equal(np.rint(actual[:,1]/.01), np.rint(expected[:,1]/.01))
