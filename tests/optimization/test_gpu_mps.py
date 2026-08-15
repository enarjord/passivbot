import numpy as np
import pytest


torch = pytest.importorskip("torch")

from optimization.gpu.model import ProxyMarket, ProxyRun, build_mps_data
from optimization.gpu.mps_kernel import MpsEmaAnchorRunner


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_ema_anchor_shader_smoke():
    count = 512
    phase = np.linspace(0.0, 10.0 * np.pi, count)
    close = 100.0 + np.sin(phase) * 5.0
    high = close * 1.01
    low = close * 0.99
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    market = ProxyMarket(
        qty_step=0.001,
        price_step=0.01,
        min_qty=0.001,
        min_cost=5.0,
        c_mult=1.0,
        maker_fee=0.0002,
    )
    run = ProxyRun(
        starting_balance=1_000.0,
        warmup_bars=10,
        trade_start_idx=10,
        guard_ts_ms=int(timestamps[0]),
        first_ts_ms=int(timestamps[0]),
        interval_ms=60_000,
        liquidation_threshold=0.05,
        first_valid_idx=0,
        last_valid_idx=count - 1,
    )
    data = build_mps_data(high, low, close, timestamps, run, market)
    parameters = np.array(
        [[0.1, 10.0, 30.0, 1.5, 0.01, 0.0, 0.0, 0.0, 60.0, 60.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    output = MpsEmaAnchorRunner(market, run, data).run(parameters)
    torch.mps.synchronize()

    assert output["balance"].device.type == "mps"
    assert output["balance"].shape == (1,)
    assert torch.isfinite(output["balance"]).all()
    assert output["day_has_fill"].sum().item() > 0
