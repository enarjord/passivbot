"""Small factual cashflows retain their contribution to GPU account sizing."""
from functools import lru_cache

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from optimization.gpu.runtime import gpu_device

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="GPU required"
)

_PROBE = r"""
kernel void balance_state_bytes(device int* output, uint b [[thread_position_in_grid]]) {
    if (b == 0) output[0] = int(sizeof(JointPortfolioAccount));
}
kernel void accumulate_cashflows(
    constant float* values, device JointPortfolioAccount* saved,
    device float* output, constant int& begin, constant int& end,
    uint b [[thread_position_in_grid]]
) {
    if (b > 0) return;
    JointPortfolioAccount account = begin == 0
        ? init_joint_portfolio_account(1000.0f) : saved[0];
    for (int i = begin; i < end; ++i)
        record_joint_portfolio_fill(account, values[i], (i & 1) == 0);
    saved[0] = account;
    output[0] = account.balance;
}
"""

@lru_cache(maxsize=2)
def _library(strategy):
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension
    from test_gpu_mps import compile_shader

    verify_loaded_runtime_extension()
    source = getattr(passivbot_rust, f"mps_{strategy}_multicoin_source_py")()
    return compile_shader(source + _PROBE)

@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("pattern", ["fees", "profits_and_fees", "loss_recovery"])
def test_cashflow_accumulation_matches_precise_encoded_sum_across_dispatches(strategy, pattern):
    if pattern == "fees":
        values = np.full(100_000, -.0002, dtype=np.float32)
    elif pattern == "profits_and_fees":
        values = np.tile(np.array([.03, -.0002], dtype=np.float32), 25_000)
    else:
        values = np.concatenate([np.full(10_000, -.003, dtype=np.float32),
                                 np.full(10_000, .0031, dtype=np.float32)])
    # Sum the actual encoded factual cashflows in f64, independent of the kernel.
    expected = np.float32(1000. + values.astype(np.float64).sum())
    device = gpu_device()
    library = _library(strategy)
    payload = torch.tensor(values, device=device)
    size = torch.empty(1, dtype=torch.int32, device=device)
    library.balance_state_bytes(size, threads=1)
    state = torch.empty(int(size.item()), dtype=torch.uint8, device=device)
    output = torch.empty(1, dtype=torch.float32, device=device)
    library.accumulate_cashflows(payload, state, output, 0, len(values), threads=1)
    whole = output.item()
    assert abs(whole - float(expected)) <= float(np.spacing(expected)), (whole, expected)
    # Resume the native account state through real separate device dispatches.
    cuts = [0, 97, 313, len(values) // 2, len(values)]
    for begin, end in zip(cuts, cuts[1:]):
        library.accumulate_cashflows(payload, state, output, begin, end, threads=1)
    assert output.item() == whole
