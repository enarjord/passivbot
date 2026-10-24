"""Original-precision execution inputs survive packing, caching and runner admission."""
from dataclasses import replace

import numpy as np
import pytest
import passivbot_rust

from optimization.gpu.model import (
    MARKET_EXECUTION_TICKS_ABI, ProxyMarket, ProxyRun, _pack_multicoin_price_arrays,
)


def _case():
    n = 9
    values = np.tile([168.0, 167.0, 167.84391568013112, 1.0], (n, 2, 1))
    values[:2, 1] = np.nan
    values[-2:, 1] = np.nan
    runs = [ProxyRun(1000, 0, 0, 0, 0, 0, 60000, .05, 0, n - 1)]
    runs.append(replace(runs[0], first_valid_idx=2, last_valid_idx=n - 3))
    markets = [ProxyMarket(.001, .01, .001, 0, 1, 0)] * 2
    return values, runs, markets


def test_native_execution_packing_retains_strided_f64_input_and_global_chunk_windows(monkeypatch):
    values, runs, markets = _case()
    original = passivbot_rust.prepare_gpu_market_execution_ticks
    calls = []
    def prepare(closes, steps, slippage, first, last, row_offset=0):
        assert closes.dtype == np.float64
        assert not closes.flags.c_contiguous
        assert np.shares_memory(closes, values)
        calls.append((len(closes), first, last, row_offset, slippage))
        return original(closes, steps, slippage, first, last, row_offset)
    monkeypatch.setattr(passivbot_rust, "prepare_gpu_market_execution_ticks", prepare)
    eager, _ = _pack_multicoin_price_arrays(values, markets, np.int32, 0.0, runs, .0005)
    pieces = []
    for start in range(0, len(values), 3):
        arrays, _ = _pack_multicoin_price_arrays(values[start:start + 3], markets, np.int32, 0.0, runs, .0005, start)
        pieces.append(arrays["market_execution_ticks"])
    np.testing.assert_array_equal(np.concatenate(pieces), eager["market_execution_ticks"])
    assert eager["market_execution_ticks"][0, 0, 0] == 16775
    assert np.all(eager["market_execution_ticks"][:2, 1] == 0)
    assert np.all(eager["market_execution_ticks"][-2:, 1] == 0)
    assert [row[3] for row in calls] == [0, 0, 3, 6]
    assert all(row[1:3] == ([0, 2], [8, 6]) and row[4] == .0005 for row in calls)


def test_slippage_changes_only_named_execution_ticks_during_price_packing():
    values, runs, markets = _case()
    plain, _ = _pack_multicoin_price_arrays(values, markets, np.int32, 0.0, runs, 0.0)
    slipped, _ = _pack_multicoin_price_arrays(values, markets, np.int32, 0.0, runs, .0005)
    assert not np.array_equal(plain["market_execution_ticks"], slipped["market_execution_ticks"])
    for name in plain.keys() - {"market_execution_ticks"}:
        np.testing.assert_array_equal(plain[name], slipped[name], err_msg=name)


@pytest.mark.parametrize("invalid", ["missing", "abi", "f64_slippage", "shape", "dtype", "noncontiguous"])
def test_runner_rejects_stale_or_malformed_execution_inputs_before_allocation(invalid):
    torch = pytest.importorskip("torch")
    from optimization.gpu.mps_kernel import _validate_market_execution_data
    slip = .0005
    data = {"n": 3, "n_coins": 2, "bars": torch.zeros((3, 2, 4)),
            "market_execution_ticks": torch.ones((3, 2, 2), dtype=torch.int32),
            "market_execution_ticks_abi": MARKET_EXECUTION_TICKS_ABI,
            "market_order_slippage_pct": slip}
    assert _validate_market_execution_data(data, slip) is data["market_execution_ticks"]
    if invalid == "missing": del data["market_execution_ticks"]
    if invalid == "abi": data["market_execution_ticks_abi"] += 1
    if invalid == "f64_slippage": data["market_order_slippage_pct"] = np.nextafter(slip, np.inf)
    if invalid == "shape": data["market_execution_ticks"] = torch.ones((3, 2), dtype=torch.int32)
    if invalid == "dtype": data["market_execution_ticks"] = data["market_execution_ticks"].to(torch.float32)
    if invalid == "noncontiguous": data["market_execution_ticks"] = data["market_execution_ticks"].transpose(1, 2)
    with pytest.raises(ValueError, match="market execution|market execution ABI"):
        _validate_market_execution_data(data, slip)
