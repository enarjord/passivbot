"""Admission and validation peaks are bounded before history-sized GPU packing."""
import gzip
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hlcvs_manifest import load_numpy_artifact, numpy_artifact_loader
from optimization.gpu import model, residency, service
from optimize import _gpu_cached_input_preflight, validate_array
from shared_arrays import SharedArrayManager


def _torch(monkeypatch, recommended=1000):
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        mps=SimpleNamespace(recommended_max_memory=lambda: recommended,
                            driver_allocated_memory=lambda: 0),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


@pytest.mark.parametrize("hourly", [False, True])
@pytest.mark.parametrize("device,bytes_per_cell", [("mps", 52), ("cuda", 49)])
def test_exact_packed_byte_estimator(device, bytes_per_cell, hourly):
    assert residency.multicoin_invariant_bytes(17, 3, device=device,
        include_hourly_ranges=hourly) == 17 * 3 * (bytes_per_cell + 4 * hourly)


@pytest.mark.parametrize("spill", [False, True])
def test_oversized_history_rejected_before_packing(monkeypatch, tmp_path, spill):
    _torch(monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail("oversized history reached price packing")
    monkeypatch.setattr(model, "_spill_multicoin_arrays", forbidden)
    monkeypatch.setattr(model, "_pack_multicoin_price_arrays", forbidden)
    values = np.tile([101., 99., 100., 1.], (6, 2, 1))
    run = model.ProxyRun(1000., 0, 0, 0, 0, 0, 60000, .05, 0, 5)
    market = model.ProxyMarket(.001, .01, .001, 1., 1., .0002)
    with pytest.raises(MemoryError, match="before packing"):
        model.build_mps_multicoin_data(values, np.arange(6) * 60000,
            [run] * 2, [market] * 2, spill_dir=tmp_path if spill else None)
    assert not list(tmp_path.iterdir())


def test_compressed_header_admission_precedes_payload_decode(monkeypatch, tmp_path):
    _torch(monkeypatch, recommended=12 * 2**30)
    path = tmp_path / "hlcvs.npy.gz"
    # The payload deliberately does not exist: refusal must depend on header alone.
    with gzip.open(path, "wb") as handle:
        np.lib.format.write_array_header_1_0(handle, {
            "descr": "<f8", "fortran_order": False, "shape": (3_000_000, 40, 4),
        })
    manager = SharedArrayManager(file_backed=True, file_preflight=_gpu_cached_input_preflight)
    try:
        with numpy_artifact_loader(manager.load_numpy_artifact):
            with pytest.raises(MemoryError, match="45%"):
                load_numpy_artifact(path)
        assert not manager._arrays
    finally:
        manager.cleanup()


def test_validation_slices_preserve_first_coin_and_absolute_error(monkeypatch):
    monkeypatch.setattr(residency, "PREPARATION_CHUNK_BYTES", 80)
    values = np.ones((23, 3, 4), dtype=np.float64)
    values[18, 0, 0] = np.finfo(np.float64).max
    values[2, 1, 0] = np.nan  # Preserve coin-first error precedence across chunks.
    values[:, 2, :3] = np.nan  # Unexposed data must not be cast/validated.
    original = np.asarray
    sizes = []
    def tracked(value, *args, **kwargs):
        if kwargs.get("dtype") == np.float32:
            sizes.append(value.size)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(service.np, "asarray", tracked)
    with pytest.raises(ValueError, match="coin index 0, invalid candle at 18"):
        service._require_exact_safe_proxy_candles(values,
            exposure_eligible_coins=[True, True, False], first_valid_indices=[0] * 3,
            last_valid_indices=[22] * 3, require_positive_high_low=True)
    assert max(sizes) <= 12


def test_array_validation_preserves_error_priority_on_strided_values():
    values = np.ones((20, 4))[::2, ::2]
    values[0, 0], values[-1, -1] = np.inf, np.nan
    with pytest.raises(ValueError, match="contains NaN"):
        validate_array(values, "history", allow_nan=False)
    with pytest.raises(ValueError, match="contains inf"):
        validate_array(values, "history", allow_nan=True)
    with pytest.raises(ValueError, match="entirely NaN"):
        validate_array(np.array([]), "history")


@pytest.mark.asyncio
async def test_cache_memory_refusal_is_not_retried_as_download(monkeypatch):
    import backtest
    from tools.gpu_parity import build_parser, fixture_inputs
    config, *_ = fixture_inputs(build_parser().parse_args(["--fixture", "ema_anchor", "--bars", "128"]))
    refusal = MemoryError("input admission failed")
    monkeypatch.setattr(backtest, "load_hlcvs_data_override", lambda *args: None)
    def reject(*args, **kwargs):
        raise refusal
    monkeypatch.setattr(backtest, "load_coins_hlcvs_from_cache", reject)
    with pytest.raises(MemoryError) as caught:
        await backtest.prepare_hlcvs_mss(config, "binance")
    assert caught.value is refusal
