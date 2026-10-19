"""Suite market tensors are reusable; candidate/replay state is not cached."""
from copy import deepcopy
from dataclasses import replace

import pytest

from optimization.gpu import service
from optimization.gpu.model import ProxyMarket, ProxyRun


def _inputs():
    return dict(
        values=object(), timestamps=object(),
        runs=[ProxyRun(1000, 1, 1, 0, 0, 0, 60000, 0.05, 0, 9)] * 2,
        markets=[ProxyMarket(0.01, 0.1, 0.01, 1, 1, 0.0002)] * 2,
        checkpoint_contract={
            "coins": ["A", "B"],
            "hlcvs": {"shape": [10, 2, 4], "dtype": "<f8", "sha256": "candles"},
            "timestamps": {"count": 10, "first": 0, "last": 540000, "sha256": "timeline"},
            "base_params": {"n_positions": 1},
        },
    )


@pytest.fixture
def builds(monkeypatch):
    calls = []
    def build(*args, **kwargs):
        result = dict(n_coins=2, n=10, invariant_bytes=1024, token=object())
        calls.append((args, kwargs, result))
        return result
    monkeypatch.setattr(service, "build_mps_multicoin_data", build)
    return calls


def test_suite_prepared_data_reuses_equal_content_across_scenario_params(builds):
    cache = {}
    inputs = _inputs()
    first = service._prepared_multicoin_data(**inputs, cache=cache)
    other = _inputs()  # Different input objects with the same validated content identity.
    other["checkpoint_contract"]["base_params"]["n_positions"] = 2
    assert service._prepared_multicoin_data(**other, cache=cache) is first
    assert len(builds) == 1
    assert builds[0][1]["include_hourly_ranges"] is True


@pytest.mark.parametrize("field", ["coins", "shape", "dtype", "candles", "timeline", "count", "first", "last"])
def test_suite_prepared_data_rejects_changed_dataset_identity(builds, field):
    inputs = _inputs()
    cache = {}
    first = service._prepared_multicoin_data(**inputs, cache=cache)
    changed = deepcopy(inputs)
    contract = changed["checkpoint_contract"]
    if field == "coins":
        contract["coins"].reverse()
    elif field == "shape":
        contract["hlcvs"]["shape"] = [10, 2, 5]
    elif field == "dtype":
        contract["hlcvs"]["dtype"] = "<f4"
    elif field == "candles":
        contract["hlcvs"]["sha256"] = "changed"
    elif field == "timeline":
        contract["timestamps"]["sha256"] = "changed"
    else:
        contract["timestamps"][field] += 1
    assert service._prepared_multicoin_data(**changed, cache=cache) is not first
    assert len(builds) == 2


@pytest.mark.parametrize("kind,field", [
    ("runs", field) for field in ProxyRun.__dataclass_fields__
] + [("markets", field) for field in ProxyMarket.__dataclass_fields__])
def test_suite_prepared_data_rejects_changed_run_or_market_settings(builds, kind, field):
    inputs = _inputs()
    cache = {}
    first = service._prepared_multicoin_data(**inputs, cache=cache)
    changed = deepcopy(inputs)
    row = changed[kind][1]
    value = getattr(row, field)
    changed[kind][1] = replace(row, **{field: 1 if value is None else value + 1})
    assert service._prepared_multicoin_data(**changed, cache=cache) is not first
    assert len(builds) == 2


def test_prepared_data_cache_is_optional_and_scoped_to_its_suite(builds):
    first = service._prepared_multicoin_data(**_inputs(), cache={})
    second = service._prepared_multicoin_data(**_inputs(), cache={})
    third = service._prepared_multicoin_data(**_inputs())
    assert first is not second and second is not third
    assert len(builds) == 3


def test_failed_preparation_never_publishes_cache_entry(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("invalid candles")
    monkeypatch.setattr(service, "build_mps_multicoin_data", fail)
    cache = {}
    with pytest.raises(ValueError, match="invalid candles"):
        service._prepared_multicoin_data(**_inputs(), cache=cache)
    assert cache == {}


@pytest.mark.parametrize("device,cache_enabled,spilled", [
    ("cuda", True, True), ("cuda", False, False), ("mps", True, True),
])
def test_backend_scope_spills_only_cached_gpu_suite_data(
    builds, monkeypatch, device, cache_enabled, spilled,
):
    from pathlib import Path
    from optimization.gpu.residency import cuda_residency_scope, metal_residency_scope, current_gpu_residency

    monkeypatch.setattr(service, "gpu_device", lambda: device)
    scope = {'cuda': cuda_residency_scope, 'mps': metal_residency_scope}[device]
    with scope():
        cache = {} if cache_enabled else None
        data = service._prepared_multicoin_data(**_inputs(), cache=cache)
        assert current_gpu_residency().owns(data) is spilled
        if spilled:
            assert isinstance(builds[0][1]["spill_dir"], Path)
            assert service._prepared_multicoin_data(**_inputs(), cache=cache) is data
            assert len(builds) == 1
        else:
            assert "spill_dir" not in builds[0][1]


def test_prepared_data_cache_separates_original_f64_market_slippage(builds):
    import numpy as np
    cache = {}
    slip = 0.0005
    adjacent = np.nextafter(slip, np.inf)
    assert np.float32(slip) == np.float32(adjacent)
    first = service._prepared_multicoin_data(**_inputs(), cache=cache, market_order_slippage_pct=slip)
    assert service._prepared_multicoin_data(**_inputs(), cache=cache, market_order_slippage_pct=slip) is first
    second = service._prepared_multicoin_data(**_inputs(), cache=cache, market_order_slippage_pct=adjacent)
    assert second is not first
    assert [row[1]["market_order_slippage_pct"] for row in builds] == [slip, adjacent]


def test_market_execution_preparation_abi_invalidates_cached_market_data(builds, monkeypatch):
    cache = {}
    first = service._prepared_multicoin_data(**_inputs(), cache=cache)
    monkeypatch.setattr(service, "MARKET_EXECUTION_TICKS_ABI", service.MARKET_EXECUTION_TICKS_ABI + 1)
    assert service._prepared_multicoin_data(**_inputs(), cache=cache) is not first
