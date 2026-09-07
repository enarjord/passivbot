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
    changed[kind][1] = replace(row, **{field: getattr(row, field) + 1})
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
