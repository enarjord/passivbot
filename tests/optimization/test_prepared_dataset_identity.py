from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from config_utils import clean_config, get_template_config
from optimization.evaluation_contract import CONTRACT_KEY, build_evaluation_contract
from optimization.prepared_dataset_identity import (
    PREPARED_DATASET_KEY,
    _array_identity,
    build_prepared_dataset_identity,
)
from shared_arrays import SharedArrayManager


@pytest.fixture
def prepared():
    manager = SharedArrayManager()
    candles = np.arange(8 * 3 * 4, dtype=np.float64).reshape(8, 3, 4) + 100
    spec, shared = manager.create_from(candles)
    btc_spec, btc = manager.create_from(np.arange(8, dtype=np.float64) + 60_000)
    config = clean_config(get_template_config())
    config["backtest"].update(
        coins={"combined": ["A", "B", "C"]},
        start_date="2024-01-01", end_date="2024-01-10",
    )
    mss = {
        coin: {
            "exchange": "binance", "symbol": f"{coin}/USDT:USDT",
            "maker_fee": 0.0002, "taker_fee": 0.0005,
            "qty_step": 0.01, "price_step": 0.1, "min_qty": 0.01,
            "min_cost": 5.0, "c_mult": 1.0,
            "first_valid_index": 0, "last_valid_index": 7, "warmup_minutes": 0,
        }
        for coin in ("A", "B", "C")
    }
    mss["__meta__"] = {"btc_source_exchange": "binance", "data_interval_minutes": 1}
    kwargs = dict(
        config=config, hlcvs_specs={"combined": spec}, btc_usd_specs={"combined": btc_spec},
        msss={"combined": mss}, timestamps={"combined": np.arange(8, dtype=np.int64) * 60_000},
    )
    try:
        yield SimpleNamespace(manager=manager, shared=shared, btc=btc, kwargs=kwargs)
    finally:
        manager.cleanup()


def _suite_context(prepared, *, label="slice", time_slice=(1, 6), indices=(2, 0)):
    kwargs = prepared.kwargs
    config = deepcopy(kwargs["config"])
    config["backtest"]["coins"]["combined"] = [
        kwargs["config"]["backtest"]["coins"]["combined"][index] for index in indices
    ]
    return SimpleNamespace(
        label=label, config=config, exchanges=["combined"],
        hlcvs_specs=kwargs["hlcvs_specs"], btc_usd_specs=kwargs["btc_usd_specs"],
        msss=deepcopy(kwargs["msss"]),
        timestamps={"combined": kwargs["timestamps"]["combined"][slice(*time_slice)]},
        master_hlcvs_specs=kwargs["hlcvs_specs"], master_btc_specs=kwargs["btc_usd_specs"],
        time_slice={"combined": time_slice}, coin_slice_indices={"combined": list(indices)},
        coin_indices={"combined": None},
    )


@pytest.mark.parametrize("backend", ["deap", "pymoo", "gpu"])
@pytest.mark.parametrize("suite", [False, True])
@pytest.mark.parametrize(
    "changed", ["candle", "btc", "fee", "source", "ohlcv_source", "btc_source", "valid_tail", "timestamps"]
)
def test_resume_rejects_changed_actual_prepared_inputs(prepared, backend, suite, changed):
    from optimize import _resume_config_mismatches

    kwargs = prepared.kwargs
    config = kwargs["config"]
    config["optimize"]["backend"] = backend
    contexts = [_suite_context(prepared)] if suite else []
    config[PREPARED_DATASET_KEY] = build_prepared_dataset_identity(
        **kwargs, scenario_contexts=contexts
    )
    old = {**deepcopy(config), CONTRACT_KEY: build_evaluation_contract(config)}
    assert _resume_config_mismatches(old, config) == []
    mss = contexts[0].msss["combined"] if suite else kwargs["msss"]["combined"]
    if changed == "candle":
        prepared.shared[3, 0, 2] += 1
    elif changed == "btc":
        prepared.btc[3] += 1
    elif changed == "fee":
        mss["A"]["maker_fee"] += 0.0001
    elif changed == "source":
        mss["A"]["exchange"] = "bybit"
    elif changed == "ohlcv_source":
        mss["A"]["ohlcv_source"] = "bybit"
    elif changed == "btc_source":
        mss["__meta__"]["btc_source_exchange"] = "bybit"
    elif changed == "valid_tail":
        mss["A"]["last_valid_index"] -= 1
    else:
        kwargs["timestamps"]["combined"][3] += 60_000
    config[PREPARED_DATASET_KEY] = build_prepared_dataset_identity(
        **kwargs, scenario_contexts=contexts
    )
    assert any("prepared_data" in item for item in _resume_config_mismatches(old, config))


def test_identity_is_content_based_across_shared_memory_and_metadata_diagnostics(prepared):
    kwargs = prepared.kwargs
    before = build_prepared_dataset_identity(**kwargs)
    spec, _ = prepared.manager.create_from(prepared.shared.copy())
    kwargs["hlcvs_specs"] = {"combined": spec}
    kwargs["msss"]["combined"]["__meta__"]["candidate_report"] = [{"cache_dir": "/other/path"}]
    kwargs["msss"]["combined"]["A"]["info"] = {"updateTime": "later"}
    assert build_prepared_dataset_identity(**kwargs) == before


def test_lazy_suite_identity_matches_exact_materialized_slice(prepared):
    ctx = _suite_context(prepared)
    kwargs = prepared.kwargs
    lazy = build_prepared_dataset_identity(**kwargs, scenario_contexts=[ctx])
    spec, _ = prepared.manager.create_from(prepared.shared[1:6, [2, 0], :])
    btc_spec, _ = prepared.manager.create_from(prepared.btc[1:6])
    ctx.master_hlcvs_specs = None
    ctx.master_btc_specs = None
    ctx.hlcvs_specs = {"combined": spec}
    ctx.btc_usd_specs = {"combined": btc_spec}
    assert build_prepared_dataset_identity(**kwargs, scenario_contexts=[ctx]) == lazy


def test_suite_identity_tracks_each_scenario_and_only_selected_data(prepared):
    contexts = [_suite_context(prepared), _suite_context(prepared, label="later", time_slice=(6, 8))]
    kwargs = prepared.kwargs
    before = build_prepared_dataset_identity(**kwargs, scenario_contexts=contexts)
    prepared.shared[:, 1, :] += 5  # Coin B is not evaluated in either scenario.
    prepared.shared[0, :, :] += 5  # Nor is the prefix before either time slice.
    assert build_prepared_dataset_identity(**kwargs, scenario_contexts=contexts) == before
    prepared.shared[7, 2, 1] += 5
    after = build_prepared_dataset_identity(**kwargs, scenario_contexts=contexts)
    assert after["scenarios"][0] == before["scenarios"][0]
    assert after["scenarios"][1] != before["scenarios"][1]


def test_repeated_suite_slices_hash_shared_content_once(prepared, monkeypatch):
    import optimization.prepared_dataset_identity as module

    original = module._array_identity
    hashed_candle_views = []

    def count(array):
        if array is not None and array.ndim == 2:
            hashed_candle_views.append(array.shape)
        return original(array)

    monkeypatch.setattr(module, "_array_identity", count)
    build_prepared_dataset_identity(
        **prepared.kwargs,
        scenario_contexts=[_suite_context(prepared), _suite_context(prepared, label="repeat")],
    )
    assert hashed_candle_views == [(5, 4), (5, 4)]


def test_hash_noncontiguous_array_uses_bounded_copy_and_canonical_endianness(monkeypatch):
    import optimization.prepared_dataset_identity as module

    original = np.ascontiguousarray
    chunks = []

    def bounded(array):
        chunks.append(array.nbytes)
        return original(array)

    monkeypatch.setattr(module.np, "ascontiguousarray", bounded)
    array = np.arange(600_000, dtype=np.float64).reshape(100_000, 3, 2)[:, 1, :]
    identity = _array_identity(array)
    assert max(chunks) <= module._HASH_BUFFER_BYTES
    assert identity == _array_identity(array.astype(">f8"))


def test_missing_prepared_dataset_is_not_a_neutral_identity(prepared):
    prepared.kwargs["hlcvs_specs"] = {}
    with pytest.raises(ValueError, match="Missing prepared optimizer datasets"):
        build_prepared_dataset_identity(**prepared.kwargs)
