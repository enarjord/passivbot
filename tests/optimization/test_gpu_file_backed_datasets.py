from pathlib import Path

import numpy as np
import pytest

from optimization.gpu.datasets import PreparedGpuDataset
from shared_arrays import SharedArrayManager


def _dataset(manager):
    config = {
        "backtest": {"coins": {"binance": ["A", "B"]}},
        "bot": {
            side: {"universe": {"approved_coins": ["A", "B"], "ignored_coins": []}}
            for side in ("long", "short")
        },
    }
    candles = np.arange(12 * 2 * 4, dtype=np.float64).reshape(12, 2, 4)
    specs = [manager.create_file_from(value)[0] for value in (
        candles, np.arange(12, dtype=np.float64), np.arange(12, dtype=np.int64),
    )]
    dataset = PreparedGpuDataset(
        config=config, markets={"A": {}, "B": {}},
        hlcvs=specs[0], btc=specs[1], timestamps=specs[2],
        candle_coins=["A", "B"], exchange="binance", metrics=["gross_turnover_ratio_per_day"],
        time_range=(2, 9),
    )
    return dataset, config, candles


def test_prepared_dataset_preserves_file_descriptors_and_selected_views():
    manager = SharedArrayManager()
    try:
        dataset, config, candles = _dataset(manager)
        updated = dataset.with_config(config)
        assert updated.hlcvs == dataset.hlcvs
        assert updated.btc == dataset.btc
        assert updated.timestamps == dataset.timestamps
        assert all(spec.storage == "file" and spec.offset > 0 for spec in (
            updated.hlcvs, updated.btc, updated.timestamps,
        ))
        assert all(spec.file_identity is not None for spec in (
            updated.hlcvs, updated.btc, updated.timestamps,
        ))
        with updated.attach() as arrays:
            np.testing.assert_array_equal(arrays[0], candles[2:9])
            np.testing.assert_array_equal(arrays[1], np.arange(2, 9))
            np.testing.assert_array_equal(arrays[2], np.arange(2, 9))
            assert all(not array.flags.writeable for array in arrays)
    finally:
        manager.cleanup()


def test_prepared_file_attachment_failure_closes_earlier_mapping(monkeypatch):
    import optimization.gpu.datasets as module

    manager = SharedArrayManager()
    try:
        dataset, _config, _candles = _dataset(manager)
        original = module.attach_shared_array
        mappings = []
        def attach(spec):
            if spec == dataset.btc:
                raise FileNotFoundError("file disappeared")
            attachment = original(spec)
            mappings.append(attachment._mapping._mmap)
            return attachment
        monkeypatch.setattr(module, "attach_shared_array", attach)
        with pytest.raises(FileNotFoundError, match="disappeared"):
            with dataset.attach():
                pytest.fail("partial attachment must fail")
        assert len(mappings) == 1 and mappings[0].closed
        assert Path(dataset.hlcvs.name).exists()
    finally:
        manager.cleanup()
