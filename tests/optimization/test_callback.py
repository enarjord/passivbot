import numpy as np
from unittest.mock import MagicMock

from optimization.callback import PymooRecorderCallback


def _build_config(vector, overrides_fn, overrides_list, template):
    config = {
        "bot": {"long": {"a": float(vector[0]), "b": float(vector[1])}},
        "backtest": {"coins": {"binance": ["BTC/USDT:USDT"]}},
    }
    config.update({k: v for k, v in template.items() if k not in config})
    return config


def test_callback_records_offspring_with_suite_payload_cleanup():
    recorder = MagicMock()
    callback = PymooRecorderCallback(
        recorder=recorder,
        template={"optimize": {"backend": "pymoo"}},
        build_config_fn=_build_config,
        overrides_fn=object(),
        overrides_list=["x"],
    )
    individual = MagicMock()
    individual.X = np.asarray([1.0, 2.0])
    individual.data = {
        "metrics": {
            "objectives": {"w_0": -1.0},
            "constraint_violation": 0.0,
            "suite_metrics": {"scenario_labels": ["base"]},
        },
        "evaluation_vector": np.asarray([0.25, 2.0]),
    }
    algorithm = MagicMock()
    algorithm.off = [individual]

    callback.notify(algorithm)

    entry = recorder.record.call_args[0][0]
    assert entry["bot"]["long"]["a"] == 0.25
    assert entry["suite_metrics"] == {"scenario_labels": ["base"]}
    assert "coins" not in entry["backtest"]
    assert entry["metrics"]["objectives"] == {"w_0": -1.0}


def test_callback_falls_back_to_population_when_offspring_missing():
    recorder = MagicMock()
    callback = PymooRecorderCallback(
        recorder=recorder,
        template={},
        build_config_fn=_build_config,
        overrides_fn=object(),
    )
    individual = MagicMock()
    individual.X = np.asarray([1.0, 2.0])
    individual.data = {"metrics": {"constraint_violation": 0.0}}
    algorithm = MagicMock()
    algorithm.off = None
    algorithm.pop = [individual]

    callback.notify(algorithm)

    assert recorder.record.call_count == 1
