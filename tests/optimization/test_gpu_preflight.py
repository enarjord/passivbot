"""Unsupported GPU simulation policies fail on CPU before device preparation."""

from copy import deepcopy
import builtins

import pytest

from config import get_template_config
from config.gpu import validate_gpu_backtest_config
from optimize import _run_gpu_preparation_preflight


@pytest.mark.parametrize("collateral", [0.5, -0.1, float("nan"), float("inf")])
@pytest.mark.parametrize("entrypoint", ["parameters", "replay"])
def test_unsupported_collateral_rejected_before_optional_device_imports(
    monkeypatch, collateral, entrypoint
):
    from optimization.gpu.parameters import prepare_candidate_parameters
    from optimization.gpu.service import MpsMulticoinProxy

    config = get_template_config()
    config["backtest"]["btc_collateral_cap"] = collateral
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "cupy"}:
            pytest.fail("unsupported GPU config reached device preparation")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="btc_collateral_cap=0"):
        if entrypoint == "parameters":
            prepare_candidate_parameters(config, {}, "binance")
        else:
            MpsMulticoinProxy(config=config, hlcvs=None, mss={}, btc=None,
                              timestamps=None, exchange="binance", batch_size=1,
                              needed_metrics=["adg_strategy_eq"])


def test_suite_overrides_cannot_hide_unsupported_collateral():
    config = get_template_config()
    config["optimize"]["backend"] = "gpu"
    original = deepcopy(config)
    with pytest.raises(ValueError, match="btc_collateral_cap=0"):
        _run_gpu_preparation_preflight(config, dict(enabled=True, scenarios=[
            dict(label="cash", overrides={}),
            dict(label="collateral", overrides={"backtest.btc_collateral_cap": 0.5}),
        ]))
    assert config == original


def test_unsupported_strategy_has_an_explicit_cpu_option():
    config = get_template_config()
    config["live"]["strategy_kind"] = "trailing_grid_v7"
    with pytest.raises(ValueError, match="strategy.*CPU backend"):
        validate_gpu_backtest_config(config)


def test_screening_labels_checked_without_loading_device_runtime(monkeypatch):
    config = get_template_config()
    config["optimize"]["backend"] = "gpu"
    config["optimize"]["gpu"]["screening"]["scenarios"] = ["missing"]
    with pytest.raises(ValueError, match="unknown labels.*missing"):
        _run_gpu_preparation_preflight(config, dict(enabled=True, scenarios=[dict(label="base")]))
