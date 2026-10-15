"""Sizing intent survives CLI normalization and reaches native dispatch as integers."""

import argparse
from types import SimpleNamespace

import pytest

from config import prepare_config
from config.gpu import resolve_gpu_sizing
from config_utils import add_config_arguments, update_config_with_args
from test_native_backend import execute, guard_cpu, inputs, managed_arrays, FakeService


@pytest.mark.parametrize("batch", ["64", "64.0", "auto"])
@pytest.mark.parametrize("dispatch", ["2000000", "auto"])
def test_cli_sizing_reaches_native_evolution(monkeypatch, tmp_path, batch, dispatch):
    guard_cpu(monkeypatch)
    import optimization.gpu.native as native
    observed = []
    class ObserveSizing(FakeService):
        def __init__(self, **kwargs):
            observed.append((kwargs["batch_size"], kwargs["max_dispatch_candidate_bars"]))
            # The device facade resolves automatic widths; this offline replay
            # harness uses a fixed executor after observing the facade input.
            if kwargs["batch_size"] is None:
                kwargs["batch_size"] = 64
            super().__init__(**kwargs)
    monkeypatch.setattr(native, "NativeBacktestService", ObserveSizing)
    with managed_arrays() as manager:
        base = inputs(manager)
        # Canonical defaults for both fields are nullable; test that surface.
        base.config["optimize"]["gpu"].update(batch_size=None, max_dispatch_candidate_bars=None)
        parser = argparse.ArgumentParser()
        allowed = add_config_arguments(parser, base.config, command="optimize", help_all=True)
        args = parser.parse_args(["--optimize.gpu.batch_size", batch,
                                  "--optimize.gpu.max_dispatch_candidate_bars", dispatch])
        update_config_with_args(base.config, args, allowed_keys=allowed)
        prepared = prepare_config(base.config, verbose=False)
        base.config["optimize"]["gpu"] = prepared["optimize"]["gpu"]
        records = []
        execute(base, SimpleNamespace(record=records.append), tmp_path / "checkpoint.pkl")
        assert len(records) == 8
        expected = (None if batch == "auto" else 64, 500_000_000 if dispatch == "auto" else 2_000_000)
        assert observed == [expected]
        assert type(observed[0][1]) is int
        assert observed[0][0] is None or type(observed[0][0]) is int


@pytest.mark.parametrize("field", ["batch_size", "max_dispatch_candidate_bars"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.9, float("nan"), float("inf"), "64", [], {}])
def test_invalid_sizing_fails_before_device_construction(field, value):
    with pytest.raises(ValueError, match=f"optimize.gpu.{field}.*positive integer or auto"):
        resolve_gpu_sizing(value, field=field)


@pytest.mark.parametrize("value,expected", [(None, None), (" auto ", None), (64.0, 64), (64, 64)])
def test_config_sizing_preserves_auto_or_normalizes_integer_values(value, expected):
    assert resolve_gpu_sizing(value, field="batch_size") == expected
