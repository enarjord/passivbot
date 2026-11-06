"""Offline projection availability checks against real native diagnostics."""

from copy import deepcopy
import json
import math
import sys

import pytest

import trailing_diagnostics as diagnostics
from test_trailing_diagnostics import (
    _hype_trailing_martingale_close_inputs,
    _sample_config,
    _sample_snapshot,
)


DEFAULT_EXTREMA = {
    "min_since_open": sys.float_info.max,
    "max_since_min": 0.0,
    "max_since_open": 0.0,
    "min_since_max": sys.float_info.max,
}
FINITE_EXTREMA = {
    "min_since_open": 99.0,
    "max_since_min": 101.0,
    "max_since_open": 102.0,
    "min_since_max": 100.0,
}


def _native_case(monkeypatch, builder_kind, side, extrema, *, amend=None):
    assert not getattr(diagnostics.pbr, "__is_stub__", False)
    if builder_kind == "martingale_close":
        inputs = _hype_trailing_martingale_close_inputs()
        inputs["symbol"] = "BTC/USDT:USDT"
        inputs["pside"] = side
        inputs["position"]["size"] = abs(inputs["position"]["size"]) * (
            1 if side == "long" else -1
        )
        inputs["trailing"] = deepcopy(extrema)
        function = "calc_trailing_martingale_close_diagnostic_py"
        builder = diagnostics.build_trailing_martingale_close_diagnostic
    else:
        inputs = diagnostics.build_trailing_inputs_from_snapshot(
            _sample_config(), _sample_snapshot(), symbol="BTC/USDT:USDT", pside="long"
        )
        inputs["pside"] = side
        inputs["position_size"] = abs(inputs["position_size"]) * (
            1 if side == "long" else -1
        )
        inputs.update(extrema)
        function = f"calc_next_{builder_kind}_{side}_py"
        builder = getattr(diagnostics, f"build_trailing_{builder_kind}_diagnostic")
    if amend is not None:
        amend(inputs)
    native = getattr(diagnostics.pbr, function)
    calls, results = [], []

    def record(*args):
        calls.append(args)
        result = native(*args)
        results.append(result)
        return result

    monkeypatch.setattr(diagnostics.pbr, function, record)
    original = deepcopy(inputs)
    payload = builder(inputs)
    assert inputs == original
    assert len(calls) == 1
    if builder_kind == "martingale_close":
        assert json.loads(calls[0][0]) == original
        native_result = json.loads(results[0])
        for key in ("qty", "price", "order_type", "triggered", "selected_mode"):
            assert payload[key] == native_result[key]
        if extrema == FINITE_EXTREMA and amend is None:
            for key in ("threshold_met", "retracement_met", "status"):
                assert payload[key] == native_result[key]
    else:
        first_extremum = 20 if builder_kind == "entry" else 15
        assert calls[0][first_extremum:first_extremum + 4] == tuple(extrema.values())
        qty, price, order_type = results[0]
        assert (payload["qty"], payload["price"], payload["order_type"]) == (
            qty, price, order_type
        )
        expected_triggered = "trailing" in str(order_type).lower() and abs(qty) > 0 and price > 0
        assert payload["triggered"] is bool(expected_triggered)
    return payload


@pytest.mark.parametrize("builder_kind", ["entry", "close", "martingale_close"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_default_native_extrema_are_not_projected_as_trigger_prices(
    monkeypatch, builder_kind, side
):
    payload = _native_case(monkeypatch, builder_kind, side, DEFAULT_EXTREMA)

    assert payload["retracement_price"] is None
    assert payload["retracement_price_status"] == "unavailable"
    assert payload["retracement_price_reason"] == "trailing_extremum_uninitialized"
    assert payload["retracement_met"] is None
    assert payload["projection_status"] == "unavailable"
    assert payload["status"] != "armed"
    assert payload["extrema"] == DEFAULT_EXTREMA
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("builder_kind", ["entry", "close", "martingale_close"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_finite_projection_keeps_native_intent_and_exact_price_formula(
    monkeypatch, builder_kind, side
):
    payload = _native_case(monkeypatch, builder_kind, side, FINITE_EXTREMA)
    low_trigger = side == ("long" if builder_kind == "entry" else "short")
    base = FINITE_EXTREMA["min_since_open" if low_trigger else "max_since_open"]
    multiplier = 1 + payload["retracement_fraction"] * (1 if low_trigger else -1)

    assert payload["retracement_price"] == base * multiplier
    assert payload["retracement_price_status"] == "available"
    assert payload["projection_status"] == "available"
    assert payload["status"] != "unavailable"
    assert not any(key.endswith("_reason") for key in payload)
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    "builder_kind,side", [("entry", "long"), ("close", "short"), ("martingale_close", "short")]
)
def test_native_diagnostic_with_finite_overflowing_extremum_stays_strict_json(
    monkeypatch, builder_kind, side
):
    extrema = {**FINITE_EXTREMA, "min_since_open": sys.float_info.max * 0.999999}
    payload = _native_case(monkeypatch, builder_kind, side, extrema)

    assert payload["retracement_price"] is None
    assert payload["retracement_price_reason"] == "projected_price_overflow"
    assert payload["retracement_met"] is None
    assert payload["status"] != "armed"
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("field", ["threshold", "retracement"])
def test_native_null_fraction_is_unavailable_instead_of_disabling_condition(monkeypatch, field):
    def overflow_fraction(inputs):
        inputs["state"]["log_range_ema_1m"] = 1.0
        inputs["close_params"][f"{field}_base_fraction"] = 1e308
        inputs["close_params"][f"{field}_log_range_1m_coefficient"] = 1e308

    payload = _native_case(
        monkeypatch, "martingale_close", "long", FINITE_EXTREMA, amend=overflow_fraction
    )

    assert payload[f"{field}_fraction"] is None
    assert payload[f"{field}_fraction_status"] == "unavailable"
    assert payload[f"{field}_fraction_reason"] == "projection_fraction_unavailable"
    assert payload[f"{field}_price"] is None
    assert payload[f"{field}_met"] is None
    assert payload["projection_status"] == "unavailable"
    assert payload["status"] != "armed"
    json.dumps(payload, allow_nan=False)


def _projection(**changes):
    params = dict(
        low_trigger=True, position_price=100.0, current_price=100.0,
        threshold_fraction=0.01, retracement_fraction=0.01,
        extrema=deepcopy(FINITE_EXTREMA),
    )
    params.update(changes)
    result = diagnostics._trailing_projection_fields(**params)
    json.dumps(result, allow_nan=False)
    return result


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_projection_source_has_explicit_unavailability(invalid):
    result = _projection(extrema={**FINITE_EXTREMA, "min_since_open": invalid})

    assert result["retracement_price"] is None
    assert result["retracement_price_reason"] == "trailing_extremum_non_finite"
    assert result["threshold_met"] is None
    assert result["threshold_met_reason"] == "trailing_extremum_non_finite"


@pytest.mark.parametrize("low_trigger", [False, True])
def test_finite_arithmetic_overflow_is_not_an_initialization_sentinel(low_trigger):
    result = _projection(
        low_trigger=low_trigger, position_price=1e308, threshold_fraction=1.0,
        retracement_fraction=1.0, extrema={**FINITE_EXTREMA, "min_since_open": 1e308},
    )
    prefix = "retracement" if low_trigger else "threshold"

    assert result[f"{prefix}_price"] is None
    assert result[f"{prefix}_price_reason"] == "projected_price_overflow"
    assert result[f"{prefix}_met"] is None


def test_finite_price_ratio_overflow_is_explicitly_unavailable():
    result = _projection(position_price=1e-308, current_price=1e308)

    assert math.isfinite(result["threshold_price"])
    assert result["current_price_change_from_threshold_fraction"] is None
    assert result["current_price_change_from_threshold_fraction_reason"] == "projected_change_overflow"


@pytest.mark.parametrize("fraction", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_fraction_does_not_disable_the_condition(fraction):
    result = _projection(threshold_fraction=fraction)

    assert result["threshold_fraction"] is None
    assert result["threshold_price_status"] == "unavailable"
    assert result["threshold_price_reason"] == "projection_fraction_non_finite"
    assert result["threshold_met"] is None


def test_disabled_conditions_do_not_require_initialized_extrema():
    result = _projection(
        threshold_fraction=0.0, retracement_fraction=0.0, extrema=DEFAULT_EXTREMA
    )

    assert result["threshold_fraction"] == 0.0
    assert result["threshold_price"] is None
    assert result["threshold_price_status"] == "not_required"
    assert result["threshold_met"] is True
    assert result["threshold_met_status"] == "not_required"
    assert result["retracement_met"] is True
    assert result["projection_status"] == "available"


def test_genuine_zero_projection_and_zero_change_are_preserved():
    result = _projection(low_trigger=False, retracement_fraction=1.0, current_price=101.0)

    assert result["retracement_price"] == 0.0
    assert result["retracement_price_status"] == "available"
    assert result["current_price_change_from_threshold_fraction"] == 0.0
    assert result["current_price_change_from_threshold_fraction_status"] == "available"
