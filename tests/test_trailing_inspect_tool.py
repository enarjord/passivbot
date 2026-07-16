import json

import pytest

from tools import trailing_inspect


def _params():
    return {
        "entry": {
            "threshold_base_pct": 0.02,
            "threshold_we_weight": 0.5,
            "threshold_volatility_1h_weight": 2.0,
            "threshold_volatility_1m_weight": 10.0,
            "retracement_base_pct": 0.005,
            "retracement_we_weight": 0.1,
            "retracement_volatility_1h_weight": 1.0,
            "retracement_volatility_1m_weight": 3.0,
        },
        "close": {
            "threshold_base_pct": 0.01,
            "threshold_we_weight": -0.02,
            "threshold_volatility_1h_weight": 1.0,
            "threshold_volatility_1m_weight": 2.0,
            "retracement_base_pct": 0.004,
            "retracement_volatility_1h_weight": 4.0,
            "retracement_volatility_1m_weight": 5.0,
        },
    }


def _inspect(pside="long"):
    return trailing_inspect.inspect_trailing(
        symbol="COIN",
        pside=pside,
        position_size=150.0,
        position_price=20.0,
        wallet_exposure=0.6,
        effective_wallet_exposure_limit=0.9,
        volatility_ema_1m=0.007,
        volatility_ema_1h=0.0033,
        params=_params(),
        parameter_source="test",
    )


def test_inspect_trailing_matches_current_entry_and_close_formulas():
    result = _inspect()

    we_ratio = 0.6 / 0.9
    entry_threshold_multiplier = 1.0 + 0.0033 * 2.0 + 0.007 * 10.0 + we_ratio * 0.5
    entry_retracement_multiplier = 1.0 + 0.0033 * 1.0 + 0.007 * 3.0 + we_ratio * 0.1
    close_threshold = 0.01 + we_ratio * -0.02 + 0.0033 * 1.0 + 0.007 * 2.0
    close_retracement_multiplier = 1.0 + 0.0033 * 4.0 + 0.007 * 5.0

    assert result["wallet_exposure_ratio"] == pytest.approx(we_ratio)
    assert result["entry"]["threshold_multiplier"]["effective"] == pytest.approx(
        entry_threshold_multiplier
    )
    assert result["entry"]["threshold_pct"] == pytest.approx(
        0.02 * entry_threshold_multiplier
    )
    assert result["entry"]["retracement_pct"] == pytest.approx(
        0.005 * entry_retracement_multiplier
    )
    assert result["close"]["threshold_pct"] == pytest.approx(close_threshold)
    assert result["close"]["retracement_pct"] == pytest.approx(
        0.004 * close_retracement_multiplier
    )


def test_long_geometry_distinguishes_confirmation_from_order_reference():
    result = _inspect()
    entry = result["entry"]
    threshold = entry["threshold_pct"]
    retracement = entry["retracement_pct"]
    threshold_price = 20.0 * (1.0 - threshold)

    assert entry["geometry"]["threshold_direction"] == "below"
    assert entry["geometry"]["retracement_direction"] == "above"
    assert entry["geometry"]["threshold_price"] == pytest.approx(threshold_price)
    assert entry["geometry"]["nominal_confirmation_price"] == pytest.approx(
        threshold_price * (1.0 + retracement)
    )
    assert entry["geometry"]["order_reference_price"] == pytest.approx(
        20.0 * (1.0 - threshold + retracement)
    )
    assert entry["geometry"]["nominal_confirmation_price"] != pytest.approx(
        entry["geometry"]["order_reference_price"]
    )


def test_short_reverses_entry_and_close_directions():
    result = _inspect("short")

    assert result["entry"]["geometry"]["threshold_direction"] == "above"
    assert result["entry"]["geometry"]["retracement_direction"] == "below"
    assert result["close"]["geometry"]["threshold_direction"] == "below"
    assert result["close"]["geometry"]["retracement_direction"] == "above"


def test_non_positive_retracement_reports_passive_mode():
    params = _params()
    params["entry"]["retracement_base_pct"] = 0.0
    params["close"]["retracement_base_pct"] = -0.01

    result = trailing_inspect.inspect_trailing(
        symbol="COIN",
        pside="long",
        position_size=None,
        position_price=20.0,
        wallet_exposure=0.6,
        effective_wallet_exposure_limit=0.9,
        volatility_ema_1m=0.007,
        volatility_ema_1h=0.0033,
        params=params,
        parameter_source="test",
    )

    assert result["entry"]["trailing_enabled"] is False
    assert result["entry"]["retracement_pct"] == 0.0
    assert result["close"]["trailing_enabled"] is False
    assert result["close"]["retracement_pct"] == 0.0
    report = trailing_inspect.render_report(result)
    assert report.count("trailing disabled") == 2
    assert report.count("Passive threshold/reference") == 2


def test_extract_strategy_params_requires_canonical_trailing_martingale():
    config = {
        "live": {"strategy_kind": "trailing_martingale"},
        "bot": {
            "long": {
                "strategy": {
                    "trailing_martingale": _params(),
                }
            }
        },
    }

    assert trailing_inspect._extract_strategy_params(config, "long") == _params()
    config["live"]["strategy_kind"] = "ema_anchor"
    with pytest.raises(ValueError, match="supports only"):
        trailing_inspect._extract_strategy_params(config, "long")


def test_main_json_applies_parameter_override(monkeypatch, capsys):
    monkeypatch.setattr(
        trailing_inspect,
        "load_parameter_source",
        lambda config_path, pside: (_params(), "test defaults"),
    )

    assert (
        trailing_inspect.main(
            [
                "--position-price",
                "20",
                "--wallet-exposure",
                "0.6",
                "--effective-wallet-exposure-limit",
                "0.9",
                "--entry-threshold-base-pct",
                "0.03",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["entry"]["threshold_base_pct"] == pytest.approx(0.03)
    assert payload["overridden_parameters"] == ["entry.threshold_base_pct"]


def test_main_rejects_non_positive_effective_limit(monkeypatch, capsys):
    monkeypatch.setattr(
        trailing_inspect,
        "load_parameter_source",
        lambda config_path, pside: (_params(), "test defaults"),
    )

    assert (
        trailing_inspect.main(
            [
                "--position-price",
                "20",
                "--wallet-exposure",
                "0.6",
                "--effective-wallet-exposure-limit",
                "0",
            ]
        )
        == 2
    )
    assert "must be greater than zero" in capsys.readouterr().err
