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


def _context(*, active=True, cooldown=1.5, forced_mode="normal"):
    return {
        "active": active,
        "total_wallet_exposure_limit": 1.0 if active else 0.0,
        "n_positions": 1,
        "forced_mode": forced_mode,
        "entry_cooldown_minutes": cooldown,
        "entry_ema_gate_mode": "all",
        "entry_initial_ema_dist": 0.01,
        "entry_initial_qty_pct": 0.02,
        "entry_double_down_factor": 1.5,
        "close_qty_pct": 0.25,
        "volatility_ema_span_1m": 100.0,
        "volatility_ema_span_1h": 200.0,
    }


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
    assert report.count("Passive analytical reference") == 2


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


@pytest.mark.parametrize(
    "state_args",
    (
        ("--wallet-exposure", "0.6"),
        ("--effective-wallet-exposure-limit", "0.9"),
        ("--volatility-ema-1m", "0.01"),
    ),
)
def test_detailed_mode_requires_both_exposure_inputs(state_args, capsys):
    assert trailing_inspect.main([*state_args, "--json"]) == 2
    assert (
        "requires both --wallet-exposure and --effective-wallet-exposure-limit"
        in capsys.readouterr().err
    )


def test_detailed_mode_requires_explicit_position_price(capsys):
    assert (
        trailing_inspect.main(
            [
                "--wallet-exposure",
                "0.6",
                "--effective-wallet-exposure-limit",
                "0.9",
                "--json",
            ]
        )
        == 2
    )
    assert "requires an explicit --position-price" in capsys.readouterr().err


def test_overview_scenario_grid_uses_formulas_and_side_geometry():
    result = trailing_inspect.build_overview(
        sources={
            "long": {"params": _params(), "context": _context()},
            "short": {"params": _params(), "context": _context(active=False)},
        },
        parameter_source="test config",
        price_anchor=100.0,
    )

    assert result["mode"] == "overview"
    assert len(result["sides"]["long"]["scenarios"]) == 9
    normal_half = next(
        row
        for row in result["sides"]["long"]["scenarios"]
        if row["volatility_label"] == "normal" and row["exposure_ratio"] == 0.5
    )
    expected_entry_threshold = 0.02 * (1.0 + 0.0025 * 2.0 + 0.005 * 10.0 + 0.5 * 0.5)
    expected_close_threshold = 0.01 + 0.5 * -0.02 + 0.0025 * 1.0 + 0.005 * 2.0
    assert normal_half["entry"]["threshold_pct"] == pytest.approx(expected_entry_threshold)
    assert normal_half["entry"]["geometry"]["threshold_price"] == pytest.approx(
        100.0 * (1.0 - expected_entry_threshold)
    )
    assert normal_half["close"]["threshold_pct"] == pytest.approx(expected_close_threshold)

    short_row = result["sides"]["short"]["scenarios"][0]
    assert short_row["entry"]["geometry"]["threshold_price"] > 100.0
    assert short_row["close"]["geometry"]["threshold_price"] < 100.0
    assert "dormant parameters" in " ".join(
        result["sides"]["short"]["classification"]["overall_comments"]
    )


def test_extract_side_context_includes_global_forced_mode():
    config = {
        "live": {"forced_mode_long": "p"},
        "bot": {
            "long": {
                "risk": {
                    "total_wallet_exposure_limit": 1.0,
                    "n_positions": 1,
                    "entry_cooldown_minutes": 2.0,
                },
                "strategy": {
                    "trailing_martingale": {
                        **_params(),
                        "volatility_ema_span_1m": 100.0,
                        "volatility_ema_span_1h": 200.0,
                    }
                },
            }
        },
    }

    context = trailing_inspect._extract_side_context(config, "long")
    assert context["forced_mode"] == "panic"


@pytest.mark.parametrize(
    ("forced_mode", "expected"),
    (
        ("panic", "immediate panic close"),
        ("manual", "emits no strategy orders"),
        ("tp_only", "ENTRY table is dormant and the CLOSE table applies"),
        ("graceful_stop", "blocks a new initial entry while flat"),
    ),
)
def test_overview_explains_global_forced_mode(forced_mode, expected):
    result = trailing_inspect.build_overview(
        sources={
            "long": {
                "params": _params(),
                "context": _context(forced_mode=forced_mode),
            }
        },
        parameter_source="test config",
        price_anchor=100.0,
    )

    comments = " ".join(result["sides"]["long"]["classification"]["overall_comments"])
    assert f"Global forced mode '{forced_mode}'" in comments
    assert expected in comments
    assert "Long is enabled" not in comments


def test_overview_qualifies_immediate_close_by_trailing_mode():
    result = trailing_inspect.build_overview(
        sources={"long": {"params": _params(), "context": _context()}},
        parameter_source="test config",
        price_anchor=100.0,
    )

    report = trailing_inspect.render_overview(result)
    assert "With close trailing enabled, a non-positive threshold is immediate" in report
    assert "With close trailing disabled, the row is passive" in report
    assert "no extrema or reversal confirmation participate" in report


@pytest.mark.parametrize(
    ("cooldown", "expected"),
    [
        (0.0, "full recursive entry ladder simultaneously"),
        (5.0, "waits 5 minutes after an entry fill"),
    ],
)
def test_overview_explains_passive_entry_ladder_and_cooldown(cooldown, expected):
    params = _params()
    params["entry"]["retracement_base_pct"] = 0.0
    result = trailing_inspect.build_overview(
        sources={"long": {"params": params, "context": _context(cooldown=cooldown)}},
        parameter_source="test config",
        price_anchor=100.0,
    )

    comments = " ".join(result["sides"]["long"]["classification"]["entry_comments"])
    assert expected in comments


def test_main_positional_config_defaults_to_overview_and_accepts_price_anchor(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        trailing_inspect,
        "load_overview_sources",
        lambda config_path, psides: (
            {
                pside: {"params": _params(), "context": _context()}
                for pside in psides
            },
            f"config {config_path}",
        ),
    )

    assert (
        trailing_inspect.main(
            ["config.json", "--side", "long", "--price-anchor", "250", "--json"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "overview"
    assert payload["price_anchor"] == pytest.approx(250.0)
    assert list(payload["sides"]) == ["long"]


def test_render_overview_includes_formula_caveats_and_scenario_prices():
    result = trailing_inspect.build_overview(
        sources={"long": {"params": _params(), "context": _context()}},
        parameter_source="test config",
        price_anchor=100.0,
    )

    report = trailing_inspect.render_overview(result)
    assert "Entry distance = base × max(1" in report
    assert "Close threshold = base +" in report
    assert "R confirm*" in report
    assert "Order ref*" in report
    assert "Trailing extrema reset after every fill" in report


def test_overview_single_custom_scenario_does_not_invent_sensitivity_comparison():
    result = trailing_inspect.build_overview(
        sources={"long": {"params": _params(), "context": _context()}},
        parameter_source="test config",
        price_anchor=100.0,
        volatility_scenarios=(("stress", 0.02, 0.01),),
        exposure_ratios=(0.75,),
    )

    classification = result["sides"]["long"]["classification"]
    assert classification["entry_headline"].startswith("one volatility scenario shown")
    assert classification["basis"] == "Headlines use 'stress' volatility at 75% WE/WEL."
    assert "Only one exposure ratio is shown" in " ".join(classification["entry_comments"])


def test_overview_sizing_description_matches_current_position_formula():
    result = trailing_inspect.build_overview(
        sources={"long": {"params": _params(), "context": _context()}},
        parameter_source="test config",
        price_anchor=100.0,
    )

    comments = " ".join(result["sides"]["long"]["classification"]["overall_comments"])
    assert "current absolute position size × 1.5 or initial-entry quantity" in comments
    assert "minimum-quantity, rounding, and exposure-cropping rules" in comments
    assert "previous fill" not in comments


def test_volatility_sensitivity_includes_retracement_changes():
    params = _params()
    for kind in ("entry", "close"):
        params[kind]["threshold_volatility_1m_weight"] = 0.0
        params[kind]["threshold_volatility_1h_weight"] = 0.0
        params[kind]["retracement_volatility_1m_weight"] = 100.0
        params[kind]["retracement_volatility_1h_weight"] = 100.0
    result = trailing_inspect.build_overview(
        sources={"long": {"params": params, "context": _context()}},
        parameter_source="test config",
        price_anchor=100.0,
    )

    classification = result["sides"]["long"]["classification"]
    assert classification["entry_headline"].startswith("very strong volatility sensitivity")
    assert classification["close_headline"].startswith("very strong volatility sensitivity")


def test_volatility_sensitivity_selects_extremes_independent_of_input_order():
    params = _params()
    params["entry"]["threshold_volatility_1m_weight"] = 20.0
    params["entry"]["threshold_volatility_1h_weight"] = 20.0
    params["entry"]["retracement_volatility_1m_weight"] = 0.0
    params["entry"]["retracement_volatility_1h_weight"] = 0.0
    result = trailing_inspect.build_overview(
        sources={"long": {"params": params, "context": _context()}},
        parameter_source="test config",
        price_anchor=100.0,
        volatility_scenarios=(
            ("quiet", 0.001, 0.0005),
            ("extreme", 0.05, 0.025),
            ("normal", 0.005, 0.0025),
        ),
    )

    headline = result["sides"]["long"]["classification"]["entry_headline"]
    assert headline.startswith("very strong volatility sensitivity")
    assert "sensitivity spans all 3 displayed scenarios" in result["sides"]["long"][
        "classification"
    ]["basis"]


def test_volatility_sensitivity_includes_effective_middle_extreme():
    params = _params()
    params["entry"].update(
        {
            "threshold_volatility_1m_weight": 20.0,
            "threshold_volatility_1h_weight": 0.0,
            "retracement_volatility_1m_weight": 0.0,
            "retracement_volatility_1h_weight": 0.0,
        }
    )
    result = trailing_inspect.build_overview(
        sources={"long": {"params": params, "context": _context()}},
        parameter_source="test config",
        price_anchor=100.0,
        volatility_scenarios=(
            ("quiet", 0.0, 0.0),
            ("large 1m effect", 0.05, 0.0),
            ("larger summed volatility", 0.0, 0.06),
        ),
    )

    headline = result["sides"]["long"]["classification"]["entry_headline"]
    assert headline.startswith("very strong volatility sensitivity")


@pytest.mark.parametrize(("pside", "expected_reference"), [("long", 98.0), ("short", 102.0)])
def test_passive_negative_close_threshold_keeps_rust_analytical_reference(
    pside, expected_reference
):
    params = _params()
    params["close"].update(
        {
            "retracement_base_pct": 0.0,
            "threshold_base_pct": -0.02,
            "threshold_we_weight": 0.0,
            "threshold_volatility_1m_weight": 0.0,
            "threshold_volatility_1h_weight": 0.0,
        }
    )
    result = trailing_inspect.build_overview(
        sources={pside: {"params": params, "context": _context()}},
        parameter_source="test config",
        price_anchor=100.0,
        volatility_scenarios=(("only", 0.0, 0.0),),
        exposure_ratios=(0.0,),
    )

    close = result["sides"][pside]["scenarios"][0]["close"]
    assert close["geometry"]["threshold_price"] is None
    assert close["geometry"]["passive_reference_price"] == pytest.approx(expected_reference)
    table_row = trailing_inspect._scenario_rows(result["sides"][pside], "close")[0]
    assert table_row[4] == "n/a"
    assert table_row[7] == f"{expected_reference:.4f}"
    headline = result["sides"][pside]["classification"]["close_headline"]
    assert "passive" in headline
    assert "immediate" not in headline


def test_negative_threshold_exposure_change_uses_absolute_distance():
    params = _params()
    params["close"].update(
        {
            "retracement_base_pct": 0.0,
            "threshold_base_pct": -0.02,
            "threshold_we_weight": -0.1,
            "threshold_volatility_1m_weight": 0.0,
            "threshold_volatility_1h_weight": 0.0,
        }
    )
    result = trailing_inspect.build_overview(
        sources={"long": {"params": params, "context": _context()}},
        parameter_source="test config",
        price_anchor=100.0,
        volatility_scenarios=(("only", 0.0, 0.0),),
        exposure_ratios=(0.0, 0.9),
    )

    comments = " ".join(result["sides"]["long"]["classification"]["close_comments"])
    assert "widens the absolute close threshold distance from 2.0000% to 11.0000%" in comments
    assert "narrows the close threshold" not in comments


@pytest.mark.parametrize(
    ("pside", "expected_reference"), [("long", 102.0), ("short", 98.0)]
)
def test_passive_negative_entry_threshold_keeps_rust_analytical_reference(
    pside, expected_reference
):
    params = _params()
    params["entry"].update(
        {
            "retracement_base_pct": 0.0,
            "threshold_base_pct": -0.02,
            "threshold_we_weight": 0.0,
            "threshold_volatility_1m_weight": 0.0,
            "threshold_volatility_1h_weight": 0.0,
        }
    )
    result = trailing_inspect.build_overview(
        sources={pside: {"params": params, "context": _context()}},
        parameter_source="test config",
        price_anchor=100.0,
        volatility_scenarios=(("only", 0.0, 0.0),),
        exposure_ratios=(0.0,),
    )

    entry = result["sides"][pside]["scenarios"][0]["entry"]
    assert entry["threshold_pct"] == pytest.approx(-0.02)
    assert entry["geometry"]["threshold_price"] is None
    assert entry["geometry"]["passive_reference_price"] == pytest.approx(expected_reference)
    table_row = trailing_inspect._scenario_rows(result["sides"][pside], "entry")[0]
    assert table_row[4] == "n/a"
    assert table_row[7] == f"{expected_reference:.4f}"
    headline = result["sides"][pside]["classification"]["entry_headline"]
    assert "passive" in headline
    assert "immediate" not in headline


def test_trailing_negative_entry_threshold_is_clamped_like_rust():
    params = _params()
    params["entry"].update(
        {
            "retracement_base_pct": 0.01,
            "threshold_base_pct": -0.02,
            "threshold_we_weight": 0.0,
            "threshold_volatility_1m_weight": 0.0,
            "threshold_volatility_1h_weight": 0.0,
        }
    )

    result = trailing_inspect.build_overview(
        sources={"long": {"params": params, "context": _context()}},
        parameter_source="test config",
        price_anchor=100.0,
        volatility_scenarios=(("only", 0.0, 0.0),),
        exposure_ratios=(0.0,),
    )

    entry = result["sides"]["long"]["scenarios"][0]["entry"]
    assert entry["trailing_enabled"] is True
    assert entry["threshold_base_pct"] == 0.0
    assert entry["threshold_pct"] == 0.0


def test_scenario_prices_preserve_precision_for_small_anchor():
    result = trailing_inspect.build_overview(
        sources={"long": {"params": _params(), "context": _context()}},
        parameter_source="test config",
        price_anchor=0.00001,
        volatility_scenarios=(("only", 0.005, 0.0025),),
        exposure_ratios=(0.5,),
    )

    row = trailing_inspect._scenario_rows(result["sides"]["long"], "entry")[0]
    for price in (row[4], row[6], row[7]):
        assert price != "0.0000"
        assert float(price) > 0.0


def test_scenario_inputs_preserve_small_custom_values():
    params = _params()
    params["entry"]["threshold_volatility_1m_weight"] = 100_000.0
    params["entry"]["threshold_we_weight"] = 100.0
    result = trailing_inspect.build_overview(
        sources={"long": {"params": params, "context": _context()}},
        parameter_source="test config",
        price_anchor=100.0,
        volatility_scenarios=(("zero", 0.0, 0.0), ("tiny", 0.00001, 0.0)),
        exposure_ratios=(0.0, 0.004),
    )

    rows = trailing_inspect._scenario_rows(result["sides"]["long"], "entry")
    assert {row[1] for row in rows} == {"0/0%", "0.001/0%"}
    assert {row[2] for row in rows} == {"0%", "0.4%"}
    assert len({row[3] for row in rows}) == 4


def test_overview_rejects_duplicate_volatility_scenario_labels():
    with pytest.raises(ValueError, match="duplicate volatility scenario label.*same"):
        trailing_inspect.build_overview(
            sources={"long": {"params": _params(), "context": _context()}},
            parameter_source="test config",
            price_anchor=100.0,
            volatility_scenarios=(
                ("same", 0.001, 0.0005),
                ("same", 0.02, 0.01),
            ),
        )
