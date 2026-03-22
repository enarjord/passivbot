from pathlib import Path

import pytest

from trailing_diagnostics import (
    build_trailing_diagnostic,
    build_trailing_inputs_from_snapshot,
)
from trailing_diagnostics_tool import (
    TrailingDiagnosticsState,
    WIZARD_CORE_KEYS,
    execute_command,
    render_screen,
)


def _sample_config():
    params = {
        "entry_grid_double_down_factor": 1.0,
        "entry_grid_spacing_volatility_weight": 0.0,
        "entry_grid_spacing_we_weight": 0.0,
        "entry_grid_spacing_pct": 0.01,
        "entry_initial_ema_dist": 0.01,
        "entry_initial_qty_pct": 0.1,
        "entry_trailing_double_down_factor": 1.0,
        "entry_trailing_grid_ratio": 1.0,
        "entry_trailing_retracement_pct": 0.01,
        "entry_trailing_retracement_we_weight": 0.0,
        "entry_trailing_retracement_volatility_weight": 0.0,
        "entry_trailing_threshold_pct": 0.01,
        "entry_trailing_threshold_we_weight": 0.0,
        "entry_trailing_threshold_volatility_weight": 0.0,
        "risk_we_excess_allowance_pct": 0.0,
        "close_grid_markup_end": 0.01,
        "close_grid_markup_start": 0.01,
        "close_grid_qty_pct": 1.0,
        "close_trailing_grid_ratio": 1.0,
        "close_trailing_qty_pct": 1.0,
        "close_trailing_retracement_pct": 0.01,
        "close_trailing_threshold_pct": 0.01,
        "risk_wel_enforcer_threshold": 0.0,
        "total_wallet_exposure_limit": 2.0,
        "n_positions": 10,
    }
    return {"bot": {"long": dict(params), "short": dict(params)}}


def _sample_snapshot():
    return {
        "payload": {
            "account": {"balance_raw": 1000.0},
            "positions": {
                "BTC/USDT:USDT": {
                    "long": {"size": 0.001, "price": 100000.0},
                    "short": {"size": 0.0, "price": 0.0},
                }
            },
            "market": {
                "BTC/USDT:USDT": {
                    "last_price": 100500.0,
                    "qty_step": 0.001,
                    "price_step": 0.1,
                    "min_qty": 0.001,
                    "min_cost": 1.0,
                    "effective_min_cost": 5.0,
                    "c_mult": 1.0,
                    "entry_volatility_logrange_ema": {"long": 0.0, "short": 0.0},
                    "ema_bands": {
                        "long": {"lower": 100200.0, "upper": 100400.0},
                        "short": {"lower": 100200.0, "upper": 100400.0},
                    },
                    "trailing": {
                        "long": {
                            "min_since_open": 99500.0,
                            "max_since_min": 100800.0,
                            "max_since_open": 100900.0,
                            "min_since_max": 100100.0,
                        }
                    },
                }
            },
            "trailing": {
                "BTC/USDT:USDT": {
                    "long": {
                        "extrema": {
                            "min_since_open": 99500.0,
                            "max_since_min": 100800.0,
                            "max_since_open": 100900.0,
                            "min_since_max": 100100.0,
                        }
                    }
                }
            },
        }
    }


def test_build_trailing_inputs_from_snapshot_extracts_required_fields():
    inputs = build_trailing_inputs_from_snapshot(
        _sample_config(),
        _sample_snapshot(),
        symbol="BTC/USDT:USDT",
        pside="long",
    )

    assert inputs["symbol"] == "BTC/USDT:USDT"
    assert inputs["pside"] == "long"
    assert inputs["balance_raw"] == pytest.approx(1000.0)
    assert inputs["current_price"] == pytest.approx(100500.0)
    assert inputs["c_mult"] == pytest.approx(1.0)
    assert inputs["wallet_exposure_limit"] == pytest.approx(0.2)
    assert inputs["h1_log_range_ema"] == pytest.approx(0.0)
    assert inputs["ema_lower"] == pytest.approx(100200.0)
    assert inputs["min_since_open"] == pytest.approx(99500.0)


def test_build_trailing_diagnostic_matches_monitor_slice():
    inputs = build_trailing_inputs_from_snapshot(
        _sample_config(),
        _sample_snapshot(),
        symbol="BTC/USDT:USDT",
        pside="long",
    )

    diagnostic = build_trailing_diagnostic(inputs)

    assert diagnostic["entry"]["order_type"] == "entry_trailing_normal_long"
    assert diagnostic["entry"]["threshold_met"] is False
    assert diagnostic["entry"]["retracement_met"] is True
    assert diagnostic["close"]["order_type"] == "close_trailing_long"
    assert diagnostic["close"]["threshold_met"] is False


def test_trailing_diagnostics_tool_commands_render_and_set_values(tmp_path, monkeypatch):
    inputs = build_trailing_inputs_from_snapshot(
        _sample_config(),
        _sample_snapshot(),
        symbol="BTC/USDT:USDT",
        pside="long",
    )
    state = TrailingDiagnosticsState(
        source_label="test",
        symbol="BTC/USDT:USDT",
        pside="long",
        baseline_inputs=dict(inputs),
        inputs=dict(inputs),
    )

    rendered = render_screen(state, width=140)
    assert "Entry" in rendered
    assert "Close" in rendered
    assert "Config Inputs" in rendered
    assert "set <key> <value>" in rendered

    assert execute_command(state, "set current_price 99999") is False
    assert state.inputs["current_price"] == pytest.approx(99999.0)
    assert "Set current_price=99999.0" in state.status_lines[0]
    assert execute_command(state, "edit current_price 100001") is False
    assert state.inputs["current_price"] == pytest.approx(100001.0)

    monkeypatch.chdir(tmp_path)
    assert execute_command(state, "dump") is False
    dump_files = list((tmp_path / "tmp").glob("trailing_diagnostics_dump_*.json"))
    assert dump_files
    assert execute_command(state, "reset") is False
    assert state.inputs["current_price"] == pytest.approx(inputs["current_price"])


def test_wizard_core_keys_do_not_repeat_shared_params():
    assert WIZARD_CORE_KEYS.count("wallet_exposure_limit") == 1
    assert WIZARD_CORE_KEYS.count("risk_we_excess_allowance_pct") == 1
