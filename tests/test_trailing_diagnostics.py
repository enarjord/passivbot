from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from trailing_diagnostics import (
    build_trailing_diagnostic,
    build_trailing_inputs_from_snapshot,
    build_trailing_martingale_close_diagnostic,
    selected_mode_from_order_type,
)
from trailing_diagnostics_tool import (
    TrailingDiagnosticsState,
    WIZARD_CORE_KEYS,
    create_state_from_sources,
    execute_command,
    render_screen,
)


@pytest.fixture
def require_real_passivbot_rust_module():
    import passivbot_rust as pbr

    if getattr(pbr, "__is_stub__", False):
        pytest.fail(
            "trailing-martingale parity tests require the real passivbot_rust extension"
        )


def _sample_config():
    params = {
        "entry_grid_double_down_factor": 1.0,
        "entry_weight_volatility_1h": 0.0,
        "entry_weight_volatility_1m": 0.0,
        "entry_we_weight": 0.0,
        "entry_grid_spacing_pct": 0.01,
        "entry_initial_ema_dist": 0.01,
        "entry_initial_qty_pct": 0.1,
        "entry_trailing_double_down_factor": 1.0,
        "entry_trailing_retracement_pct": 0.01,
        "entry_trailing_threshold_pct": 0.01,
        "risk_we_excess_allowance_pct": 0.0,
        "close_grid_qty_pct": 1.0,
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


def test_ema_anchor_snapshot_requires_explicit_trailing_wizard(tmp_path):
    snapshot = _sample_snapshot()
    snapshot["payload"]["trailing"] = {
        "_meta": {
            "diagnostics_supported": False,
            "strategy_kind": "ema_anchor",
            "reason": "strategy_has_no_trailing_diagnostics",
        }
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    config_path = Path(__file__).parents[1] / "configs" / "examples" / "ema_anchor.json"

    with pytest.raises(
        ValueError,
        match=(
            "snapshot trailing diagnostics are not supported for strategy "
            "'ema_anchor'.*use --wizard"
        ),
    ):
        create_state_from_sources(
            config_path=str(config_path),
            monitor_root=None,
            exchange=None,
            user=None,
            snapshot_path=str(snapshot_path),
            symbol=None,
            pside="long",
            wizard=False,
        )


def test_ema_anchor_wizard_state_cannot_reload_unsupported_snapshot(
    tmp_path, monkeypatch
):
    snapshot = _sample_snapshot()
    snapshot["payload"]["trailing"] = {
        "_meta": {
            "diagnostics_supported": False,
            "strategy_kind": "ema_anchor",
            "reason": "strategy_has_no_trailing_diagnostics",
        }
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    config_path = Path(__file__).parents[1] / "configs" / "examples" / "ema_anchor.json"
    monkeypatch.setattr(
        "trailing_diagnostics_tool.prompt_manual_wizard", lambda defaults: defaults
    )

    state = create_state_from_sources(
        config_path=str(config_path),
        monitor_root=None,
        exchange=None,
        user=None,
        snapshot_path=str(snapshot_path),
        symbol=None,
        pside="long",
        wizard=True,
    )

    assert state.can_reload_from_snapshot() is False
    assert execute_command(state, "side short") is False
    assert state.pside == "short"
    assert state.status_lines == ["Updated side to short in manual mode."]
    assert execute_command(state, "symbol BTC") is False
    assert state.symbol == "BTC/USDT:USDT"
    assert state.status_lines == [
        "symbol switching requires a supported snapshot + config source"
    ]


def _hype_trailing_martingale_close_inputs():
    return {
        "symbol": "HYPE/USDT:USDT",
        "pside": "long",
        "current_price": 60.6695,
        "exchange": {
            "qty_step": 0.1,
            "price_step": 0.001,
            "min_qty": 0.1,
            "min_cost": 1.0,
            "c_mult": 1.0,
            "maker_fee": 0.0002,
            "taker_fee": 0.00055,
        },
        "state": {
            "balance": 99.88140021,
            "order_book": {"bid": 60.6695, "ask": 60.6695},
            "ema_bands": {"upper": 60.6695, "lower": 60.6695},
            "volatility_ema_1m": 0.0014553489538740175,
            "volatility_ema_1h": 0.013108943219306139,
        },
        "bot_params": {
            "wallet_exposure_limit": 0.5,
            "total_wallet_exposure_limit": 1.5,
            "n_positions": 3,
            "risk_we_excess_allowance_pct": 0.66,
            "risk_we_excess_allowance_mode": "bounded",
            "risk_wel_enforcer_enabled": False,
            "risk_wel_enforcer_threshold": 1.0,
        },
        "runtime": {"effective_wallet_exposure_limit": 0.5},
        "close_params": {
            "qty_pct": 0.23,
            "threshold_base_pct": -0.0143,
            "threshold_we_weight": -0.0278,
            "threshold_volatility_1h_weight": 0.19,
            "threshold_volatility_1m_weight": 7.93,
            "retracement_base_pct": 0.0001,
            "retracement_volatility_1h_weight": 12.11,
            "retracement_volatility_1m_weight": 4.49,
        },
        "position": {"size": 0.1, "price": 60.675},
        "trailing": {
            "min_since_open": sys.float_info.max,
            "max_since_min": 0.0,
            "max_since_open": 0.0,
            "min_since_max": sys.float_info.max,
        },
    }


def test_trailing_martingale_close_diagnostic_uses_rust_dynamic_terms(
    require_real_passivbot_rust_module,
):
    inputs = _hype_trailing_martingale_close_inputs()

    diagnostic = build_trailing_martingale_close_diagnostic(inputs)

    wallet_exposure = 0.1 * 60.675 / inputs["state"]["balance"]
    effective_wel = 0.5 * 1.66
    wallet_exposure_ratio = wallet_exposure / effective_wel
    expected_threshold = (
        -0.0143
        + wallet_exposure_ratio * -0.0278
        + inputs["state"]["volatility_ema_1m"] * 7.93
        + inputs["state"]["volatility_ema_1h"] * 0.19
    )
    expected_retracement = 0.0001 * (
        1.0
        + inputs["state"]["volatility_ema_1m"] * 4.49
        + inputs["state"]["volatility_ema_1h"] * 12.11
    )

    assert diagnostic is not None
    assert diagnostic["threshold_pct"] == pytest.approx(expected_threshold)
    assert diagnostic["retracement_pct"] == pytest.approx(expected_retracement)
    assert sum(diagnostic["threshold_terms"].values()) == pytest.approx(expected_threshold)
    assert diagnostic["effective_wallet_exposure_limit"] == pytest.approx(effective_wel)
    assert diagnostic["wallet_exposure_ratio"] == pytest.approx(wallet_exposure_ratio)
    assert diagnostic["volatility_ema_1m"] == pytest.approx(
        inputs["state"]["volatility_ema_1m"]
    )
    assert diagnostic["volatility_ema_1h"] == pytest.approx(
        inputs["state"]["volatility_ema_1h"]
    )
    assert diagnostic["threshold_pct"] != pytest.approx(-0.014500652114747214)


def test_trailing_martingale_reset_extrema_cannot_emit_ordinary_trailing_close(
    require_real_passivbot_rust_module,
):
    inputs = _hype_trailing_martingale_close_inputs()

    reset = build_trailing_martingale_close_diagnostic(inputs)
    inputs["trailing"] = {
        "min_since_open": 60.0,
        "max_since_min": 61.0,
        "max_since_open": 61.0,
        "min_since_max": 60.0,
    }
    seeded = build_trailing_martingale_close_diagnostic(inputs)

    assert reset is not None
    assert reset["order_type"] == "close_trailing_long"
    assert reset["qty"] == pytest.approx(0.0)
    assert reset["triggered"] is False
    assert reset["status"] == "waiting_retracement"
    assert seeded is not None
    assert seeded["order_type"] == "close_trailing_long"
    assert seeded["qty"] < 0.0
    assert seeded["triggered"] is True
    assert seeded["status"] == "triggered"


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
    assert inputs["total_wallet_exposure_limit"] == pytest.approx(2.0)
    assert inputs["h1_log_range_ema"] == pytest.approx(0.0)
    assert inputs["ema_lower"] == pytest.approx(100200.0)
    assert inputs["min_since_open"] == pytest.approx(99500.0)


def test_snapshot_trailing_diagnostic_caps_excess_by_total_wallet_exposure_limit():
    config = _sample_config()
    side_cfg = config["bot"]["long"]
    side_cfg.pop("total_wallet_exposure_limit")
    side_cfg.pop("n_positions")
    side_cfg.pop("risk_we_excess_allowance_pct")
    side_cfg["risk"] = {
        "total_wallet_exposure_limit": 0.2,
        "n_positions": 1,
        "we_excess_allowance_pct": 0.5,
    }
    inputs = build_trailing_inputs_from_snapshot(
        config,
        _sample_snapshot(),
        symbol="BTC/USDT:USDT",
        pside="long",
    )

    diagnostic = build_trailing_diagnostic(inputs)

    assert inputs["wallet_exposure_limit"] == pytest.approx(0.2)
    assert inputs["total_wallet_exposure_limit"] == pytest.approx(0.2)
    assert diagnostic["allowed_wallet_exposure_limit"] == pytest.approx(0.2)
    assert diagnostic["entry"]["limit_cap"] == pytest.approx(0.2)


def test_snapshot_trailing_diagnostic_legacy_raw_excess_mode_is_unbounded():
    config = _sample_config()
    side_cfg = config["bot"]["long"]
    side_cfg.pop("total_wallet_exposure_limit")
    side_cfg.pop("n_positions")
    side_cfg.pop("risk_we_excess_allowance_pct")
    side_cfg["risk"] = {
        "total_wallet_exposure_limit": 0.2,
        "n_positions": 1,
        "we_excess_allowance_pct": 0.5,
        "we_excess_allowance_mode": "legacy_raw",
    }
    inputs = build_trailing_inputs_from_snapshot(
        config,
        _sample_snapshot(),
        symbol="BTC/USDT:USDT",
        pside="long",
    )

    diagnostic = build_trailing_diagnostic(inputs)

    assert inputs["risk_we_excess_allowance_mode"] == "legacy_raw"
    assert diagnostic["allowed_wallet_exposure_limit"] == pytest.approx(0.3)
    assert diagnostic["entry"]["limit_cap"] == pytest.approx(0.3)


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
    assert diagnostic["close"]["selected_mode"] == "trailing"
    assert diagnostic["close"]["threshold_met"] is False


def test_trailing_diagnostic_keeps_threshold_state_when_next_close_is_grid():
    config = _sample_config()
    config["bot"]["long"]["close_trailing_retracement_pct"] = 0.0
    inputs = build_trailing_inputs_from_snapshot(
        config,
        _sample_snapshot(),
        symbol="BTC/USDT:USDT",
        pside="long",
    )

    diagnostic = build_trailing_diagnostic(inputs)

    assert diagnostic["close"] is not None
    assert diagnostic["close"]["order_type"] == "close_grid_long"
    assert diagnostic["close"]["selected_mode"] == "grid"
    assert diagnostic["close"]["triggered"] is False
    assert diagnostic["close"]["status"] == "waiting_threshold"
    assert diagnostic["close"]["threshold_met"] is False
    assert diagnostic["close"]["threshold_price"] == pytest.approx(101000.0)
    assert diagnostic["close"]["retracement_pct"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("order_type", "has_order", "expected"),
    [
        ("close_trailing_long", True, "trailing"),
        ("close_auto_reduce_wel_long", True, "auto_reduce"),
        ("close_unstuck_long", True, "unstuck"),
        ("close_grid_long", True, "grid"),
        ("empty", False, "none"),
        ("unknown_custom_order", True, "other"),
    ],
)
def test_selected_mode_from_order_type_uses_specific_non_trailing_labels(
    order_type, has_order, expected
):
    assert selected_mode_from_order_type(order_type, has_order=has_order) == expected


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
    assert "total_wallet_exposure_limit=2" in rendered
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


def test_trailing_diagnostics_tool_help_runs_without_import_errors():
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = (
        src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    result = subprocess.run(
        [sys.executable, "src/tools/trailing_diagnostics.py", "--help"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Interactive trailing diagnostics explorer" in result.stdout
    assert "--wizard" in result.stdout
    assert "--snapshot-path" in result.stdout
