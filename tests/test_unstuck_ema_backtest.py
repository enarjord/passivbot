"""Offline end-to-end replay through the canonical config loader and Rust backtester."""
from config.universe import set_side_universe_sources

from copy import deepcopy
import numpy as np
import pytest
from config import get_template_config, prepare_config
from config.overrides import parse_overrides
from config.migrations.v9 import migrate_v8_to_v9
from backtest import run_backtest
from hsl_reporting import hsl_report as collect_hsl_report


def _legacy_replay_config():
    c = get_template_config()
    c["config_version"] = "v8.2.0"
    # This fixture migrates unrelated strategy fields with old HSL disabled,
    # not newly generated current-schema HSL restart authorization.
    for side in ("long", "short"):
        c["bot"][side]["hsl"].update(enabled=False, restart_after_red_policy=None)
    c["optimize"]["fixed_runtime_overrides"] = {
        key: value
        for key, value in c["optimize"]["fixed_runtime_overrides"].items()
        if ".hsl." not in key
    }
    for root in (c["bot"], c["optimize"]["bounds"]):
        for side in ("long", "short"):
            strategy = root[side]["strategy"]["trailing_martingale"]
            for key in ("ema_span_0", "ema_span_1"):
                strategy[key] = strategy["entry"].pop(key)
    for side in ("long", "short"):
        for key in ("ema_span_0", "ema_span_1"):
            c["bot"][side]["unstuck"].pop(key)
            c["optimize"]["bounds"][side]["unstuck"].pop(key)
    c["backtest"].update(
        start_date="2024-01-01",
        end_date="2024-01-15",
        starting_balance=1000,
        exchanges=["binance"],
        suite_enabled=False,
        candle_interval_minutes=1,
    )
    set_side_universe_sources(c, 'approved_coins', {'long': ['BTC', 'ETH'], 'short': []})
    set_side_universe_sources(c, 'ignored_coins', {'long': [], 'short': []})
    c['bot']['global']['warmup']['ratio'] = 1
    c['bot']['global']['warmup']['max_minutes'] = 100
    for side in ("long", "short"):
        b = c["bot"][side]
        b['risk'].update({'max_positions': 2, 'total_wallet_exposure_limit': 1.0 if side == 'long' else 0.0, 'we_excess_allowance_pct': 0, 'position_exposure_enforcer': {**b['risk']['position_exposure_enforcer'], 'enabled': False}, 'total_exposure_enforcer': {**b['risk']['total_exposure_enforcer'], 'enabled': False}})
        b["entry_cooldown"]["base_duration_minutes"] = 0
        b["hsl"]["enabled"] = False
        b["forager"].update(volume_ema_span_1m=10, volatility_ema_span_1m=10)
        t = b["strategy"]["trailing_martingale"]
        t.update(
            ema_span_0=10.5,
            ema_span_1=40.5,
            volatility_ema_span_1m=1,
            volatility_ema_span_1h=1,
        )
        t["entry"].update(
            initial_qty_pct=0.25,
            initial_ema_dist=-0.001,
            double_down_factor=1,
            ema_gate_mode="all",
            threshold_base_pct=0.015,
            retracement_base_pct=0,
        )
        t["close"].update(
            qty_pct=0.2,
            threshold_base_pct=0.01,
            retracement_base_pct=0,
            threshold_we_weight=0,
        )
        for group in ("entry", "close"):
            for key in t[group]:
                if key.endswith("_weight"):
                    t[group][key] = 0
        b["unstuck"].update(
            enabled=True,
            ema_gating_enabled=True,
            close_pct=0.1,
            ema_dist=-0.005,
            loss_allowance_pct=0.02,
            exposure_trigger_ratio=0.3,
        )
    c["coin_overrides"] = {
        "ETH": {
            "bot": {
                "long": {
                    "strategy": {
                        "trailing_martingale": {
                            "ema_span_0": 21.5,
                            "ema_span_1": 110.25,
                        }
                    }
                }
            }
        }
    }
    return c


def _prepare_released_replay(raw):
    before = deepcopy(raw)
    with pytest.raises(ValueError, match="migrate-config-v9"):
        prepare_config(raw, verbose=False)
    migrated = migrate_v8_to_v9(raw, memory_span_minutes=43200.0).config
    assert raw == before
    return prepare_config(migrated, verbose=False)


def _replay(config, *, return_payload=False):
    n = 20_000
    t = np.arange(n)
    close = 100 + 4 * np.sin(t / 23) + 8 * np.sin(t / 510) - 12 * (t / n)
    hlcvs = np.empty((n, 2, 4))
    for i in range(2):
        prices = close * (1 + 0.05 * i) + np.sin(t / 41 + i)
        hlcvs[:, i, 0] = prices + 0.3
        hlcvs[:, i, 1] = prices - 0.3
        hlcvs[:, i, 2] = prices
        hlcvs[:, i, 3] = 1000
    start = 1704067200000
    ts = start + t * 60000
    mss = {
        coin: dict(
            maker=0.0002,
            taker=0.0005,
            qty_step=0.001,
            price_step=0.01,
            min_qty=0.001,
            min_cost=1,
            c_mult=1,
            first_valid_index=0,
            last_valid_index=n - 1,
            warmup_minutes=100,
        )
        for coin in ("BTC", "ETH")
    }
    mss["__meta__"] = {
        "requested_start_ts": start + 100 * 60000,
        "requested_start_date": "2024-01-01",
        "warmup_minutes_requested": 100,
    }
    config["backtest"]["coins"] = {"binance": ["BTC", "ETH"]}
    return run_backtest(
        hlcvs, mss, config, "binance", np.full(n, 50000.0), ts,
        return_payload=return_payload,
    )


def test_migrated_replay_matches_explicit_legacy_pairs_and_independent_spans_change_fills():
    import passivbot_rust as pbr

    assert not getattr(pbr, "__is_stub__", False)
    raw = _legacy_replay_config()
    migrated = parse_overrides(
        _prepare_released_replay(raw),
        verbose=False,
    )
    explicit = deepcopy(raw)
    for side in ("long", "short"):
        strategy = explicit["bot"][side]["strategy"]["trailing_martingale"]
        for key in ("ema_span_0", "ema_span_1"):
            explicit["bot"][side]["unstuck"][key] = strategy[key]
    explicit["coin_overrides"]["ETH"]["bot"]["long"]["unstuck"] = {
        "ema_span_0": 21.5,
        "ema_span_1": 110.25,
    }
    explicit = parse_overrides(
        _prepare_released_replay(explicit),
        verbose=False,
    )
    old_fills, old_equity, _ = _replay(migrated)
    expected_fills, expected_equity, _ = _replay(explicit)
    np.testing.assert_array_equal(old_fills, expected_fills)
    np.testing.assert_array_equal(old_equity, expected_equity)
    assert sum(row[13] == "close_unstuck_long" for row in old_fills) > 100
    independent = deepcopy(migrated)
    independent["bot"]["long"]["unstuck"].update(ema_span_0=3.5, ema_span_1=5.5)
    independent["coin_overrides"]["ETH"]["bot"]["long"].pop("unstuck")
    new_fills, _, _ = _replay(independent)
    assert not np.array_equal(new_fills, old_fills)


@pytest.mark.parametrize("hsl_enabled", [False, True])
@pytest.mark.parametrize("active_side", ["long", "short"])
def test_orchestrator_profiling_preserves_replay_and_records_stages(
    monkeypatch, tmp_path, hsl_enabled, active_side
):
    import json
    import passivbot_rust as pbr

    assert not getattr(pbr, "__is_stub__", False)
    # Author a current-schema fixture; do not activate HSL in legacy input.
    migrated = _prepare_released_replay(_legacy_replay_config())
    raw = get_template_config()
    for section in ("bot", "coin_overrides", "backtest", "live"):
        raw[section] = deepcopy(migrated[section])
    inactive_side = "short" if active_side == "long" else "long"
    raw["bot"][active_side]["risk"].update(
        max_positions=2, total_wallet_exposure_limit=1.0
    )
    raw["bot"][inactive_side]["risk"].update(
        max_positions=0, total_wallet_exposure_limit=0.0
    )
    set_side_universe_sources(raw, 'approved_coins', {
        active_side: ["BTC", "ETH"], inactive_side: []
    })
    raw["bot"][active_side]["hsl"].update(
        enabled=hsl_enabled,
        drawdown_ratio_ema_span_minutes=1.0, red_threshold=0.001,
        cooldown_minutes_after_red=7.0,
    )
    config = parse_overrides(prepare_config(raw, verbose=False), verbose=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PASSIVBOT_ORCH_PROFILE", raising=False)
    expected_fills, expected_equity, expected_analysis, expected_payload = _replay(
        deepcopy(config), return_payload=True
    )
    assert any(row[13].endswith(f"_{active_side}") for row in expected_fills)
    panic_fills = [row for row in expected_fills if row[13] == f"close_panic_{active_side}"]
    if hsl_enabled:
        assert panic_fills, "the enabled-HSL fixture must execute a panic close"
        report = collect_hsl_report(expected_payload.hard_stop_plot_data)
        assert report is not None
        side = 0 if active_side == "long" else 1
        events = [event for event in report["events"] if event["side"] == side]
        assert any(event["kind"] == "red" for event in events)
        assert any(event["kind"] == "restart" for event in events)
    else:
        assert not panic_fills
    profile_path = tmp_path / "measurements" / "orch_profile_orchestrator.json"
    assert not profile_path.exists()

    monkeypatch.setenv("PASSIVBOT_ORCH_PROFILE", "1")
    fills, equity, analysis, payload = _replay(deepcopy(config), return_payload=True)
    np.testing.assert_array_equal(fills, expected_fills)
    np.testing.assert_array_equal(equity, expected_equity)
    np.testing.assert_equal(analysis, expected_analysis)
    np.testing.assert_equal(payload.hard_stop_plot_data, expected_payload.hard_stop_plot_data)
    profile = json.loads(profile_path.read_text())
    stages = ("clear_orders", "peek_hints", "input_update", "compute", "distribute")
    assert set(profile) == {
        "mode", "steps", "total_ns", "sort_bundles_ns",
        *(f"{stage}_ns" for stage in stages),
    }
    assert profile["mode"] == "orchestrator"
    assert profile["steps"] > 0
    for stage in stages:
        assert profile[f"{stage}_ns"] >= 0
    assert profile["input_update_ns"] > 0
    assert profile["compute_ns"] > 0
    assert profile["sort_bundles_ns"] == 0
    assert profile["total_ns"] >= sum(profile[f"{stage}_ns"] for stage in stages)
