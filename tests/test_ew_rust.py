"""Behavioral checks against the rebuilt Rust analysis implementation."""
import math

import pytest

DAY_MS = 86_400_000


@pytest.fixture(scope="module")
def require_real_passivbot_rust_module():
    import passivbot_rust as pbr
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(pbr, "__is_stub__", False)
    verify_loaded_runtime_extension()
    return pbr


def _curve(daily, interval_minutes=1):
    n = 1440 // interval_minutes
    interval = interval_minutes * 60_000
    equity = [daily[i // n] for i in range((len(daily) - 1) * n + 1)]
    timestamps = [DAY_MS - interval + i * interval for i in range(len(equity))]
    return equity, timestamps


@pytest.mark.parametrize("interval_minutes", [1, 5])
def test_ew_metrics_match_daily_hand_calculation(require_real_passivbot_rust_module, interval_minutes):
    pbr = require_real_passivbot_rust_module
    equity, timestamps = _curve([100.0, 110.0, 99.0, 99.0], interval_minutes)
    result = pbr.calculate_ew_metrics(equity, timestamps, 1.0, interval_minutes)
    weights = [0.25, 0.5, 1.0]
    returns = [0.1, -0.1, 0.0]
    mu = sum(w * r for w, r in zip(weights, returns)) / sum(weights)
    variance = sum(w * (r - mu)**2 for w, r in zip(weights, returns)) / sum(weights)
    downside = sum(w * min(r, 0)**2 for w, r in zip(weights, returns)) / sum(weights)
    assert result["metrics"] == pytest.approx({
        "adg": math.expm1(sum(w * math.log1p(r) for w, r in zip(weights, returns)) / sum(weights)),
        "sharpe_ratio": mu / math.sqrt(variance),
        "sortino_ratio": mu / math.sqrt(downside),
        "omega_ratio": 0.5,
    })
    metadata = result["metadata"]
    assert metadata["status"] == "available"
    assert metadata["definition_version"] == 1
    assert metadata["observation_count"] == 3
    assert metadata["first_return_end_ms"] == 2 * DAY_MS
    assert metadata["last_return_end_ms"] == 4 * DAY_MS
    assert metadata["effective_sample_size"] == pytest.approx(sum(weights)**2 / sum(w*w for w in weights))


def test_ew_partial_days_and_ruin_are_not_removed(require_real_passivbot_rust_module):
    pbr = require_real_passivbot_rust_module
    equity, timestamps = _curve([100.0, 110.0, 99.0, 99.0])
    partial = pbr.calculate_ew_metrics(equity[1:-1], timestamps[1:-1], 180.5)
    assert partial["metrics"] is None
    assert partial["metadata"]["observation_count"] == 1
    equity[2] = 0.0
    ruined = pbr.calculate_ew_metrics(equity, timestamps, 180.0)
    assert ruined["metrics"] is None
    assert ruined["metadata"]["status"] == "nonpositive_equity"
    equity[2] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        pbr.calculate_ew_metrics(equity, timestamps, 180.0)


@pytest.mark.parametrize("half_life", [0.9, float("nan"), float("inf")])
def test_ew_half_life_is_validated_by_rust(require_real_passivbot_rust_module, half_life):
    with pytest.raises(ValueError, match="half_life_days"):
        require_real_passivbot_rust_module.calculate_ew_metrics([], [], half_life)


def test_ew_time_gaps_are_explicitly_unavailable(require_real_passivbot_rust_module):
    equity, timestamps = _curve([100.0, 110.0, 99.0])
    del equity[10]
    del timestamps[10]
    result = require_real_passivbot_rust_module.calculate_ew_metrics(equity, timestamps, 180.0)
    assert result["metrics"] is None
    assert result["metadata"]["status"] == "missing_samples"


def test_native_backtest_ew_wiring_and_metrics_only_agree(require_real_passivbot_rust_module):
    import numpy as np
    from backtest import build_backtest_payload
    from test_backtest_directional_eligibility import _ema_anchor_config, _synthetic_inputs

    pbr = require_real_passivbot_rust_module
    config = _ema_anchor_config(True)
    config["backtest"].update(ew_half_life_days=180.5, end_date="2024-01-05")
    initial_hlcvs, markets, _, initial_timestamps = _synthetic_inputs()
    n = 4 * 1440
    timestamps = initial_timestamps[0] + np.arange(n, dtype=np.int64) * 60_000
    hlcvs = np.tile(initial_hlcvs[:1], (n, 1, 1))
    for name in ("LONGCOIN", "SHORTCOIN"):
        markets[name]["last_valid_index"] = n - 1
    payload = build_backtest_payload(
        hlcvs, markets, config, "binance", np.full(n, 50_000.0), timestamps
    )

    def run():
        return pbr.run_backtest_bundle(
            payload.bundle, payload.bot_params_list, payload.strategy_params_list,
            payload.exchange_params, payload.backtest_params,
        )

    full = run()
    eq = np.asarray(full[1])
    clocks = eq[:, 0].astype(np.uint64).tolist()
    for column, analysis in [(1, full[2]), (2, full[3])]:
        canonical = pbr.calculate_ew_metrics(eq[:, column].tolist(), clocks, 180.5)
        assert canonical["metadata"]["status"] == "available"
        for name, value in canonical["metrics"].items():
            assert analysis[f"{name}_ew"] == pytest.approx(value)
        strategy = pbr.calculate_ew_metrics(eq[:, 3].tolist(), clocks, 180.5)
        for name, value in strategy["metrics"].items():
            assert analysis[f"{name}_strategy_eq_ew"] == pytest.approx(value)
    assert full[2]["_ew_metadata"]["strategy_eq"] == strategy["metadata"]
    for analysis in full[2:4]:
        assert not any(key.endswith("_w") for key in analysis)
    payload.backtest_params["metrics_only"] = True
    metrics_only = run()
    assert metrics_only[2] == full[2]
    assert metrics_only[3] == full[3]
    payload.backtest_params["calculate_legacy_w"] = False
    with pytest.raises(ValueError, match="calculate_legacy_w is retired"):
        run()
