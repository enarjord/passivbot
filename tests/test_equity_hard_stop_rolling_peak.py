import pytest

try:
    import passivbot_rust as pbr
except Exception:  # pragma: no cover
    pbr = None

pbr_is_stub = bool(getattr(pbr, "__is_stub__", False)) if pbr is not None else False


@pytest.mark.skipif(
    pbr is None or pbr_is_stub,
    reason="passivbot_rust extension not available",
)
def test_red_episode_finalization_binding_returns_explicit_disposition():
    out = pbr.hsl_red_episode_finalization(
        restart_after_red_policy="threshold",
        stop_timestamp_ms=125_500,
        stop_equity=90.0,
        stop_peak_strategy_equity=100.0,
        previous_no_restart_peak_strategy_equity=120.0,
        drawdown_ema=0.10,
        red_threshold=0.20,
        no_restart_drawdown_threshold=0.90,
        cooldown_minutes_after_red=5.0,
    )

    assert out == {
        "no_restart_peak_strategy_equity": pytest.approx(120.0),
        "no_restart_drawdown_raw": pytest.approx(0.25),
        "no_restart_latched": False,
        "cooldown_until_ms": 425_500,
        "disposition": "cooldown",
    }


@pytest.mark.skipif(
    pbr is None or pbr_is_stub,
    reason="passivbot_rust extension not available",
)
def test_equity_hard_stop_rolling_peak_accepts_negative_values():
    tracker = pbr.EquityHardStopRollingPeak()

    assert tracker.update(1_000, -5.0, 1_000) == pytest.approx(-5.0)
    assert tracker.update(1_500, -2.0, 1_000) == pytest.approx(-2.0)
    assert tracker.update(2_100, -7.0, 1_000) == pytest.approx(-2.0)


@pytest.mark.skipif(
    pbr is None or pbr_is_stub,
    reason="passivbot_rust extension not available",
)
def test_equity_hard_stop_runtime_same_minute_recall_is_cached():
    runtime = pbr.EquityHardStopRuntime()
    kwargs = {
        "red_threshold": 0.25,
        "ema_span_minutes": 60.0,
        "tier_ratio_yellow": 0.5,
        "tier_ratio_orange": 0.75,
    }

    runtime.apply_sample(
        timestamp_ms=60_000,
        equity=100.0,
        peak_strategy_equity=100.0,
        **kwargs,
    )
    first = runtime.apply_sample(
        timestamp_ms=120_000,
        equity=90.0,
        peak_strategy_equity=100.0,
        **kwargs,
    )
    cached = runtime.apply_sample(
        timestamp_ms=120_500,
        equity=80.0,
        peak_strategy_equity=100.0,
        **kwargs,
    )

    assert cached["elapsed_minutes"] == 0
    assert cached["changed"] is False
    assert cached["drawdown_raw"] == pytest.approx(first["drawdown_raw"])
    assert cached["drawdown_score"] == pytest.approx(first["drawdown_score"])
    assert runtime.drawdown_ema() == pytest.approx(first["drawdown_ema"])


@pytest.mark.skipif(
    pbr is None or pbr_is_stub,
    reason="passivbot_rust extension not available",
)
def test_equity_hard_stop_runtime_multi_minute_gap_matches_repeated_steps():
    kwargs = {
        "red_threshold": 0.25,
        "ema_span_minutes": 60.0,
        "tier_ratio_yellow": 0.5,
        "tier_ratio_orange": 0.75,
    }
    gap_runtime = pbr.EquityHardStopRuntime()
    iter_runtime = pbr.EquityHardStopRuntime()

    for runtime in (gap_runtime, iter_runtime):
        runtime.apply_sample(
            timestamp_ms=60_000,
            equity=100.0,
            peak_strategy_equity=100.0,
            **kwargs,
        )

    gap_step = gap_runtime.apply_sample(
        timestamp_ms=360_000,
        equity=90.0,
        peak_strategy_equity=100.0,
        **kwargs,
    )
    for minute in range(2, 7):
        iter_runtime.apply_sample(
            timestamp_ms=minute * 60_000,
            equity=90.0,
            peak_strategy_equity=100.0,
            **kwargs,
        )

    assert gap_step["elapsed_minutes"] == 5
    assert gap_runtime.drawdown_ema() == pytest.approx(iter_runtime.drawdown_ema())


@pytest.mark.skipif(
    pbr is None or pbr_is_stub,
    reason="passivbot_rust extension not available",
)
def test_equity_hard_stop_runtime_can_replay_red_without_latching():
    runtime = pbr.EquityHardStopRuntime()
    kwargs = {
        "red_threshold": 0.25,
        "ema_span_minutes": 1.0,
        "tier_ratio_yellow": 0.5,
        "tier_ratio_orange": 0.75,
        "latch_red": False,
    }

    runtime.apply_sample(
        timestamp_ms=60_000,
        equity=100.0,
        peak_strategy_equity=100.0,
        **kwargs,
    )
    red = runtime.apply_sample(
        timestamp_ms=120_000,
        equity=60.0,
        peak_strategy_equity=100.0,
        **kwargs,
    )
    recovered = runtime.apply_sample(
        timestamp_ms=180_000,
        equity=100.0,
        peak_strategy_equity=100.0,
        **kwargs,
    )

    assert red["tier"] == "red"
    assert red["red_latched"] is False
    assert recovered["tier"] == "green"
    assert runtime.red_latched() is False


@pytest.mark.skipif(
    pbr is None or pbr_is_stub,
    reason="passivbot_rust extension not available",
)
def test_equity_hard_stop_runtime_default_still_latches_red():
    runtime = pbr.EquityHardStopRuntime()
    kwargs = {
        "red_threshold": 0.25,
        "ema_span_minutes": 1.0,
        "tier_ratio_yellow": 0.5,
        "tier_ratio_orange": 0.75,
    }

    runtime.apply_sample(
        timestamp_ms=60_000,
        equity=100.0,
        peak_strategy_equity=100.0,
        **kwargs,
    )
    red = runtime.apply_sample(
        timestamp_ms=120_000,
        equity=60.0,
        peak_strategy_equity=100.0,
        **kwargs,
    )
    recovered = runtime.apply_sample(
        timestamp_ms=180_000,
        equity=100.0,
        peak_strategy_equity=100.0,
        **kwargs,
    )

    assert red["tier"] == "red"
    assert red["red_latched"] is True
    assert recovered["tier"] == "red"
    assert runtime.red_latched() is True


@pytest.mark.skipif(
    pbr is None or pbr_is_stub,
    reason="passivbot_rust extension not available",
)
def test_equity_hard_stop_runtime_same_minute_replay_red_can_latch_current_sample():
    runtime = pbr.EquityHardStopRuntime()
    kwargs = {
        "red_threshold": 0.25,
        "ema_span_minutes": 1.0,
        "tier_ratio_yellow": 0.5,
        "tier_ratio_orange": 0.75,
    }

    runtime.apply_sample(
        timestamp_ms=60_000,
        equity=100.0,
        peak_strategy_equity=100.0,
        **kwargs,
        latch_red=False,
    )
    replay_red = runtime.apply_sample(
        timestamp_ms=120_000,
        equity=60.0,
        peak_strategy_equity=100.0,
        **kwargs,
        latch_red=False,
    )
    current_red = runtime.apply_sample(
        timestamp_ms=120_500,
        equity=60.0,
        peak_strategy_equity=100.0,
        **kwargs,
    )

    assert replay_red["tier"] == "red"
    assert replay_red["red_latched"] is False
    assert current_red["tier"] == "red"
    assert current_red["red_latched"] is True
    assert runtime.red_latched() is True
