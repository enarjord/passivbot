"""Bounded completed-episode replay versus the source-verified Rust controller."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from test_gpu_hsl_window import run_controller

pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="Apple MPS and NVIDIA CUDA unavailable",
)


def _reference(events, span, never, lookback):
    import passivbot_rust

    episodes, points, opened, output = [], [], None, []
    for minute, budget, pnl, upnl, exposed, terminal in events:
        if not points and episodes:
            points = [dict(episodes[-1]["points"][-1], flatten=False)]
        if episodes and exposed and opened is None:
            opened = minute * 60_000
        points.append(dict(timestamp=minute * 60_000, pnl=pnl, upnl=upnl,
                           exposed=exposed, flatten=terminal))
        episode = dict(points=points, opened_at=opened)
        payload = dict(episodes=episodes + [episode], now=minute * 60_000,
                       start=(minute - lookback) * 60_000, budget=budget, span=span,
                       threshold=0.1, cooldown_ms=600_000,
                       restart="never" if never else "always")
        decision = json.loads(passivbot_rust.hsl_controller(json.dumps(payload)))[-1]
        output.append([{"normal": 0, "halted": 1, "panic": 3}[decision["action"]],
                       decision["raw"], decision["ema"]])
        if terminal:
            episodes.append(episode)
            points, opened = [], None
    return np.asarray(output)


@pytest.mark.parametrize("span", [1.0, 3.5])
@pytest.mark.parametrize("positive_peak", [0, 100, 300])
@pytest.mark.parametrize("lookback", [10, 65, 1440])
@pytest.mark.parametrize("never", [False, True])
def test_completed_episode_forgets_expired_peak(span, positive_peak, lookback, never):
    events = [(0, 1000, 0, 0, False, False), (1, 1000, 0, 0, True, False),
              (2, 1000, 0, positive_peak, True, False),
              (3, 1000, 0, -200, True, False), (4, 800, -200, 0, False, True)]
    events.extend((minute, 800, -200, 0, False, False)
                  for minute in range(5, lookback + 6))
    actual = run_controller(events, span, never, lookback=lookback)
    expected = _reference(events, span, never, lookback)
    assert actual[4, 0] == expected[4, 0] == 1
    # The peak expires before the terminal loss, without a new fill or deposit.
    assert expected[lookback + 3, 0] == 0
    np.testing.assert_array_equal(actual[:, 0], expected[:, 0])
    np.testing.assert_allclose(actual[:, 1:], expected[:, 1:], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("span", [1.0, 3.5])
def test_active_exposure_retains_current_entry_loss_estimate(span):
    import passivbot_rust

    lookback = 10
    events = [(0, 1000, 0, 0, False, False), (1, 1000, 0, 0, True, False)]
    events.extend((minute, 1000, 0, -200, True, False) for minute in range(2, 16))
    actual = run_controller(events, span, True, lookback=lookback)
    now = events[-1][0] * 60_000
    position = dict(size=10, basis=100, mark=80, multiplier=1, inverse=False,
                    pside="long", quantity_step=1)
    pair = dict(symbol="COIN", position=position, position_at=now, mark_at=now,
                fills_started_at=now, fills_at=now, prices_at=now, fills=[], prices={},
                revisions=[0] * 4, fills_position_anchor=None)
    snapshot = dict(now=now, start=now-lookback*60_000, balance=1000,
                    balance_at=now, config_at=now, max_current_age_ms=0,
                    mode="unified", pside=None, symbol=None, pairs=[pair])
    reference = json.loads(passivbot_rust.hsl_evaluate(json.dumps(dict(
        snapshot=snapshot, slots=1, span=span, threshold=0.1,
        cooldown_ms=600_000, restart="never"))))
    decision = reference["decision"]
    assert "estimated_current_opening" in reference["reasons"]
    assert decision["action"] == "panic"
    np.testing.assert_allclose(actual[-1], [3, decision["raw"], decision["ema"]],
                               rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode", ["coin", "pside"])
def test_native_closed_episode_expiry_matches_cpu(strategy, mode):
    if not torch.cuda.is_available():
        pytest.skip("native CUDA service required")
    from optimization.gpu.parity import MetricTolerance
    from tools.gpu_parity import DEFAULT_TOLERANCES, build_parser, fixture_inputs, run_comparison

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "2",
        "--bars", "3000", "--seed", "43", "--hsl", mode,
    ]))
    for side in ("long", "short"):
        inputs[0]["bot"][side]["hsl"].update(
            red_threshold=0.002, ema_span_minutes=2.5, cooldown_minutes_after_red=10000,
        )
    inputs[1][1500:, 0, :3] *= 0.7
    inputs[1][1800:, 1, :3] *= 1.3
    durations = ("hard_stop_duration_minutes_mean", "hard_stop_duration_minutes_max")
    counters = ("hard_stop_triggers_per_year", "hard_stop_restarts_per_year",
                "hard_stop_post_restart_retrigger_pct", "hard_stop_trigger_drawdown_mean",
                "hard_stop_restarts_per_year_long", "hard_stop_restarts_per_year_short")
    metrics = (*durations, *counters, "hard_stop_time_in_red_pct", "adg_strategy_eq",
               "drawdown_worst_strategy_eq", "fills_per_day", "backtest_completion_ratio")
    policies = {**DEFAULT_TOLERANCES,
                **{name: MetricTolerance(1e-4, 0) for name in durations},
                **{name: MetricTolerance(1e-5, 1e-3) for name in counters},
                "hard_stop_time_in_red_pct": MetricTolerance(1e-8, 1e-5),
                # Bounded seed-43 trajectory residuals, separate from exact expiry.
                "adg_strategy_eq": MetricTolerance(5e-5, 0),
                "fills_per_day": MetricTolerance(0, 1e-3)}
    if strategy == "ema_anchor":
        # Two fills differ across 1,942/1,902 CPU fills in these measured cases.
        # Keep a local per-day count bound; HSL duration and RED time stay strict.
        policies["fills_per_day"] = MetricTolerance(1, 0)
    report = run_comparison(inputs, "binance", metrics, policies, gpu_engine="native")
    assert report["metrics"]["hard_stop_triggers_per_year"]["cpu"] > 0
    assert report["metrics"][durations[1]]["cpu"] > 1000
    assert report["passed"], report["metrics"]


@pytest.mark.parametrize("span", [1.0, 3.5])
def test_clipped_active_entry_loss_is_evaluated_before_terminal_cooldown(span):
    import passivbot_rust

    events = [(0, 1000, 0, 0, False, False), (1, 1000, 0, 0, True, False)]
    events.extend((minute, 1000, 0, -200, True, False) for minute in range(2, 16))
    events.append((16, 800, -200, 0, False, True))
    actual = run_controller(events, span, True, lookback=10)
    now = 16 * 60_000
    position = dict(size=0, basis=0, mark=80, multiplier=1, inverse=False,
                    pside="long", quantity_step=1)
    fill = dict(identity="close", timestamp=now, delta=-10, price=80,
                realized=-200, fee=0, sequence=0, revision=0)
    anchor = dict(position_at=now, revision=0,
                  **{key: position[key] for key in ("size", "basis", "multiplier", "inverse", "pside")})
    pair = dict(symbol="COIN", position=position, position_at=now, mark_at=now,
                fills_started_at=now, fills_at=now, prices_at=now, fills=[fill], prices={},
                revisions=[0] * 4, fills_position_anchor=anchor)
    snapshot = dict(global_fill_sequence=True, fills_before_same_time_price=False,
                    now=now, start=now-10*60_000, balance=800, balance_at=now,
                    config_at=now, max_current_age_ms=0, mode="unified",
                    pside=None, symbol=None, pairs=[pair])
    reference = json.loads(passivbot_rust.hsl_evaluate(json.dumps(dict(
        snapshot=snapshot, slots=1, span=span, threshold=0.1,
        cooldown_ms=600_000, restart="never"))))
    assert "estimated_entry_peak" in reference["reasons"]
    assert reference["decision"]["action"] == "halted"
    assert actual[-2, 0] == 3
    assert actual[-1, 0] == 1
    np.testing.assert_allclose(actual[-1, 1:], [0.2, 0.2], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("span", [1.0, 3.5])
@pytest.mark.parametrize("flat_budget", [800, 1600])
def test_completed_entry_reference_is_cache_independent(span, flat_budget, monkeypatch):
    import passivbot_rust
    import test_gpu_hsl_window as probe

    events = [(0, 1000, 0, 0, False, False), (1, 1000, 0, 0, True, False)]
    events.extend((minute, 1000, 0, -200, True, False) for minute in range(2, 16))
    events.extend([(16, 800, -200, 0, False, True),
                   (16, flat_budget, -200, 0, False, False),
                   (17, flat_budget, -200, 0, False, False)])
    cached = run_controller(events, span, True, lookback=10)
    original_compile = probe.compile_shader

    def force_rebuild(source, *args, **kwargs):
        marker = "        bool valid = hsl_observe("
        assert source.count(marker) == 1
        source = source.replace(marker, "        h.scalar_ready = false;\n" + marker)
        return original_compile(source, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(probe, "compile_shader", force_rebuild)
            probe.controller_library.cache_clear()
            rebuilt = run_controller(events, span, True, lookback=10)
    finally:
        probe.controller_library.cache_clear()
    np.testing.assert_allclose(cached, rebuilt, rtol=2e-5, atol=2e-6)
    # Rust's estimated reference belongs to the first retained sample. It remains
    # at the same timestamp/budget change and disappears when that sample expires.
    points = [dict(timestamp=minute*60_000, pnl=0, upnl=-200,
                   exposed=True, flatten=False) for minute in range(6, 16)]
    points.append(dict(timestamp=16*60_000, pnl=-200, upnl=0,
                       exposed=False, flatten=True))
    episode = dict(points=points, opened_at=None, entry_reference_delta=200)
    flat = dict(points[-1], flatten=False)
    for minute, index in ((16, -2), (17, -1)):
        current = dict(flat, timestamp=minute*60_000)
        payload = dict(episodes=[episode, dict(points=[flat, current])],
                       now=minute*60_000, start=(minute-10)*60_000, budget=flat_budget,
                       span=span, threshold=0.1, cooldown_ms=600_000, restart="never")
        expected = json.loads(passivbot_rust.hsl_controller(json.dumps(payload)))[-1]
        assert cached[index, 0] == {"normal": 0, "halted": 1}[expected["action"]]
    np.testing.assert_array_equal(cached[-2:, 0], [1, 0])
