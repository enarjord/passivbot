"""V9 GPU policy transport does not retain v8 restart or mode aliases."""

from types import SimpleNamespace

import pytest

from optimization.gpu.hsl import pack_cooldown_ms, pack_params, project_bot


def _bot():
    return {
        "hsl_enabled": True,
        "hsl_red_threshold": .1,
        "hsl_ema_span_minutes": 3.5,
        "hsl_cooldown_minutes_after_red": 5.25,
        "hsl_lookback_ema_span_minutes": 43200.5,
        "hsl_slot_count": 2,
    }


@pytest.mark.parametrize("mode,code", [("coin", 2), ("pside", 1), ("portfolio", 0)])
def test_policy_packs_new_global_span_in_removed_policy_word(mode, code):
    packed = pack_params(_bot(), mode)
    assert packed["hsl_signal_mode"] == code
    assert packed["hsl_lookback_ema_span_minutes"] == 43200.5
    assert packed["hsl_ema_span_minutes"] == 3.5
    assert not any("restart" in key for key in packed)


@pytest.mark.parametrize("policy", ["always", "never"])
def test_v8_restart_field_requires_migration(policy):
    bot = _bot()
    bot["hsl_restart_after_red_policy"] = policy
    with pytest.raises(ValueError, match="migration"):
        pack_params(bot, "coin")


def test_old_portfolio_mode_name_is_not_a_runtime_alias():
    with pytest.raises(ValueError, match="portfolio"):
        pack_params(_bot(), "unified")


def test_projection_uses_explicit_portfolio_policy_and_shared_memory():
    policy = {"enabled": True, "red_threshold": .2, "ema_span_minutes": 15.5,
              "cooldown_minutes_after_red": 20.0, "panic_close_order_type": "market"}
    payload = SimpleNamespace(
        bot_params_list=[{"long": {"n_positions": 2}}],
        backtest_params={"lookback_ema_span_minutes": 1234.5,
                         "equity_hard_stop_loss": {"engine": "hsl", "mode": "portfolio",
                                                   "portfolio": policy}},
    )
    projected = project_bot(payload, 0, "long", {})
    assert projected["hsl_red_threshold"] == .2
    assert projected["hsl_lookback_ema_span_minutes"] == 1234.5


@pytest.mark.parametrize("span", [float("nan"), .5, 1e30])
def test_memory_span_cannot_silently_stop_decaying_during_packing(span):
    bot = _bot()
    bot["hsl_lookback_ema_span_minutes"] = span
    with pytest.raises(ValueError, match="lookback_ema_span_minutes"):
        pack_params(bot, "coin")


@pytest.mark.parametrize("duration", [0, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 48) - 1])
def test_exact_duration_limbs_preserve_carries(duration):
    low, high = pack_cooldown_ms(duration / 60000.0)
    assert int(low) + (int(high) << 24) == duration
    assert low < 1 << 24 and high < 1 << 24


def test_fractional_duration_keeps_f64_rounding_at_minute_boundary():
    low, high = pack_cooldown_ms(.99999165)
    assert int(low) + (int(high) << 24) == 59999


@pytest.mark.parametrize("minutes", [-1, float("nan"), float("inf"), (1 << 48) / 60000.0])
def test_unsupported_duration_fails_before_replay(minutes):
    with pytest.raises(ValueError, match="cooldown"):
        pack_cooldown_ms(minutes)


@pytest.mark.parametrize("name", ["MpsSingleCoinProxy", "MpsEmaAnchorProxy"])
def test_retired_directional_service_fails_before_device_or_input_access(name):
    from optimization.gpu import service

    with pytest.raises(ValueError, match="v9.*NativeBacktestService"):
        getattr(service, name)(config={}, hlcvs=None, mss={}, btc=None, timestamps=None,
                               exchange="binance", batch_size=1, needed_metrics=())


@pytest.mark.parametrize("name", ["MpsEmaAnchorRunner", "MpsTrailingMartingaleRunner"])
def test_retired_directional_runner_fails_before_device_or_input_access(name):
    pytest.importorskip("torch")
    from optimization.gpu import mps_kernel

    with pytest.raises(ValueError, match="v9.*NativeBacktestService"):
        getattr(mps_kernel, name)(None, None, {})


@pytest.mark.parametrize("option", ["--hsl-ema-span-minutes", "--lookback-ema-span-minutes"])
@pytest.mark.parametrize("span", ["1e9", "3.4028234663852886e38"])
def test_fixture_preflight_rejects_memory_that_rounds_to_no_decay(option, span):
    from tools.gpu_parity import build_parser, resolve_fixture_args

    args = build_parser().parse_args(["--fixture", "ema_anchor", option, span])
    with pytest.raises(ValueError, match=option):
        resolve_fixture_args(args)
