"""Effective consumer proofs and full-output controls for unstuck EMA ablation."""

from itertools import product

import numpy as np
import pytest

from optimization.gpu.specialization import unstuck_ema_required

KEYS = ("unstuck_enabled", "other", "unstuck_ema_gating_enabled")


@pytest.fixture(scope="module")
def require_real_passivbot_rust_module():
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("GPU unavailable")
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(passivbot_rust, "__is_stub__", False)
    verify_loaded_runtime_extension()
    return passivbot_rust


@pytest.mark.parametrize("enabled,gating", product((0.0, 1.0), repeat=2))
def test_effective_coin_flags_match_independent_scalar_inheritance(enabled, gating):
    rows = np.array([[enabled, 9.0, gating], [1.0 - enabled, 9.0, 1.0 - gating]])
    for overrides in product((np.nan, 0.0, 1.0, np.inf), repeat=4):
        coins = np.asarray(overrides).reshape(2, 2)
        expected = any(
            (e if np.isfinite(e) else row[0]) > 0.5
            and (g if np.isfinite(g) else row[2]) > 0.5
            for row in rows for e, g in coins
        )
        assert unstuck_ema_required(rows, KEYS, (coins,)) == expected


def test_fused_sides_do_not_combine_unrelated_enabled_and_gating_flags():
    matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    inherited = np.full((2, 2), np.nan)
    assert not unstuck_ema_required(matrix, KEYS, (inherited, inherited))
    short = inherited.copy()
    short[1, 0] = 1.0
    assert unstuck_ema_required(matrix, KEYS, (inherited, short))


def test_proof_uses_shader_float32_flags_and_retains_unknown_base_flags():
    inherited = np.full((2, 2), np.nan)
    assert not unstuck_ema_required([[0.500000001, 0.0, 1.0]], KEYS, (inherited,))
    assert unstuck_ema_required([[np.nan, 0.0, 0.0]], KEYS, (inherited,))
    assert unstuck_ema_required(np.empty((0, 3)), KEYS, (inherited,))


@pytest.mark.parametrize("matrix,overrides", [
    ([[0.0, 0.0]], (np.full((2, 2), np.nan),)),
    ([[0.0, 0.0, 0.0]], ()),
    ([[0.0, 0.0, 0.0]], (np.full((2, 1), np.nan),)),
    ([[0.0, 0.0, 0.0]], (np.empty((0, 2)),)),
])
def test_proof_rejects_misaligned_parameter_and_override_views(matrix, overrides):
    with pytest.raises(ValueError, match="unstuck EMA proof"):
        unstuck_ema_required(matrix, KEYS, overrides)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("policy", ["ungated", "disabled", "gated", "coin_gated", "coins_ungated"])
def test_multicoin_dispatch_ablation_preserves_all_outputs(
    require_real_passivbot_rust_module, strategy, sides, policy,
):
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("GPU unavailable")
    from optimization.gpu.service import MpsMulticoinProxy
    from test_gpu_unstuck_lookback import make_proxy
    from test_gpu_hsl_multicoin import raw

    original, inputs = make_proxy(sides, strategy=strategy)
    del original
    candles, markets, config, exchange, btc, timestamps = inputs
    for side in sides:
        config["bot"][side]["unstuck"].update(
            enabled=policy != "disabled",
            ema_gating_enabled=policy in {"gated", "coins_ungated"},
        )
    if policy == "coin_gated":
        config["coin_overrides"] = {"ETH": {"bot": {
            sides[-1]: {"unstuck": {"ema_gating_enabled": True}}
        }}}
    if policy == "coins_ungated":
        config["coin_overrides"] = {coin: {"bot": {
            side: {"unstuck": {"ema_gating_enabled": False}} for side in sides
        }} for coin in ("BTC", "ETH")}
    proxy = MpsMulticoinProxy(
        config=config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
        exchange=exchange, batch_size=2, needed_metrics={
            "adg_strategy_eq", "adg_strategy_eq_w", "volume_pct_per_day_avg_w",
            "strategy_eq_recovery_days_p95", "drawdown_worst_strategy_eq",
            "drawdown_worst_mean_1pct_strategy_eq", "hard_stop_time_in_red_pct",
        },
    )
    runner, output = raw(proxy, [{}, {}])
    enabled = policy in {"gated", "coin_gated"}
    assert runner.dispatch_unstuck_ema_enabled == enabled
    assert output["fill_count"].min().item() >= len(sides)
    specialized = {key: value.clone() if isinstance(value, torch.Tensor) else value
                   for key, value in output.items()}

    def assert_same(expected, actual):
        assert expected.keys() == actual.keys()
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                # Recovery tapes intentionally leave unobserved slots as NaN.
                # Compare their masks as well as every finite returned value.
                torch.testing.assert_close(
                    value, actual[key], rtol=0, atol=0, equal_nan=True, msg=key,
                )
            else:
                assert value == actual[key]
    if strategy == "trailing_martingale":
        runner.max_dispatch_candidate_bars = 2 * 2 * len(sides) * 7
        _, temporal = raw(proxy, [{}, {}])
        assert_same(specialized, temporal)
        specialized_bytes = runner._replay_state_bytes
    runner.unstuck_ema_specialization = False
    _, general = raw(proxy, [{}, {}])
    assert runner.dispatch_unstuck_ema_enabled
    assert_same(specialized, general)
    if strategy == "trailing_martingale" and not enabled:
        assert runner._replay_state_bytes > specialized_bytes
    runner.unstuck_ema_specialization = True
    _, restored = raw(proxy, [{}])
    assert runner.dispatch_unstuck_ema_enabled == enabled
    first = {key: value[:1] if isinstance(value, torch.Tensor) else value
             for key, value in specialized.items()}
    assert_same(first, restored)
    if policy == "ungated":
        # One consuming candidate must keep EMA state for the whole dispatch,
        # independently of base config and previously cached disabled variants.
        candidates = [{}, {f"{sides[-1]}_unstuck_ema_gating_enabled": 1.0}]
        _, mixed = raw(proxy, candidates)
        assert runner.dispatch_unstuck_ema_enabled
        runner.unstuck_ema_specialization = False
        _, mixed_general = raw(proxy, candidates)
        assert_same(mixed, mixed_general)
        runner.unstuck_ema_specialization = True
        _, disabled_again = raw(proxy, [{}, {}])
        assert not runner.dispatch_unstuck_ema_enabled
        assert_same(specialized, disabled_again)
