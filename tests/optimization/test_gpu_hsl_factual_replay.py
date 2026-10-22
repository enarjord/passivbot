"""Native v9 forward-state ownership, continuation and service boundaries."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="actual supported Metal/CUDA execution required",
)


@pytest.fixture(scope="module")
def reference():
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    return passivbot_rust


def _equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            np.testing.assert_array_equal(actual[key].cpu().numpy(), value.cpu().numpy(), err_msg=key)
        else:
            assert actual[key] == value, key


@pytest.fixture
def forbid_cpu(monkeypatch, reference):
    import backtest

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU backtest called during native GPU forward replay")

    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(reference, "run_backtest_bundle", forbidden)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coin_count", [1, 2])
def test_native_forward_dispatch(reference, forbid_cpu, strategy, mode, sides, coin_count):
    from test_gpu_hsl_multicoin import make_proxy, raw

    proxy = make_proxy(mode, strategy, sides, minutes=128, coin_count=coin_count)
    runner, first = raw(proxy, [{}, {}])
    assert float(first["fill_count"].min()) > 0
    assert runner.hsl_fact_capacity == 0
    assert runner._hsl_history_bytes_per_candidate() == 0
    _, repeated = raw(proxy, [{}, {}])
    _equal(repeated, first)


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coin_count", [1, 2])
def test_forward_context_rebinds_after_every_temporal_chunk(reference, forbid_cpu, mode, sides, coin_count):
    from test_gpu_hsl_multicoin import make_proxy, raw

    proxy = make_proxy(mode, "trailing_martingale", sides, minutes=128, coin_count=coin_count)
    runner, full = raw(proxy, [{}, {}])
    runner.max_dispatch_candidate_bars = 64
    _, chunked = raw(proxy, [{}, {}])
    assert runner._last_temporal_dispatch["dispatch_count"] > 1
    _equal(chunked, full)


@pytest.mark.parametrize("explicit_callback", [False, True])
def test_shared_scenario_inputs_keep_compatible_interruption_ownership(reference, forbid_cpu, explicit_callback):
    from test_gpu_hsl_multicoin import make_proxy

    cache = {}
    callback = (lambda: None) if explicit_callback else None
    kwargs = dict(mode="coin", strategy="trailing_martingale", sides=("long",),
                  minutes=64, prepared_data_cache=cache, interrupt_check=callback)
    first, second = make_proxy(**kwargs), make_proxy(**kwargs)
    assert first.data is second.data
    assert first.suite_batch_key() is not None
    assert first.suite_batch_key() == second.suite_batch_key()
    other = make_proxy(**(kwargs | {"interrupt_check": lambda: None}))
    assert other.data is first.data
    assert other.suite_batch_key() != first.suite_batch_key()


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
def test_policy_disable_restore_does_not_leak_episode_state(reference, forbid_cpu, strategy, mode):
    from test_gpu_hsl_multicoin import make_proxy, raw

    proxy = make_proxy(mode, strategy, minutes=128)
    runner, baseline = raw(proxy, [{}])
    _, disabled = raw(proxy, [{"long_hsl_enabled": 0., "short_hsl_enabled": 0., "hsl_enabled": 0.}])
    assert not (disabled["hsl_triggers_long"] + disabled["hsl_triggers_short"]).any()
    _, restored = raw(proxy, [{}])
    _equal(restored, baseline)
    assert runner.hsl_fact_capacity == 0
    assert runner.last_hsl_fact_retries == 0


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_coin_override_enables_forward_risk_with_inherited_policy_off(reference, forbid_cpu, strategy):
    from test_gpu_hsl_multicoin import make_proxy, raw

    proxy = make_proxy("coin", strategy, ("long",), minutes=128, coin_count=1,
                       enabled=False, override={"enabled": True, "red_threshold": 1e-8})
    runner, actual = raw(proxy, [{}])
    assert actual["hsl_triggers_long"].item() > 0
    assert runner.hsl_fact_capacity == 0


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_authoritative_service_never_calls_cpu_simulation(reference, forbid_cpu, strategy):
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import NativeBacktestService
    from tools.gpu_parity import build_parser, fixture_inputs, _native_dataset

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "2", "--bars", "128",
        "--hsl", "portfolio", "--hsl-red-threshold", "1e-8", "--hsl-ema-span-minutes", "1",
    ]))
    metrics = ("fills_per_day", "hard_stop_triggers_per_year")
    with _native_dataset(inputs, "binance", metrics) as dataset:
        with NativeBacktestService(batch_size=1, tuning_mode="off") as service:
            service.register_dataset("forward", dataset)
            actual = service.submit(BacktestRequest("one", "forward", {})).result(timeout=120)
    assert actual.metrics["fills_per_day"] > 0
    assert actual.metrics["hard_stop_triggers_per_year"] > 0
