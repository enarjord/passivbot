"""Actual native simulators use factual HSL reconstruction without CPU replay."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")


@pytest.fixture(scope="module")
def reference():
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    return passivbot_rust


def _enable(proxy, capacity=64):
    runners = [proxy.fused_runner] if proxy.fused_runner else list(proxy.runners.values())
    for runner in runners:
        runner.hsl_fact_capacity = capacity
        runner.hsl_factual_replay = True


def _equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            np.testing.assert_array_equal(actual[key].cpu().numpy(), value.cpu().numpy(), err_msg=key)
        else:
            assert actual[key] == value, key


@pytest.fixture
def forbid_cpu(monkeypatch):
    import backtest

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU backtest called during native GPU factual replay")

    monkeypatch.setattr(backtest, "execute_backtest", forbidden)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coin_count", [1, 2])
def test_native_factual_replay_dispatch(reference, forbid_cpu, strategy, mode, sides, coin_count):
    from test_gpu_hsl_multicoin import make_proxy, raw

    proxy = make_proxy(mode, strategy, sides, minutes=128, coin_count=coin_count)
    assert len(proxy.checkpoint_contract["coins"]) == coin_count
    _enable(proxy)
    runner, first = raw(proxy, [{}, {}])
    assert float(first["fill_count"].min()) > 0
    assert runner.settings[-1] == 1
    _, repeated = raw(proxy, [{}, {}])
    _equal(repeated, first)


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coin_count", [1, 2])
def test_factual_context_rebinds_after_every_temporal_chunk(reference, forbid_cpu, mode, sides, coin_count):
    from test_gpu_hsl_multicoin import make_proxy, raw

    proxy = make_proxy(mode, "trailing_martingale", sides, minutes=128, coin_count=coin_count)
    _enable(proxy)
    runner, full = raw(proxy, [{}, {}])
    runner.max_dispatch_candidate_bars = 64
    _, chunked = raw(proxy, [{}, {}])
    assert runner._last_temporal_dispatch["dispatch_count"] > 1
    _equal(chunked, full)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_factual_replay_capacity_growth_repeats_only_gpu_work(reference, forbid_cpu, strategy):
    from test_gpu_hsl_multicoin import make_proxy, raw

    proxy = make_proxy("unified", strategy, ("long", "short"), minutes=256)
    _enable(proxy)
    runner, baseline = raw(proxy, [{}])
    _enable(proxy, capacity=1)
    compiled_identity = runner._library_cache_call()
    _, grown = raw(proxy, [{}])
    assert runner.hsl_fact_capacity > 1
    assert runner.last_hsl_fact_retries > 0
    assert runner._library_cache_call() == compiled_identity
    _equal(grown, baseline)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", [("long",), ("long", "short")])
def test_disabled_policy_factual_route_preserves_all_metrics(reference, forbid_cpu, strategy, sides):
    from test_gpu_hsl_multicoin import make_proxy, raw

    proxy = make_proxy("unified", strategy, sides, minutes=128, enabled=False)
    _, baseline = raw(proxy, [{}])
    _enable(proxy)
    _, replay = raw(proxy, [{}])
    _equal(replay, baseline)


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
    if explicit_callback:
        assert first.runners["long"].interrupt_check is callback
    # Different cancellation ownership still prevents combined execution.
    other = make_proxy(**(kwargs | {"interrupt_check": lambda: None}))
    assert other.data is first.data
    assert other.suite_batch_key() != first.suite_batch_key()
