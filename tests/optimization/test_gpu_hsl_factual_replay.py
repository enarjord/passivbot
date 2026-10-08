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


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
def test_native_policy_transitions_preserve_learned_storage_and_factual_results(reference, forbid_cpu, strategy, mode, sides):
    from test_gpu_hsl_multicoin import make_proxy, raw

    native = make_proxy(mode, strategy, sides, minutes=256, factual_hsl=True)
    explicit = make_proxy(mode, strategy, sides, minutes=256)
    _enable(explicit)
    _, expected = raw(explicit, [{}])
    runner = native.fused_runner or native.runners[sides[0]]
    runner.hsl_fact_capacity_learned = 1
    _, actual = raw(native, [{}])
    _equal(actual, expected)
    assert runner.last_hsl_fact_retries > 0
    learned = runner.hsl_fact_capacity_learned
    assert learned == runner.hsl_fact_capacity > 1
    active_library = runner._library_cache_call()
    active_cost = runner._history_bytes_per_candidate()

    disabled = {f"{side}_hsl_enabled": 0.0 for side in sides}
    general = make_proxy(mode, strategy, sides, minutes=256)
    _, off_expected = raw(general, [disabled])
    _, off_actual = raw(native, [disabled])
    _equal(off_actual, off_expected)
    assert runner.hsl_fact_capacity == 0 and runner.settings[-2:].tolist() == [0.0, 0.0]
    assert runner.hsl_fact_capacity_learned == learned
    assert runner._history_bytes_per_candidate() < active_cost
    assert runner._library_cache_call() != active_library
    if strategy == "ema_anchor" and len(sides) == 1 and mode != "coin":
        assert runner.dispatch_hsl_disabled

    _, restored = raw(native, [{}])
    _equal(restored, expected)
    assert runner.hsl_fact_capacity == learned and runner.last_hsl_fact_retries == 0
    assert runner._library_cache_call() == active_library


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides,override_side", [
    (("long",), "long"), (("short",), "short"),
    (("long", "short"), "long"), (("long", "short"), "short"),
])
def test_native_coin_override_enables_factual_replay_with_base_hsl_off(reference, forbid_cpu, strategy, sides, override_side):
    from test_gpu_hsl_multicoin import make_proxy, raw

    options = dict(mode="coin", strategy=strategy, sides=sides, minutes=256,
                   enabled=False, override={"enabled": True}, override_side=override_side)
    native, explicit = make_proxy(**options, factual_hsl=True), make_proxy(**options)
    _enable(explicit)
    _, expected = raw(explicit, [{}])
    runner, actual = raw(native, [{}])
    _equal(actual, expected)
    assert runner.hsl_factual_replay and runner.hsl_fact_capacity > 0
    assert not runner.dispatch_hsl_disabled


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("sides", ["long", "both"])
def test_authoritative_service_uses_factual_replay_without_cpu_backtests(reference, forbid_cpu, monkeypatch, strategy, mode, sides):
    from optimization.gpu.datasets import PreparedGpuDataset
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from optimization.gpu.service import MpsMulticoinProxy
    from shared_arrays import SharedArrayManager
    from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS

    def forbidden(*_args, **_kwargs):
        pytest.fail("Rust CPU simulation called during authoritative service replay")
    monkeypatch.setattr(reference, "run_backtest_bundle", forbidden)
    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", sides, "--coins", "2", "--bars", "512",
        "--hsl", mode, "--hsl-red-threshold", ".002", "--hsl-ema-span-minutes", "2.5",
        "--price-shock", "0", "256", ".7",
    ]))
    config, candles, markets, btc, timestamps = inputs
    metrics = (*DEFAULT_METRICS, "hard_stop_time_in_red_pct", "hard_stop_duration_minutes_max",
               "hard_stop_triggers_per_year", "hard_stop_restarts_per_year")
    explicit = MpsMulticoinProxy(config=config, hlcvs=candles, mss=markets, btc=btc,
        timestamps=timestamps, exchange="binance", batch_size=1, needed_metrics=metrics)
    _enable(explicit, 1024)
    expected = explicit.evaluate_results([{}])[0]
    observed = []
    original = MpsMulticoinProxy.evaluate_results
    def inspect(self, parameters):
        result = original(self, parameters)
        runners = [self.fused_runner] if self.fused_runner else list(self.runners.values())
        observed.extend((r.native_factual_hsl, r.hsl_factual_replay, r.hsl_fact_capacity) for r in runners)
        return result
    monkeypatch.setattr(MpsMulticoinProxy, "evaluate_results", inspect)
    manager = SharedArrayManager()
    try:
        specs = [manager.create_from(a)[0] for a in (candles, btc, timestamps)]
        dataset = PreparedGpuDataset(config=config, markets=markets, exchange="binance",
            hlcvs=specs[0], btc=specs[1], timestamps=specs[2],
            candle_coins=tuple(config["backtest"]["coins"]["binance"]), metrics=metrics)
        with CudaBacktestService(batch_size=1, tuning_mode="off") as service:
            service.register_dataset("factual", dataset)
            actual = service.submit(BacktestRequest("one", "factual", {})).result(timeout=120)
        assert actual.metrics == expected.metrics
        assert actual.liquidated == expected.liquidated
        assert observed and all(native and factual and cap > 0 for native, factual, cap in observed)
    finally:
        manager.cleanup()


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
def test_disabling_every_effective_coin_policy_omits_factual_storage(reference, forbid_cpu, strategy, sides):
    from test_gpu_hsl_multicoin import make_proxy, raw

    # Fused: disable the inherited long policy in the candidate and override
    # the enabled short policy off. Both effective sides are then off.
    override_side = sides[-1]
    options = dict(mode="coin", strategy=strategy, sides=sides, minutes=256,
                   coin_count=1, enabled=True, override={"enabled": False},
                   override_side=override_side)
    native = make_proxy(**options, factual_hsl=True)
    legacy = make_proxy(**options)
    candidate = {"long_hsl_enabled": 0.0} if len(sides) == 2 else {}
    fully_disabled = candidate | {f"{side}_hsl_enabled": 0.0 for side in sides}
    _, expected = raw(legacy, [fully_disabled])
    runner, actual = raw(native, [candidate])
    _equal(actual, expected)
    assert runner.hsl_fact_capacity == 0 and not runner.hsl_factual_replay
    assert runner.settings[-2:].tolist() == [0.0, 0.0]
    # Omitting unused history must not silently accept malformed policy values.
    with pytest.raises(ValueError, match="Invalid GPU HSL policy"):
        raw(native, [candidate | {f"{override_side}_hsl_red_threshold": 0.0}])
    # Removing that long-side disable restores one effective inherited policy.
    if len(sides) == 2:
        explicit = make_proxy(**options)
        _enable(explicit)
        _, expected = raw(explicit, [{}])
        _, actual = raw(native, [{}])
        _equal(actual, expected)
        assert runner.hsl_factual_replay and runner.hsl_fact_capacity > 0
