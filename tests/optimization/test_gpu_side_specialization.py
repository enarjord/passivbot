"""Prepared one-side identity can remove the unused directional kernel path."""
from inspect import signature

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


@pytest.fixture(scope="module")
def verified_rust():
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension
    info = verify_loaded_runtime_extension()
    assert not info.get("skipped")
    assert info["runtime_compiled_source_stamp"] == info["expected_source_fingerprint"]
    return passivbot_rust


def equal_outputs(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, right[key], rtol=0, atol=0, equal_nan=True, msg=key)
        else:
            assert value == right[key]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coins", [1, 3])
@pytest.mark.parametrize("hsl", [False, True])
def test_prepared_side_preserves_every_raw_output_and_general_cache(
    verified_rust, strategy, sides, coins, hsl,
):
    from test_gpu_hsl_multicoin import make_proxy, raw

    proxy = make_proxy("unified", strategy, sides, minutes=512, coin_count=coins,
                       enabled=hsl, factual_hsl=True)
    candidates = [{"hsl_red_threshold": 1e-6}, {"hsl_red_threshold": 1e-6}]
    runner, specialized = raw(proxy, candidates)
    assert specialized["fill_count"].min().item() > 0
    if hsl:
        assert (specialized["hsl_triggers_long"] + specialized["hsl_triggers_short"]).min().item() > 0
    loader, specialized_args = runner._library_cache_call()
    bound = signature(loader).bind(*specialized_args).arguments
    assert bound["side"] == (sides[0] if len(sides) == 1 else None)
    if coins == 1:
        assert runner.cuda_coin_capacity == 1
    if strategy == "trailing_martingale":
        runner.max_dispatch_candidate_bars = len(candidates) * coins * len(sides) * 31
        _, temporal = raw(proxy, candidates)
        equal_outputs(specialized, temporal)
        temporal_args = runner._library_cache_call()[1]
        assert temporal_args in runner._replay_state_sizes
    runner.side_specialization = False
    _, general = raw(proxy, candidates)
    equal_outputs(specialized, general)
    loader, general_args = runner._library_cache_call()
    assert signature(loader).bind(*general_args).arguments["side"] is None
    if len(sides) == 1:
        assert general_args != (temporal_args if strategy == "trailing_martingale" else specialized_args)
    runner.side_specialization = True
    _, restored = raw(proxy, candidates[:1])
    equal_outputs({key: value[:1] if isinstance(value, torch.Tensor) else value
                   for key, value in specialized.items()}, restored)


def test_side_compile_contract_rejects_invalid_or_missing_source(verified_rust):
    from optimization.gpu.mps_kernel import _with_multicoin_side
    with pytest.raises(ValueError, match="compile side"):
        _with_multicoin_side("", "buy")
    with pytest.raises(RuntimeError, match="side contract"):
        _with_multicoin_side("", "short")


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("suite", [False, True])
async def test_native_one_side_cli_uses_specialized_replay_without_cpu(
    verified_rust, monkeypatch, tmp_path, strategy, side, suite,
):
    from optimization.gpu.service import MpsMulticoinProxy
    from test_native_backend_cuda import test_native_optimizer_cli_runs_cuda_and_resumes_without_cpu

    evaluate = MpsMulticoinProxy.evaluate_results
    observations = []
    def observed(proxy, candidates):
        results = evaluate(proxy, candidates)
        assert proxy.fused_runner is None and list(proxy.sides) == [side]
        loader, args = proxy.runners[side]._library_cache_call()
        assert signature(loader).bind(*args).arguments["side"] == side
        observations.append(len(results))
        return results
    monkeypatch.setattr(MpsMulticoinProxy, "evaluate_results", observed)
    await test_native_optimizer_cli_runs_cuda_and_resumes_without_cpu(
        monkeypatch, tmp_path, suite, True, True, screening=suite, strategy_kind=strategy, sides=side,
    )
    assert observations and sum(observations) > 0
