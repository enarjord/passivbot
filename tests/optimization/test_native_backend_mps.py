"""Offline Metal optimizer CLI, persistence and resume qualification."""

import pytest

from test_native_backend_cuda import test_native_optimizer_cli_runs_cuda_and_resumes_without_cpu as cli_smoke


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", [False, True])
async def test_metal_optimizer_coupled_unstuck_spans_resume(monkeypatch, tmp_path, suite):
    await cli_smoke(monkeypatch, tmp_path, suite, True, False,
                    coupled=True, device="mps")


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", [False, True])
async def test_metal_cli_with_enabled_passing_prefilter(monkeypatch, tmp_path, suite):
    await cli_smoke(monkeypatch, tmp_path, suite, False, False, screening=suite,
                    device="mps", prefilter=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("suite", [False, True])
async def test_metal_optimizer_persists_and_resumes_without_cpu(monkeypatch, tmp_path, strategy, suite):
    await cli_smoke(monkeypatch, tmp_path, suite, True, True, screening=suite,
                    strategy_kind=strategy, device="mps")


@pytest.mark.asyncio
async def test_metal_max_bootstraps_fifty_seeds_and_resumes(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Metal device required")
    from optimization.gpu.mps_kernel import _MulticoinReplayRunner

    observed_widths = []
    original = _MulticoinReplayRunner._dispatch_replay

    def dispatch(runner, *args, **kwargs):
        original(runner, *args, **kwargs)
        observed_widths.append(kwargs["batch_size"])

    monkeypatch.setattr(_MulticoinReplayRunner, "_dispatch_replay", dispatch)
    seed_values = tuple(0.012 + index * 0.0003 for index in range(50))
    await cli_smoke(monkeypatch, tmp_path, True, True, True, device="mps",
                    tuning_mode="max", seed_values=seed_values)
    assert observed_widths[:2] == [58, 58]  # One full bootstrap dispatch per scenario.


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
async def test_metal_weighted_objectives_persist_and_resume(monkeypatch, tmp_path, strategy):
    await cli_smoke(monkeypatch, tmp_path, True, True, True, screening=True,
                    strategy_kind=strategy, additional_equity=True, device="mps")


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
async def test_metal_hsl_tail_scores_and_resumes(monkeypatch, tmp_path, strategy):
    await cli_smoke(monkeypatch, tmp_path, False, True, True,
                    strategy_kind=strategy, scaled_hsl=True, hsl_tail=True, device="mps")
