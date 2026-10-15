"""Offline Metal optimizer CLI, persistence and resume qualification."""

import pytest

from test_native_backend_cuda import test_native_optimizer_cli_runs_cuda_and_resumes_without_cpu as cli_smoke


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
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
async def test_metal_weighted_objectives_persist_and_resume(monkeypatch, tmp_path, strategy):
    await cli_smoke(monkeypatch, tmp_path, True, True, True, screening=True,
                    strategy_kind=strategy, weighted_equity=True, device="mps")


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
async def test_metal_hsl_tail_scores_and_resumes(monkeypatch, tmp_path, strategy):
    await cli_smoke(monkeypatch, tmp_path, False, True, True,
                    strategy_kind=strategy, scaled_hsl=True, hsl_tail=True, device="mps")
