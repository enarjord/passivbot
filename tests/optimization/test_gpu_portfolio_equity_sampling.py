"""Explicit portfolio strategy risk retains raw marks, separately from account risk."""
import pytest


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "both"])
@pytest.mark.parametrize("terminal_fill", [False, True], ids=["mark", "panic-fill"])
def test_native_portfolio_raw_risk_retains_liquidation_mark(strategy, sides, terminal_fill):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from test_gpu_hsl_ordering import _liquidation_inputs
    from tools.gpu_parity import run_comparison
    from optimization.gpu.parity import MetricTolerance

    metrics = ("drawdown_worst_strategy_eq", "drawdown_worst_mean_1pct_strategy_eq",
               "strategy_eq_underwater_pct_mean", "drawdown_worst_usd")
    report = run_comparison(
        _liquidation_inputs(strategy, sides, terminal_fill), "bybit", metrics,
        {name: MetricTolerance(1e-8, 1e-6) for name in metrics},
        diagnostics=True, gpu_engine="native",
    )
    assert report["diagnostics"]["gpu"]["native_result"]["liquidated"]
    assert report["metrics"]["drawdown_worst_strategy_eq"]["cpu"] > 3
    assert report["metrics"]["drawdown_worst_usd"]["cpu"] < 1
    assert report["passed"], report["metrics"]


@pytest.mark.parametrize("days", [2, 200])
def test_raw_daily_risk_keeps_account_aliases_and_excludes_padding(days):
    from types import SimpleNamespace
    torch = pytest.importorskip("torch")
    from optimization.gpu.metrics import compute_objectives
    from optimization.gpu.model import GAP_BINS

    shape = (1, days + 1)
    day_min = torch.full(shape, 100.0, dtype=torch.float64)
    day_min[:, -1] = float('inf')
    account_dd = torch.full(shape, .1, dtype=torch.float64)
    raw_dd = torch.full(shape, .2, dtype=torch.float64)
    raw_dd[:, 0] = 3.0
    raw_dd[:, 1] = 2.0
    raw_dd[:, -1] = 1000.0  # Inactive padding cannot enter risk summaries.
    zero = torch.zeros(1, dtype=torch.float64)
    out = {
        'day_end_eq': torch.full(shape, 100.0, dtype=torch.float64),
        'day_min_eq': day_min, 'day_max_dd': account_dd,
        'raw_strategy_day_max_dd': raw_dd,
        'day_volume': torch.zeros(shape),
        'day_has_fill': torch.zeros(shape, dtype=torch.bool),
        'max_dd': torch.tensor([.1]), 'fill_count': zero,
        'held_max_ms': zero, 'gap_max_ms': zero,
        'gap_hist': torch.zeros((1, GAP_BINS), dtype=torch.int32),
        'first_fill_ts': zero, 'last_fill_ts': zero,
        'first_eq_ts': zero, 'last_eq_ts': torch.tensor([(days-1)*86400000.0]),
        'last_high_ts': zero, 'recovery_max_ms': zero,
        'liq_step': torch.tensor([-1]),
    }
    names = {'drawdown_worst_strategy_eq', 'drawdown_worst_mean_1pct_strategy_eq',
             'strategy_eq_underwater_pct_mean', 'drawdown_worst_usd',
             'drawdown_worst_mean_1pct_usd'}
    values = compute_objectives(
        out, SimpleNamespace(interval_ms=86400000, requested_start_ts_ms=0),
        {'ts0': 0, 'n': days}, needed=names,
    )
    assert set(values) == names
    assert values['drawdown_worst_strategy_eq'].item() == pytest.approx(3.0)
    assert values['drawdown_worst_mean_1pct_strategy_eq'].item() == pytest.approx(
        3.0 if days == 2 else 2.5)
    assert values['strategy_eq_underwater_pct_mean'].item() == pytest.approx(
        (5.0 + (days-2)*.2) / days)
    for name in ('drawdown_worst_usd', 'drawdown_worst_mean_1pct_usd'):
        assert values[name].item() == pytest.approx(.1)


@pytest.mark.parametrize('strategy', ['ema_anchor', 'trailing_martingale'])
@pytest.mark.parametrize('side', ['long', 'short'])
@pytest.mark.parametrize('btc_risk', [False, True])
def test_raw_risk_ablation_preserves_other_outputs_and_budgets_days(strategy, side, btc_risk):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    import numpy as np
    from test_gpu_mps import _multicoin_exposure_fixture
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorMulticoinRunner, MpsTrailingMartingaleMulticoinRunner,
    )

    count = 1513
    _, row, run, data = _multicoin_exposure_fixture(
        strategy, side, count=count, return_context=True,
    )
    cls = (MpsEmaAnchorMulticoinRunner if strategy == 'ema_anchor'
           else MpsTrailingMartingaleMulticoinRunner)
    kwargs = dict(side=side, btc_risk_enabled=btc_risk,
                  btc_prices=np.full(count, 30000.0) if btc_risk else None)
    baseline = cls(run, data, **kwargs)
    enabled = cls(run, data, raw_strategy_risk_enabled=True, **kwargs)
    assert enabled.daily_cols == baseline.daily_cols + 1
    assert (enabled._history_bytes_per_candidate() - baseline._history_bytes_per_candidate()
            == 4 * data['n_days'])
    params = np.asarray([row, row], dtype=np.float64)
    expected = baseline.run(params)
    actual = enabled.run(params)
    assert actual.keys() == expected.keys() | {'raw_strategy_day_max_dd'}
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0, equal_nan=True)
        else:
            assert actual[name] == value
    assert actual['raw_strategy_day_max_dd'].shape == (2, data['n_days'])
    assert torch.isfinite(actual['raw_strategy_day_max_dd']).all()


@pytest.mark.parametrize('strategy', ['ema_anchor', 'trailing_martingale'])
@pytest.mark.parametrize('sides', ['long', 'short', 'both'])
@pytest.mark.parametrize('hsl', ['disabled', 'coin', 'pside', 'unified'])
def test_native_portfolio_raw_risk_on_active_shock_replay(strategy, sides, hsl):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from test_gpu_side_equity_sampling import _side_equity_inputs
    from tools.gpu_parity import run_comparison
    from optimization.gpu.parity import MetricTolerance

    metrics = ('drawdown_worst_strategy_eq', 'drawdown_worst_mean_1pct_strategy_eq',
               'strategy_eq_underwater_pct_mean', 'drawdown_worst_usd')
    policies = {name: MetricTolerance(1e-6, 1e-4) for name in metrics}
    if strategy == 'trailing_martingale' and sides == 'both' and hsl == 'disabled':
        # This public seed-43 fixture retains the independently traced .017/.018
        # float32 partial-entry difference. Its accepted-source portfolio DD gap
        # is 5.24e-5; keep a fixture-local one-basis-point risk bound. The other
        # 23 scenarios and general parity policy remain strict.
        policies = {name: MetricTolerance(1e-4, 0) for name in metrics}
    account_ablation = strategy == 'ema_anchor' and sides == 'both' and hsl == 'coin'
    if account_ablation:
        # Accepted-source account DD has an existing 2.985e-6 residual in this
        # public fixture. Raw capture on/off and the accepted shader produce
        # exactly the same account value. Bound only this account comparison;
        # all three new raw-risk values retain the strict policy above.
        policies['drawdown_worst_usd'] = MetricTolerance(3.1e-6, 0)
    report = run_comparison(
        _side_equity_inputs(strategy, sides, hsl), 'binance', metrics, policies,
        diagnostics=True, gpu_engine='native',
    )
    assert not report['diagnostics']['gpu']['native_result']['liquidated']
    assert report['passed'], report['metrics']
    if account_ablation:
        account_only = run_comparison(
            _side_equity_inputs(strategy, sides, hsl), 'binance',
            ('drawdown_worst_usd',), {'drawdown_worst_usd': policies['drawdown_worst_usd']},
            gpu_engine='native',
        )
        assert account_only['passed'], account_only['metrics']
        assert (account_only['metrics']['drawdown_worst_usd']['gpu']
                == report['metrics']['drawdown_worst_usd']['gpu'])
