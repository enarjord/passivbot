"""Shared current replay fixtures for feature ablation regressions."""
import numpy as np
from optimization.gpu.model import adaptive_params

def _runner_context(strategy, sides, *, chunked=False,
                    raw_growth=False, raw_risk=False, btc_risk=False, hsl_tail=False, shock=False,
                    factual_hsl=True, count=1513):
    from test_gpu_mps import _multicoin_exposure_fixture
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorMulticoinRunner, MpsEmaAnchorMulticoinFusedRunner,
        MpsTrailingMartingaleMulticoinRunner, MpsTrailingMartingaleMulticoinFusedRunner,
    )
    closes = np.tile([100., 120.], (count, 1))
    if shock:
        closes[100:700, 0] *= .7
        closes[200:800, 1] *= 1.3
    _, row, run, data = _multicoin_exposure_fixture(
        strategy, "long" if sides == "both" else sides,
        count=count, requested_start_index=31, return_context=True, closes=closes,
    )
    cls = {
        ("ema_anchor", False): MpsEmaAnchorMulticoinRunner,
        ("ema_anchor", True): MpsEmaAnchorMulticoinFusedRunner,
        ("trailing_martingale", False): MpsTrailingMartingaleMulticoinRunner,
        ("trailing_martingale", True): MpsTrailingMartingaleMulticoinFusedRunner,
    }[strategy, sides == "both"]
    if shock:
        from optimization.gpu.model import EMA_ANCHOR_MULTICOIN_PARAM_KEYS, TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
        keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS if strategy == "ema_anchor" else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
        for key, value in {"hsl_enabled": 1, "hsl_signal_mode": 2,
                           "hsl_red_threshold": .01, "hsl_drawdown_ratio_ema_span_minutes": 2.5,
                           "hsl_cooldown_ms_lo": 300000, "hsl_cooldown_ms_hi": 0,
                           "hsl_memory_ema_span_minutes": 1440}.items():
            row[keys.index(key)] = value
    kwargs = dict(recovery_distribution_enabled=True,
                  hsl_ema_tail_enabled=hsl_tail,
                  factual_hsl=factual_hsl,
                  raw_strategy_growth_enabled=raw_growth, raw_strategy_risk_enabled=raw_risk,
                  btc_risk_enabled=btc_risk,
                  btc_prices=np.full(count, 30000.0) if btc_risk else None)
    if shock:
        kwargs["pnl_lookback_bars"] = 1440
    if sides != "both":
        kwargs["side"] = sides
    if chunked:
        assert strategy == "trailing_martingale" and sides != "both"
        kwargs["max_dispatch_candidate_bars"] = 47 * 2 * 3
    params = np.asarray([row] * 3, dtype=np.float64)
    if sides == "both":
        params = np.concatenate((params, params), axis=1)
    return cls(run, data, **kwargs), params


def _base_parameter_values() -> dict[str, float]:
    return {
        **adaptive_params({}),
        "base_qty_pct": 0.08,
        "ema_span_0": 60.0,
        "ema_span_1": 240.0,
        "entry_double_down_factor": 1.25,
        "offset": 0.01,
        "offset_psize_weight": 0.0,
        "offset_volatility_1h_weight": 0.0,
        "offset_volatility_1m_weight": 0.0,
        "offset_volatility_ema_span_1h": 120.0,
        "offset_volatility_ema_span_1m": 60.0,
        "volatility_ema_span_1h": 120.0,
        "volatility_ema_span_1m": 60.0,
        "entry_initial_ema_dist": 0.01,
        "entry_initial_qty_pct": 0.08,
        "entry_threshold_base_pct": 0.01,
        "entry_threshold_we_weight": 0.0,
        "entry_threshold_volatility_1h_weight": 0.0,
        "entry_threshold_volatility_1m_weight": 0.0,
        "entry_retracement_base_pct": 0.001,
        "entry_retracement_we_weight": 0.0,
        "entry_retracement_volatility_1h_weight": 0.0,
        "entry_retracement_volatility_1m_weight": 0.0,
        "close_qty_pct": 0.5,
        "close_threshold_base_pct": 0.01,
        "close_threshold_we_weight": 0.0,
        "close_threshold_volatility_1h_weight": 0.0,
        "close_threshold_volatility_1m_weight": 0.0,
        "close_retracement_base_pct": 0.001,
        "close_retracement_volatility_1h_weight": 0.0,
        "close_retracement_volatility_1m_weight": 0.0,
        "entry_cooldown_minutes": 0.0,
        "total_wallet_exposure_limit": 1.0,
        "gate_initial": 1.0,
        "gate_reentry": 1.0,
        "forager_volume_ema_span_1m": 120.0,
        "forager_volatility_ema_span_1m": 60.0,
        "forager_score_weights_volume": 1.0,
        "forager_score_weights_ema_readiness": 0.0,
        "forager_score_weights_volatility": 0.0,
        "n_positions": 4.0,
        "we_excess_allowance_pct": 0.0,
        "hsl_scale_budget_with_excess_allowance": 0.0,
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold": 1.0,
        "wel_enforcer_enabled": 0.0,
        "wel_enforcer_threshold": 1.0,
        "twel_enforcer_enabled": 0.0,
        "twel_enforcer_reduce_portfolio": 0.0,
        "unstuck_ema_span_0": 17.25,
        "unstuck_ema_span_1": 211.75,
        "unstuck_enabled": 0.0,
        "unstuck_ema_gating_enabled": 1.0,
        "unstuck_close_pct": 0.1,
        "unstuck_ema_dist": 0.0,
        "unstuck_loss_allowance_pct": 0.1,
        "unstuck_threshold": 0.5,
        "hsl_enabled": 0.0,
        "hsl_red_threshold": 0.2,
        "hsl_drawdown_ratio_ema_span_minutes": 60.0,
        "hsl_cooldown_ms_lo": 0.0,
        "hsl_cooldown_ms_hi": 0.0,
        "hsl_memory_ema_span_minutes": 1440.0,
        "hsl_signal_mode": 0.0,
        "unstuck_memory_ema_span_minutes": 1440.0,
        "loss_gate_memory_ema_span_minutes": 1440.0,
    }


def _build_case(name, *, candidates, dispatch_batch_size, single_bars,
                multicoin_bars, coins, seed):
    """A current native shared-account fixture; no retired directional engine."""
    from tools.gpu_parity import build_parser, fixture_inputs, _identity
    from optimization.gpu.service import SharedAccountReplay
    strategies = {"ema-multicoin-overhead": "ema_anchor",
                  "tm-multicoin-overhead": "trailing_martingale"}
    strategy = strategies[name]
    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "long", "--coins", str(coins),
        "--bars", str(multicoin_bars), "--seed", str(seed)]))
    config, candles, markets, btc, timestamps = inputs
    metrics = {"adg_strategy_eq", "drawdown_worst_strategy_eq", "fills_per_day"}
    replay = SharedAccountReplay(config=config, hlcvs=candles, mss=markets,
        btc=btc, timestamps=timestamps, exchange="binance", batch_size=dispatch_batch_size,
        needed_metrics=metrics, factual_hsl=True)
    rng = np.random.default_rng(seed)
    key = "long_base_qty_pct" if strategy == "ema_anchor" else "long_entry_initial_qty_pct"
    base = replay.base_params["long"][key.removeprefix("long_")]
    population = [{key: float(base * factor)} for factor in rng.uniform(.75, 1.25, candidates)]
    return replay, population, multicoin_bars, coins, 1, _identity(config, (candles, btc, timestamps), markets, "binance")
