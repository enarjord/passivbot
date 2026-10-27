"""Shared current replay fixtures for feature ablation regressions."""
import numpy as np
from optimization.gpu.model import adaptive_params

def _base_parameter_values() -> dict[str, float]:
    return {
        **adaptive_params({}),
        "base_qty_fraction": 0.08,
        "ema_span_0_minutes": 60.0,
        "ema_span_1_minutes": 240.0,
        "entry_ema_span_0_minutes": 60.0,
        "entry_ema_span_1_minutes": 240.0,
        "entry_qty_exposure_coefficient": 1.25,
        "entry_reentry_qty_multiplier": 1.25,
        "offset_fraction": 0.01,
        "inventory_offset_exposure_coefficient": 0.0,
        "offset_log_range_1h_coefficient": 0.0,
        "offset_log_range_1m_coefficient": 0.0,
        "offset_log_range_ema_span_1h_candles": 120.0,
        "offset_log_range_ema_span_minutes": 60.0,
        "log_range_ema_span_1h_candles": 120.0,
        "log_range_ema_span_minutes": 60.0,
        "entry_initial_ema_distance_fraction": 0.01,
        "entry_initial_qty_fraction": 0.08,
        "entry_threshold_base_fraction": 0.01,
        "entry_threshold_exposure_ratio_coefficient": 0.0,
        "entry_threshold_log_range_1h_coefficient": 0.0,
        "entry_threshold_log_range_1m_coefficient": 0.0,
        "entry_retracement_base_fraction": 0.001,
        "entry_retracement_exposure_ratio_coefficient": 0.0,
        "entry_retracement_log_range_1h_coefficient": 0.0,
        "entry_retracement_log_range_1m_coefficient": 0.0,
        "close_qty_fraction": 0.5,
        "close_threshold_base_fraction": 0.01,
        "close_threshold_exposure_ratio_coefficient": 0.0,
        "close_threshold_log_range_1h_coefficient": 0.0,
        "close_threshold_log_range_1m_coefficient": 0.0,
        "close_retracement_base_fraction": 0.001,
        "close_retracement_log_range_1h_coefficient": 0.0,
        "close_retracement_log_range_1m_coefficient": 0.0,
        "entry_cooldown_minutes": 0.0,
        "total_wallet_exposure_limit_ratio": 1.0,
        "gate_initial": 1.0,
        "gate_reentry": 1.0,
        "forager_volume_ema_span_minutes": 120.0,
        "forager_log_range_ema_span_minutes": 60.0,
        "forager_score_weights_volume": 1.0,
        "forager_score_weights_ema_readiness": 0.0,
        "forager_score_weights_log_range": 0.0,
        "n_positions": 4.0,
        "we_excess_allowance_fraction": 0.0,
        "hsl_scale_budget_with_excess_allowance": 0.0,
        "twel_entry_gate_enabled": 1.0,
        "twel_enforcer_threshold_ratio": 1.0,
        "wel_enforcer_enabled": 0.0,
        "wel_enforcer_threshold_ratio": 1.0,
        "twel_enforcer_enabled": 0.0,
        "twel_enforcer_reduce_portfolio": 0.0,
        "unstuck_ema_span_0_minutes": 17.25,
        "unstuck_ema_span_1_minutes": 211.75,
        "unstuck_enabled": 0.0,
        "unstuck_ema_gating_enabled": 1.0,
        "unstuck_close_fraction": 0.1,
        "unstuck_ema_distance_fraction": 0.0,
        "unstuck_loss_allowance_fraction": 0.1,
        "unstuck_exposure_trigger_ratio": 0.5,
        "hsl_enabled": 0.0,
        "hsl_drawdown_threshold_fraction": 0.2,
        "hsl_drawdown_fraction_ema_span_minutes": 60.0,
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
    metrics = {"cagr_fraction_strategy_equity", "drawdown_max_fraction_strategy_equity", "gross_turnover_ratio_per_day"}
    replay = SharedAccountReplay(config=config, hlcvs=candles, mss=markets,
        btc=btc, timestamps=timestamps, exchange="binance", batch_size=dispatch_batch_size,
        needed_metrics=metrics, factual_hsl=True)
    rng = np.random.default_rng(seed)
    key = "long_base_qty_fraction" if strategy == "ema_anchor" else "long_entry_initial_qty_fraction"
    base = replay.base_params["long"][key.removeprefix("long_")]
    population = [{key: float(base * factor)} for factor in rng.uniform(.75, 1.25, candidates)]
    return replay, population, multicoin_bars, coins, 1, _identity(config, (candles, btc, timestamps), markets, "binance")
