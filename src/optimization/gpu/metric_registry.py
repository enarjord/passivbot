"""Torch-free native core metric capabilities for GPU replay."""

from config.metrics import CORE_METRIC_CATALOG

CORE_GPU_METRICS = frozenset(CORE_METRIC_CATALOG)
HSL_METRICS = frozenset(name for name in CORE_GPU_METRICS if name.startswith("hsl_"))
UTILIZATION_METRICS = frozenset(name for name in CORE_GPU_METRICS
                              if name.startswith("position_exposure_utilization_"))


def requested_metric_features(metric_names):
    names = set(metric_names)
    features = set()
    groups = {
        "daily_returns": {"volatility_annual_fraction_strategy_equity", "sharpe_ratio_strategy_equity",
                          "sortino_ratio_strategy_equity", "omega_ratio_strategy_equity"},
        "minute_path": {"drawdown_max_fraction_strategy_equity", "ulcer_index_fraction_strategy_equity",
                        "pain_index_fraction_strategy_equity", "peak_recovery_max_days_strategy_equity"},
        "calendar_weighted_growth": {"cagr_fraction_strategy_equity_ew"},
        "fill_turnover": {"gross_turnover_ratio_per_day"},
        "net_cashflow": {"loss_profit_ratio"},
        "side_net_profit": {"short_net_profit_contribution_fraction"},
        "gross_exposure": {"gross_exposure_rms_ratio_strategy_equity", "gross_exposure_max_ratio_strategy_equity"},
        "position_utilization": UTILIZATION_METRICS,
        "hsl": HSL_METRICS,
    }
    for feature, metrics in groups.items():
        if names & metrics:
            features.add(feature)
    return frozenset(features)
