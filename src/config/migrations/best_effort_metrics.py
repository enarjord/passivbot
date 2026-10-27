"""Observable formula choices for the offline released-v8 converter only."""

from dataclasses import dataclass
import math

from .released_v8_metrics import RELEASED_V8_METRIC_NAMES, RELEASED_V8_STAT_SUFFIXES, released_v8_metric_name


@dataclass(frozen=True)
class MetricInference:
    target: str | None
    reason: str
    transform: str = "identity"
    outer_reducer: str | None = None


# Exact historical formula families, not a general permission to strip modifiers.
_SHARED_WEIGHTED = {
    "positions_held_per_day_w", "win_rate_w", "volume_pct_per_day_avg_w",
    "loss_profit_ratio_w", "adg_pnl_w", "mdg_pnl_w", "sharpe_ratio_pnl_w",
    "sortino_ratio_pnl_w", "adg_strategy_eq_w", "mdg_strategy_eq_w",
    "sharpe_ratio_strategy_eq_w", "sortino_ratio_strategy_eq_w",
    "omega_ratio_strategy_eq_w", "calmar_ratio_strategy_eq_w", "sterling_ratio_strategy_eq_w",
    *(f"{metric}_strategy_pnl_rebased_w" for metric in
      ("adg", "mdg", "sharpe_ratio", "sortino_ratio", "omega_ratio", "calmar_ratio", "sterling_ratio")),
}
_CURRENCY_WEIGHTED = {
    "adg_w", "adg_w_per_exposure_long", "adg_w_per_exposure_short",
    "mdg_w", "mdg_w_per_exposure_long", "mdg_w_per_exposure_short",
    "calmar_ratio_w", "exposure_mean_ratio_w", "exposure_ratio_w",
    "equity_choppiness_w", "equity_jerkiness_w", "exponential_fit_error_w",
    "omega_ratio_w", "paper_loss_mean_ratio_w", "paper_loss_ratio_w",
    "sharpe_ratio_w", "sortino_ratio_w", "sterling_ratio_w",
}
_VERIFIED_WEIGHTED = frozenset({
    *_SHARED_WEIGHTED, *_CURRENCY_WEIGHTED,
    *(f"{metric}_{currency}" for metric in _CURRENCY_WEIGHTED for currency in ("usd", "btc")),
    *(f"{currency}_{metric}" for metric in _SHARED_WEIGHTED | _CURRENCY_WEIGHTED for currency in ("usd", "btc")),
})
_EXTRA_RELEASED = frozenset({"strategy_eq_underwater_pct_mean", "strategy_eq_underwater_pct_median"})
_INFERRED = {
    "adg_strategy_eq": ("cagr_fraction_strategy_equity", "annualize_daily",
                        "replace smoothed daily growth with unsmoothed endpoint annual CAGR"),
    "adg_strategy_pnl_rebased": ("cagr_fraction_strategy_equity", "annualize_daily",
                                "replace smoothed daily growth with unsmoothed endpoint annual CAGR"),
    "adg_strategy_eq_w": ("cagr_fraction_strategy_equity_ew", "annualize_daily",
                          "replace legacy weighted daily growth with calendar-weighted annual log growth"),
    "adg_strategy_pnl_rebased_w": ("cagr_fraction_strategy_equity_ew", "annualize_daily",
                                  "replace legacy weighted daily growth with calendar-weighted annual log growth"),
    "volume_pct_per_day_avg": ("gross_turnover_ratio_per_day", "identity",
                               "replace the active-day volume proxy with all-day pre-fill-balance-normalized gross turnover"),
    "volume_pct_per_day_avg_w": ("gross_turnover_ratio_per_day", "identity",
                                 "drop legacy recency weighting and replace the active-day volume proxy with all-day pre-fill-balance-normalized gross turnover"),
    "strategy_eq_underwater_pct_mean": ("pain_index_fraction_strategy_equity", "identity",
                                         "replace mean daily worst drawdown with minute-sampled mean drawdown (Pain Index)"),
    "strategy_eq_recovery_days_max": ("peak_recovery_max_days_strategy_equity", "identity",
                                       "strictly-higher waiting times become running-peak recovery episodes; peak ties recover and flat equity has zero recovery"),
    "peak_recovery_days_strategy_eq": ("peak_recovery_max_days_strategy_equity", "identity",
                                       "strictly-higher waiting times become running-peak recovery episodes; peak ties recover and flat equity has zero recovery"),
    "position_held_hours_max": ("position_held_max_days", "hours_to_days",
                                "express maximum holding duration in days instead of hours"),
    "long_short_profit_ratio": ("short_net_profit_contribution_fraction", "complement",
                                "replace realized pre-fee long profit share with short net profit contribution; fees and open-position PnL now count and combined net profit must be positive"),
    "pnl_ratio_long_short": ("short_net_profit_contribution_fraction", "complement",
                             "replace realized pre-fee long profit share with short net profit contribution; fees and open-position PnL now count and combined net profit must be positive"),
}


def _verified_base(name):
    if name in RELEASED_V8_METRIC_NAMES or name in _EXTRA_RELEASED:
        return True
    if name in _VERIFIED_WEIGHTED:
        return True
    for prefix in ("usd_", "btc_"):
        if name.startswith(prefix) and (name[len(prefix):] in RELEASED_V8_METRIC_NAMES or name[len(prefix):] in _EXTRA_RELEASED):
            return True
    return False


def _split_released(name):
    base, outer = name, None
    if not _verified_base(base):
        for stat in RELEASED_V8_STAT_SUFFIXES:
            if name.endswith("_" + stat) and _verified_base(name[:-(len(stat) + 1)]):
                base, outer = name[:-(len(stat) + 1)], stat
                break
    if not _verified_base(base):
        return None
    for prefix in ("usd_", "btc_"):
        if not base.startswith(prefix):
            continue
        core = base[len(prefix):]
        if core in _SHARED_WEIGHTED or core in _EXTRA_RELEASED:
            base = core
        elif core in RELEASED_V8_METRIC_NAMES and released_v8_metric_name(base) == RELEASED_V8_METRIC_NAMES[core]:
            base = core
        break
    return base, outer


def legacy_metric_identity(name):
    """Resolve frozen aliases before looking up an original reducer policy."""
    split = _split_released(name)
    if split is None:
        return name
    base, outer = split
    base = RELEASED_V8_METRIC_NAMES.get(base, base.replace("_strategy_pnl_rebased_w", "_strategy_eq_w"))
    return base + ("_" + outer if outer else "")


def infer_metric(name):
    """Return a verified replacement/removal, or None for an unknown selector."""
    from config.metrics import KNOWN_METRICS, metric_base_name

    split = _split_released(name)
    if split is not None:
        base, outer = split
        if base in _INFERRED:
            target, transform, reason = _INFERRED[base]
            return MetricInference(target, reason, transform, outer)
        if base in RELEASED_V8_METRIC_NAMES:
            target = RELEASED_V8_METRIC_NAMES[base]
            if metric_base_name(target) in KNOWN_METRICS:
                try:
                    released_v8_metric_name(name)
                except ValueError:
                    return MetricInference(target, "preserve verified shared presentation and outer statistic", outer_reducer=outer)
                return None  # Existing retained-definition conversion owns this case.
        return MetricInference(None, "retired metric has no meaningful compact-core counterpart", outer_reducer=outer)
    # The frozen converter also verifies historical currency-prefix selectors.
    try:
        target = released_v8_metric_name(name)
    except ValueError:
        return None
    if metric_base_name(target) in KNOWN_METRICS:
        return None
    return MetricInference(None, "retired currency/presentation metric has no meaningful compact-core counterpart")


def transform_bound(value, transform):
    """Convert a finite bound, raising when a nominal inference is unsupported."""
    if not math.isfinite(value):
        raise ValueError("metric bounds must be finite")
    if transform == "hours_to_days":
        return value / 24.0
    if transform == "complement":
        return 1.0 - value
    if transform == "annualize_daily":
        if value == -1:
            return -1.0
        if value < -1:
            raise ValueError("daily growth below -1 has no nominal annual equivalent")
        try:
            result = math.expm1(365.25 * math.log1p(value))
        except OverflowError as exc:
            raise ValueError("nominal annual growth bound overflows") from exc
        if not math.isfinite(result):
            raise ValueError("nominal annual growth bound must be finite")
        return result
    return value
