"""Descriptive quantities for owned analysis fields; never supplies trading formulas.

Objective direction, accepted selectors, reductions and backend support remain with
their existing owners. This registry neither resolves aliases nor converts inputs.
"""

from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    unit: str
    denominator: str | None
    basis: str
    sampling: str
    aggregation: str
    sign: str
    domain: str
    weighting: str = "uniform"
    curve: str | None = None
    position_side: str | None = None
    notes: str = ""


def _definitions():
    from .metrics import CORE_METRIC_CATALOG

    descriptions = {
        "cagr_fraction_strategy_equity": (
            "starting strategy equity",
            "Annualized raw endpoint growth over actual elapsed years",
            "evaluation boundaries",
            "geometric annualization",
        ),
        "volatility_annual_fraction_strategy_equity": (
            None,
            "Sample daily simple-return standard deviation times sqrt(365.25)",
            "complete UTC daily close pairs",
            "sample standard deviation",
        ),
        "sharpe_ratio_strategy_equity": (
            "sample daily return standard deviation",
            "Arithmetic mean daily return divided by sample dispersion, times sqrt(365.25); no risk-free subtraction",
            "complete UTC daily close pairs",
            "quotient",
        ),
        "sortino_ratio_strategy_equity": (
            "RMS negative daily shortfall over ALL daily observations",
            "Arithmetic mean daily return divided by all-observation downside RMS, times sqrt(365.25)",
            "complete UTC daily close pairs",
            "quotient",
        ),
        "omega_ratio_strategy_equity": (
            "negative daily return magnitude sum",
            "Positive daily return mass divided by negative mass, zero threshold, no annualization",
            "complete UTC daily close pairs",
            "quotient",
        ),
        "drawdown_max_fraction_strategy_equity": (
            "running strategy-equity peak",
            "Largest decline from the running peak",
            "minute marks plus initial baseline",
            "maximum",
        ),
        "ulcer_index_fraction_strategy_equity": (
            "running strategy-equity peak",
            "Root mean square drawdown including zeros at highs",
            "minute marks plus initial baseline",
            "RMS",
        ),
        "pain_index_fraction_strategy_equity": (
            "running strategy-equity peak",
            "Mean drawdown including zeros at highs",
            "minute marks plus initial baseline",
            "mean",
        ),
        "cagr_fraction_strategy_equity_ew": (
            "calendar-age-weighted elapsed days",
            "Annualized weighted log growth; one fixed positive campaign half-life",
            "minute intervals",
            "exponential calendar-age weighting",
        ),
        "backtest_completion_fraction": (
            "intended evaluation duration",
            "Contiguous covered evaluation duration fraction",
            "actual evaluation boundaries",
            "duration quotient",
        ),
        "fills_gap_max_days": (
            None,
            "Maximum no-fill interval including leading and trailing boundaries",
            "actual fills and evaluation boundaries",
            "maximum",
        ),
        "position_held_max_days": (
            None,
            "Longest coin/position-side open-to-flat holding episode including open tails",
            "position lifecycle events",
            "maximum",
        ),
        "peak_recovery_max_days_strategy_equity": (
            None,
            "Longest last-high-to-regained-high underwater episode including unfinished tail",
            "minute marks",
            "maximum",
        ),
        "gross_turnover_ratio_per_day": (
            "each fill's raw pre-fill account balance and all elapsed days",
            "Sum absolute executed fill notional/pre-fill balance, divided by elapsed days; entries and exits count",
            "actual fills",
            "normalized notional sum per day",
        ),
        "position_held_time_weighted_mean_hours": (
            "sum of episode hours",
            "Sum of squared episode hours divided by total episode hours including open tails",
            "position lifecycle events",
            "duration-weighted mean",
        ),
        "hsl_panic_episode_loss_max_fraction": (
            "raw account balance before episode's first HSL fill",
            "HSL episode loss-only net realized cashflow fraction including signed fees once",
            "actual HSL execution episodes",
            "maximum",
        ),
        "hsl_panic_episode_loss_mean_fraction": (
            "raw account balance before episode's first HSL fill",
            "HSL episode loss-only net realized cashflow fraction including zero-loss executed episodes",
            "actual HSL execution episodes",
            "arithmetic mean",
        ),
        "hsl_triggers_per_year": (
            "actual elapsed years of 365.25 days",
            "Actual observed HSL trigger count per year",
            "controller lifecycle events",
            "count per elapsed year",
        ),
        "gross_exposure_rms_ratio_strategy_equity": (
            "strategy equity",
            "Gross marked position notional, long and short added",
            "minute marks including flat periods",
            "RMS",
        ),
        "gross_exposure_max_ratio_strategy_equity": (
            "strategy equity",
            "Gross marked position notional, long and short added",
            "minute marks including flat periods",
            "maximum",
        ),
        "hsl_red_time_fraction": (
            "observed evaluated HSL duration",
            "Union elapsed time with any enabled controller RED",
            "controller lifecycle and actual clocks",
            "union duration quotient",
        ),
        "loss_profit_ratio": (
            "positive net realized event cashflow sum",
            "Negative net realized event cashflow magnitude divided by positive sum, fees included once",
            "actual fill/cashflow events",
            "quotient",
        ),
        "short_net_profit_contribution_fraction": (
            "positive combined net long and short strategy P&L",
            "Net short P&L share including fees and change in marked UPNL; signed losing-side contribution",
            "actual side cashflows and boundary UPNL",
            "quotient",
        ),
    }
    for side in ("long", "short"):
        descriptions[f"position_exposure_utilization_rms_ratio_{side}"] = (
            "contemporaneous nominal position WEL and side TWEL",
            "RMS budget-relative position load using entry-price notional/raw account balance, no excess allowance",
            "minute marks including flat periods",
            "budget-relative RMS",
        )
    result = {}
    signed = {
        "cagr_fraction_strategy_equity",
        "cagr_fraction_strategy_equity_ew",
        "sharpe_ratio_strategy_equity",
        "sortino_ratio_strategy_equity",
        "short_net_profit_contribution_fraction",
    }
    for name, item in CORE_METRIC_CATALOG.items():
        denominator, basis, sampling, aggregation = descriptions[name]
        result[name] = MetricDefinition(
            name,
            item["unit"],
            denominator,
            basis,
            sampling,
            aggregation,
            "signed" if name in signed else "nonnegative",
            "extended real with explicit status",
            weighting="calendar exponential" if name.endswith("_ew") else "uniform",
            curve="strategy_equity" if "strategy_equity" in name else None,
            position_side=next(
                (side for side in ("long", "short") if name.endswith("_" + side)), None
            ),
            notes="Core definition version 1; unavailable and genuine infinite endpoints retain explicit status",
        )
    return result


METRIC_DEFINITIONS: Mapping[str, MetricDefinition] = MappingProxyType(_definitions())


def get_metric_definition(name: str) -> MetricDefinition:
    """Describe one exact owned name, without selector or compatibility rewriting."""
    return METRIC_DEFINITIONS[name]


def describe_metric(name: str) -> dict:
    return asdict(get_metric_definition(name))
