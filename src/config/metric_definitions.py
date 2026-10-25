"""Descriptive quantities for owned analysis fields; never supplies trading formulas.

Objective direction, accepted selectors, reductions and backend support remain with
their existing owners. This registry neither resolves aliases nor converts inputs.
"""

from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Mapping

from .metrics import ANALYSIS_SHARED_KEYS, CURRENCY_METRICS


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


def _definitions() -> dict[str, MetricDefinition]:
    result: dict[str, MetricDefinition] = {}

    def add(names, *, unit, denominator=None, basis, sampling, aggregation,
            sign="nonnegative", domain="[0, infinity)", weighting="uniform",
            curve=None, position_side=None, notes=""):
        for name in names.split():
            if name in result:
                raise ValueError(f"duplicate metric definition: {name}")
            result[name] = MetricDefinition(
                name, unit, denominator, basis, sampling, aggregation,
                sign, domain, weighting, curve, position_side, notes,
            )

    add("gain", unit="multiplier", denominator="first daily end equity",
        basis="Mean of the last up to three daily end-equity values divided by the first.",
        sampling="daily end equity", aggregation="smoothed terminal quotient",
        sign="signed", domain="real",
        notes="Native nonpositive-terminal and insufficient-history sentinels are retained.")
    add("adg", unit="fraction", denominator="first daily end equity",
        basis="Smoothed terminal gain raised to one over the number of daily samples, minus one.",
        sampling="daily end equity", aggregation="geometric daily growth",
        sign="signed", domain="real")
    add("mdg", unit="fraction", denominator="absolute preceding daily end equity, floored at 1e-12",
        basis="Simple returns between adjacent daily end-equity values.",
        sampling="daily end equity", aggregation="median", sign="signed", domain="real")
    add("sharpe_ratio", unit="ratio", denominator="RMS daily minimum-equity return deviation from geometric ADG",
        basis="Geometric ADG divided by daily minimum-equity return dispersion.",
        sampling="daily end equity for ADG; daily minimum equity for dispersion",
        aggregation="quotient", sign="signed", domain="real",
        notes="Daily and unannualized; a zero denominator returns zero.")
    add("sortino_ratio", unit="ratio", denominator="RMS of negative daily minimum-equity returns",
        basis="Geometric ADG divided by downside deviation of negative returns only.",
        sampling="daily end equity for ADG; daily minimum equity for downside",
        aggregation="quotient", sign="signed", domain="real",
        notes="Daily and unannualized; a zero denominator returns zero.")
    add("omega_ratio", unit="ratio", denominator="absolute sum of negative daily end-equity returns",
        basis="Sum of positive daily returns divided by the absolute sum of negative returns.",
        sampling="daily end equity", aggregation="quotient", domain="[0, 1000]",
        notes="Capped at 1000. Negligible loss with positive gain returns 1000; negligible gain and loss returns zero.")
    add("expected_shortfall_1pct", unit="fraction", denominator="absolute preceding daily minimum equity, floored at 1e-12",
        basis="Mean absolute magnitude of the lowest one percent of daily minimum-equity returns.",
        sampling="daily minimum equity", aggregation="worst one percent mean",
        notes="At least one return is selected; positive-only histories still use absolute magnitudes.")
    add("drawdown_worst", unit="fraction", denominator="running peak equity",
        basis="Absolute decline from the running equity peak.",
        sampling="every retained equity observation", aggregation="maximum",
        notes="Includes no-fill observations; values may exceed one for negative equity.")
    add("drawdown_worst_mean_1pct", unit="fraction", denominator="running peak equity",
        basis="Daily worst drawdown reduced from the complete retained equity curve.",
        sampling="worst observation per UTC day", aggregation="worst one percent mean",
        notes="A tail mean, not the percentile at its cutoff; at least one day is selected.")
    add("calmar_ratio", unit="ratio", denominator="maximum full-resolution drawdown fraction, floored at 1e-12",
        basis="Geometric ADG divided by maximum drawdown.",
        sampling="daily end equity and full-resolution drawdown", aggregation="quotient",
        sign="signed", domain="real")
    add("sterling_ratio", unit="ratio", denominator="mean of worst one percent daily worst drawdown fractions, floored at 1e-12",
        basis="Geometric ADG divided by the daily-worst drawdown tail mean.",
        sampling="daily end equity and daily worst drawdown", aggregation="quotient",
        sign="signed", domain="real")

    for polarity, sign in (("neg", "negative"), ("pos", "positive")):
        for stat in ("max", "mean"):
            add(f"equity_balance_diff_{polarity}_{stat}", unit="fraction", denominator="balance at the observation",
                basis=f"Magnitude of the {sign} normalized gap (equity - balance) / balance; opposite signs are excluded.",
                sampling="every retained equity observation", aggregation=stat,
                notes="An empty matching-sign population returns zero.")
    for name, denominator in (
        ("paper_loss_ratio", "maximum absolute equity-below-balance fraction"),
        ("paper_loss_mean_ratio", "mean absolute equity-below-balance fraction"),
        ("exposure_ratio", "maximum recorded absolute net wallet exposure ratio"),
        ("exposure_mean_ratio", "mean recorded absolute net wallet exposure ratio"),
    ):
        add(name, unit="ratio", denominator=denominator,
            basis="Geometric ADG divided by the named risk denominator, floored at 1e-12.",
            sampling="daily end equity and retained risk observations", aggregation="quotient",
            sign="signed", domain="real")
    add("equity_choppiness", unit="ratio", denominator="absolute terminal minus initial daily end equity",
        basis="Sum of absolute adjacent equity differences divided by absolute net equity change.",
        sampling="daily end equity", aggregation="total variation quotient",
        notes="A near-zero net change returns infinity; fewer than two samples returns zero.")
    add("equity_jerkiness", unit="ratio", denominator="absolute mean of each three-observation equity window",
        basis="Absolute second equity differences normalized within each window.",
        sampling="three adjacent daily end-equity observations", aggregation="mean of normalized second differences",
        notes="Near-zero window denominators contribute zero; fewer than three samples returns zero.")
    add("exponential_fit_error", unit="squared_log_ratio",
        basis="Squared residual of log equity from its least-squares linear time fit.",
        sampling="daily end equity", aggregation="mean squared residual",
        notes="Fewer than two samples, a nonpositive equity value, or a degenerate fit returns infinity.")

    for unit in ("hours", "days"):
        add(f"peak_recovery_{unit}_equity", unit=unit,
            basis="Longest equity peak-to-recovery interval, including the unrecovered final tail.",
            sampling="every retained equity observation", aggregation="maximum",
            notes="An equal peak resets the account-equity recovery clock.")
        add(f"peak_recovery_{unit}_pnl", unit=unit,
            basis="Longest cumulative realized-PnL peak-to-recovery interval, including the final tail.",
            sampling="fill timestamps and analysis end", aggregation="maximum", curve="realized_pnl")

    for metric in tuple(CURRENCY_METRICS):
        if "_per_exposure_" in metric or metric.endswith("_ew"):
            continue
        if metric not in result:
            raise ValueError(f"missing currency quantity definition: {metric}")
    for metric in ("adg", "mdg", "gain"):
        numerator = result[metric]
        for side in ("long", "short"):
            add(f"{metric}_per_exposure_{side}", unit="ratio", denominator="authored position-side total wallet exposure limit",
                basis=f"{numerator.basis} The resulting {numerator.unit} is divided by authored TWEL.",
                sampling=numerator.sampling, aggregation="quotient", sign=numerator.sign,
                domain=numerator.domain, position_side=side,
                notes="TWEL at or below zero returns zero; an unavailable numerator remains None. This is not measured-exposure normalization.")

    ew_bases = ("adg", "sharpe_ratio", "sortino_ratio", "omega_ratio")
    for base in ew_bases:
        basis, denominator, aggregation = {
            "adg": ("Geometric mean of complete-day simple returns using calendar-age weights.",
                    "preceding complete UTC daily end equity", "weighted geometric growth"),
            "sharpe_ratio": ("Weighted arithmetic daily return mean divided by weighted population standard deviation.",
                             "weighted population daily return standard deviation", "quotient"),
            "sortino_ratio": ("Weighted arithmetic daily return mean divided by downside RMS, including nonnegative returns as zeros.",
                              "weighted RMS of negative daily returns and zeros", "quotient"),
            "omega_ratio": ("Weighted positive daily return sum divided by weighted negative return magnitude sum.",
                            "weighted negative return magnitude sum", "quotient"),
        }[base]
        add(f"{base}_ew", unit="fraction" if base == "adg" else "ratio",
            denominator=denominator, basis=basis, sampling="complete UTC daily end-equity returns",
            aggregation=aggregation, sign="nonnegative" if base == "omega_ratio" else "signed",
            domain="[0, 1000]" if base == "omega_ratio" else "[-1000, 1000]" if base != "adg" else "real",
            weighting="exponential calendar age with configured half-life",
            notes="Not the ordinary estimator with weights added. Availability and denominator floors belong to config.ew and the native analysis owner.")

    currency_bases = dict(result)
    for base in CURRENCY_METRICS:
        definition = currency_bases[base]
        for currency in ("usd", "btc"):
            name = f"{base}_{currency}"
            result[name] = replace(definition, name=name, curve=currency)

    for base in ("gain", "adg", "mdg", "sharpe_ratio", "sortino_ratio", "omega_ratio",
                 "expected_shortfall_1pct", "calmar_ratio", "sterling_ratio",
                 "drawdown_worst", "drawdown_worst_mean_1pct"):
        name = f"{base}_strategy_eq"
        result[name] = replace(result[base], name=name, curve="strategy_equity")
    for base in ew_bases:
        name = f"{base}_strategy_eq_ew"
        result[name] = replace(result[f"{base}_ew"], name=name, curve="strategy_equity")
    add("adg_rolling_hmean_strategy_eq", unit="fraction", denominator="strategy equity at each rolling-window start",
        basis="Harmonic mean growth factors within each automatic horizon; equal-horizon geometric daily growth.",
        sampling="daily strategy end equity and complete rolling windows", aggregation="multi-horizon harmonic/geometric growth",
        sign="signed", domain="real", curve="strategy_equity")
    add("adg_time_integrated_strategy_eq", unit="fraction", denominator="initial strategy equity",
        basis="Dailyized trapezoidal integral of log equity relative to its initial value.",
        sampling="daily strategy end equity", aggregation="log-area daily growth",
        sign="signed", domain="real", curve="strategy_equity")
    add("positive_gain_participation_strategy_eq", unit="score", denominator="interval count times sum of squared positive log returns",
        basis="Squared sum of positive log returns divided by the normalization denominator.",
        sampling="daily strategy end-equity return intervals", aggregation="effective participation score",
        domain="[0, 1]", curve="strategy_equity",
        notes="Concentration-sensitive score; it is not simply the fraction of profitable days.")
    for name, aggregation in (("pain_index_strategy_eq", "mean drawdown"),
                              ("ulcer_index_strategy_eq", "root mean square drawdown")):
        add(name, unit="fraction", denominator="absolute running peak strategy equity",
            basis="Decline from running strategy-equity peak, including zero at peaks.",
            sampling="every retained strategy-equity observation", aggregation=aggregation,
            curve="strategy_equity", notes="No daily-max reduction or plot downsampling; negative equity may produce values above one.")

    for stat in ("mean", "median", "p95", "p99", "mean_worst_5pct", "mean_worst_1pct", "max"):
        add(f"strategy_eq_recovery_days_{stat}", unit="days",
            basis="For every observation, elapsed time to the first strictly higher future value, or the analysis end.",
            sampling="every retained strategy-equity observation", aggregation=stat,
            curve="strategy_equity", notes="Includes unrecovered tails and all observation starts, not just running-peak episodes.")
    for unit in ("hours", "days"):
        name = f"peak_recovery_{unit}_strategy_eq"
        result[name] = replace(result["strategy_eq_recovery_days_max"], name=name, unit=unit,
                               notes="Same maximum first-higher recovery duration, converted to the named time unit.")
    for name, aggregation in (("drawdown_worst_ema_strategy_eq", "maximum"),
                              ("drawdown_worst_mean_1pct_ema_strategy_eq", "worst one percent mean")):
        add(name, unit="fraction", denominator="positive equity plus remembered decaying drawdown",
            basis="Observed normalized remembered-drawdown EMA of actual enabled HSL controllers.",
            sampling="retained HSL signal observations", aggregation=aggregation, curve="hsl_signal",
            notes="The overall series takes the maximum enabled-controller signal at each observation. It is not ordinary strategy-equity drawdown smoothing.")
    for base in ("drawdown_worst_strategy_eq", "drawdown_worst_mean_1pct_strategy_eq",
                 "drawdown_worst_ema_strategy_eq", "drawdown_worst_mean_1pct_ema_strategy_eq",
                 "peak_recovery_hours_strategy_eq", "peak_recovery_days_strategy_eq"):
        for side in ("long", "short"):
            name = f"{base}_{side}"
            result[name] = replace(result[base], name=name, position_side=side)

    for stat in ("mean", "median"):
        add(f"{'adg' if stat == 'mean' else 'mdg'}_pnl", unit="fraction", denominator="absolute last recorded USD balance on that fill day, floored at 1e-12",
            basis="Daily sum of realized fill PnL plus signed fees divided by the daily balance denominator.",
            sampling="UTC days containing fills", aggregation=stat, sign="signed", domain="real",
            curve="realized_pnl", notes="Days with no fills are absent; nonfinite daily sums or balances are skipped by the owner.")
    for base in ("sharpe_ratio", "sortino_ratio"):
        add(f"{base}_pnl", unit="ratio", denominator="daily realized-PnL ratio dispersion" if base == "sharpe_ratio" else "negative daily realized-PnL ratio downside deviation",
            basis="Arithmetic mean of net daily realized-PnL ratios divided by the named dispersion.",
            sampling="UTC days containing fills", aggregation="quotient", sign="signed", domain="real",
            curve="realized_pnl", notes="Daily and unannualized; uses the realized-PnL estimator, not geometric equity ADG.")

    add("liquidated", unit="boolean", basis="Native replay stopped at the configured equity floor.",
        sampling="replay lifecycle", aggregation="flag", domain="false or true")
    add("backtest_completion_ratio", unit="fraction", denominator="requested replay duration",
        basis="Time from requested start to the final covered candle end, bounded by the requested end.",
        sampling="actual equity clocks and configured replay boundaries", aggregation="duration quotient",
        domain="[0, 1]", notes="The first equity observation does not subtract leading warmup. Empty or unavailable history can be absent or zero according to the producing path; with equity observations, a nonpositive requested interval returns one.")
    add("n_days fills_analysis_duration_days", unit="days",
        basis="Elapsed time between first and last actual equity-analysis timestamps.",
        sampling="equity-analysis boundaries", aggregation="elapsed duration",
        notes="Empty or single-observation histories have zero duration; the owner retains its sample-index fallback when clocks are absent or degenerate. Both current output spellings describe the same quantity.")
    add("positions_held_per_day", unit="per_day", denominator="analysis duration in days",
        basis="Count of observed per-coin, per-position-side holding episodes, including open tails.",
        sampling="holding episodes", aggregation="count per elapsed day")
    for unit in ("hours", "days"):
        for stat in ("mean", "median", "max"):
            add(f"position_held_{unit}_{stat}", unit=unit,
                basis="Opening-to-full-close duration per coin and position side; open tails end at analysis end.",
                sampling="holding episodes", aggregation=stat)
        add(f"position_unchanged_{unit}_max", unit=unit,
            basis="Elapsed intervals between fills on an open position, including its final open tail.",
            sampling="per-coin, per-position-side fill gaps", aggregation="maximum")
    add("position_held_time_weighted_mean_hours", unit="hours", denominator="sum of holding-episode hours",
        basis="Sum of squared episode hours divided by sum of episode hours; includes open tails.",
        sampling="per-coin, per-position-side holding episodes", aggregation="duration-weighted mean",
        weighting="holding duration", notes="Neither position size nor exposure weights the episodes.")
    add("win_rate", unit="fraction", denominator="fully closed position episodes",
        basis="Fully closed episodes with summed realized fill PnL strictly above zero.",
        sampling="fully closed per-coin, per-position-side episodes", aggregation="winning episode share",
        domain="[0, 1]", notes="The trade accumulator uses fill.pnl without fees; open episodes are excluded.")
    for stat in ("max", "mean", "median"):
        add(f"trade_loss_{stat}", unit="fraction", denominator="balance recorded at the episode opening fill",
            basis="Absolute nonpositive summed realized fill PnL of a fully closed position episode.",
            sampling="fully closed losing or zero-PnL episodes", aggregation=stat,
            notes="Fees are excluded; open episodes are excluded. Nonpositive opening balance contributes zero.")
    for stat in ("mean", "median", "p95", "p99", "max"):
        add(f"entry_interval_hours_{stat}", unit="hours",
            basis="Time between successive normal initial-entry fills of the same coin and position side.",
            sampling="normal initial-entry fill intervals", aggregation=stat,
            notes="This population follows the owner's normal-initial-entry order types, not every entry fill.")

    add("fills_count fills_count_entry fills_count_close fills_count_long fills_count_short fills_active_symbols_count",
        unit="count", basis="Observed fill population or its distinct active dates/symbols.",
        sampling="recorded fills", aggregation="count")
    add("fills_active_days_count", unit="count",
        basis="Distinct 24-hour buckets containing in-window fills, anchored to the analysis start.",
        sampling="24-hour buckets from analysis start", aggregation="count")
    add("fills_active_days_ratio", unit="fraction", denominator="ceiling of elapsed analysis days, floored at one",
        basis="Active 24-hour buckets anchored to the analysis start divided by the duration denominator.",
        sampling="in-window fills and analysis boundaries", aggregation="active bucket share",
        notes="These are not UTC calendar days. Inclusive terminal-boundary fills can make this fraction exceed one.")
    add("fills_top_symbol_share", unit="fraction", denominator="all recorded fills",
        basis="Fill count of the most active symbol divided by total fill count.",
        sampling="recorded fills", aggregation="maximum symbol share", domain="[0, 1]")
    add("fills_entry_per_close", unit="ratio", denominator="close fill count, floored at one",
        basis="Entry fill count divided by close fill count.",
        sampling="recorded fills", aggregation="count quotient")
    for name in ("fills_per_day", "fills_per_day_entry", "fills_per_day_close", "fills_per_day_long", "fills_per_day_short"):
        add(name, unit="per_day", denominator="analysis duration in days",
            basis="The named fill count divided by elapsed analysis days.",
            sampling="fills and complete analysis boundaries", aggregation="count per elapsed day")
    for suffix in ("", "_long", "_short"):
        add(f"fills_per_day_per_position_slot{suffix}", unit="per_day_per_position_slot",
            denominator="analysis days times configured position-side slot count",
            basis="Side fill frequency per configured position slot; overall is the mean of enabled-side rates.",
            sampling="fills and configured side capacity", aggregation="normalized frequency")
    add("fills_gap_longest_days", unit="days",
        basis="No-fill gap across the portfolio, including analysis leading and trailing boundaries.",
        sampling="in-window fill timestamps and analysis boundaries", aggregation="maximum")
    for stat in ("mean", "median", "p95", "p99", "time_weighted_mean"):
        add(f"fills_gap_{stat}_hours", unit="hours", denominator="sum of gap hours" if stat == "time_weighted_mean" else None,
            basis="Portfolio no-fill gaps, including leading and trailing analysis boundaries.",
            sampling=("unique in-window fill timestamps and analysis boundaries" if stat == "time_weighted_mean"
                      else "all in-window fill timestamps and analysis boundaries, including duplicate timestamps"), aggregation=stat,
            weighting="gap duration" if stat == "time_weighted_mean" else "uniform",
            notes="A zero-fill run has one gap equal to the full analysis duration.")
    add("volume_pct_per_day_avg", unit="ratio", denominator="signed balance recorded at each fill, without a floor",
        basis="Daily sum of abs(fill_qty) * fill_price / fill.usd_total_balance.",
        sampling="UTC days containing fills", aggregation="mean of active-day turnover",
        sign="signed", domain="real or nonfinite",
        notes="Scale one, not percent. The helper does not apply c_mult; negative balance can yield a negative result and zero balance can be nonfinite. Days without fills are absent.")

    for suffix in ("", "_long", "_short"):
        add(f"loss_profit_ratio{suffix}", unit="ratio", denominator="sum of positive realized fill PnL",
            basis="Absolute sum of negative realized fill PnL divided by summed positive PnL.",
            sampling="fills", aggregation="loss/profit quotient", domain="[0, 1000]",
            notes="Fees are excluded. Capped at 1000; negligible profit with positive loss returns 1000, negligible profit and loss returns one. Public CPU/GPU no-trade defaults are one; the internal starting-state analyzer can return zero for no fills.")
    add("pnl_ratio_long_short long_short_profit_ratio", unit="ratio", denominator="total realized long plus short fill PnL",
        basis="Summed long fill PnL divided by total summed long and short fill PnL.",
        sampling="fills", aggregation="signed PnL quotient", sign="signed", domain="real",
        notes="May lie outside [0,1]; zero total PnL returns 0.5. Fees are excluded. Both current output spellings are identical.")
    for stat in ("max", "mean", "median"):
        add(f"total_wallet_exposure_{stat}", unit="ratio", denominator="account balance",
            basis="Absolute recorded net long-minus-short wallet exposure.",
            sampling="retained exposure observations; fill observations or starting exposure when absent",
            aggregation=stat)
    for side in ("long", "short", "gross"):
        add(f"total_wallet_exposure_rms_{side}", unit="ratio", denominator="raw account balance at each observation",
            basis="Absolute side exposure; gross sums long and short before squaring.",
            sampling="every retained active candle, including flat capacity", aggregation="root mean square",
            position_side=None if side == "gross" else side,
            notes="Empty history is undefined; gross hedges do not cancel.")
    for side in ("long", "short"):
        add(f"total_wallet_exposure_utilization_rms_{side}", unit="ratio", denominator="current nominal side TWEL at each observation",
            basis="Side total wallet exposure divided by nominal side TWEL, excluding excess allowance.",
            sampling="every retained active candle, including flat capacity", aggregation="root mean square",
            position_side=side, notes="Positive exposure with invalid budget is undefined, not silently zero.")
        add(f"position_exposure_utilization_rms_{side}", unit="ratio", denominator="current position WEL and current nominal side TWEL",
            basis="Square root of mean summed e * (e / WEL) / TWEL over positions.",
            sampling="every retained active candle, including flat capacity", aggregation="budget-weighted root mean square",
            position_side=side, notes="Budgets exclude excess allowance; invalid budgets with positive exposure are undefined.")
        add(f"entry_initial_balance_pct_{side}", unit="fraction", denominator="raw balance",
            basis="Nominal side TWEL / effective positions times initial quantity fraction and excess-allowance multiplier.",
            sampling="terminal effective configured capacity", aggregation="derived sizing diagnostic",
            position_side=side, notes="Not a mean or a measurement of actual initial entry fills.")
        for unit in ("hours", "days"):
            for stat in ("mean", "max"):
                add(f"high_exposure_{unit}_{stat}_{side}", unit=unit,
                    basis="Intervals where fill-sampled side exposure exceeds its calendar-day-averaged reference level.",
                    sampling="fill observations", aggregation=stat, position_side=side,
                    notes="Reference daily means include empty days between first and last fill; terminal interval ends at the final fill.")

    for name in ("hard_stop_triggers", "hard_stop_restarts"):
        add(name, unit="count", basis="Observed HSL lifecycle events.",
            sampling="HSL lifecycle", aggregation="count")
        for side in ("long", "short"):
            side_name = f"{name}_{side}"
            result[side_name] = replace(result[name], name=side_name, position_side=side)
    for name in ("hard_stop_triggers_per_year", "hard_stop_restarts_per_year",
                 "hard_stop_restarts_per_year_long", "hard_stop_restarts_per_year_short"):
        add(name, unit="per_year", denominator="observed replay minutes / (365.25 * 1440)",
            basis="Observed HSL lifecycle count annualized over replay duration.",
            sampling="HSL lifecycle and replay boundaries", aggregation="count per elapsed year")
    add("hard_stop_halt_to_restart_equity_loss_pct", unit="fraction", denominator="starting balance, floored at floating epsilon",
        basis="Total loss-only net panic-fill PnL divided by starting balance.",
        sampling="HSL panic fills", aggregation="total loss fraction",
        notes="Not halt-to-restart mark-to-market equity change; profitable panic fills do not offset losses.")
    add("hard_stop_time_in_red_pct", unit="fraction", denominator="observed HSL minutes",
        basis="Minutes with any scope red divided by observed minutes.",
        sampling="HSL observed time", aggregation="time share", domain="[0, 1]")
    for stat in ("mean", "max"):
        add(f"hard_stop_duration_minutes_{stat}", unit="minutes",
            basis="HSL RED episode duration.", sampling="HSL lifecycle episodes", aggregation=stat)
    add("hard_stop_trigger_drawdown_mean", unit="fraction", denominator="positive equity plus remembered decaying drawdown",
        basis="Minimum of raw normalized remembered drawdown and its EMA at each observed RED trigger.",
        sampling="HSL trigger events", aggregation="mean")
    for stat in ("sum", "max"):
        add(f"hard_stop_panic_close_loss_{stat}", unit="usd",
            basis="Loss-only magnitude of realized panic-fill PnL including signed fees.",
            sampling="individual HSL panic fills", aggregation=stat,
            notes="The maximum is an individual fill, not an entire episode.")
    for stat in ("min", "mean", "max"):
        add(f"hard_stop_panic_close_loss_drawdown_pct_{stat}", unit="fraction", denominator="account equity recorded at the episode's first panic fill, floored at floating epsilon",
            basis="Episode loss-only net panic-fill loss divided by first-panic-fill account equity.",
            sampling="HSL panic execution episodes", aggregation=stat,
            notes="An unfinished observed episode contributes its current loss on report finalization.")
    add("hard_stop_flatten_time_minutes_mean", unit="minutes",
        basis="Elapsed panic-exit-start to observed flat time.",
        sampling="HSL flattening episodes", aggregation="mean")
    add("hard_stop_post_restart_retrigger_pct", unit="fraction", denominator="observed restart count",
        basis="Observed post-restart retriggers divided by restarts.",
        sampling="HSL lifecycle", aggregation="retrigger share",
        notes="Each restart can contribute its first subsequent trigger; there is no fixed elapsed-time window. No restarts returns zero.")

    for name, definition in tuple(result.items()):
        if definition.position_side is None:
            for side in ("long", "short"):
                if name.endswith(f"_{side}"):
                    result[name] = replace(definition, position_side=side)
                    break

    missing = ANALYSIS_SHARED_KEYS - result.keys()
    if missing:
        raise ValueError(f"missing shared metric definitions: {sorted(missing)}")
    return result


METRIC_DEFINITIONS: Mapping[str, MetricDefinition] = MappingProxyType(_definitions())


def get_metric_definition(name: str) -> MetricDefinition:
    """Describe one exact owned name, without selector or compatibility rewriting."""
    return METRIC_DEFINITIONS[name]


def describe_metric(name: str) -> dict:
    return asdict(get_metric_definition(name))
