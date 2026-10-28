//! Versioned, bounded backtest arithmetic. This module never changes trading intent.
//!
//! All equity observations use causal economic boundary timestamps. Strategy P&L
//! excludes collateral appreciation and collateral top-up conversion costs: those
//! are collateral operations, not derivatives trading cashflows. Actual account
//! balances still include their effects. Fills use position side, never buy/sell.
//! The owning HSL reporter supplies actual execution-episode aggregates and the
//! union of reported RED time, including flat restart cooldown. No controller or
//! historical stop is reconstructed here.

use serde::Serialize;
use std::collections::BTreeMap;

pub const CORE_METRIC_DEFINITION_VERSION: u32 = 1;
pub const YEAR_DAYS: f64 = 365.25;
pub const DEFAULT_EW_HALF_LIFE_DAYS: f64 = 180.0;
pub const DAY_MS: u64 = 86_400_000;
pub const MINUTE_MS: u64 = 60_000;
const HOUR_MS: f64 = 3_600_000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Metric {
    Cagr,
    Volatility,
    Sharpe,
    Sortino,
    Omega,
    DrawdownMax,
    Ulcer,
    Pain,
    CagrEw,
    Completion,
    FillGapMax,
    HoldingMax,
    PeakRecoveryMax,
    Turnover,
    HoldingTimeWeightedMean,
    PositionUtilizationLong,
    PositionUtilizationShort,
    HslEpisodeLossMax,
    HslEpisodeLossMean,
    HslTriggersPerYear,
    GrossExposureRms,
    GrossExposureMax,
    HslRedTime,
    LossProfit,
    ShortContribution,
    HslExecutionEpisodeCount,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricRole {
    Maximize,
    Minimize,
    Limit,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct MetricDefinition {
    #[serde(skip)]
    pub metric: Metric,
    pub name: &'static str,
    pub unit: &'static str,
    pub role: MetricRole,
    pub requires_minute_observations: bool,
}

macro_rules! definition {
    ($metric:ident, $name:literal, $unit:literal, $role:ident) => {
        MetricDefinition {
            metric: Metric::$metric,
            name: $name,
            unit: $unit,
            role: MetricRole::$role,
            requires_minute_observations: Metric::$metric.requires_minute_observations(),
        }
    };
}

/// Twenty-five quantities, with two explicit position-utilization side scopes.
/// Roles are recommendations; every selector supports either limit direction.
pub const METRIC_CATALOG: [MetricDefinition; 26] = [
    definition!(Cagr, "cagr_fraction_strategy_equity", "fraction", Maximize),
    definition!(
        Volatility,
        "volatility_annual_fraction_strategy_equity",
        "fraction",
        Limit
    ),
    definition!(Sharpe, "sharpe_ratio_strategy_equity", "ratio", Maximize),
    definition!(Sortino, "sortino_ratio_strategy_equity", "ratio", Maximize),
    definition!(Omega, "omega_ratio_strategy_equity", "ratio", Maximize),
    definition!(
        DrawdownMax,
        "drawdown_max_fraction_strategy_equity",
        "fraction",
        Limit
    ),
    definition!(
        Ulcer,
        "ulcer_index_fraction_strategy_equity",
        "fraction",
        Minimize
    ),
    definition!(
        Pain,
        "pain_index_fraction_strategy_equity",
        "fraction",
        Minimize
    ),
    definition!(
        CagrEw,
        "cagr_fraction_strategy_equity_ew",
        "fraction",
        Maximize
    ),
    definition!(
        Completion,
        "backtest_completion_fraction",
        "fraction",
        Limit
    ),
    definition!(FillGapMax, "fills_gap_max_days", "days", Limit),
    definition!(HoldingMax, "position_held_max_days", "days", Limit),
    definition!(
        PeakRecoveryMax,
        "peak_recovery_max_days_strategy_equity",
        "days",
        Limit
    ),
    definition!(
        Turnover,
        "gross_turnover_ratio_per_day",
        "per_day",
        Maximize
    ),
    definition!(
        HoldingTimeWeightedMean,
        "position_held_time_weighted_mean_hours",
        "hours",
        Minimize
    ),
    definition!(
        PositionUtilizationLong,
        "position_exposure_utilization_rms_ratio_long",
        "ratio",
        Minimize
    ),
    definition!(
        PositionUtilizationShort,
        "position_exposure_utilization_rms_ratio_short",
        "ratio",
        Minimize
    ),
    definition!(
        HslEpisodeLossMax,
        "hsl_panic_episode_loss_max_fraction",
        "fraction",
        Limit
    ),
    definition!(
        HslEpisodeLossMean,
        "hsl_panic_episode_loss_mean_fraction",
        "fraction",
        Limit
    ),
    definition!(
        HslTriggersPerYear,
        "hsl_triggers_per_year",
        "per_year",
        Limit
    ),
    definition!(
        GrossExposureRms,
        "gross_exposure_rms_ratio_strategy_equity",
        "ratio",
        Minimize
    ),
    definition!(
        GrossExposureMax,
        "gross_exposure_max_ratio_strategy_equity",
        "ratio",
        Limit
    ),
    definition!(HslRedTime, "hsl_red_time_fraction", "fraction", Limit),
    definition!(LossProfit, "loss_profit_ratio", "ratio", Minimize),
    definition!(
        ShortContribution,
        "short_net_profit_contribution_fraction",
        "fraction",
        Limit
    ),
    definition!(
        HslExecutionEpisodeCount,
        "hsl_execution_episode_count",
        "count",
        Limit
    ),
];

impl Metric {
    pub const fn requires_minute_observations(self) -> bool {
        matches!(
            self,
            Self::DrawdownMax
                | Self::Ulcer
                | Self::Pain
                | Self::PeakRecoveryMax
                | Self::CagrEw
                | Self::PositionUtilizationLong
                | Self::PositionUtilizationShort
                | Self::GrossExposureRms
                | Self::GrossExposureMax
        )
    }
    pub fn definition(self) -> &'static MetricDefinition {
        &METRIC_CATALOG[self as usize]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetricRequest(u32);
impl MetricRequest {
    pub fn all() -> Self {
        Self((1 << METRIC_CATALOG.len()) - 1)
    }
    pub fn from_metrics(metrics: impl IntoIterator<Item = Metric>) -> Self {
        let mut request = Self::default();
        for metric in metrics {
            request.0 |= 1 << metric as u8;
        }
        request
    }
    pub fn from_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<Self, String> {
        let mut request = Self::default();
        for name in names {
            let entry = METRIC_CATALOG
                .iter()
                .find(|entry| entry.name == name)
                .ok_or_else(|| format!("unknown core metric '{name}'"))?;
            request.0 |= 1 << entry.metric as u8;
        }
        Ok(request)
    }
    pub fn contains(self, metric: Metric) -> bool {
        self.0 & (1 << metric as u8) != 0
    }
    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    fn any(self, metrics: &[Metric]) -> bool {
        metrics.iter().any(|&m| self.contains(m))
    }
    pub fn needs_daily_returns(self) -> bool {
        self.any(&[
            Metric::Volatility,
            Metric::Sharpe,
            Metric::Sortino,
            Metric::Omega,
        ])
    }
    pub fn needs_exposure(self) -> bool {
        self.any(&[
            Metric::PositionUtilizationLong,
            Metric::PositionUtilizationShort,
            Metric::GrossExposureRms,
            Metric::GrossExposureMax,
        ])
    }
    pub fn needs_holding(self) -> bool {
        self.any(&[Metric::HoldingMax, Metric::HoldingTimeWeightedMean])
    }
    pub fn needs_hsl(self) -> bool {
        self.any(&[
            Metric::HslEpisodeLossMax,
            Metric::HslEpisodeLossMean,
            Metric::HslTriggersPerYear,
            Metric::HslRedTime,
            Metric::HslExecutionEpisodeCount,
        ])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricStatus {
    Finite,
    PositiveInfinity,
    NegativeInfinity,
    Undefined,
    Invalid,
}

/// Standard-JSON-safe representation. Only finite status has a numeric value.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct MetricValue {
    pub status: MetricStatus,
    pub value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
}
impl MetricValue {
    pub fn finite(value: f64) -> Self {
        if value.is_finite() {
            Self {
                status: MetricStatus::Finite,
                value: Some(value),
                reason: None,
            }
        } else {
            Self::invalid("numeric_overflow")
        }
    }
    pub fn invalid(reason: &'static str) -> Self {
        Self {
            status: MetricStatus::Invalid,
            value: None,
            reason: Some(reason),
        }
    }
    pub fn undefined(reason: &'static str) -> Self {
        Self {
            status: MetricStatus::Undefined,
            value: None,
            reason: Some(reason),
        }
    }
    fn ratio(numerator: f64, denominator: f64) -> Self {
        if !numerator.is_finite() || !denominator.is_finite() || denominator < 0.0 {
            return Self::invalid("invalid_ratio_input");
        }
        if denominator == 0.0 {
            return if numerator == 0.0 {
                Self::undefined("zero_over_zero")
            } else {
                Self {
                    status: if numerator > 0.0 {
                        MetricStatus::PositiveInfinity
                    } else {
                        MetricStatus::NegativeInfinity
                    },
                    value: None,
                    reason: None,
                }
            };
        }
        Self::finite(numerator / denominator)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EvaluationWindow {
    pub start_timestamp_ms: u64,
    pub intended_end_timestamp_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PositionSide {
    Long,
    Short,
}
impl PositionSide {
    pub fn index(self) -> usize {
        match self {
            Self::Long => 0,
            Self::Short => 1,
        }
    }
    fn valid_size(self, size: f64) -> bool {
        size.is_finite()
            && match self {
                Self::Long => size >= 0.0,
                Self::Short => size <= 0.0,
            }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FillObservation {
    pub timestamp_ms: u64,
    pub coin_index: usize,
    pub position_side: PositionSide,
    pub position_size_before: f64,
    pub position_size_after: f64,
    /// Gross executed notional including the canonical instrument multiplier.
    pub notional_usd: f64,
    pub pre_fill_balance_usd: f64,
    /// Gross price P&L plus signed fee, exactly once. Entry fees are included.
    pub net_realized_pnl_usd: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ExposureObservation {
    pub timestamp_ms: u64,
    pub strategy_equity: f64,
    pub gross_marked_notional_usd: f64,
    /// sum_p((abs(entry_notional_p) / raw_balance)^2 / WEL_p) / TWEL_side.
    /// Flat sides are zero. The owning replay supplies contemporaneous budgets.
    pub position_utilization_squared: [f64; 2],
}

#[derive(Clone, Copy, Debug)]
pub struct HslSummary {
    pub enabled: bool,
    pub trigger_count: u64,
    pub execution_episode_count: u64,
    pub execution_episode_count_valid: bool,
    pub episode_loss_fraction_sum: f64,
    pub episode_loss_fraction_max: f64,
    pub episode_loss_valid: bool,
    pub red_duration_ms: u64,
    pub observed_duration_ms: u64,
    pub observation_valid: bool,
}
impl Default for HslSummary {
    fn default() -> Self {
        Self {
            enabled: false,
            trigger_count: 0,
            execution_episode_count: 0,
            execution_episode_count_valid: true,
            episode_loss_fraction_sum: 0.0,
            episode_loss_fraction_max: 0.0,
            episode_loss_valid: true,
            red_duration_ms: 0,
            observed_duration_ms: 0,
            observation_valid: true,
        }
    }
}

impl HslSummary {
    fn clock_valid(self, elapsed_ms: u64) -> bool {
        if !self.observation_valid || self.red_duration_ms > self.observed_duration_ms {
            return false;
        }
        if self.enabled {
            self.observed_duration_ms == elapsed_ms
        } else {
            (self.observed_duration_ms == 0 || self.observed_duration_ms == elapsed_ms)
                && self.trigger_count == 0
                && self.red_duration_ms == 0
                && self.execution_episode_count == 0
                && self.episode_loss_fraction_sum == 0.0
                && self.episode_loss_fraction_max == 0.0
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Finalization {
    pub actual_end_timestamp_ms: u64,
    /// Each side's net realized trading cashflows plus change in marked UPNL.
    /// Excludes collateral operations. These two amounts must reconcile to the
    /// portfolio strategy-equity change in the owning replay integration.
    pub net_side_profit_usd: [f64; 2],
    pub hsl: HslSummary,
}

#[derive(Clone, Debug, Serialize)]
pub struct MetricsMetadata {
    pub definition_version: u32,
    pub year_days: f64,
    pub observation_interval_ms: u64,
    pub start_timestamp_ms: u64,
    pub intended_end_timestamp_ms: u64,
    pub actual_end_timestamp_ms: u64,
    pub equity_observation_count: u64,
    pub daily_return_count: u64,
    pub daily_first_close_timestamp_ms: Option<u64>,
    pub daily_last_close_timestamp_ms: Option<u64>,
    pub fill_count: u64,
    pub holding_episode_count: u64,
    /// Present only for a selected recovery metric with valid path support.
    pub peak_recovery_tail_censored: Option<bool>,
    pub hsl_enabled: bool,
    pub hsl_trigger_count: u64,
    pub hsl_execution_episode_count: u64,
    pub ew_half_life_days: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CoreMetricsResult {
    pub metrics: BTreeMap<&'static str, MetricValue>,
    pub metadata: MetricsMetadata,
}

/// Backend-neutral completed statistics. A producer must include baseline and
/// unfinished tails once. No observation or fill histories cross this boundary.
#[derive(Clone, Debug, Default)]
pub struct CoreMetricsSummary {
    pub equity: EquitySummary,
    pub daily: DailySummary,
    pub fills: FillSummary,
    pub holding: HoldingSummary,
    pub exposure: ExposureSummary,
}
#[derive(Clone, Debug, Default)]
pub struct EquitySummary {
    pub end_timestamp_ms: u64,
    pub final_strategy_equity: f64,
    pub observation_count: u64,
    pub clock_valid: bool,
    pub all_equity_finite: bool,
    pub all_equity_positive: bool,
    pub drawdown_sum: f64,
    pub drawdown_squares: f64,
    pub drawdown_max: f64,
    pub recovery_max_ms: u64,
    /// A summary carries the producer's tail fact without reconstructing a start.
    pub peak_recovery_tail_censored: Option<bool>,
    pub ew_log_sum: f64,
    pub ew_days_sum: f64,
    pub ew_valid: bool,
}
#[derive(Clone, Debug, Default)]
pub struct DailySummary {
    pub first_close_timestamp_ms: Option<u64>,
    pub last_close_timestamp_ms: Option<u64>,
    pub return_count: u64,
    pub mean: f64,
    pub m2: f64,
    pub downside_squares: f64,
    pub gains: f64,
    pub losses: f64,
    pub valid: bool,
}
#[derive(Clone, Debug, Default)]
pub struct FillSummary {
    pub count: u64,
    pub last_timestamp_ms: u64,
    pub max_gap_ms: u64,
    pub turnover_sum: f64,
    pub realized_profit: f64,
    pub realized_loss: f64,
    pub clock_valid: bool,
    pub turnover_valid: bool,
    pub realized_valid: bool,
}
#[derive(Clone, Debug, Default)]
pub struct HoldingSummary {
    pub count: u64,
    pub sum_hours: f64,
    pub sum_squares: f64,
    pub max_hours: f64,
    pub valid: bool,
}
#[derive(Clone, Debug, Default)]
pub struct ExposureSummary {
    pub first_timestamp_ms: Option<u64>,
    pub last_timestamp_ms: Option<u64>,
    pub observation_count: u64,
    pub gross_squares: f64,
    pub gross_max: f64,
    pub utilization_sums: [f64; 2],
    pub clock_valid: bool,
    pub gross_valid: bool,
    pub utilization_valid: [bool; 2],
}

#[derive(Clone, Debug, Default)]
struct DailyStats {
    first: Option<u64>,
    previous: Option<(u64, f64)>,
    count: u64,
    mean: f64,
    m2: f64,
    downside_squares: f64,
    gains: f64,
    losses: f64,
    invalid: bool,
}
impl DailyStats {
    fn observe(&mut self, timestamp: u64, equity: f64, window: EvaluationWindow) {
        if timestamp % DAY_MS != 0
            || timestamp < window.start_timestamp_ms
            || timestamp > window.intended_end_timestamp_ms
            || !equity.is_finite()
            || equity <= 0.0
        {
            self.invalid = true;
            return;
        }
        if let Some((previous_ts, previous_equity)) = self.previous {
            if timestamp.checked_sub(previous_ts) != Some(DAY_MS) {
                self.invalid = true;
                return;
            }
            let r = equity / previous_equity - 1.0;
            if !r.is_finite() {
                self.invalid = true;
                return;
            }
            self.count += 1;
            let delta = r - self.mean;
            self.mean += delta / self.count as f64;
            self.m2 += delta * (r - self.mean);
            self.downside_squares += r.min(0.0).powi(2);
            self.gains += r.max(0.0);
            self.losses += (-r).max(0.0);
            if ![
                self.mean,
                self.m2,
                self.downside_squares,
                self.gains,
                self.losses,
            ]
            .iter()
            .all(|x| x.is_finite())
                || self.m2 < 0.0
            {
                self.invalid = true;
            }
        } else {
            self.first = Some(timestamp);
        }
        self.previous = Some((timestamp, equity));
    }
    fn metric(&self, metric: Metric, start: u64, end: u64) -> MetricValue {
        if self.invalid || self.previous.is_some_and(|(ts, _)| ts > end) {
            return MetricValue::invalid("invalid_daily_close_population");
        }
        let first_required = (start / DAY_MS + u64::from(start % DAY_MS != 0)).checked_mul(DAY_MS);
        let last_required = end / DAY_MS * DAY_MS;
        if first_required.is_some_and(|first| first <= last_required) {
            if self.first != first_required
                || self.previous.map(|(ts, _)| ts) != Some(last_required)
            {
                return MetricValue::invalid("missing_daily_boundary");
            }
        }
        let minimum = if matches!(metric, Metric::Volatility | Metric::Sharpe) {
            2
        } else {
            1
        };
        if self.count < minimum {
            return MetricValue::undefined("insufficient_daily_returns");
        }
        match metric {
            Metric::Volatility => {
                MetricValue::finite((self.m2 / (self.count - 1) as f64).sqrt() * YEAR_DAYS.sqrt())
            }
            Metric::Sharpe => MetricValue::ratio(
                self.mean * YEAR_DAYS.sqrt(),
                (self.m2 / (self.count - 1) as f64).sqrt(),
            ),
            Metric::Sortino => MetricValue::ratio(
                self.mean * YEAR_DAYS.sqrt(),
                (self.downside_squares / self.count as f64).sqrt(),
            ),
            Metric::Omega => MetricValue::ratio(self.gains, self.losses),
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct HoldingStats {
    open: BTreeMap<(usize, PositionSide), u64>,
    count: u64,
    sum_hours: f64,
    sum_squares: f64,
    max_hours: f64,
    invalid: bool,
}
impl HoldingStats {
    fn complete(&mut self, start: u64, end: u64) {
        let Some(duration) = end.checked_sub(start) else {
            self.invalid = true;
            return;
        };
        let hours = duration as f64 / HOUR_MS;
        self.count += 1;
        self.sum_hours += hours;
        self.sum_squares += hours * hours;
        self.max_hours = self.max_hours.max(hours);
        if !self.sum_squares.is_finite() || !self.sum_hours.is_finite() {
            self.invalid = true;
        }
    }
    fn fill(&mut self, fill: FillObservation) {
        if !fill.position_side.valid_size(fill.position_size_before)
            || !fill.position_side.valid_size(fill.position_size_after)
        {
            self.invalid = true;
            return;
        }
        let key = (fill.coin_index, fill.position_side);
        let before = fill.position_size_before != 0.0;
        let after = fill.position_size_after != 0.0;
        if before != self.open.contains_key(&key) {
            self.invalid = true;
            return;
        }
        if !before && after {
            self.open.insert(key, fill.timestamp_ms);
        }
        if before && !after {
            let start = self.open.remove(&key).unwrap();
            self.complete(start, fill.timestamp_ms);
        }
    }
    fn with_tails(&self, end: u64) -> Self {
        let mut result = self.clone();
        for &start in self.open.values() {
            result.complete(start, end);
        }
        result
    }
}

#[derive(Clone, Debug, Default)]
struct ExposureStats {
    pending: Option<ExposureObservation>,
    first: Option<u64>,
    count: u64,
    gross_squares: f64,
    gross_max: f64,
    utilization_sums: [f64; 2],
    invalid_clock: bool,
    invalid_gross: bool,
    invalid_utilization: [bool; 2],
    terminal: bool,
    last_timestamp_ms: Option<u64>,
}
impl ExposureStats {
    fn commit(&mut self, request: MetricRequest) {
        let Some(observation) = self.pending.take() else {
            return;
        };
        self.count += 1;
        if request.any(&[Metric::GrossExposureRms, Metric::GrossExposureMax]) {
            let e = observation.strategy_equity;
            let n = observation.gross_marked_notional_usd;
            if !e.is_finite() || e <= 0.0 || !n.is_finite() || n < 0.0 {
                self.invalid_gross = true;
            } else {
                let gross = n / e;
                self.gross_squares += gross * gross;
                self.gross_max = self.gross_max.max(gross);
                if !self.gross_squares.is_finite() {
                    self.invalid_gross = true;
                }
            }
        }
        for (side, metric) in [
            Metric::PositionUtilizationLong,
            Metric::PositionUtilizationShort,
        ]
        .into_iter()
        .enumerate()
        {
            if request.contains(metric) {
                let value = observation.position_utilization_squared[side];
                if !value.is_finite() || value < 0.0 {
                    self.invalid_utilization[side] = true;
                } else {
                    self.utilization_sums[side] += value;
                    if !self.utilization_sums[side].is_finite() {
                        self.invalid_utilization[side] = true;
                    }
                }
            }
        }
    }
    fn observe(
        &mut self,
        observation: ExposureObservation,
        request: MetricRequest,
        start: u64,
        intended_end: u64,
    ) {
        if observation.timestamp_ms < start || observation.timestamp_ms > intended_end {
            self.invalid_clock = true;
            return;
        }
        if let Some(previous) = self.pending {
            if observation.timestamp_ms == previous.timestamp_ms {
                self.pending = Some(observation);
                return;
            }
            let Some(delta) = observation.timestamp_ms.checked_sub(previous.timestamp_ms) else {
                self.invalid_clock = true;
                return;
            };
            if delta > MINUTE_MS || self.terminal {
                self.invalid_clock = true;
            }
            self.terminal = delta < MINUTE_MS;
            self.commit(request);
        } else {
            self.first = Some(observation.timestamp_ms);
            if observation.timestamp_ms != start {
                self.invalid_clock = true;
            }
        }
        self.last_timestamp_ms = Some(observation.timestamp_ms);
        self.pending = Some(observation);
    }
}

/// Constant-size state except for one entry per currently open coin/side position.
/// The pending equity/exposure mark permits replacing a same-clock terminal mark
/// without counting it twice. Never feed per-fill snapshots into minute averages.
#[derive(Clone, Debug)]
pub struct CoreMetrics {
    request: MetricRequest,
    window: EvaluationWindow,
    observation_interval_ms: u64,
    initial_equity: f64,
    pending_equity: Option<(u64, f64)>,
    previous_equity: Option<(u64, f64)>,
    last_timestamp_ms: u64,
    count: u64,
    clock_invalid: bool,
    equity_invalid: bool,
    equity_nonpositive: bool,
    path_moments_invalid: bool,
    terminal: bool,
    peak: f64,
    peak_timestamp_ms: u64,
    underwater_start: Option<u64>,
    peak_recovery_tail_censored: Option<bool>,
    recovery_max_ms: u64,
    drawdown_sum: f64,
    drawdown_squares: f64,
    drawdown_max: f64,
    half_life_days: Option<f64>,
    ew_log_sum: f64,
    ew_days_sum: f64,
    ew_invalid: bool,
    daily: DailyStats,
    fill_count: u64,
    last_fill_timestamp_ms: u64,
    fill_gap_max_ms: u64,
    fill_clock_invalid: bool,
    turnover_sum: f64,
    turnover_invalid: bool,
    realized_profit: f64,
    realized_loss: f64,
    realized_invalid: bool,
    holding: HoldingStats,
    exposure: ExposureStats,
}
impl CoreMetrics {
    pub fn new(
        request: MetricRequest,
        window: EvaluationWindow,
        initial_strategy_equity: f64,
        ew_half_life_days: Option<f64>,
    ) -> Result<Self, String> {
        Self::new_with_interval(
            request,
            window,
            initial_strategy_equity,
            ew_half_life_days,
            MINUTE_MS,
        )
    }
    pub fn new_with_interval(
        request: MetricRequest,
        window: EvaluationWindow,
        initial_strategy_equity: f64,
        ew_half_life_days: Option<f64>,
        observation_interval_ms: u64,
    ) -> Result<Self, String> {
        if observation_interval_ms == 0 || observation_interval_ms % MINUTE_MS != 0 {
            return Err("observation interval must be a positive whole number of minutes".into());
        }
        if window.intended_end_timestamp_ms <= window.start_timestamp_ms {
            return Err("evaluation window must have positive duration".into());
        }
        if request.contains(Metric::CagrEw)
            && !ew_half_life_days.is_some_and(|h| h.is_finite() && h > 0.0)
        {
            return Err("CAGR EW requires a finite positive half-life in days".into());
        }
        Ok(Self {
            request,
            window,
            observation_interval_ms,
            initial_equity: initial_strategy_equity,
            pending_equity: Some((window.start_timestamp_ms, initial_strategy_equity)),
            previous_equity: None,
            last_timestamp_ms: window.start_timestamp_ms,
            count: 0,
            clock_invalid: false,
            equity_invalid: !initial_strategy_equity.is_finite(),
            equity_nonpositive: initial_strategy_equity <= 0.0,
            path_moments_invalid: false,
            terminal: false,
            peak: initial_strategy_equity,
            peak_timestamp_ms: window.start_timestamp_ms,
            underwater_start: None,
            peak_recovery_tail_censored: Some(false),
            recovery_max_ms: 0,
            drawdown_sum: 0.0,
            drawdown_squares: 0.0,
            drawdown_max: 0.0,
            half_life_days: ew_half_life_days.filter(|_| request.contains(Metric::CagrEw)),
            ew_log_sum: 0.0,
            ew_days_sum: 0.0,
            ew_invalid: false,
            daily: DailyStats::default(),
            fill_count: 0,
            last_fill_timestamp_ms: window.start_timestamp_ms,
            fill_gap_max_ms: 0,
            fill_clock_invalid: false,
            turnover_sum: 0.0,
            turnover_invalid: false,
            realized_profit: 0.0,
            realized_loss: 0.0,
            realized_invalid: false,
            holding: HoldingStats::default(),
            exposure: ExposureStats::default(),
        })
    }
    /// Restore already-finalized device moments and reuse the ordinary finish
    /// formulas/status handling. This observer is sealed against further samples.
    pub fn from_summary(
        request: MetricRequest,
        window: EvaluationWindow,
        initial_strategy_equity: f64,
        ew_half_life_days: Option<f64>,
        observation_interval_ms: u64,
        summary: CoreMetricsSummary,
    ) -> Result<Self, String> {
        let mut state = Self::new_with_interval(
            request,
            window,
            initial_strategy_equity,
            ew_half_life_days,
            observation_interval_ms,
        )?;
        let e = summary.equity;
        let elapsed = e.end_timestamp_ms.checked_sub(window.start_timestamp_ms);
        let expected_count = elapsed
            .map(|d| d / observation_interval_ms + 1 + u64::from(d % observation_interval_ms != 0));
        state.pending_equity = None;
        state.previous_equity = Some((e.end_timestamp_ms, e.final_strategy_equity));
        state.last_timestamp_ms = e.end_timestamp_ms;
        state.count = e.observation_count;
        state.clock_invalid = !e.clock_valid || expected_count != Some(e.observation_count);
        state.equity_invalid |= !e.all_equity_finite || !e.final_strategy_equity.is_finite();
        state.equity_nonpositive |= !e.all_equity_positive || e.final_strategy_equity <= 0.0;
        state.path_moments_invalid = ![e.drawdown_sum, e.drawdown_squares, e.drawdown_max]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
            || elapsed.is_none_or(|duration| e.recovery_max_ms > duration);
        state.terminal = true;
        state.drawdown_sum = e.drawdown_sum;
        state.drawdown_squares = e.drawdown_squares;
        state.drawdown_max = e.drawdown_max;
        state.recovery_max_ms = e.recovery_max_ms;
        state.peak_recovery_tail_censored = e.peak_recovery_tail_censored;
        state.ew_log_sum = e.ew_log_sum;
        state.ew_days_sum = e.ew_days_sum;
        state.ew_invalid = !e.ew_valid || !e.ew_log_sum.is_finite() || !e.ew_days_sum.is_finite();
        let d = summary.daily;
        state.daily = DailyStats {
            first: d.first_close_timestamp_ms,
            previous: d.last_close_timestamp_ms.map(|t| (t, 1.0)),
            count: d.return_count,
            mean: d.mean,
            m2: d.m2,
            downside_squares: d.downside_squares,
            gains: d.gains,
            losses: d.losses,
            invalid: !d.valid
                || ![d.mean, d.m2, d.downside_squares, d.gains, d.losses]
                    .iter()
                    .all(|v| v.is_finite())
                || d.m2 < 0.0
                || d.downside_squares < 0.0
                || d.gains < 0.0
                || d.losses < 0.0
                || match (d.first_close_timestamp_ms, d.last_close_timestamp_ms) {
                    (Some(a), Some(b)) => {
                        b.checked_sub(a).map(|n| n / DAY_MS) != Some(d.return_count)
                    }
                    (None, None) => d.return_count != 0,
                    _ => true,
                },
        };
        let f = summary.fills;
        state.fill_count = f.count;
        state.last_fill_timestamp_ms = f.last_timestamp_ms;
        state.fill_gap_max_ms = f.max_gap_ms;
        state.fill_clock_invalid = !f.clock_valid
            || f.last_timestamp_ms < window.start_timestamp_ms
            || (f.count == 0 && f.last_timestamp_ms != window.start_timestamp_ms)
            || elapsed.is_none_or(|duration| f.max_gap_ms > duration);
        state.turnover_sum = f.turnover_sum;
        state.turnover_invalid =
            !f.turnover_valid || !f.turnover_sum.is_finite() || f.turnover_sum < 0.0;
        state.realized_profit = f.realized_profit;
        state.realized_loss = f.realized_loss;
        state.realized_invalid = !f.realized_valid
            || !f.realized_profit.is_finite()
            || !f.realized_loss.is_finite()
            || f.realized_profit < 0.0
            || f.realized_loss < 0.0;
        let h = summary.holding;
        state.holding = HoldingStats {
            open: BTreeMap::new(),
            count: h.count,
            sum_hours: h.sum_hours,
            sum_squares: h.sum_squares,
            max_hours: h.max_hours,
            invalid: !h.valid
                || ![h.sum_hours, h.sum_squares, h.max_hours]
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0.0),
        };
        let x = summary.exposure;
        state.exposure = ExposureStats {
            pending: None,
            first: x.first_timestamp_ms,
            count: x.observation_count,
            gross_squares: x.gross_squares,
            gross_max: x.gross_max,
            utilization_sums: x.utilization_sums,
            invalid_clock: !x.clock_valid
                || x.first_timestamp_ms != Some(window.start_timestamp_ms)
                || x.last_timestamp_ms != Some(e.end_timestamp_ms)
                || x.observation_count != e.observation_count,
            invalid_gross: !x.gross_valid
                || !x.gross_squares.is_finite()
                || !x.gross_max.is_finite()
                || x.gross_squares < 0.0
                || x.gross_max < 0.0,
            invalid_utilization: [0, 1].map(|i| {
                !x.utilization_valid[i]
                    || !x.utilization_sums[i].is_finite()
                    || x.utilization_sums[i] < 0.0
            }),
            terminal: true,
            last_timestamp_ms: x.last_timestamp_ms,
        };
        Ok(state)
    }
    pub fn request(&self) -> MetricRequest {
        self.request
    }
    pub fn observe_minute(&mut self, timestamp_ms: u64, strategy_equity: f64) {
        self.observe_equity(timestamp_ms, strategy_equity, false);
    }
    pub fn observe_terminal(&mut self, timestamp_ms: u64, strategy_equity: f64) {
        self.observe_equity(timestamp_ms, strategy_equity, true);
    }
    fn observe_equity(&mut self, timestamp: u64, equity: f64, terminal: bool) {
        if timestamp > self.window.intended_end_timestamp_ms
            || timestamp < self.last_timestamp_ms
            || self.terminal
        {
            self.clock_invalid = true;
            return;
        }
        let delta = timestamp - self.last_timestamp_ms;
        if (!terminal && delta != self.observation_interval_ms)
            || (terminal && delta > self.observation_interval_ms)
        {
            self.clock_invalid = true;
            return;
        }
        if delta > 0 {
            self.commit_equity();
        }
        self.pending_equity = Some((timestamp, equity));
        self.last_timestamp_ms = timestamp;
        self.terminal = terminal;
    }
    fn commit_equity(&mut self) {
        let Some((timestamp, equity)) = self.pending_equity.take() else {
            return;
        };
        self.count += 1;
        if !equity.is_finite() {
            self.equity_invalid = true;
        }
        if equity <= 0.0 {
            self.equity_nonpositive = true;
        }
        if self.request.any(&[
            Metric::DrawdownMax,
            Metric::Ulcer,
            Metric::Pain,
            Metric::PeakRecoveryMax,
        ]) {
            if !self.equity_invalid && self.peak.is_finite() && self.peak > 0.0 {
                if equity >= self.peak {
                    if let Some(start) = self.underwater_start.take() {
                        self.recovery_max_ms = self.recovery_max_ms.max(timestamp - start);
                    }
                    self.peak = equity;
                    self.peak_timestamp_ms = timestamp;
                } else if self.underwater_start.is_none() {
                    self.underwater_start = Some(self.peak_timestamp_ms);
                }
                self.peak_recovery_tail_censored = Some(self.underwater_start.is_some());
                let d = (self.peak - equity) / self.peak;
                self.drawdown_sum += d;
                self.drawdown_squares += d * d;
                self.drawdown_max = self.drawdown_max.max(d);
            }
        }
        if self.request.contains(Metric::CagrEw) {
            if let Some((previous_timestamp, previous_equity)) = self.previous_equity {
                if !self.equity_invalid && !self.equity_nonpositive {
                    let days = (timestamp - previous_timestamp) as f64 / DAY_MS as f64;
                    let decay = 2.0_f64.powf(-days / self.half_life_days.unwrap());
                    self.ew_log_sum = self.ew_log_sum * decay + log_ratio(equity, previous_equity);
                    self.ew_days_sum = self.ew_days_sum * decay + days;
                    if !self.ew_log_sum.is_finite() || !self.ew_days_sum.is_finite() {
                        self.ew_invalid = true;
                    }
                }
            }
        }
        self.previous_equity = Some((timestamp, equity));
    }
    /// Feed only actual UTC daily boundaries inside the evaluation window. The
    /// first close seeds a reference and creates no fabricated zero return.
    pub fn observe_daily_close(&mut self, timestamp_ms: u64, strategy_equity: f64) {
        if self.request.needs_daily_returns() {
            self.daily
                .observe(timestamp_ms, strategy_equity, self.window);
        }
    }
    pub fn observe_fill(&mut self, fill: FillObservation) {
        if fill.timestamp_ms < self.last_fill_timestamp_ms
            || fill.timestamp_ms > self.window.intended_end_timestamp_ms
        {
            self.fill_clock_invalid = true;
            return;
        }
        self.fill_count += 1;
        if self.request.contains(Metric::FillGapMax) {
            self.fill_gap_max_ms = self
                .fill_gap_max_ms
                .max(fill.timestamp_ms - self.last_fill_timestamp_ms);
        }
        self.last_fill_timestamp_ms = fill.timestamp_ms;
        if self.request.contains(Metric::Turnover) {
            if !fill.notional_usd.is_finite()
                || fill.notional_usd < 0.0
                || !fill.pre_fill_balance_usd.is_finite()
                || fill.pre_fill_balance_usd <= 0.0
            {
                self.turnover_invalid = true;
            } else {
                self.turnover_sum += fill.notional_usd / fill.pre_fill_balance_usd;
                if !self.turnover_sum.is_finite() {
                    self.turnover_invalid = true;
                }
            }
        }
        if self.request.contains(Metric::LossProfit) {
            let net = fill.net_realized_pnl_usd;
            if !net.is_finite() {
                self.realized_invalid = true;
            } else {
                self.realized_profit += net.max(0.0);
                self.realized_loss += (-net).max(0.0);
                if !self.realized_profit.is_finite() || !self.realized_loss.is_finite() {
                    self.realized_invalid = true;
                }
            }
        }
        if self.request.needs_holding() {
            self.holding.fill(fill);
        }
    }
    /// Optional initial inventory must have a factually known analysis boundary.
    /// Only the observed portion starting at evaluation start is scored.
    pub fn seed_open_position(&mut self, coin_index: usize, side: PositionSide) {
        if self.request.needs_holding() {
            if self.fill_count != 0 || self.last_timestamp_ms != self.window.start_timestamp_ms {
                self.holding.invalid = true;
                return;
            }
            if self
                .holding
                .open
                .insert((coin_index, side), self.window.start_timestamp_ms)
                .is_some()
            {
                self.holding.invalid = true;
            }
        }
    }
    pub fn observe_exposure(&mut self, observation: ExposureObservation) {
        if self.request.needs_exposure() && self.observation_interval_ms == MINUTE_MS {
            self.exposure.observe(
                observation,
                self.request,
                self.window.start_timestamp_ms,
                self.window.intended_end_timestamp_ms,
            );
        }
    }
    /// Produces a non-mutating view, including open holding and HSL tails once.
    pub fn finish(&self, finalization: Finalization) -> CoreMetricsResult {
        let mut state = self.clone();
        state.commit_equity();
        let end = finalization.actual_end_timestamp_ms;
        let time_valid = !state.clock_invalid
            && end == state.last_timestamp_ms
            && end > state.window.start_timestamp_ms
            && end <= state.window.intended_end_timestamp_ms;
        let days = end.saturating_sub(state.window.start_timestamp_ms) as f64 / DAY_MS as f64;
        let years = days / YEAR_DAYS;
        let final_equity = state
            .previous_equity
            .map(|(_, e)| e)
            .unwrap_or(state.initial_equity);
        let holding = state.holding.with_tails(end);
        let exposure_end = state.exposure.last_timestamp_ms;
        state.exposure.commit(state.request);
        let mut metrics = BTreeMap::new();
        for definition in METRIC_CATALOG
            .iter()
            .filter(|d| state.request.contains(d.metric))
        {
            let metric = definition.metric;
            let value = if !time_valid {
                MetricValue::invalid("invalid_evaluation_clock")
            } else if metric.requires_minute_observations()
                && state.observation_interval_ms != MINUTE_MS
            {
                MetricValue::invalid("unsupported_observation_interval")
            } else {
                match metric {
                    Metric::Cagr | Metric::CagrEw => {
                        if state.equity_invalid || state.equity_nonpositive {
                            MetricValue::invalid("nonpositive_or_nonfinite_strategy_equity")
                        } else if metric == Metric::Cagr {
                            MetricValue::finite(
                                (log_ratio(final_equity, state.initial_equity) / years).exp_m1(),
                            )
                        } else if state.ew_invalid {
                            MetricValue::invalid("numeric_overflow")
                        } else if state.ew_days_sum <= 0.0 {
                            MetricValue::undefined("no_growth_intervals")
                        } else {
                            MetricValue::finite(
                                (YEAR_DAYS * state.ew_log_sum / state.ew_days_sum).exp_m1(),
                            )
                        }
                    }
                    Metric::Volatility | Metric::Sharpe | Metric::Sortino | Metric::Omega => state
                        .daily
                        .metric(metric, state.window.start_timestamp_ms, end),
                    Metric::DrawdownMax
                    | Metric::Ulcer
                    | Metric::Pain
                    | Metric::PeakRecoveryMax => {
                        if state.equity_invalid
                            || state.path_moments_invalid
                            || !state.initial_equity.is_finite()
                            || state.initial_equity <= 0.0
                        {
                            MetricValue::invalid("invalid_drawdown_curve")
                        } else if metric == Metric::PeakRecoveryMax
                            && state.peak_recovery_tail_censored.is_none()
                        {
                            MetricValue::invalid("missing_recovery_tail_status")
                        } else {
                            MetricValue::finite(match metric {
                                Metric::DrawdownMax => state.drawdown_max,
                                Metric::Ulcer => {
                                    (state.drawdown_squares / state.count as f64).sqrt()
                                }
                                Metric::Pain => state.drawdown_sum / state.count as f64,
                                Metric::PeakRecoveryMax => {
                                    state
                                        .recovery_max_ms
                                        .max(state.underwater_start.map_or(0, |t| end - t))
                                        as f64
                                        / DAY_MS as f64
                                }
                                _ => unreachable!(),
                            })
                        }
                    }
                    Metric::Completion => MetricValue::finite(
                        (end - state.window.start_timestamp_ms) as f64
                            / (state.window.intended_end_timestamp_ms
                                - state.window.start_timestamp_ms)
                                as f64,
                    ),
                    Metric::FillGapMax => {
                        if state.fill_clock_invalid || state.last_fill_timestamp_ms > end {
                            MetricValue::invalid("invalid_fill_clock")
                        } else {
                            MetricValue::finite(
                                state
                                    .fill_gap_max_ms
                                    .max(end - state.last_fill_timestamp_ms)
                                    as f64
                                    / DAY_MS as f64,
                            )
                        }
                    }
                    Metric::Turnover => {
                        if state.fill_clock_invalid
                            || state.last_fill_timestamp_ms > end
                            || state.turnover_invalid
                        {
                            MetricValue::invalid("invalid_turnover_input")
                        } else {
                            MetricValue::finite(state.turnover_sum / days)
                        }
                    }
                    Metric::HoldingMax | Metric::HoldingTimeWeightedMean => {
                        if state.fill_clock_invalid
                            || state.last_fill_timestamp_ms > end
                            || holding.invalid
                        {
                            MetricValue::invalid("invalid_position_episode")
                        } else {
                            MetricValue::finite(if metric == Metric::HoldingMax {
                                holding.max_hours / 24.0
                            } else if holding.sum_hours == 0.0 {
                                0.0
                            } else {
                                holding.sum_squares / holding.sum_hours
                            })
                        }
                    }
                    Metric::GrossExposureRms
                    | Metric::GrossExposureMax
                    | Metric::PositionUtilizationLong
                    | Metric::PositionUtilizationShort => {
                        if state.exposure.invalid_clock
                            || exposure_end != Some(end)
                            || state.exposure.count == 0
                        {
                            MetricValue::invalid("invalid_exposure_population")
                        } else if matches!(
                            metric,
                            Metric::GrossExposureRms | Metric::GrossExposureMax
                        ) {
                            if state.exposure.invalid_gross
                                || state.equity_invalid
                                || state.equity_nonpositive
                            {
                                MetricValue::invalid("invalid_gross_exposure_input")
                            } else {
                                MetricValue::finite(if metric == Metric::GrossExposureMax {
                                    state.exposure.gross_max
                                } else {
                                    (state.exposure.gross_squares / state.exposure.count as f64)
                                        .sqrt()
                                })
                            }
                        } else {
                            let side = usize::from(metric == Metric::PositionUtilizationShort);
                            if state.exposure.invalid_utilization[side] {
                                MetricValue::invalid("invalid_utilization_input")
                            } else {
                                MetricValue::finite(
                                    (state.exposure.utilization_sums[side]
                                        / state.exposure.count as f64)
                                        .sqrt(),
                                )
                            }
                        }
                    }
                    Metric::LossProfit => {
                        if state.realized_invalid
                            || state.fill_clock_invalid
                            || state.last_fill_timestamp_ms > end
                        {
                            MetricValue::invalid("invalid_realized_cashflow")
                        } else {
                            MetricValue::ratio(state.realized_loss, state.realized_profit)
                        }
                    }
                    Metric::ShortContribution => {
                        let [long, short] = finalization.net_side_profit_usd;
                        let total = long + short;
                        if !long.is_finite() || !short.is_finite() || !total.is_finite() {
                            MetricValue::invalid("invalid_side_pnl")
                        } else if total <= 0.0 {
                            MetricValue::undefined("nonpositive_combined_net_profit")
                        } else {
                            MetricValue::ratio(short, total)
                        }
                    }
                    Metric::HslEpisodeLossMax | Metric::HslEpisodeLossMean => {
                        let h = finalization.hsl;
                        let valid = h.episode_loss_valid
                            && (h.enabled
                                || (h.execution_episode_count == 0
                                    && h.episode_loss_fraction_sum == 0.0))
                            && h.episode_loss_fraction_sum.is_finite()
                            && h.episode_loss_fraction_sum >= 0.0
                            && h.episode_loss_fraction_max.is_finite()
                            && h.episode_loss_fraction_max >= 0.0
                            && h.episode_loss_fraction_max <= h.episode_loss_fraction_sum
                            && (h.execution_episode_count > 0
                                || h.episode_loss_fraction_sum == 0.0);
                        if !valid {
                            MetricValue::invalid("invalid_hsl_execution_episode")
                        } else {
                            MetricValue::finite(if metric == Metric::HslEpisodeLossMax {
                                h.episode_loss_fraction_max
                            } else if h.execution_episode_count == 0 {
                                0.0
                            } else {
                                h.episode_loss_fraction_sum / h.execution_episode_count as f64
                            })
                        }
                    }
                    Metric::HslExecutionEpisodeCount => {
                        let h = finalization.hsl;
                        if !h.execution_episode_count_valid
                            || !h.clock_valid(end - state.window.start_timestamp_ms)
                        {
                            MetricValue::invalid("invalid_hsl_execution_episode_count")
                        } else {
                            MetricValue::finite(h.execution_episode_count as f64)
                        }
                    }
                    Metric::HslTriggersPerYear => {
                        if !finalization
                            .hsl
                            .clock_valid(end - state.window.start_timestamp_ms)
                        {
                            MetricValue::invalid("invalid_hsl_observation")
                        } else {
                            MetricValue::finite(finalization.hsl.trigger_count as f64 / years)
                        }
                    }
                    Metric::HslRedTime => {
                        let h = finalization.hsl;
                        if !h.clock_valid(end - state.window.start_timestamp_ms) {
                            MetricValue::invalid("invalid_hsl_duration")
                        } else if !h.enabled && h.observed_duration_ms == 0 {
                            MetricValue::finite(0.0)
                        } else {
                            MetricValue::ratio(
                                h.red_duration_ms as f64,
                                h.observed_duration_ms as f64,
                            )
                        }
                    }
                }
            };
            metrics.insert(definition.name, value);
        }
        let peak_recovery_tail_censored = metrics
            .get(Metric::PeakRecoveryMax.definition().name)
            .filter(|value| value.status == MetricStatus::Finite)
            .and(state.peak_recovery_tail_censored);
        CoreMetricsResult {
            metrics,
            metadata: MetricsMetadata {
                definition_version: CORE_METRIC_DEFINITION_VERSION,
                year_days: YEAR_DAYS,
                observation_interval_ms: state.observation_interval_ms,
                start_timestamp_ms: state.window.start_timestamp_ms,
                intended_end_timestamp_ms: state.window.intended_end_timestamp_ms,
                actual_end_timestamp_ms: end,
                equity_observation_count: state.count,
                daily_return_count: state.daily.count,
                daily_first_close_timestamp_ms: state.daily.first,
                daily_last_close_timestamp_ms: state.daily.previous.map(|(t, _)| t),
                fill_count: state.fill_count,
                holding_episode_count: holding.count,
                peak_recovery_tail_censored,
                hsl_enabled: finalization.hsl.enabled,
                hsl_trigger_count: finalization.hsl.trigger_count,
                hsl_execution_episode_count: finalization.hsl.execution_episode_count,
                ew_half_life_days: state.half_life_days,
            },
        }
    }
}

fn log_ratio(value: f64, reference: f64) -> f64 {
    let ratio = value / reference;
    if ratio.is_finite() && ratio > 0.0 {
        ratio.ln()
    } else {
        value.ln() - reference.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(metrics: &[Metric]) -> MetricRequest {
        MetricRequest::from_metrics(metrics.iter().copied())
    }
    fn new(metrics: &[Metric], end: u64) -> CoreMetrics {
        CoreMetrics::new(
            request(metrics),
            EvaluationWindow {
                start_timestamp_ms: 0,
                intended_end_timestamp_ms: end,
            },
            100.0,
            Some(180.0),
        )
        .unwrap()
    }
    fn flat(metrics: &[Metric], end: u64) -> CoreMetrics {
        let mut state = new(metrics, end);
        for t in (MINUTE_MS..=end).step_by(MINUTE_MS as usize) {
            state.observe_minute(t, 100.0);
        }
        if end % MINUTE_MS != 0 {
            state.observe_terminal(end, 100.0);
        }
        state
    }
    fn finalization(end: u64) -> Finalization {
        Finalization {
            actual_end_timestamp_ms: end,
            net_side_profit_usd: [0.0; 2],
            hsl: HslSummary::default(),
        }
    }
    fn value(result: &CoreMetricsResult, metric: Metric) -> MetricValue {
        result.metrics[metric.definition().name]
    }
    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-10_f64.max(expected.abs() * 1e-10),
            "{actual} != {expected}"
        );
    }
    fn number(result: &CoreMetricsResult, metric: Metric) -> f64 {
        let v = value(result, metric);
        assert_eq!(v.status, MetricStatus::Finite, "{metric:?}: {v:?}");
        v.value.unwrap()
    }
    fn daily(levels: &[f64], selected: &[Metric]) -> CoreMetrics {
        let end = (levels.len() as u64 - 1) * DAY_MS;
        let mut state = CoreMetrics::new(
            request(selected),
            EvaluationWindow {
                start_timestamp_ms: 0,
                intended_end_timestamp_ms: end,
            },
            levels[0],
            Some(180.0),
        )
        .unwrap();
        for t in (MINUTE_MS..=end).step_by(MINUTE_MS as usize) {
            state.observe_minute(t, levels[(t / DAY_MS) as usize]);
        }
        for (day, &level) in levels.iter().enumerate() {
            state.observe_daily_close(day as u64 * DAY_MS, level);
        }
        state
    }
    fn fill(t: u64, coin: usize, side: PositionSide, before: f64, after: f64) -> FillObservation {
        FillObservation {
            timestamp_ms: t,
            coin_index: coin,
            position_side: side,
            position_size_before: before,
            position_size_after: after,
            notional_usd: 0.0,
            pre_fill_balance_usd: 1000.0,
            net_realized_pnl_usd: 0.0,
        }
    }
    fn exposure(t: u64, gross: f64, utilization: [f64; 2]) -> ExposureObservation {
        ExposureObservation {
            timestamp_ms: t,
            strategy_equity: 100.0,
            gross_marked_notional_usd: gross * 100.0,
            position_utilization_squared: utilization,
        }
    }

    #[test]
    fn catalog_has_only_canonical_names_and_requests_are_deduplicated() {
        let mut names = std::collections::BTreeSet::new();
        for (i, definition) in METRIC_CATALOG.iter().enumerate() {
            assert_eq!(definition.metric as usize, i);
            assert!(names.insert(definition.name));
        }
        assert_eq!(MetricRequest::all().len(), 26);
        assert!(MetricRequest::default().is_empty());
        let r = MetricRequest::from_names(["loss_profit_ratio", "loss_profit_ratio"]).unwrap();
        assert_eq!(r.len(), 1);
        assert!(r.contains(Metric::LossProfit));
        assert!(MetricRequest::from_names(["profit_factor"]).is_err());
        assert!(MetricRequest::from_names(["adg_fraction_strategy_equity"]).is_err());
    }

    #[test]
    fn statuses_distinguish_true_infinity_undefined_and_overflow_in_standard_json() {
        for (numerator, expected) in [
            (1.0, MetricStatus::PositiveInfinity),
            (-1.0, MetricStatus::NegativeInfinity),
            (0.0, MetricStatus::Undefined),
        ] {
            let v = MetricValue::ratio(numerator, 0.0);
            assert_eq!(v.status, expected);
            assert!(v.value.is_none());
            let json = serde_json::to_value(v).unwrap();
            assert!(json["value"].is_null());
            assert!(json["status"].is_string());
        }
        assert_eq!(
            MetricValue::ratio(f64::MAX, f64::MIN_POSITIVE).status,
            MetricStatus::Invalid
        );
        assert_eq!(
            MetricValue::finite(f64::INFINITY).status,
            MetricStatus::Invalid
        );
    }

    #[test]
    fn minute_drawdown_moments_include_baseline_and_peak_ties_recover() {
        let selected = [
            Metric::DrawdownMax,
            Metric::Pain,
            Metric::Ulcer,
            Metric::PeakRecoveryMax,
        ];
        let mut state = new(&selected, 5 * MINUTE_MS);
        for (i, equity) in [90.0, 80.0, 100.0, 110.0, 99.0].into_iter().enumerate() {
            state.observe_minute((i as u64 + 1) * MINUTE_MS, equity);
        }
        let result = state.finish(finalization(5 * MINUTE_MS));
        close(number(&result, Metric::DrawdownMax), 0.2);
        close(number(&result, Metric::Pain), 0.4 / 6.0);
        close(number(&result, Metric::Ulcer), 0.1);
        close(number(&result, Metric::PeakRecoveryMax), 3.0 / 1440.0);
        assert_eq!(result.metadata.equity_observation_count, 6);
        let mut plateau = new(&[Metric::PeakRecoveryMax], 3 * MINUTE_MS);
        for (i, equity) in [100.0, 90.0, 100.0].into_iter().enumerate() {
            plateau.observe_minute((i as u64 + 1) * MINUTE_MS, equity);
        }
        close(
            number(
                &plateau.finish(finalization(3 * MINUTE_MS)),
                Metric::PeakRecoveryMax,
            ),
            2.0 / 1440.0,
        );
        close(
            number(
                &flat(&[Metric::PeakRecoveryMax], MINUTE_MS).finish(finalization(MINUTE_MS)),
                Metric::PeakRecoveryMax,
            ),
            0.0,
        );
    }

    #[test]
    fn partial_terminal_mark_counts_once_and_same_clock_terminal_replaces_pending_mark() {
        let mut state = new(
            &[Metric::Pain, Metric::Ulcer, Metric::Completion],
            2 * MINUTE_MS,
        );
        state.observe_minute(MINUTE_MS, 90.0);
        state.observe_terminal(MINUTE_MS + MINUTE_MS / 2, 80.0);
        let result = state.finish(finalization(MINUTE_MS + MINUTE_MS / 2));
        assert_eq!(result.metadata.equity_observation_count, 3);
        close(number(&result, Metric::Pain), 0.1);
        close(number(&result, Metric::Completion), 0.75);
        let mut same = new(&[Metric::Pain], MINUTE_MS);
        same.observe_minute(MINUTE_MS, 90.0);
        same.observe_terminal(MINUTE_MS, 80.0);
        let result = same.finish(finalization(MINUTE_MS));
        assert_eq!(result.metadata.equity_observation_count, 2);
        close(number(&result, Metric::Pain), 0.1);
    }

    #[test]
    fn cagr_is_unsmoothed_and_uses_actual_time_and_ew_recurrence_has_calendar_weights() {
        let selected = [Metric::Cagr, Metric::CagrEw];
        let mut state = CoreMetrics::new(
            request(&selected),
            EvaluationWindow {
                start_timestamp_ms: 0,
                intended_end_timestamp_ms: 2 * MINUTE_MS,
            },
            100.0,
            Some(1.0 / 1440.0),
        )
        .unwrap();
        let e1 = 100.0 * 1.000001;
        let e2 = e1 * 1.000002;
        state.observe_minute(MINUTE_MS, e1);
        state.observe_minute(2 * MINUTE_MS, e2);
        let result = state.finish(finalization(2 * MINUTE_MS));
        close(
            number(&result, Metric::Cagr),
            ((e2 / 100.0).ln() * 365.25 * 1440.0 / 2.0).exp_m1(),
        );
        let expected =
            (365.25 * (0.5 * 1.000001_f64.ln() + 1.000002_f64.ln()) / (1.5 / 1440.0)).exp_m1();
        close(number(&result, Metric::CagrEw), expected);
        let mut long_half_life =
            CoreMetrics::new(request(&selected), state.window, 100.0, Some(1e20)).unwrap();
        long_half_life.observe_minute(MINUTE_MS, e1);
        long_half_life.observe_minute(2 * MINUTE_MS, e2);
        let result = long_half_life.finish(finalization(2 * MINUTE_MS));
        close(
            number(&result, Metric::CagrEw),
            number(&result, Metric::Cagr),
        );
    }

    #[test]
    fn daily_ratios_use_arithmetic_mean_sample_variance_and_all_period_downside_rms() {
        let selected = [
            Metric::Volatility,
            Metric::Sharpe,
            Metric::Sortino,
            Metric::Omega,
        ];
        let state = daily(&[100.0, 110.0, 99.0, 108.9], &selected);
        let result = state.finish(finalization(3 * DAY_MS));
        assert_eq!(result.metadata.daily_return_count, 3);
        close(
            number(&result, Metric::Volatility),
            (0.04_f64 / 3.0).sqrt() * YEAR_DAYS.sqrt(),
        );
        close(
            number(&result, Metric::Sharpe),
            YEAR_DAYS.sqrt() / (2.0 * 3.0_f64.sqrt()),
        );
        close(
            number(&result, Metric::Sortino),
            YEAR_DAYS.sqrt() / 3.0_f64.sqrt(),
        );
        close(number(&result, Metric::Omega), 2.0);
    }

    #[test]
    fn constant_daily_returns_keep_sign_and_zero_denominator_status() {
        let selected = [
            Metric::Volatility,
            Metric::Sharpe,
            Metric::Sortino,
            Metric::Omega,
        ];
        let positive = daily(&[100.0, 200.0, 400.0], &selected).finish(finalization(2 * DAY_MS));
        for metric in [Metric::Sharpe, Metric::Sortino, Metric::Omega] {
            assert_eq!(
                value(&positive, metric).status,
                MetricStatus::PositiveInfinity
            );
        }
        let negative = daily(&[100.0, 50.0, 25.0], &selected).finish(finalization(2 * DAY_MS));
        assert_eq!(
            value(&negative, Metric::Sharpe).status,
            MetricStatus::NegativeInfinity
        );
        close(number(&negative, Metric::Sortino), -YEAR_DAYS.sqrt());
        close(number(&negative, Metric::Omega), 0.0);
        let unchanged = daily(&[100.0, 100.0, 100.0], &selected).finish(finalization(2 * DAY_MS));
        close(number(&unchanged, Metric::Volatility), 0.0);
        for metric in [Metric::Sharpe, Metric::Sortino, Metric::Omega] {
            assert_eq!(value(&unchanged, metric).status, MetricStatus::Undefined);
        }
    }

    #[test]
    fn missing_leading_trailing_interior_or_non_utc_daily_boundaries_are_invalid() {
        for closes in [
            vec![DAY_MS, 2 * DAY_MS, 3 * DAY_MS],
            vec![0, DAY_MS, 2 * DAY_MS],
            vec![0, 2 * DAY_MS, 3 * DAY_MS],
            vec![0, DAY_MS + MINUTE_MS, 3 * DAY_MS],
        ] {
            let mut state = flat(&[Metric::Omega], 3 * DAY_MS);
            for t in closes {
                state.observe_daily_close(t, 100.0);
            }
            assert_eq!(
                value(&state.finish(finalization(3 * DAY_MS)), Metric::Omega).status,
                MetricStatus::Invalid
            );
        }
        let start = MINUTE_MS;
        let end = 3 * DAY_MS - MINUTE_MS;
        let mut state = CoreMetrics::new(
            request(&[Metric::Omega]),
            EvaluationWindow {
                start_timestamp_ms: start,
                intended_end_timestamp_ms: end,
            },
            100.0,
            None,
        )
        .unwrap();
        for t in ((start + MINUTE_MS)..=end).step_by(MINUTE_MS as usize) {
            state.observe_minute(t, 100.0);
        }
        state.observe_daily_close(DAY_MS, 100.0);
        state.observe_daily_close(2 * DAY_MS, 110.0);
        let result = state.finish(finalization(end));
        assert_eq!(result.metadata.daily_return_count, 1);
        assert_eq!(
            value(&result, Metric::Omega).status,
            MetricStatus::PositiveInfinity
        );
    }

    #[test]
    fn ruin_and_overflow_are_invalid_growth_even_if_the_endpoint_recovers() {
        let mut state = new(
            &[Metric::Cagr, Metric::CagrEw, Metric::DrawdownMax],
            2 * MINUTE_MS,
        );
        state.observe_minute(MINUTE_MS, 0.0);
        state.observe_minute(2 * MINUTE_MS, 100.0);
        let result = state.finish(finalization(2 * MINUTE_MS));
        for metric in [Metric::Cagr, Metric::CagrEw] {
            assert_eq!(value(&result, metric).status, MetricStatus::Invalid);
        }
        close(number(&result, Metric::DrawdownMax), 1.0);
        let mut overflow = new(&[Metric::Cagr], MINUTE_MS);
        overflow.observe_minute(MINUTE_MS, 200.0);
        assert_eq!(
            value(&overflow.finish(finalization(MINUTE_MS)), Metric::Cagr).status,
            MetricStatus::Invalid
        );
    }

    #[test]
    fn turnover_uses_gross_pre_fill_balance_and_all_elapsed_days() {
        for (end, notional, expected) in [(DAY_MS, 500.0, 0.5), (2 * DAY_MS, 1200.0, 0.6)] {
            let mut state = flat(&[Metric::Turnover], end);
            let mut f = fill(MINUTE_MS, 0, PositionSide::Long, 0.0, 1.0);
            f.notional_usd = notional;
            state.observe_fill(f);
            close(
                number(&state.finish(finalization(end)), Metric::Turnover),
                expected,
            );
        }
        let mut state = flat(&[Metric::Turnover], DAY_MS);
        let mut f = fill(0, 0, PositionSide::Long, 0.0, 1.0);
        f.notional_usd = 500.0;
        state.observe_fill(f);
        f.timestamp_ms = MINUTE_MS;
        f.pre_fill_balance_usd = 2000.0;
        state.observe_fill(f);
        close(
            number(&state.finish(finalization(DAY_MS)), Metric::Turnover),
            0.75,
        );
        f.pre_fill_balance_usd = 0.0;
        state.observe_fill(f);
        assert_eq!(
            value(&state.finish(finalization(DAY_MS)), Metric::Turnover).status,
            MetricStatus::Invalid
        );
    }

    #[test]
    fn fill_gap_includes_both_boundaries_and_duplicate_timestamps() {
        let mut state = flat(&[Metric::FillGapMax], DAY_MS);
        close(
            number(&state.finish(finalization(DAY_MS)), Metric::FillGapMax),
            1.0,
        );
        for t in [DAY_MS / 4, DAY_MS / 4, DAY_MS / 2] {
            state.observe_fill(fill(t, 0, PositionSide::Long, 0.0, 0.0));
        }
        close(
            number(&state.finish(finalization(DAY_MS)), Metric::FillGapMax),
            0.5,
        );
    }

    #[test]
    fn holding_additions_partial_closes_and_open_tails_use_observed_episode_hours() {
        let end = 3 * HOUR_MS as u64;
        let selected = [Metric::HoldingMax, Metric::HoldingTimeWeightedMean];
        let mut state = flat(&selected, end);
        for f in [
            fill(0, 0, PositionSide::Long, 0.0, 1.0),
            fill(0, 1, PositionSide::Short, 0.0, -1.0),
            fill(20 * MINUTE_MS, 0, PositionSide::Long, 1.0, 2.0),
            fill(30 * MINUTE_MS, 0, PositionSide::Long, 2.0, 1.0),
            fill(HOUR_MS as u64, 0, PositionSide::Long, 1.0, 0.0),
        ] {
            state.observe_fill(f);
        }
        assert_eq!(state.holding.open.len(), 1);
        let result = state.finish(finalization(end));
        close(number(&result, Metric::HoldingMax), 3.0 / 24.0);
        close(number(&result, Metric::HoldingTimeWeightedMean), 2.5);
        assert_eq!(result.metadata.holding_episode_count, 2);
        assert_eq!(
            state
                .finish(finalization(end))
                .metadata
                .holding_episode_count,
            2
        );
        assert_eq!(state.holding.count, 1);
        let empty = flat(&selected, end).finish(finalization(end));
        close(number(&empty, Metric::HoldingTimeWeightedMean), 0.0);
        close(number(&empty, Metric::HoldingMax), 0.0);
    }

    #[test]
    fn missing_holding_prefix_is_invalid_and_known_initial_inventory_can_be_seeded() {
        let mut state = new(&[Metric::HoldingMax], MINUTE_MS);
        state.observe_fill(fill(0, 0, PositionSide::Long, 1.0, 0.0));
        state.observe_minute(MINUTE_MS, 100.0);
        assert_eq!(
            value(&state.finish(finalization(MINUTE_MS)), Metric::HoldingMax).status,
            MetricStatus::Invalid
        );
        let mut seeded = new(&[Metric::HoldingMax], MINUTE_MS);
        seeded.seed_open_position(0, PositionSide::Long);
        seeded.observe_minute(MINUTE_MS, 100.0);
        close(
            number(&seeded.finish(finalization(MINUTE_MS)), Metric::HoldingMax),
            1.0 / 1440.0,
        );
    }

    #[test]
    fn gross_and_budget_relative_exposure_include_flat_baseline_and_keep_separate_scopes() {
        let selected = [
            Metric::GrossExposureRms,
            Metric::GrossExposureMax,
            Metric::PositionUtilizationLong,
            Metric::PositionUtilizationShort,
        ];
        let mut state = flat(&selected, 2 * MINUTE_MS);
        state.observe_exposure(exposure(0, 0.0, [0.0, 0.0]));
        state.observe_exposure(exposure(MINUTE_MS, 2.0, [0.25, 0.0]));
        state.observe_exposure(exposure(2 * MINUTE_MS, 1.0, [1.0, 0.0]));
        let result = state.finish(finalization(2 * MINUTE_MS));
        close(
            number(&result, Metric::GrossExposureRms),
            (5.0_f64 / 3.0).sqrt(),
        );
        close(number(&result, Metric::GrossExposureMax), 2.0);
        close(
            number(&result, Metric::PositionUtilizationLong),
            (1.25_f64 / 3.0).sqrt(),
        );
        close(number(&result, Metric::PositionUtilizationShort), 0.0);
        let mut missing = flat(&[Metric::GrossExposureRms], MINUTE_MS);
        missing.observe_exposure(exposure(MINUTE_MS, 0.0, [0.0, 0.0]));
        assert_eq!(
            value(
                &missing.finish(finalization(MINUTE_MS)),
                Metric::GrossExposureRms
            )
            .status,
            MetricStatus::Invalid
        );
    }

    #[test]
    fn hsl_owner_supplies_episode_aggregates_and_union_time_without_histories() {
        let selected = [
            Metric::HslEpisodeLossMax,
            Metric::HslEpisodeLossMean,
            Metric::HslTriggersPerYear,
            Metric::HslRedTime,
        ];
        let state = flat(&selected, DAY_MS);
        let mut f = finalization(DAY_MS);
        f.hsl = HslSummary {
            enabled: true,
            trigger_count: 2,
            execution_episode_count: 2,
            execution_episode_count_valid: true,
            episode_loss_fraction_sum: 0.15,
            episode_loss_fraction_max: 0.1,
            episode_loss_valid: true,
            red_duration_ms: DAY_MS * 6 / 10,
            observed_duration_ms: DAY_MS,
            observation_valid: true,
        };
        let result = state.finish(f);
        close(number(&result, Metric::HslEpisodeLossMax), 0.1);
        close(number(&result, Metric::HslEpisodeLossMean), 0.075);
        close(number(&result, Metric::HslTriggersPerYear), 730.5);
        close(number(&result, Metric::HslRedTime), 0.6);
        let empty = state.finish(finalization(DAY_MS));
        close(number(&empty, Metric::HslEpisodeLossMean), 0.0);
        assert_eq!(empty.metadata.hsl_execution_episode_count, 0);
        f.hsl.episode_loss_valid = false;
        assert_eq!(
            value(&state.finish(f), Metric::HslEpisodeLossMean).status,
            MetricStatus::Invalid
        );
        f.hsl.observed_duration_ms = 0;
        assert_eq!(
            value(&state.finish(f), Metric::HslRedTime).status,
            MetricStatus::Invalid
        );
    }

    #[test]
    fn hsl_execution_count_requires_fills_but_not_losses_or_restart() {
        let state = flat(&[Metric::HslExecutionEpisodeCount], DAY_MS);
        let mut f = finalization(DAY_MS);
        close(
            number(&state.finish(f), Metric::HslExecutionEpisodeCount),
            0.0,
        );
        f.hsl.enabled = true;
        f.hsl.observed_duration_ms = DAY_MS;
        f.hsl.trigger_count = 3;
        // Trigger-only episodes and unfilled panic orders do not count.
        close(
            number(&state.finish(f), Metric::HslExecutionEpisodeCount),
            0.0,
        );
        for count in [1, 3] {
            f.hsl.execution_episode_count = count;
            // Zero-loss and unfinished executed episodes still count. Loss
            // anchor validity belongs to the loss metrics, not this count.
            f.hsl.episode_loss_valid = false;
            close(
                number(&state.finish(f), Metric::HslExecutionEpisodeCount),
                count as f64,
            );
        }
        f.hsl.execution_episode_count_valid = false;
        assert_eq!(
            value(&state.finish(f), Metric::HslExecutionEpisodeCount).status,
            MetricStatus::Invalid
        );
        f.hsl.execution_episode_count_valid = true;
        f.hsl.observed_duration_ms = 0;
        assert_eq!(
            value(&state.finish(f), Metric::HslExecutionEpisodeCount).status,
            MetricStatus::Invalid
        );
        f.hsl.enabled = false;
        assert_eq!(
            value(&state.finish(f), Metric::HslExecutionEpisodeCount).status,
            MetricStatus::Invalid
        );
    }

    #[test]
    fn hsl_duration_requires_complete_evaluation_support_and_disabled_events_are_invalid() {
        let selected = [Metric::HslRedTime, Metric::HslTriggersPerYear];
        let state = flat(&selected, DAY_MS);
        for observed in [0, DAY_MS - 1, DAY_MS + 1, 2 * DAY_MS] {
            let mut f = finalization(DAY_MS);
            f.hsl.enabled = true;
            f.hsl.observed_duration_ms = observed;
            let result = state.finish(f);
            for metric in selected {
                assert_eq!(value(&result, metric).status, MetricStatus::Invalid);
            }
        }
        for trigger in [false, true] {
            let mut f = finalization(DAY_MS);
            f.hsl.observed_duration_ms = DAY_MS;
            if trigger {
                f.hsl.trigger_count = 1;
            } else {
                f.hsl.red_duration_ms = 1;
            }
            let result = state.finish(f);
            for metric in selected {
                assert_eq!(value(&result, metric).status, MetricStatus::Invalid);
            }
        }
        let disabled = state.finish(finalization(DAY_MS));
        for metric in selected {
            close(number(&disabled, metric), 0.0);
        }
    }

    #[test]
    fn loss_profit_counts_signed_net_cashflow_and_keeps_empty_and_infinite_cases() {
        let mut state = flat(&[Metric::LossProfit], MINUTE_MS);
        assert_eq!(
            value(&state.finish(finalization(MINUTE_MS)), Metric::LossProfit).status,
            MetricStatus::Undefined
        );
        let mut f = fill(0, 0, PositionSide::Long, 0.0, 0.0);
        f.net_realized_pnl_usd = -5.0; // An entry fee, not omitted for zero gross P&L.
        state.observe_fill(f);
        assert_eq!(
            value(&state.finish(finalization(MINUTE_MS)), Metric::LossProfit).status,
            MetricStatus::PositiveInfinity
        );
        f.net_realized_pnl_usd = 105.0 - 5.0;
        state.observe_fill(f);
        f.net_realized_pnl_usd = -55.0 - 10.0;
        state.observe_fill(f);
        close(
            number(&state.finish(finalization(MINUTE_MS)), Metric::LossProfit),
            0.7,
        );
    }

    #[test]
    fn short_contribution_preserves_losing_sides_and_rejects_nonpositive_combined_profit() {
        let state = flat(&[Metric::ShortContribution], MINUTE_MS);
        for (profits, expected) in [
            ([100.0, 0.0], 0.0),
            ([0.0, 100.0], 1.0),
            ([100.0, 100.0], 0.5),
            ([900.0, 100.0], 0.1),
            ([100.0, 900.0], 0.9),
            ([1000.0, -100.0], -1.0 / 9.0),
            ([-100.0, 1000.0], 10.0 / 9.0),
        ] {
            let mut f = finalization(MINUTE_MS);
            f.net_side_profit_usd = profits;
            close(
                number(&state.finish(f), Metric::ShortContribution),
                expected,
            );
        }
        for profits in [[0.0, 0.0], [-100.0, -100.0], [100.0, -100.0]] {
            let mut f = finalization(MINUTE_MS);
            f.net_side_profit_usd = profits;
            assert_eq!(
                value(&state.finish(f), Metric::ShortContribution).status,
                MetricStatus::Undefined
            );
        }
    }

    #[test]
    fn selected_metrics_ignore_unrequested_invalid_inputs_and_match_all_core_values() {
        let end = 2 * MINUTE_MS;
        let mut all = CoreMetrics::new(
            MetricRequest::all(),
            EvaluationWindow {
                start_timestamp_ms: 0,
                intended_end_timestamp_ms: end,
            },
            100.0,
            Some(180.0),
        )
        .unwrap();
        let mut selected = new(&[Metric::Pain, Metric::FillGapMax, Metric::LossProfit], end);
        for state in [&mut all, &mut selected] {
            state.observe_minute(MINUTE_MS, 90.0);
            state.observe_minute(end, 100.0);
            let mut f = fill(MINUTE_MS, 0, PositionSide::Long, 0.0, 0.0);
            f.net_realized_pnl_usd = 100.0;
            state.observe_fill(f);
        }
        let a = all.finish(finalization(end));
        let s = selected.finish(finalization(end));
        assert_eq!(s.metrics.len(), 3);
        for (&name, &v) in &s.metrics {
            assert_eq!(v, a.metrics[name]);
        }
        let mut only_cagr = new(&[Metric::Cagr], MINUTE_MS);
        let mut f = fill(0, 0, PositionSide::Long, f64::NAN, f64::NAN);
        f.notional_usd = f64::NAN;
        f.net_realized_pnl_usd = f64::NAN;
        f.pre_fill_balance_usd = -1.0;
        only_cagr.observe_fill(f);
        only_cagr.observe_minute(MINUTE_MS, 100.0);
        close(
            number(&only_cagr.finish(finalization(MINUTE_MS)), Metric::Cagr),
            0.0,
        );
    }

    #[test]
    fn long_flat_stream_uses_fixed_state_and_missing_minute_support_is_invalid() {
        let end = 100_000 * MINUTE_MS;
        let state = flat(&[Metric::Pain, Metric::HoldingMax], end);
        assert!(state.holding.open.is_empty());
        let result = state.finish(finalization(end));
        assert_eq!(result.metadata.equity_observation_count, 100_001);
        close(number(&result, Metric::Pain), 0.0);
        let mut missing = new(&[Metric::Cagr], 2 * MINUTE_MS);
        missing.observe_minute(2 * MINUTE_MS, 100.0);
        assert_eq!(
            value(&missing.finish(finalization(2 * MINUTE_MS)), Metric::Cagr).status,
            MetricStatus::Invalid
        );
    }
    fn summary_of(state: &CoreMetrics, end: u64) -> CoreMetricsSummary {
        let mut e = state.clone();
        e.commit_equity();
        let mut exposure = e.exposure.clone();
        let exposure_end = exposure.last_timestamp_ms;
        exposure.commit(e.request);
        let holding = e.holding.with_tails(end);
        CoreMetricsSummary {
            equity: EquitySummary {
                end_timestamp_ms: end,
                final_strategy_equity: e.previous_equity.unwrap().1,
                observation_count: e.count,
                clock_valid: !e.clock_invalid,
                all_equity_finite: !e.equity_invalid,
                all_equity_positive: !e.equity_nonpositive,
                drawdown_sum: e.drawdown_sum,
                drawdown_squares: e.drawdown_squares,
                drawdown_max: e.drawdown_max,
                recovery_max_ms: e
                    .recovery_max_ms
                    .max(e.underwater_start.map_or(0, |t| end - t)),
                peak_recovery_tail_censored: e.peak_recovery_tail_censored,
                ew_log_sum: e.ew_log_sum,
                ew_days_sum: e.ew_days_sum,
                ew_valid: !e.ew_invalid,
            },
            daily: DailySummary {
                first_close_timestamp_ms: e.daily.first,
                last_close_timestamp_ms: e.daily.previous.map(|(t, _)| t),
                return_count: e.daily.count,
                mean: e.daily.mean,
                m2: e.daily.m2,
                downside_squares: e.daily.downside_squares,
                gains: e.daily.gains,
                losses: e.daily.losses,
                valid: !e.daily.invalid,
            },
            fills: FillSummary {
                count: e.fill_count,
                last_timestamp_ms: e.last_fill_timestamp_ms,
                max_gap_ms: e.fill_gap_max_ms,
                turnover_sum: e.turnover_sum,
                realized_profit: e.realized_profit,
                realized_loss: e.realized_loss,
                clock_valid: !e.fill_clock_invalid,
                turnover_valid: !e.turnover_invalid,
                realized_valid: !e.realized_invalid,
            },
            holding: HoldingSummary {
                count: holding.count,
                sum_hours: holding.sum_hours,
                sum_squares: holding.sum_squares,
                max_hours: holding.max_hours,
                valid: !holding.invalid,
            },
            exposure: ExposureSummary {
                first_timestamp_ms: exposure.first,
                last_timestamp_ms: exposure_end,
                observation_count: exposure.count,
                gross_squares: exposure.gross_squares,
                gross_max: exposure.gross_max,
                utilization_sums: exposure.utilization_sums,
                clock_valid: !exposure.invalid_clock,
                gross_valid: !exposure.invalid_gross,
                utilization_valid: exposure.invalid_utilization.map(|v| !v),
            },
        }
    }
    #[test]
    fn completed_sufficient_statistics_reuse_the_exact_streaming_finalizer() {
        let levels = [100.0, 110.0, 99.0, 108.9];
        let end = 3 * DAY_MS;
        let mut state = daily(&levels, &METRIC_CATALOG.map(|d| d.metric));
        for t in (0..=end).step_by(MINUTE_MS as usize) {
            let mut x = exposure(t, 2.0, [0.25, 0.5]);
            x.strategy_equity = levels[(t / DAY_MS) as usize];
            state.observe_exposure(x);
        }
        state.observe_fill(fill(MINUTE_MS, 0, PositionSide::Long, 0.0, 1.0));
        state.observe_fill(fill(DAY_MS, 0, PositionSide::Long, 1.0, 0.0));
        let summary = summary_of(&state, end);
        let reconstructed = CoreMetrics::from_summary(
            MetricRequest::all(),
            state.window,
            state.initial_equity,
            state.half_life_days,
            MINUTE_MS,
            summary.clone(),
        )
        .unwrap();
        assert_eq!(
            state.finish(finalization(end)).metrics,
            reconstructed.finish(finalization(end)).metrics
        );
        assert_eq!(
            state
                .finish(finalization(end))
                .metadata
                .equity_observation_count,
            reconstructed
                .finish(finalization(end))
                .metadata
                .equity_observation_count
        );
        let mut bad = summary;
        bad.equity.observation_count -= 1;
        let bad = CoreMetrics::from_summary(
            request(&[Metric::Cagr]),
            state.window,
            100.0,
            None,
            MINUTE_MS,
            bad,
        )
        .unwrap();
        assert_eq!(
            value(&bad.finish(finalization(end)), Metric::Cagr).status,
            MetricStatus::Invalid
        );
    }
    #[test]
    fn coarser_statistics_are_labeled_and_minute_only_quantities_are_unavailable() {
        let request = request(&[Metric::Cagr, Metric::Pain, Metric::Volatility]);
        let end = 2 * DAY_MS;
        let mut state = CoreMetrics::new_with_interval(
            request,
            EvaluationWindow {
                start_timestamp_ms: 0,
                intended_end_timestamp_ms: end,
            },
            100.0,
            None,
            5 * MINUTE_MS,
        )
        .unwrap();
        for t in (5 * MINUTE_MS..=end).step_by((5 * MINUTE_MS) as usize) {
            state.observe_minute(t, 100.0);
        }
        for t in [0, DAY_MS, 2 * DAY_MS] {
            state.observe_daily_close(t, 100.0);
        }
        let result = state.finish(finalization(end));
        assert_eq!(result.metadata.observation_interval_ms, 5 * MINUTE_MS);
        close(number(&result, Metric::Cagr), 0.0);
        close(number(&result, Metric::Volatility), 0.0);
        assert_eq!(
            value(&result, Metric::Pain).reason,
            Some("unsupported_observation_interval")
        );
        for d in METRIC_CATALOG {
            assert_eq!(
                d.requires_minute_observations,
                d.metric.requires_minute_observations()
            );
        }
    }

    #[test]
    fn recovery_tail_censoring_distinguishes_flat_recovered_and_unfinished_curves() {
        for (curve, censored, recovery_minutes) in [
            ([100.0, 100.0, 100.0], false, 0.0),
            ([90.0, 95.0, 100.0], false, 3.0),
            ([90.0, 95.0, 99.0], true, 3.0),
        ] {
            let mut state = new(&[Metric::PeakRecoveryMax], 3 * MINUTE_MS);
            for (i, equity) in curve.into_iter().enumerate() {
                state.observe_minute((i as u64 + 1) * MINUTE_MS, equity);
            }
            let result = state.finish(finalization(3 * MINUTE_MS));
            assert_eq!(result.metadata.peak_recovery_tail_censored, Some(censored));
            close(
                number(&result, Metric::PeakRecoveryMax),
                recovery_minutes / 1440.0,
            );
        }
        let mut unselected = new(&[Metric::Pain], MINUTE_MS);
        unselected.observe_minute(MINUTE_MS, 90.0);
        assert_eq!(
            unselected
                .finish(finalization(MINUTE_MS))
                .metadata
                .peak_recovery_tail_censored,
            None
        );
        let mut invalid = new(&[Metric::PeakRecoveryMax], MINUTE_MS);
        invalid.observe_minute(MINUTE_MS, f64::NAN);
        assert_eq!(
            invalid
                .finish(finalization(MINUTE_MS))
                .metadata
                .peak_recovery_tail_censored,
            None
        );
    }

    #[test]
    fn imported_recovery_tail_preserves_producer_censoring_without_inventing_duration() {
        let metrics = [Metric::PeakRecoveryMax, Metric::Pain];
        let mut original = new(&metrics, 3 * MINUTE_MS);
        for (i, equity) in [100.0, 90.0, 95.0].into_iter().enumerate() {
            original.observe_minute((i as u64 + 1) * MINUTE_MS, equity);
        }
        let summary = summary_of(&original, 3 * MINUTE_MS);
        assert_eq!(summary.equity.peak_recovery_tail_censored, Some(true));
        let restore = |summary| {
            CoreMetrics::from_summary(
                request(&metrics),
                original.window,
                original.initial_equity,
                None,
                MINUTE_MS,
                summary,
            )
            .unwrap()
            .finish(finalization(3 * MINUTE_MS))
        };
        let imported = restore(summary.clone());
        assert_eq!(imported.metadata.peak_recovery_tail_censored, Some(true));
        assert_eq!(
            imported.metrics,
            original.finish(finalization(3 * MINUTE_MS)).metrics
        );
        let mut missing_tail = summary;
        missing_tail.equity.peak_recovery_tail_censored = None;
        let incomplete = restore(missing_tail);
        assert_eq!(
            value(&incomplete, Metric::PeakRecoveryMax).status,
            MetricStatus::Invalid
        );
        assert_eq!(incomplete.metadata.peak_recovery_tail_censored, None);
        assert_eq!(
            value(&incomplete, Metric::Pain),
            value(&imported, Metric::Pain)
        );
    }
}
