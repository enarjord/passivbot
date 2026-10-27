//! Compact GPU observations, finalized by the ordinary Rust core metric owner.
//!
//! The packet contains sufficient statistics only. Its order and shader writes
//! are generated here; Python must not duplicate its layout or metric formulas.

use crate::core_metrics::{
    CoreMetrics, CoreMetricsResult, CoreMetricsSummary, DailySummary, EquitySummary,
    EvaluationWindow, ExposureSummary, FillSummary, Finalization, HoldingSummary, HslSummary,
    Metric, MetricRequest, CORE_METRIC_DEFINITION_VERSION, METRIC_CATALOG, MINUTE_MS,
};

pub const OBSERVATION_FIELDS: &[&str] = &[
    "start_step",
    "end_step",
    "initial_equity",
    "last_equity",
    "minute_count",
    "clock_invalid",
    "equity_invalid",
    "equity_nonpositive",
    "drawdown_sum",
    "drawdown_squares",
    "drawdown_max",
    "recovery_max_steps",
    "ew_log_sum",
    "ew_days_sum",
    "ew_invalid",
    "daily_first_step",
    "daily_last_step",
    "daily_count",
    "daily_mean",
    "daily_m2",
    "daily_downside_squares",
    "daily_gains",
    "daily_losses",
    "daily_invalid",
    "fill_count",
    "last_fill_step",
    "fill_gap_max_steps",
    "fill_invalid",
    "turnover_sum",
    "turnover_invalid",
    "realized_profit",
    "realized_loss",
    "realized_invalid",
    "holding_count",
    "holding_sum_hours",
    "holding_sum_squares",
    "holding_max_hours",
    "holding_invalid",
    "exposure_count",
    "gross_squares",
    "gross_max",
    "gross_invalid",
    "utilization_long_sum",
    "utilization_short_sum",
    "utilization_long_invalid",
    "utilization_short_invalid",
    "net_profit_long",
    "net_profit_short",
    "hsl_enabled",
    "hsl_trigger_count",
    "hsl_episode_count",
    "hsl_episode_sum",
    "hsl_episode_max",
    "hsl_episode_invalid",
    "hsl_red_steps",
    "hsl_observed_steps",
    "hsl_invalid",
    "liquidated",
    "peak_recovery_tail_censored",
];

pub fn observation_catalog() -> serde_json::Value {
    let bits = METRIC_CATALOG
        .iter()
        .map(|entry| (entry.name, 1u32 << entry.metric as u8))
        .collect::<std::collections::BTreeMap<_, _>>();
    serde_json::json!({"definition_version": CORE_METRIC_DEFINITION_VERSION,
        "fields": OBSERVATION_FIELDS, "metric_bits": bits})
}

pub fn shader_source() -> String {
    let fields = OBSERVATION_FIELDS
        .iter()
        .map(|name| format!("    float {name};\n"))
        .collect::<String>();
    let initialize = OBSERVATION_FIELDS
        .iter()
        .map(|name| format!("    s.{name} = 0.0f;\n"))
        .collect::<String>();
    let writes = OBSERVATION_FIELDS
        .iter()
        .enumerate()
        .map(|(index, name)| format!("    output[offset + {index}] = s.{name};\n"))
        .collect::<String>();
    let ids = [
        ("CAGR", Metric::Cagr),
        ("VOLATILITY", Metric::Volatility),
        ("SHARPE", Metric::Sharpe),
        ("SORTINO", Metric::Sortino),
        ("OMEGA", Metric::Omega),
        ("DRAWDOWN", Metric::DrawdownMax),
        ("ULCER", Metric::Ulcer),
        ("PAIN", Metric::Pain),
        ("EW", Metric::CagrEw),
        ("RECOVERY", Metric::PeakRecoveryMax),
        ("TURNOVER", Metric::Turnover),
        ("LOSS_PROFIT", Metric::LossProfit),
        ("GROSS_RMS", Metric::GrossExposureRms),
        ("GROSS_MAX", Metric::GrossExposureMax),
        ("UTIL_LONG", Metric::PositionUtilizationLong),
        ("UTIL_SHORT", Metric::PositionUtilizationShort),
        ("HOLD_MAX", Metric::HoldingMax),
        ("HOLD_MEAN", Metric::HoldingTimeWeightedMean),
    ]
    .into_iter()
    .map(|(name, metric)| format!("#define CORE_ID_{name} {}\n", metric as u8))
    .collect::<String>();
    format!(
        "{ids}\n#define CORE_GPU_PACKET_COLS {}\n{}",
        OBSERVATION_FIELDS.len(),
        include_str!("gpu/mps_core_metrics.metal")
            .replace("// CORE_OBSERVATION_FIELDS", &fields)
            .replace("// CORE_INITIALIZE_FIELDS", &initialize)
            .replace("// CORE_WRITE_FIELDS", &writes)
    )
}

struct Packet<'a>(&'a [f32]);
impl Packet<'_> {
    fn get(&self, name: &str) -> f64 {
        self.0[OBSERVATION_FIELDS
            .iter()
            .position(|field| *field == name)
            .unwrap()] as f64
    }
    fn count(&self, name: &str) -> Result<u64, String> {
        let value = self.get(name);
        if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= (1u64 << 24) as f64
        {
            Ok(value as u64)
        } else {
            Err(format!("invalid GPU integer observation {name}"))
        }
    }
    fn flag(&self, name: &str) -> Result<bool, String> {
        match self.get(name) {
            0.0 => Ok(false),
            1.0 => Ok(true),
            _ => Err(format!("invalid GPU flag {name}")),
        }
    }
    fn timestamp(&self, name: &str, origin: u64, interval: u64) -> Result<u64, String> {
        self.count(name)?
            .checked_mul(interval)
            .and_then(|x| origin.checked_add(x))
            .ok_or_else(|| format!("GPU clock overflow for {name}"))
    }
    fn optional_timestamp(
        &self,
        name: &str,
        origin: u64,
        interval: u64,
    ) -> Result<Option<u64>, String> {
        if self.get(name) == -1.0 {
            Ok(None)
        } else {
            self.timestamp(name, origin, interval).map(Some)
        }
    }
    fn duration(&self, name: &str, interval: u64) -> Result<u64, String> {
        self.count(name)?
            .checked_mul(interval)
            .ok_or_else(|| format!("GPU duration overflow for {name}"))
    }
}

/// Each row is an immutable, completed packet from one actual GPU replay.
/// Returns typed core results; the Python boundary only flattens their values
/// and status metadata, without changing arithmetic or unavailability policy.
pub fn finalize_observations(
    values: &[f32],
    batch_size: usize,
    metric_names: &[String],
    origin_timestamp_ms: u64,
    observation_interval_ms: u64,
    intended_end_timestamp_ms: u64,
    ew_half_life_days: Option<f64>,
    hsl_effective_policy_enabled: Option<&[bool]>,
) -> Result<Vec<(CoreMetricsResult, bool)>, String> {
    if observation_interval_ms != MINUTE_MS {
        return Err("core GPU metric observations currently require 1m replay".into());
    }
    if values.len()
        != batch_size
            .checked_mul(OBSERVATION_FIELDS.len())
            .ok_or("GPU observation shape overflow")?
    {
        return Err("invalid GPU core observation packet shape".into());
    }
    let request = MetricRequest::from_names(metric_names.iter().map(String::as_str))?;
    if hsl_effective_policy_enabled.is_some_and(|flags| flags.len() != batch_size) {
        return Err("GPU effective HSL policy flag count differs from candidate count".into());
    }
    values
        .chunks_exact(OBSERVATION_FIELDS.len())
        .enumerate()
        .map(|(index, row)| {
            let p = Packet(row);
            let start = p.timestamp("start_step", origin_timestamp_ms, observation_interval_ms)?;
            let end = p.timestamp("end_step", origin_timestamp_ms, observation_interval_ms)?;
            let summary = CoreMetricsSummary {
                equity: EquitySummary {
                    end_timestamp_ms: end,
                    final_strategy_equity: p.get("last_equity"),
                    observation_count: p.count("minute_count")?,
                    clock_valid: !p.flag("clock_invalid")?,
                    all_equity_finite: !p.flag("equity_invalid")?,
                    all_equity_positive: !p.flag("equity_nonpositive")?,
                    drawdown_sum: p.get("drawdown_sum"),
                    drawdown_squares: p.get("drawdown_squares"),
                    drawdown_max: p.get("drawdown_max"),
                    recovery_max_ms: p.duration("recovery_max_steps", observation_interval_ms)?,
                    peak_recovery_tail_censored: Some(p.flag("peak_recovery_tail_censored")?),
                    ew_log_sum: p.get("ew_log_sum"),
                    ew_days_sum: p.get("ew_days_sum"),
                    ew_valid: !p.flag("ew_invalid")?,
                },
                daily: DailySummary {
                    first_close_timestamp_ms: p.optional_timestamp(
                        "daily_first_step",
                        origin_timestamp_ms,
                        observation_interval_ms,
                    )?,
                    last_close_timestamp_ms: p.optional_timestamp(
                        "daily_last_step",
                        origin_timestamp_ms,
                        observation_interval_ms,
                    )?,
                    return_count: p.count("daily_count")?,
                    mean: p.get("daily_mean"),
                    m2: p.get("daily_m2"),
                    downside_squares: p.get("daily_downside_squares"),
                    gains: p.get("daily_gains"),
                    losses: p.get("daily_losses"),
                    valid: !p.flag("daily_invalid")?,
                },
                fills: FillSummary {
                    count: p.count("fill_count")?,
                    last_timestamp_ms: p.timestamp(
                        "last_fill_step",
                        origin_timestamp_ms,
                        observation_interval_ms,
                    )?,
                    max_gap_ms: p.duration("fill_gap_max_steps", observation_interval_ms)?,
                    turnover_sum: p.get("turnover_sum"),
                    realized_profit: p.get("realized_profit"),
                    realized_loss: p.get("realized_loss"),
                    clock_valid: !p.flag("fill_invalid")?,
                    turnover_valid: !p.flag("turnover_invalid")?,
                    realized_valid: !p.flag("realized_invalid")?,
                },
                holding: HoldingSummary {
                    count: p.count("holding_count")?,
                    sum_hours: p.get("holding_sum_hours"),
                    sum_squares: p.get("holding_sum_squares"),
                    max_hours: p.get("holding_max_hours"),
                    valid: !p.flag("holding_invalid")?,
                },
                exposure: ExposureSummary {
                    first_timestamp_ms: Some(start),
                    last_timestamp_ms: Some(end),
                    observation_count: p.count("exposure_count")?,
                    gross_squares: p.get("gross_squares"),
                    gross_max: p.get("gross_max"),
                    utilization_sums: [
                        p.get("utilization_long_sum"),
                        p.get("utilization_short_sum"),
                    ],
                    clock_valid: !p.flag("clock_invalid")?,
                    gross_valid: !p.flag("gross_invalid")?,
                    utilization_valid: [
                        !p.flag("utilization_long_invalid")?,
                        !p.flag("utilization_short_invalid")?,
                    ],
                },
            };
            let state = CoreMetrics::from_summary(
                request,
                EvaluationWindow {
                    start_timestamp_ms: start,
                    intended_end_timestamp_ms,
                },
                p.get("initial_equity"),
                ew_half_life_days,
                observation_interval_ms,
                summary,
            )?;
            // Producer arithmetic overflow is an unavailable selected quantity, not
            // a reason to discard unrelated metrics from the same factual replay.
            let triggers = p.count("hsl_trigger_count").ok();
            let episodes = p.count("hsl_episode_count").ok();
            let controller_enabled = p.flag("hsl_enabled")?;
            let policy_enabled =
                hsl_effective_policy_enabled.map_or(controller_enabled, |flags| flags[index]);
            if controller_enabled && !policy_enabled {
                return Err(
                    "GPU active HSL controller contradicts effective policy metadata".into(),
                );
            }
            let mut observed_duration =
                p.duration("hsl_observed_steps", observation_interval_ms)?;
            if policy_enabled && !controller_enabled {
                // Effective enabled zero-slot scopes are known inactive for this
                // flat-start replay. Metric coverage follows the verified NAV
                // clock independently of the packed trading controller bit.
                if triggers != Some(0)
                    || episodes != Some(0)
                    || p.get("hsl_red_steps") != 0.0
                    || observed_duration != 0
                {
                    return Err("GPU inactive HSL scopes contain controller activity".into());
                }
                observed_duration = end
                    .checked_sub(start)
                    .ok_or("GPU HSL clock precedes baseline")?;
            }
            let finalization = Finalization {
                actual_end_timestamp_ms: end,
                net_side_profit_usd: [p.get("net_profit_long"), p.get("net_profit_short")],
                hsl: HslSummary {
                    enabled: policy_enabled,
                    trigger_count: triggers.unwrap_or(0),
                    execution_episode_count: episodes.unwrap_or(0),
                    episode_loss_fraction_sum: p.get("hsl_episode_sum"),
                    episode_loss_fraction_max: p.get("hsl_episode_max"),
                    episode_loss_valid: episodes.is_some() && !p.flag("hsl_episode_invalid")?,
                    red_duration_ms: p.duration("hsl_red_steps", observation_interval_ms)?,
                    observed_duration_ms: observed_duration,
                    observation_valid: triggers.is_some() && !p.flag("hsl_invalid")?,
                },
            };
            Ok((state.finish(finalization), p.flag("liquidated")?))
        })
        .collect()
}
