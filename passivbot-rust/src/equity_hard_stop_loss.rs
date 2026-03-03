#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardStopTier {
    Green,
    Yellow,
    Orange,
    Red,
}

impl Default for HardStopTier {
    fn default() -> Self {
        Self::Green
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HardStopTierRatios {
    pub yellow: f64,
    pub orange: f64,
}

impl Default for HardStopTierRatios {
    fn default() -> Self {
        Self {
            yellow: 0.5,
            orange: 0.75,
        }
    }
}

impl HardStopTierRatios {
    pub fn validate(self) -> Result<(), String> {
        if !(self.yellow.is_finite() && self.orange.is_finite()) {
            return Err("tier ratios must be finite".to_string());
        }
        if !(0.0 < self.yellow && self.yellow < self.orange && self.orange < 1.0) {
            return Err("tier ratios must satisfy 0 < yellow < orange < 1".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HardStopConfig {
    pub threshold: f64,
    pub ema_span_minutes: f64,
    pub tier_ratios: HardStopTierRatios,
}

impl HardStopConfig {
    pub fn validate(self) -> Result<(), String> {
        if !self.threshold.is_finite() || self.threshold <= 0.0 {
            return Err("threshold must be finite and > 0".to_string());
        }
        if !self.ema_span_minutes.is_finite() || self.ema_span_minutes <= 0.0 {
            return Err("ema_span_minutes must be finite and > 0".to_string());
        }
        self.tier_ratios.validate()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HardStopState {
    pub equity_peak: f64,
    pub drawdown_ema: f64,
    pub tier: HardStopTier,
    pub red_latched: bool,
    pub initialized: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct HardStopStep {
    pub drawdown_raw: f64,
    pub drawdown_ema: f64,
    pub drawdown_score: f64,
    pub tier: HardStopTier,
    pub changed: bool,
    pub red_latched: bool,
    pub span_samples: f64,
    pub alpha: f64,
}

#[derive(Debug, Clone, Default)]
pub struct RollingPeakTracker {
    peaks: VecDeque<(u64, f64)>,
    last_timestamp_ms: Option<u64>,
}

impl RollingPeakTracker {
    pub fn reset(&mut self) {
        self.peaks.clear();
        self.last_timestamp_ms = None;
    }

    pub fn len(&self) -> usize {
        self.peaks.len()
    }

    pub fn update(&mut self, timestamp_ms: u64, equity: f64, lookback_ms: u64) -> Result<f64, String> {
        if !equity.is_finite() || equity <= 0.0 {
            return Err("equity must be finite and > 0".to_string());
        }
        if lookback_ms == 0 {
            return Err("lookback_ms must be > 0".to_string());
        }
        if let Some(prev_ts) = self.last_timestamp_ms {
            if timestamp_ms < prev_ts {
                return Err(format!(
                    "timestamp_ms must be non-decreasing, got {} after {}",
                    timestamp_ms, prev_ts
                ));
            }
        }
        self.last_timestamp_ms = Some(timestamp_ms);

        while let Some((old_ts, _)) = self.peaks.front() {
            if timestamp_ms.saturating_sub(*old_ts) > lookback_ms {
                self.peaks.pop_front();
            } else {
                break;
            }
        }
        while let Some((_, peak_eq)) = self.peaks.back() {
            if *peak_eq <= equity {
                self.peaks.pop_back();
            } else {
                break;
            }
        }
        self.peaks.push_back((timestamp_ms, equity));
        Ok(self.peaks.front().map(|(_, eq)| *eq).unwrap_or(equity))
    }
}

pub fn span_samples(ema_span_minutes: f64, sample_minutes: f64) -> Result<f64, String> {
    if !ema_span_minutes.is_finite() || ema_span_minutes <= 0.0 {
        return Err("ema_span_minutes must be finite and > 0".to_string());
    }
    if !sample_minutes.is_finite() || sample_minutes <= 0.0 {
        return Err("sample_minutes must be finite and > 0".to_string());
    }
    Ok(ema_span_minutes / sample_minutes)
}

pub fn step(
    state: &mut HardStopState,
    config: HardStopConfig,
    equity: f64,
    sample_minutes: f64,
) -> Result<HardStopStep, String> {
    let next_equity_peak = if !state.initialized {
        equity
    } else {
        state.equity_peak.max(equity)
    };
    step_with_equity_peak(state, config, equity, next_equity_peak, sample_minutes)
}

pub fn step_with_equity_peak(
    state: &mut HardStopState,
    config: HardStopConfig,
    equity: f64,
    equity_peak: f64,
    sample_minutes: f64,
) -> Result<HardStopStep, String> {
    config.validate()?;
    if !equity.is_finite() || equity <= 0.0 {
        return Err("equity must be finite and > 0".to_string());
    }
    if !equity_peak.is_finite() || equity_peak <= 0.0 {
        return Err("equity_peak must be finite and > 0".to_string());
    }
    if equity_peak + f64::EPSILON < equity {
        return Err("equity_peak must be >= equity".to_string());
    }
    let span_samples = span_samples(config.ema_span_minutes, sample_minutes)?;
    let alpha = 2.0 / (span_samples + 1.0);
    if !alpha.is_finite() || !(0.0 < alpha && alpha <= 1.0) {
        return Err("computed alpha is invalid".to_string());
    }

    let prev_tier = state.tier;
    if !state.initialized {
        state.initialized = true;
        state.equity_peak = equity_peak;
        state.drawdown_ema = 0.0;
        state.tier = if state.red_latched {
            HardStopTier::Red
        } else {
            HardStopTier::Green
        };
        return Ok(HardStopStep {
            drawdown_raw: 0.0,
            drawdown_ema: 0.0,
            drawdown_score: 0.0,
            tier: state.tier,
            changed: state.tier != prev_tier,
            red_latched: state.red_latched,
            span_samples,
            alpha,
        });
    }

    state.equity_peak = equity_peak;
    let drawdown_raw = (1.0 - (equity / state.equity_peak.max(f64::EPSILON))).max(0.0);
    state.drawdown_ema = alpha * drawdown_raw + (1.0 - alpha) * state.drawdown_ema;
    let drawdown_score = drawdown_raw.min(state.drawdown_ema);

    let threshold_yellow = config.tier_ratios.yellow * config.threshold;
    let threshold_orange = config.tier_ratios.orange * config.threshold;
    let next_tier = if state.red_latched {
        HardStopTier::Red
    } else if drawdown_score >= config.threshold {
        HardStopTier::Red
    } else if drawdown_score >= threshold_orange {
        HardStopTier::Orange
    } else if drawdown_score >= threshold_yellow {
        HardStopTier::Yellow
    } else {
        HardStopTier::Green
    };
    if next_tier == HardStopTier::Red {
        state.red_latched = true;
    }
    state.tier = if state.red_latched {
        HardStopTier::Red
    } else {
        next_tier
    };
    Ok(HardStopStep {
        drawdown_raw,
        drawdown_ema: state.drawdown_ema,
        drawdown_score,
        tier: state.tier,
        changed: state.tier != prev_tier,
        red_latched: state.red_latched,
        span_samples,
        alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> HardStopConfig {
        HardStopConfig {
            threshold: 0.25,
            ema_span_minutes: 60.0,
            tier_ratios: HardStopTierRatios {
                yellow: 0.5,
                orange: 0.75,
            },
        }
    }

    #[test]
    fn minutes_to_samples_keeps_float_precision() {
        let samples = span_samples(47.5, 2.0).unwrap();
        assert!((samples - 23.75).abs() < 1e-12);
    }

    #[test]
    fn tier_boundaries_follow_configurable_ratios() {
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            threshold: 0.2,
            ema_span_minutes: 1.0, // one-sample EMA for direct threshold checks
            tier_ratios: HardStopTierRatios {
                yellow: 0.4,
                orange: 0.8,
            },
        };
        // initialize at equity peak
        let _ = step(&mut state, config, 100.0, 1.0).unwrap();
        // 8% dd => yellow (0.4 * 0.2 = 0.08)
        let s1 = step(&mut state, config, 92.0, 1.0).unwrap();
        assert_eq!(s1.tier, HardStopTier::Yellow);
        // 16% dd => orange (0.8 * 0.2 = 0.16)
        let s2 = step(&mut state, config, 84.0, 1.0).unwrap();
        assert_eq!(s2.tier, HardStopTier::Orange);
        // 20% dd => red
        let s3 = step(&mut state, config, 80.0, 1.0).unwrap();
        assert_eq!(s3.tier, HardStopTier::Red);
    }

    #[test]
    fn min_raw_and_ema_prevents_stale_ema_red_after_recovery() {
        let mut state = HardStopState::default();
        let config = cfg();

        let _ = step(&mut state, config, 100.0, 1.0).unwrap();
        // slow drop builds EMA
        for _ in 0..30 {
            let _ = step(&mut state, config, 80.0, 1.0).unwrap();
        }
        // recover strongly: raw drawdown low, ema still high.
        let recovered = step(&mut state, config, 98.0, 1.0).unwrap();
        assert!(recovered.drawdown_raw < config.threshold);
        assert!(recovered.drawdown_ema > config.threshold * 0.5);
        assert!(recovered.drawdown_score < config.threshold);
        assert_ne!(recovered.tier, HardStopTier::Red);
    }

    #[test]
    fn red_is_latched_once_triggered() {
        let mut state = HardStopState::default();
        let config = cfg();
        let _ = step(&mut state, config, 100.0, 1.0).unwrap();
        let red = step(&mut state, config, 60.0, 1.0).unwrap();
        assert_eq!(red.tier, HardStopTier::Red);
        assert!(red.red_latched);
        let after_recovery = step(&mut state, config, 100.0, 1.0).unwrap();
        assert_eq!(after_recovery.tier, HardStopTier::Red);
        assert!(after_recovery.red_latched);
    }

    #[test]
    fn rolling_peak_override_allows_peak_to_decrease() {
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            threshold: 0.25,
            ema_span_minutes: 1.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        let _ = step_with_equity_peak(&mut state, config, 100.0, 100.0, 1.0).unwrap();
        // Simulate old peaks aged out from lookback: peak now 90, equity 90.
        let s = step_with_equity_peak(&mut state, config, 90.0, 90.0, 1.0).unwrap();
        assert!((s.drawdown_raw - 0.0).abs() < 1e-12);
        assert!((state.equity_peak - 90.0).abs() < 1e-12);
    }

    #[test]
    fn preexisting_red_latch_is_preserved_on_first_sample() {
        let config = cfg();
        let mut state = HardStopState {
            red_latched: true,
            ..Default::default()
        };
        let first = step(&mut state, config, 100.0, 1.0).unwrap();
        assert_eq!(first.tier, HardStopTier::Red);
        assert!(first.red_latched);
    }

    #[test]
    fn rolling_peak_tracker_enforces_lookback_window() {
        let mut tracker = RollingPeakTracker::default();
        let lookback_ms = 1_000;
        let p0 = tracker.update(1_000, 100.0, lookback_ms).unwrap();
        assert!((p0 - 100.0).abs() < 1e-12);
        let p1 = tracker.update(1_500, 90.0, lookback_ms).unwrap();
        assert!((p1 - 100.0).abs() < 1e-12);
        let p2 = tracker.update(2_100, 95.0, lookback_ms).unwrap();
        assert!((p2 - 95.0).abs() < 1e-12);
    }

    #[test]
    fn rolling_peak_tracker_rejects_out_of_order_timestamps() {
        let mut tracker = RollingPeakTracker::default();
        let _ = tracker.update(1_000, 100.0, 1_000).unwrap();
        let err = tracker.update(999, 101.0, 1_000).unwrap_err();
        assert!(err.contains("non-decreasing"));
    }
}
use std::collections::VecDeque;
