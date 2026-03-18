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
    pub red_threshold: f64,
    pub ema_span_minutes: f64,
    pub tier_ratios: HardStopTierRatios,
}

impl HardStopConfig {
    pub fn validate(self) -> Result<(), String> {
        if !self.red_threshold.is_finite() || self.red_threshold <= 0.0 {
            return Err("red_threshold must be finite and > 0".to_string());
        }
        if !self.ema_span_minutes.is_finite() || self.ema_span_minutes <= 0.0 {
            return Err("ema_span_minutes must be finite and > 0".to_string());
        }
        self.tier_ratios.validate()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HardStopState {
    pub peak_strategy_equity: f64,
    pub drawdown_ema: f64,
    pub tier: HardStopTier,
    pub red_latched: bool,
    pub initialized: bool,
    /// Minute number of the last processed sample (timestamp_ms / 60_000).
    /// `None` until the first sample is processed.
    pub last_minute: Option<u64>,
    /// Cached step result returned on intra-minute re-calls.
    pub cached_step: Option<HardStopStep>,
}

#[derive(Debug, Clone, Copy)]
pub struct HardStopStep {
    pub drawdown_raw: f64,
    pub drawdown_score: f64,
    pub tier: HardStopTier,
    pub changed: bool,
    pub alpha: f64,
    /// Number of whole-minute steps applied in this call.
    /// 0 for intra-minute cached returns.
    pub elapsed_minutes: u64,
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

    pub fn update(
        &mut self,
        timestamp_ms: u64,
        equity: f64,
        lookback_ms: u64,
    ) -> Result<f64, String> {
        if !equity.is_finite() {
            return Err("equity must be finite".to_string());
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

/// Convenience helper that tracks peak internally (peak = max of all equity seen).
#[allow(dead_code)]
pub fn step(
    state: &mut HardStopState,
    config: HardStopConfig,
    equity: f64,
    timestamp_ms: u64,
) -> Result<HardStopStep, String> {
    let next_peak_strategy_equity = if !state.initialized {
        equity
    } else {
        state.peak_strategy_equity.max(equity)
    };
    step_with_peak_strategy_equity(
        state,
        config,
        equity,
        next_peak_strategy_equity,
        timestamp_ms,
    )
}

/// Core minute-quantized HSL step.
///
/// Timing is derived from `timestamp_ms`:
/// - Quantized to whole minutes (`timestamp_ms / 60_000`).
/// - If the minute has not advanced since the last call, returns a cached
///   result with `elapsed_minutes = 0` and no state mutation.
/// - If N >= 1 minutes have elapsed, applies the EMA update using a
///   closed-form multi-step formula: `decay = (1 - alpha).powf(N)`.
///
/// This gives exact parity with the backtester (which calls once per candle
/// with candle-aligned timestamps) regardless of how often the live bot
/// invokes this function.
pub fn step_with_peak_strategy_equity(
    state: &mut HardStopState,
    config: HardStopConfig,
    equity: f64,
    peak_strategy_equity: f64,
    timestamp_ms: u64,
) -> Result<HardStopStep, String> {
    config.validate()?;
    if !equity.is_finite() || equity <= 0.0 {
        return Err("equity must be finite and > 0".to_string());
    }
    if !peak_strategy_equity.is_finite() || peak_strategy_equity <= 0.0 {
        return Err("peak_strategy_equity must be finite and > 0".to_string());
    }
    if peak_strategy_equity + f64::EPSILON < equity {
        return Err("peak_strategy_equity must be >= equity".to_string());
    }
    let alpha = (2.0 / (config.ema_span_minutes + 1.0)).min(1.0);
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err("computed alpha is invalid".to_string());
    }

    let current_minute = timestamp_ms / 60_000;

    let prev_tier = state.tier;
    if !state.initialized {
        state.initialized = true;
        state.peak_strategy_equity = peak_strategy_equity;
        state.drawdown_ema = 0.0;
        state.last_minute = Some(current_minute);
        state.tier = if state.red_latched {
            HardStopTier::Red
        } else {
            HardStopTier::Green
        };
        let step = HardStopStep {
            drawdown_raw: 0.0,
            drawdown_score: 0.0,
            tier: state.tier,
            changed: state.tier != prev_tier,
            alpha,
            elapsed_minutes: 0,
        };
        state.cached_step = Some(step);
        return Ok(step);
    }

    // Enforce monotonic timestamps (at minute resolution).
    let last_minute = state.last_minute.ok_or_else(|| {
        "invariant violation: initialized but last_minute is None".to_string()
    })?;
    if current_minute < last_minute {
        return Err(format!(
            "timestamp must be non-decreasing: minute {} < last {}",
            current_minute, last_minute
        ));
    }
    let elapsed_minutes = current_minute - last_minute;

    // Intra-minute re-call: return cached result, no state mutation.
    if elapsed_minutes == 0 {
        let mut cached = state.cached_step.ok_or_else(|| {
            "invariant violation: initialized but cached_step is None".to_string()
        })?;
        cached.elapsed_minutes = 0;
        return Ok(cached);
    }

    // --- Minute boundary crossed: full update ---
    state.peak_strategy_equity = peak_strategy_equity;
    let drawdown_raw = (1.0 - (equity / state.peak_strategy_equity.max(f64::EPSILON))).max(0.0);
    // Closed-form N-step EMA: equivalent to applying alpha N times with constant input.
    let decay = (1.0 - alpha).powf(elapsed_minutes as f64);
    state.drawdown_ema = drawdown_raw + (state.drawdown_ema - drawdown_raw) * decay;
    // Effective trigger metric: min(raw, EMA).
    // Prevents false RED after recovery (stale EMA) and flash-crash bottom exits (raw spike).
    let drawdown_score = drawdown_raw.min(state.drawdown_ema);

    let threshold_yellow = config.tier_ratios.yellow * config.red_threshold;
    let threshold_orange = config.tier_ratios.orange * config.red_threshold;
    let cmp_eps = 1e-12;
    let next_tier = if state.red_latched {
        HardStopTier::Red
    } else if drawdown_score + cmp_eps >= config.red_threshold {
        HardStopTier::Red
    } else if drawdown_score + cmp_eps >= threshold_orange {
        HardStopTier::Orange
    } else if drawdown_score + cmp_eps >= threshold_yellow {
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
    state.last_minute = Some(current_minute);
    let step = HardStopStep {
        drawdown_raw,
        drawdown_score,
        tier: state.tier,
        changed: state.tier != prev_tier,
        alpha,
        elapsed_minutes,
    };
    state.cached_step = Some(step);
    Ok(step)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN_MS: u64 = 60_000;

    fn cfg() -> HardStopConfig {
        HardStopConfig {
            red_threshold: 0.25,
            ema_span_minutes: 60.0,
            tier_ratios: HardStopTierRatios {
                yellow: 0.5,
                orange: 0.75,
            },
        }
    }

    #[test]
    fn tier_boundaries_follow_configurable_ratios() {
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            red_threshold: 0.2,
            ema_span_minutes: 1.0, // alpha=1.0 → EMA tracks raw exactly
            tier_ratios: HardStopTierRatios {
                yellow: 0.4,
                orange: 0.8,
            },
        };
        // initialize at minute 0
        let _ = step(&mut state, config, 100.0, 0).unwrap();
        // 8% dd => yellow (0.4 * 0.2 = 0.08)
        let s1 = step(&mut state, config, 92.0, MIN_MS).unwrap();
        assert_eq!(s1.tier, HardStopTier::Yellow);
        // 16% dd => orange (0.8 * 0.2 = 0.16)
        let s2 = step(&mut state, config, 84.0, 2 * MIN_MS).unwrap();
        assert_eq!(s2.tier, HardStopTier::Orange);
        // 20% dd => red
        let s3 = step(&mut state, config, 80.0, 3 * MIN_MS).unwrap();
        assert_eq!(s3.tier, HardStopTier::Red);
    }

    #[test]
    fn small_ema_span_clamps_alpha_to_one() {
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            red_threshold: 0.2,
            ema_span_minutes: 0.5, // alpha = 2/1.5 = 1.33 → clamped to 1.0
            tier_ratios: HardStopTierRatios::default(),
        };
        let _ = step(&mut state, config, 100.0, 0).unwrap();
        let s = step(&mut state, config, 90.0, MIN_MS).unwrap();
        assert!((s.alpha - 1.0).abs() < 1e-12);
        // With alpha=1.0, EMA = raw immediately
        assert!((state.drawdown_ema - s.drawdown_raw).abs() < 1e-12);
        assert!((s.drawdown_score - s.drawdown_raw).abs() < 1e-12);
    }

    #[test]
    fn score_equals_min_raw_ema() {
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            red_threshold: 0.2,
            ema_span_minutes: 15.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        let _ = step(&mut state, config, 100.0, 0).unwrap();
        let s = step(&mut state, config, 90.0, MIN_MS).unwrap();
        let expected = s.drawdown_raw.min(state.drawdown_ema);
        assert!((s.drawdown_score - expected).abs() < 1e-12);
    }

    #[test]
    fn score_uses_raw_when_ema_is_stale_after_recovery() {
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            red_threshold: 0.5,
            ema_span_minutes: 60.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        let _ = step(&mut state, config, 100.0, 0).unwrap();
        // Large drawdown to push EMA up
        let _ = step(&mut state, config, 80.0, MIN_MS).unwrap();
        let _ = step(&mut state, config, 80.0, 2 * MIN_MS).unwrap();
        // Now recover — raw goes to 0 but EMA is still elevated
        let s = step(&mut state, config, 100.0, 3 * MIN_MS).unwrap();
        assert!(
            s.drawdown_raw < state.drawdown_ema,
            "raw should be lower after recovery"
        );
        assert!(
            (s.drawdown_score - s.drawdown_raw).abs() < 1e-12,
            "score should follow raw (min)"
        );
    }

    #[test]
    fn red_is_latched_once_triggered() {
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            red_threshold: 0.25,
            ema_span_minutes: 1.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        let _ = step(&mut state, config, 100.0, 0).unwrap();
        let red = step(&mut state, config, 60.0, MIN_MS).unwrap();
        assert_eq!(red.tier, HardStopTier::Red);
        assert!(state.red_latched);
        let after_recovery = step(&mut state, config, 100.0, 2 * MIN_MS).unwrap();
        assert_eq!(after_recovery.tier, HardStopTier::Red);
        assert!(state.red_latched);
    }

    #[test]
    fn rolling_peak_override_allows_peak_to_decrease() {
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            red_threshold: 0.25,
            ema_span_minutes: 1.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        let _ =
            step_with_peak_strategy_equity(&mut state, config, 100.0, 100.0, 0).unwrap();
        // Simulate old peaks aged out from lookback: peak now 90, equity 90.
        let s =
            step_with_peak_strategy_equity(&mut state, config, 90.0, 90.0, MIN_MS).unwrap();
        assert!((s.drawdown_raw - 0.0).abs() < 1e-12);
        assert!((state.peak_strategy_equity - 90.0).abs() < 1e-12);
    }

    #[test]
    fn preexisting_red_latch_is_preserved_on_first_sample() {
        let config = cfg();
        let mut state = HardStopState {
            red_latched: true,
            ..Default::default()
        };
        let first = step(&mut state, config, 100.0, 0).unwrap();
        assert_eq!(first.tier, HardStopTier::Red);
        assert!(state.red_latched);
    }

    // --- Minute-quantization tests ---

    #[test]
    fn intra_minute_returns_cached_result() {
        let mut state = HardStopState::default();
        let config = cfg();
        let _ = step(&mut state, config, 100.0, 0).unwrap();
        let s1 = step(&mut state, config, 90.0, MIN_MS).unwrap();
        assert!(s1.elapsed_minutes == 1);
        let ema_after_s1 = state.drawdown_ema;
        // Call again within the same minute — different equity, but should be cached
        let s2 = step(&mut state, config, 80.0, MIN_MS + 30_000).unwrap();
        assert_eq!(s2.elapsed_minutes, 0);
        assert!((s2.drawdown_raw - s1.drawdown_raw).abs() < 1e-12);
        assert!((s2.drawdown_score - s1.drawdown_score).abs() < 1e-12);
        // State should not have changed
        assert!((state.drawdown_ema - ema_after_s1).abs() < 1e-12);
    }

    #[test]
    fn gap_filling_uses_closed_form() {
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            red_threshold: 0.5,
            ema_span_minutes: 60.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        let _ = step(&mut state, config, 100.0, 0).unwrap();
        // Skip 5 minutes at once
        let s = step(&mut state, config, 90.0, 5 * MIN_MS).unwrap();
        assert_eq!(s.elapsed_minutes, 5);
        let alpha: f64 = 2.0 / (60.0 + 1.0);
        let decay = (1.0_f64 - alpha).powf(5.0);
        let expected_ema = 0.1 + (0.0 - 0.1) * decay; // raw=0.1, old_ema=0
        assert!((state.drawdown_ema - expected_ema).abs() < 1e-12);
    }

    #[test]
    fn gap_filling_matches_sequential_steps() {
        let config = HardStopConfig {
            red_threshold: 0.5,
            ema_span_minutes: 60.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        // Sequential: 5 individual 1-minute steps
        let mut state_seq = HardStopState::default();
        let _ = step(&mut state_seq, config, 100.0, 0).unwrap();
        for i in 1..=5u64 {
            let _ = step(&mut state_seq, config, 90.0, i * MIN_MS).unwrap();
        }
        // Batch: one 5-minute gap
        let mut state_batch = HardStopState::default();
        let _ = step(&mut state_batch, config, 100.0, 0).unwrap();
        let _ = step(&mut state_batch, config, 90.0, 5 * MIN_MS).unwrap();
        assert!(
            (state_seq.drawdown_ema - state_batch.drawdown_ema).abs() < 1e-12,
            "sequential ({}) != batch ({})",
            state_seq.drawdown_ema,
            state_batch.drawdown_ema
        );
    }

    #[test]
    fn timestamp_regression_rejected() {
        let mut state = HardStopState::default();
        let config = cfg();
        let _ = step(&mut state, config, 100.0, 2 * MIN_MS).unwrap();
        let err = step(&mut state, config, 100.0, MIN_MS).unwrap_err();
        assert!(err.contains("non-decreasing"));
    }

    #[test]
    fn large_gap_converges_ema_to_raw() {
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            red_threshold: 0.5,
            ema_span_minutes: 60.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        let _ = step(&mut state, config, 100.0, 0).unwrap();
        // 100,000 minutes gap (~69 days) with 10% drawdown
        let s = step(&mut state, config, 90.0, 100_000 * MIN_MS).unwrap();
        assert_eq!(s.elapsed_minutes, 100_000);
        // EMA should have fully converged to drawdown_raw
        assert!((state.drawdown_ema - s.drawdown_raw).abs() < 1e-12);
    }

    #[test]
    fn first_sample_reports_zero_elapsed() {
        let mut state = HardStopState::default();
        let config = cfg();
        let s = step(&mut state, config, 100.0, 5 * MIN_MS).unwrap();
        assert_eq!(s.elapsed_minutes, 0);
        assert!((s.drawdown_raw - 0.0).abs() < 1e-12);
    }

    // --- RollingPeakTracker tests (unchanged) ---

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

    #[test]
    fn rolling_peak_tracker_accepts_negative_finite_values() {
        let mut tracker = RollingPeakTracker::default();
        let lookback_ms = 1_000;
        let p0 = tracker.update(1_000, -5.0, lookback_ms).unwrap();
        assert!((p0 - -5.0).abs() < 1e-12);
        let p1 = tracker.update(1_500, -2.0, lookback_ms).unwrap();
        assert!((p1 - -2.0).abs() < 1e-12);
        let p2 = tracker.update(2_100, -7.0, lookback_ms).unwrap();
        assert!((p2 - -2.0).abs() < 1e-12);
    }
}
use std::collections::VecDeque;
