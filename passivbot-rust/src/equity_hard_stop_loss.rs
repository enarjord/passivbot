use std::collections::VecDeque;

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
    /// True once any sample in the current episode crossed RED
    /// (`red_active_now`), regardless of latching. Post-episode
    /// cooldown/restart evidence per the clarified B2.1 contract; cleared
    /// only on episode reset. Unlike `red_latched`, this never pins the tier.
    pub red_seen_in_episode: bool,
    pub initialized: bool,
    pub last_minute: Option<u64>,
    pub cached_step: Option<HardStopStep>,
}

#[derive(Debug, Clone, Copy)]
pub struct HardStopStep {
    pub drawdown_raw: f64,
    pub drawdown_ema: f64,
    pub drawdown_score: f64,
    /// Whether THIS sample's score crosses the RED threshold, independent of
    /// latching. The only signal that may authorize new panic orders per the
    /// clarified B2.1 contract; a latched tier alone must not.
    pub red_active_now: bool,
    pub tier: HardStopTier,
    pub changed: bool,
    pub alpha: f64,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoinDrawdownSignal {
    pub slot_budget: f64,
    pub drawdown_usd: f64,
    pub drawdown_raw: f64,
}

/// Derive the shared coin-HSL drawdown signal for live and backtest callers.
///
/// HSL is a drawdown stop, not an exposure scaler. The slot budget therefore
/// uses account balance divided by the caller's applicable slot count; TWEL is
/// intentionally not an input to this contract. Live supplies configured
/// slots, while backtests may supply effective tradability-aware slots.
pub fn coin_drawdown_signal(
    balance: f64,
    n_positions: usize,
    peak_realized: f64,
    last_realized: f64,
    current_upnl: f64,
) -> Result<CoinDrawdownSignal, String> {
    if !balance.is_finite() || balance <= 0.0 {
        return Err("balance must be finite and > 0".to_string());
    }
    if n_positions == 0 {
        return Err("n_positions must be > 0".to_string());
    }
    if !peak_realized.is_finite() {
        return Err("peak_realized must be finite".to_string());
    }
    if !last_realized.is_finite() {
        return Err("last_realized must be finite".to_string());
    }
    if !current_upnl.is_finite() {
        return Err("current_upnl must be finite".to_string());
    }

    let slot_budget = balance / n_positions as f64;
    if !slot_budget.is_finite() || slot_budget <= 0.0 {
        return Err("slot_budget must be finite and > 0".to_string());
    }
    let drawdown_usd = (peak_realized - (last_realized + current_upnl)).max(0.0);
    let drawdown_raw = drawdown_usd / slot_budget;
    Ok(CoinDrawdownSignal {
        slot_budget,
        drawdown_usd,
        drawdown_raw,
    })
}

/// Whether the permanent no-restart halt trips for a finalized RED stop.
///
/// Contract (fable audit plan, clarified 2026-07-06): the no-restart trigger
/// is conservative and uses `max(drawdown_raw, drawdown_ema)` so it catches
/// either catastrophic instantaneous damage or sustained smoothed damage,
/// while the RED/panic-now tier score stays `min(raw, ema)`.
pub fn no_restart_triggered(
    restart_after_red_policy: &str,
    drawdown_raw: f64,
    drawdown_ema: f64,
    no_restart_drawdown_threshold: f64,
) -> Result<bool, String> {
    match restart_after_red_policy {
        "always" => Ok(false),
        "threshold" => Ok(drawdown_raw.max(drawdown_ema) >= no_restart_drawdown_threshold),
        "never" => Ok(true),
        raw => Err(format!(
            "hsl_restart_after_red_policy must be one of always, threshold, never; got {:?}",
            raw
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RedEpisodeFinalization {
    pub no_restart_peak_strategy_equity: f64,
    pub no_restart_drawdown_raw: f64,
    pub no_restart_latched: bool,
    pub cooldown_until_ms: Option<u64>,
    pub disposition: RedEpisodeDisposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedEpisodeDisposition {
    NoRestart,
    Cooldown,
    HaltedNoCooldown,
}

impl RedEpisodeDisposition {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoRestart => "no_restart",
            Self::Cooldown => "cooldown",
            Self::HaltedNoCooldown => "halted_no_cooldown",
        }
    }
}

/// Evaluate the state transition after a RED episode has fully flattened.
///
/// The caller owns exchange/history proof and supplies the previous
/// no-restart peak for the applicable lookback. This pure transition owns the
/// shared live/backtest policy math. A positive configured cooldown is rounded
/// to milliseconds with a canonical 1 ms floor; bar backtests observe the
/// exact deadline on their next available sample instead of redefining it.
pub fn evaluate_red_episode_finalization(
    restart_after_red_policy: &str,
    stop_timestamp_ms: u64,
    stop_equity: f64,
    stop_peak_strategy_equity: f64,
    previous_no_restart_peak_strategy_equity: f64,
    drawdown_ema: f64,
    red_threshold: f64,
    no_restart_drawdown_threshold: f64,
    cooldown_minutes_after_red: f64,
) -> Result<RedEpisodeFinalization, String> {
    if !stop_equity.is_finite() || stop_equity <= 0.0 {
        return Err("stop_equity must be finite and > 0".to_string());
    }
    if !stop_peak_strategy_equity.is_finite() || stop_peak_strategy_equity <= 0.0 {
        return Err("stop_peak_strategy_equity must be finite and > 0".to_string());
    }
    if stop_peak_strategy_equity + f64::EPSILON < stop_equity {
        return Err("stop_peak_strategy_equity must be >= stop_equity".to_string());
    }
    if !previous_no_restart_peak_strategy_equity.is_finite()
        || previous_no_restart_peak_strategy_equity < 0.0
    {
        return Err("previous_no_restart_peak_strategy_equity must be finite and >= 0".to_string());
    }
    if !drawdown_ema.is_finite() || drawdown_ema < 0.0 {
        return Err("drawdown_ema must be finite and >= 0".to_string());
    }
    if !red_threshold.is_finite() || red_threshold <= 0.0 {
        return Err("red_threshold must be finite and > 0".to_string());
    }
    if !no_restart_drawdown_threshold.is_finite()
        || !(red_threshold <= no_restart_drawdown_threshold && no_restart_drawdown_threshold <= 1.0)
    {
        return Err(
            "no_restart_drawdown_threshold must be finite and satisfy red_threshold <= threshold <= 1"
                .to_string(),
        );
    }
    if !cooldown_minutes_after_red.is_finite() || cooldown_minutes_after_red < 0.0 {
        return Err("cooldown_minutes_after_red must be finite and >= 0".to_string());
    }

    let no_restart_peak_strategy_equity = previous_no_restart_peak_strategy_equity
        .max(stop_peak_strategy_equity)
        .max(stop_equity);
    let no_restart_drawdown_raw =
        (1.0 - stop_equity / no_restart_peak_strategy_equity.max(f64::EPSILON)).max(0.0);
    let no_restart_latched = no_restart_triggered(
        restart_after_red_policy,
        no_restart_drawdown_raw,
        drawdown_ema,
        no_restart_drawdown_threshold,
    )?;
    let cooldown_until_ms = if no_restart_latched || cooldown_minutes_after_red <= 0.0 {
        None
    } else {
        let cooldown_ms_f64 = (cooldown_minutes_after_red * 60_000.0).round();
        if !cooldown_ms_f64.is_finite() || cooldown_ms_f64 > u64::MAX as f64 {
            return Err("cooldown_minutes_after_red is too large".to_string());
        }
        let cooldown_ms = (cooldown_ms_f64 as u64).max(1);
        Some(stop_timestamp_ms.saturating_add(cooldown_ms))
    };
    let disposition = if no_restart_latched {
        RedEpisodeDisposition::NoRestart
    } else if cooldown_until_ms.is_some() {
        RedEpisodeDisposition::Cooldown
    } else {
        RedEpisodeDisposition::HaltedNoCooldown
    };

    Ok(RedEpisodeFinalization {
        no_restart_peak_strategy_equity,
        no_restart_drawdown_raw,
        no_restart_latched,
        cooldown_until_ms,
        disposition,
    })
}

#[allow(dead_code)] // Kept as a convenience helper for callers that want internal peak tracking.
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

pub fn step_with_peak_strategy_equity(
    state: &mut HardStopState,
    config: HardStopConfig,
    equity: f64,
    peak_strategy_equity: f64,
    timestamp_ms: u64,
) -> Result<HardStopStep, String> {
    step_with_peak_strategy_equity_latch(
        state,
        config,
        equity,
        peak_strategy_equity,
        timestamp_ms,
        true,
    )
}

pub fn step_with_peak_strategy_equity_latch(
    state: &mut HardStopState,
    config: HardStopConfig,
    equity: f64,
    peak_strategy_equity: f64,
    timestamp_ms: u64,
    latch_red: bool,
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
    let alpha = 2.0 / (config.ema_span_minutes + 1.0);
    if !alpha.is_finite() || !(0.0 < alpha && alpha <= 1.0) {
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
            drawdown_ema: state.drawdown_ema.max(0.0),
            drawdown_score: 0.0,
            red_active_now: false,
            tier: state.tier,
            changed: state.tier != prev_tier,
            alpha,
            elapsed_minutes: 0,
        };
        state.cached_step = Some(step);
        return Ok(step);
    }

    let last_minute = state
        .last_minute
        .ok_or_else(|| "initialized hard-stop state missing last_minute".to_string())?;
    if current_minute < last_minute {
        return Err(format!(
            "timestamp minute must be non-decreasing, got {} after {}",
            current_minute, last_minute
        ));
    }
    let elapsed_minutes = current_minute - last_minute;
    if elapsed_minutes == 0 {
        let mut step = state
            .cached_step
            .ok_or_else(|| "initialized hard-stop state missing cached_step".to_string())?;
        if latch_red && step.tier == HardStopTier::Red && !state.red_latched {
            state.red_latched = true;
            state.tier = HardStopTier::Red;
            step.tier = HardStopTier::Red;
        }
        step.changed = false;
        step.elapsed_minutes = 0;
        state.cached_step = Some(step);
        return Ok(step);
    }

    state.peak_strategy_equity = peak_strategy_equity;
    let drawdown_raw = (1.0 - (equity / state.peak_strategy_equity.max(f64::EPSILON))).max(0.0);
    let decay = (1.0 - alpha).powf(elapsed_minutes as f64);
    state.drawdown_ema = drawdown_raw + (state.drawdown_ema - drawdown_raw) * decay;
    // Effective trigger metric: min(raw, EMA).
    // Prevents false RED after recovery (stale EMA) and flash-crash bottom exits (raw spike).
    let drawdown_score = drawdown_raw.min(state.drawdown_ema);

    let threshold_yellow = config.tier_ratios.yellow * config.red_threshold;
    let threshold_orange = config.tier_ratios.orange * config.red_threshold;
    let cmp_eps = 1e-12;
    let red_active_now = drawdown_score + cmp_eps >= config.red_threshold;
    if red_active_now {
        state.red_seen_in_episode = true;
    }
    let next_tier = if state.red_latched {
        HardStopTier::Red
    } else if red_active_now {
        HardStopTier::Red
    } else if drawdown_score + cmp_eps >= threshold_orange {
        HardStopTier::Orange
    } else if drawdown_score + cmp_eps >= threshold_yellow {
        HardStopTier::Yellow
    } else {
        HardStopTier::Green
    };
    if latch_red && next_tier == HardStopTier::Red {
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
        drawdown_ema: state.drawdown_ema.max(0.0),
        drawdown_score,
        red_active_now,
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

    #[test]
    fn red_active_now_splits_from_seen_and_latch() {
        // Clarified B2.1 contract: red_active_now reflects only the current
        // sample; red_seen_in_episode persists after recovery; the tier latch
        // may pin display red but must not report red_active_now.
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            red_threshold: 0.2,
            ema_span_minutes: 1.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        let s =
            step_with_peak_strategy_equity_latch(&mut state, config, 100.0, 100.0, 60_000, true)
                .unwrap();
        assert!(!s.red_active_now);
        assert!(!state.red_seen_in_episode);

        // Crash through RED with latching on.
        let s =
            step_with_peak_strategy_equity_latch(&mut state, config, 70.0, 100.0, 120_000, true)
                .unwrap();
        assert!(s.red_active_now);
        assert!(state.red_seen_in_episode);
        assert!(state.red_latched);
        assert_eq!(s.tier, HardStopTier::Red);

        // Full recovery: latch pins the tier, but the current sample is no
        // longer RED, and the episode memory persists.
        let s =
            step_with_peak_strategy_equity_latch(&mut state, config, 100.0, 100.0, 180_000, true)
                .unwrap();
        assert!(!s.red_active_now);
        assert!(state.red_seen_in_episode);
        assert_eq!(s.tier, HardStopTier::Red);

        // Episode reset clears both.
        state = HardStopState::default();
        assert!(!state.red_seen_in_episode);
        assert!(!state.red_latched);
    }

    #[test]
    fn red_seen_in_episode_tracks_without_latching() {
        // With latching off (replay), the tier recovers but the episode
        // memory still records that RED was active.
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            red_threshold: 0.2,
            ema_span_minutes: 1.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        step_with_peak_strategy_equity_latch(&mut state, config, 100.0, 100.0, 60_000, false)
            .unwrap();
        let s =
            step_with_peak_strategy_equity_latch(&mut state, config, 70.0, 100.0, 120_000, false)
                .unwrap();
        assert!(s.red_active_now);
        assert!(!state.red_latched);
        let s =
            step_with_peak_strategy_equity_latch(&mut state, config, 100.0, 100.0, 180_000, false)
                .unwrap();
        assert!(!s.red_active_now);
        assert!(state.red_seen_in_episode);
        assert_ne!(s.tier, HardStopTier::Red);
    }

    #[test]
    fn coin_drawdown_signal_uses_caller_supplied_slot_count() {
        let signal = coin_drawdown_signal(100.0, 4, 2.0, -1.0, -2.5).unwrap();
        assert!((signal.slot_budget - 25.0).abs() < 1e-12);
        assert!((signal.drawdown_usd - 5.5).abs() < 1e-12);
        assert!((signal.drawdown_raw - 0.22).abs() < 1e-12);

        let fewer_slots = coin_drawdown_signal(100.0, 2, 2.0, -1.0, -2.5).unwrap();
        assert!((fewer_slots.slot_budget - 50.0).abs() < 1e-12);
        assert!((fewer_slots.drawdown_raw - 0.11).abs() < 1e-12);
    }

    #[test]
    fn coin_drawdown_signal_rejects_invalid_inputs() {
        assert!(coin_drawdown_signal(0.0, 1, 0.0, 0.0, 0.0).is_err());
        assert!(coin_drawdown_signal(100.0, 0, 0.0, 0.0, 0.0).is_err());
        assert!(coin_drawdown_signal(100.0, 1, f64::NAN, 0.0, 0.0).is_err());
        assert!(coin_drawdown_signal(100.0, 1, 0.0, f64::INFINITY, 0.0).is_err());
        assert!(coin_drawdown_signal(100.0, 1, 0.0, 0.0, f64::NAN).is_err());
    }

    #[test]
    fn red_episode_finalization_uses_persistent_peak_and_fill_timestamp() {
        let out = evaluate_red_episode_finalization(
            "threshold",
            125_500,
            70.0,
            80.0,
            100.0,
            0.10,
            0.20,
            0.25,
            5.0,
        )
        .unwrap();
        assert!((out.no_restart_peak_strategy_equity - 100.0).abs() < 1e-12);
        assert!((out.no_restart_drawdown_raw - 0.30).abs() < 1e-12);
        assert!(out.no_restart_latched);
        assert_eq!(out.cooldown_until_ms, None);
        assert_eq!(out.disposition, RedEpisodeDisposition::NoRestart);

        let restartable = evaluate_red_episode_finalization(
            "threshold",
            125_500,
            90.0,
            100.0,
            0.0,
            0.10,
            0.20,
            0.25,
            5.0,
        )
        .unwrap();
        assert!(!restartable.no_restart_latched);
        assert_eq!(restartable.cooldown_until_ms, Some(425_500));
        assert_eq!(restartable.disposition, RedEpisodeDisposition::Cooldown);
    }

    #[test]
    fn red_episode_finalization_honors_policy_and_canonical_minimum() {
        let always = evaluate_red_episode_finalization(
            "always", 1_000, 1.0, 1.0, 0.0, 1.0, 0.25, 0.5, 0.000_001,
        )
        .unwrap();
        assert!(!always.no_restart_latched);
        assert_eq!(always.cooldown_until_ms, Some(1_001));

        let halted_no_cooldown =
            evaluate_red_episode_finalization("always", 1_000, 1.0, 1.0, 0.0, 0.0, 0.25, 0.5, 0.0)
                .unwrap();
        assert_eq!(
            halted_no_cooldown.disposition,
            RedEpisodeDisposition::HaltedNoCooldown
        );
        assert_eq!(halted_no_cooldown.cooldown_until_ms, None);

        let never =
            evaluate_red_episode_finalization("never", 1_000, 1.0, 1.0, 0.0, 0.0, 0.25, 0.5, 5.0)
                .unwrap();
        assert!(never.no_restart_latched);
        assert_eq!(never.cooldown_until_ms, None);
    }

    #[test]
    fn red_episode_finalization_rejects_invalid_inputs() {
        assert!(evaluate_red_episode_finalization(
            "threshold",
            1_000,
            0.0,
            1.0,
            0.0,
            0.0,
            0.25,
            0.5,
            5.0,
        )
        .is_err());
        assert!(evaluate_red_episode_finalization(
            "threshold",
            1_000,
            1.0,
            1.0,
            0.0,
            f64::NAN,
            0.25,
            0.5,
            5.0,
        )
        .is_err());
        assert!(evaluate_red_episode_finalization(
            "sometimes",
            1_000,
            1.0,
            1.0,
            0.0,
            0.0,
            0.25,
            0.5,
            5.0,
        )
        .is_err());
    }

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
            ema_span_minutes: 1.0, // one-sample EMA for direct red_threshold checks
            tier_ratios: HardStopTierRatios {
                yellow: 0.4,
                orange: 0.8,
            },
        };
        // initialize at equity peak
        let _ = step(&mut state, config, 100.0, 60_000).unwrap();
        // 8% dd => yellow (0.4 * 0.2 = 0.08)
        let s1 = step(&mut state, config, 92.0, 120_000).unwrap();
        assert_eq!(s1.tier, HardStopTier::Yellow);
        // 16% dd => orange (0.8 * 0.2 = 0.16)
        let s2 = step(&mut state, config, 84.0, 180_000).unwrap();
        assert_eq!(s2.tier, HardStopTier::Orange);
        // 20% dd => red
        let s3 = step(&mut state, config, 80.0, 240_000).unwrap();
        assert_eq!(s3.tier, HardStopTier::Red);
    }

    #[test]
    fn same_minute_recall_returns_cached_step_without_advancing() {
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            red_threshold: 0.2,
            ema_span_minutes: 60.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        let _ = step(&mut state, config, 100.0, 60_000).unwrap();
        let first = step(&mut state, config, 90.0, 120_000).unwrap();
        let ema_after_first = state.drawdown_ema;
        let cached = step(&mut state, config, 80.0, 120_500).unwrap();
        assert_eq!(cached.elapsed_minutes, 0);
        assert!(!cached.changed);
        assert!((cached.drawdown_raw - first.drawdown_raw).abs() < 1e-12);
        assert!((cached.drawdown_score - first.drawdown_score).abs() < 1e-12);
        assert!((state.drawdown_ema - ema_after_first).abs() < 1e-12);
    }

    #[test]
    fn multi_minute_gap_matches_repeated_one_minute_steps() {
        let config = HardStopConfig {
            red_threshold: 0.5,
            ema_span_minutes: 60.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        let mut gap_state = HardStopState::default();
        let mut iter_state = HardStopState::default();

        let _ = step(&mut gap_state, config, 100.0, 60_000).unwrap();
        let _ = step(&mut iter_state, config, 100.0, 60_000).unwrap();

        let gap_step = step(&mut gap_state, config, 90.0, 6 * 60_000).unwrap();
        for minute in 2..=6 {
            let _ = step(&mut iter_state, config, 90.0, minute * 60_000).unwrap();
        }

        assert_eq!(gap_step.elapsed_minutes, 5);
        assert!((gap_state.drawdown_ema - iter_state.drawdown_ema).abs() < 1e-12);
    }

    #[test]
    fn score_equals_min_raw_ema() {
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            red_threshold: 0.2,
            ema_span_minutes: 15.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        let _ = step(&mut state, config, 100.0, 60_000).unwrap();
        let s = step(&mut state, config, 90.0, 120_000).unwrap();
        let expected = s.drawdown_raw.min(state.drawdown_ema);
        assert!((s.drawdown_score - expected).abs() < 1e-12);
    }

    #[test]
    fn score_uses_raw_when_ema_is_stale_after_recovery() {
        let mut state = HardStopState::default();
        // Use 1-sample EMA so EMA tracks raw closely on drawdown
        let config = HardStopConfig {
            red_threshold: 0.5,
            ema_span_minutes: 60.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        let _ = step(&mut state, config, 100.0, 60_000).unwrap();
        // Large drawdown to push EMA up
        let _ = step(&mut state, config, 80.0, 120_000).unwrap();
        let _ = step(&mut state, config, 80.0, 180_000).unwrap();
        // Now recover — raw goes to 0 but EMA is still elevated
        let s = step(&mut state, config, 100.0, 240_000).unwrap();
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
        let _ = step(&mut state, config, 100.0, 60_000).unwrap();
        let red = step(&mut state, config, 60.0, 120_000).unwrap();
        assert_eq!(red.tier, HardStopTier::Red);
        assert!(state.red_latched);
        let after_recovery = step(&mut state, config, 100.0, 180_000).unwrap();
        assert_eq!(after_recovery.tier, HardStopTier::Red);
        assert!(state.red_latched);
    }

    #[test]
    fn red_crossing_can_be_replayed_without_latching() {
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            red_threshold: 0.25,
            ema_span_minutes: 1.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        let _ =
            step_with_peak_strategy_equity_latch(&mut state, config, 100.0, 100.0, 60_000, false)
                .unwrap();
        let red =
            step_with_peak_strategy_equity_latch(&mut state, config, 60.0, 100.0, 120_000, false)
                .unwrap();
        assert_eq!(red.tier, HardStopTier::Red);
        assert!(!state.red_latched);
        let after_recovery =
            step_with_peak_strategy_equity_latch(&mut state, config, 100.0, 100.0, 180_000, false)
                .unwrap();
        assert_eq!(after_recovery.tier, HardStopTier::Green);
        assert!(!state.red_latched);
    }

    #[test]
    fn rolling_peak_override_allows_peak_to_decrease() {
        let mut state = HardStopState::default();
        let config = HardStopConfig {
            red_threshold: 0.25,
            ema_span_minutes: 1.0,
            tier_ratios: HardStopTierRatios::default(),
        };
        let _ = step_with_peak_strategy_equity(&mut state, config, 100.0, 100.0, 60_000).unwrap();
        // Simulate old peaks aged out from lookback: peak now 90, equity 90.
        let s = step_with_peak_strategy_equity(&mut state, config, 90.0, 90.0, 120_000).unwrap();
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
        let first = step(&mut state, config, 100.0, 60_000).unwrap();
        assert_eq!(first.tier, HardStopTier::Red);
        assert!(state.red_latched);
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
