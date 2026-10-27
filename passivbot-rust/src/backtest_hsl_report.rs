//! Observational HSL reporting. No field here is read by trading code.
use super::*;
use crate::hsl_intent::{Action, LifecycleEvent};
use crate::hsl_support::Output;
use serde::Serialize;
use std::collections::BTreeMap;

pub(super) type Key = (Option<usize>, Option<usize>);

#[derive(Clone, Debug, Default, Serialize)]
pub struct Summary {
    pub triggers: u32,
    pub restarts: u32,
    pub triggers_long: u32,
    pub triggers_short: u32,
    pub restarts_long: u32,
    pub restarts_short: u32,
    pub red_minutes: f64,
    pub observed_minutes: f64,
    pub red_duration_ms: u64,
    pub observed_duration_ms: u64,
    pub panic_close_fills: u32,
    pub panic_fill_loss_sum_usd: f64,
    pub signal_raw_max_fraction: f64,
    pub signal_ema_max_fraction: f64,
}

#[derive(Debug, Serialize)]
pub struct Sample {
    pub sequence: u64,
    pub timestamp: i64,
    pub side: Option<usize>,
    pub coin: Option<usize>,
    pub phase: &'static str,
    pub action: Option<Action>,
    pub signal_raw_fraction: Option<f64>,
    pub signal_ema_fraction: Option<f64>,
    pub red_at: Option<i64>,
    pub flat_at: Option<i64>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct Event {
    pub sequence: u64,
    pub observed_at: i64,
    pub side: Option<usize>,
    pub coin: Option<usize>,
    pub kind: &'static str,
    pub reconstructed_at: Option<i64>,
    pub reason: &'static str,
}

#[derive(Default)]
struct Scope {
    // Events at a previous capture timestamp can be extended by more same-time
    // executions. Count occurrences per kind, not global fill IDs or red_at,
    // which may change under a later balance rebase.
    watermark: Option<i64>,
    consumed: BTreeMap<(i64, &'static str), usize>,
    red: bool,
    exit_started: Option<i64>,
    panic_loss: f64,
    panic_balance: Option<f64>,
}

#[derive(Clone, Default)]
struct EpisodeLossMoments {
    sum: f64,
    max: f64,
}
impl EpisodeLossMoments {
    fn push(&mut self, value: f64) {
        self.sum += value;
        self.max = self.max.max(value);
    }
}

#[derive(Clone, Default)]
struct LifecycleStats {
    panic_loss_ratios: EpisodeLossMoments,
    execution_episode_count: u64,
    episode_loss_invalid: bool,
}

#[derive(Default)]
pub(super) struct Report {
    scopes: BTreeMap<Key, Scope>,
    pub summary: Summary,
    pub samples: Vec<Sample>,
    pub events: Vec<Event>,
    timestamp: Option<i64>,
    detailed: bool,
    retain_events: bool,
    sequence: u64,
    stats: LifecycleStats,
}

impl Report {
    pub(super) fn new(detailed: bool, retain_events: bool) -> Self {
        Self {
            detailed,
            retain_events,
            ..Self::default()
        }
    }

    pub(super) fn advance(&mut self, now: i64) {
        if let Some(previous) = self.timestamp {
            let minutes = (now - previous).max(0) as f64 / 60_000.0;
            self.summary.observed_minutes += minutes;
            let elapsed_ms = now.saturating_sub(previous).max(0) as u64;
            self.summary.observed_duration_ms += elapsed_ms;
            if self.scopes.values().any(|s| s.red) {
                self.summary.red_minutes += minutes;
                self.summary.red_duration_ms += elapsed_ms;
            }
        }
        self.timestamp = Some(self.timestamp.map_or(now, |last| last.max(now)));
    }

    pub(super) fn observe(&mut self, key: Key, now: i64, phase: &'static str, output: &Output) {
        self.advance(now);
        let mut scope = self.scopes.remove(&key).unwrap_or_default();
        let watermark = scope.watermark.unwrap_or(now);
        let mut counts = BTreeMap::new();
        for event in &output.events {
            // A newly reconstructed historical breach is not an observed stop.
            // Keep same-capture lifecycle events, including zero-time round trips;
            // current non-GREEN permission is still counted independently below.
            if event.timestamp < watermark || event.timestamp < now {
                continue;
            }
            let identity = (event.timestamp, event.kind);
            let count = counts.entry(identity).or_insert(0);
            *count += 1;
            if *count > scope.consumed.get(&identity).copied().unwrap_or(0) {
                self.event(key, now, event, &mut scope);
            }
        }
        scope.watermark = Some(watermark.max(now));
        if now >= watermark {
            scope.consumed = counts.into_iter().filter(|((t, _), _)| *t == now).collect();
        }
        let decision = output.decision.as_ref();
        let action = decision.map_or(Action::Normal, |d| d.action);
        // A current RED reconstructed from an earlier sample is first observed
        // now. Do not count all historical hypothetical transitions as executions.
        if action != Action::Normal && !scope.red {
            self.trigger(
                key,
                now,
                decision.and_then(|d| d.red_at),
                "reconstructed_red",
                &mut scope,
            );
        } else if action == Action::Normal && scope.red {
            self.restart(key, now, None, "current_permission", &mut scope);
        }
        if let Some(d) = decision {
            self.summary.signal_raw_max_fraction = self
                .summary
                .signal_raw_max_fraction
                .max(d.signal_raw_fraction);
            self.summary.signal_ema_max_fraction = self
                .summary
                .signal_ema_max_fraction
                .max(d.signal_ema_fraction);
        }
        self.sequence += 1;
        if self.detailed {
            self.samples.push(Sample {
                sequence: self.sequence,
                timestamp: now,
                side: key.0,
                coin: key.1,
                phase,
                action: decision.map(|d| d.action),
                signal_raw_fraction: decision.map(|d| d.signal_raw_fraction),
                signal_ema_fraction: decision.map(|d| d.signal_ema_fraction),
                red_at: decision.and_then(|d| d.red_at),
                flat_at: decision.and_then(|d| d.flat_at),
                reasons: output.reasons.iter().cloned().collect(),
            });
        }
        self.scopes.insert(key, scope);
    }

    fn trigger(
        &mut self,
        key: Key,
        now: i64,
        reconstructed: Option<i64>,
        reason: &'static str,
        scope: &mut Scope,
    ) {
        self.summary.triggers += 1;
        match key.0 {
            Some(LONG) => self.summary.triggers_long += 1,
            Some(SHORT) => self.summary.triggers_short += 1,
            _ => {} // One unified event, without inventing two side controllers.
        }
        scope.exit_started = Some(now);
        scope.red = true;
        self.sequence += 1;
        if self.retain_events {
            self.events.push(Event {
                sequence: self.sequence,
                observed_at: now,
                side: key.0,
                coin: key.1,
                kind: "red",
                reconstructed_at: reconstructed,
                reason,
            });
        }
    }

    fn restart(
        &mut self,
        key: Key,
        now: i64,
        reconstructed: Option<i64>,
        reason: &'static str,
        scope: &mut Scope,
    ) {
        if !scope.red {
            return;
        }
        self.summary.restarts += 1;
        match key.0 {
            Some(LONG) => self.summary.restarts_long += 1,
            Some(SHORT) => self.summary.restarts_short += 1,
            _ => {}
        }
        Self::finish_loss(&mut self.stats, scope);
        scope.exit_started = None;
        scope.red = false;
        self.sequence += 1;
        if self.retain_events {
            self.events.push(Event {
                sequence: self.sequence,
                observed_at: now,
                side: key.0,
                coin: key.1,
                kind: "restart",
                reconstructed_at: reconstructed,
                reason,
            });
        }
    }

    fn event(&mut self, key: Key, now: i64, event: &LifecycleEvent, scope: &mut Scope) {
        if let Some(raw) = event.signal_raw_fraction {
            self.summary.signal_raw_max_fraction = self.summary.signal_raw_max_fraction.max(raw);
        }
        if let Some(ema) = event.signal_ema_fraction {
            self.summary.signal_ema_max_fraction = self.summary.signal_ema_max_fraction.max(ema);
        }
        match event.kind {
            "red" => {
                self.trigger(key, now, Some(event.timestamp), event.reason, scope);
            }
            "flat" => {
                if !scope.red {
                    self.trigger(key, now, Some(event.red_at), "reconstructed_red", scope);
                }
                scope.exit_started = None;
                Self::finish_loss(&mut self.stats, scope);
                self.sequence += 1;
                if self.retain_events {
                    self.events.push(Event {
                        sequence: self.sequence,
                        observed_at: now,
                        side: key.0,
                        coin: key.1,
                        kind: "flat",
                        reconstructed_at: Some(event.timestamp),
                        reason: event.reason,
                    });
                }
            }
            "restart" => self.restart(key, now, Some(event.timestamp), event.reason, scope),
            _ => unreachable!("Rust lifecycle producer emitted unknown kind"),
        }
    }

    fn record_episode_loss(stats: &mut LifecycleStats, loss: f64, balance: f64) {
        stats.execution_episode_count += 1;
        if balance.is_finite() && balance > 0.0 && loss.is_finite() && loss >= 0.0 {
            let fraction = loss / balance;
            if fraction.is_finite() {
                stats.panic_loss_ratios.push(fraction);
            } else {
                stats.episode_loss_invalid = true;
            }
        } else {
            stats.episode_loss_invalid = true;
        }
    }
    fn finish_loss(stats: &mut LifecycleStats, scope: &mut Scope) {
        if let Some(balance) = scope.panic_balance.take() {
            Self::record_episode_loss(stats, scope.panic_loss, balance);
            scope.panic_loss = 0.0;
        }
    }

    pub(super) fn panic_fill(&mut self, key: Key, net_pnl: f64, account_balance: f64) {
        if let Some(scope) = self.scopes.get_mut(&key).filter(|s| s.red) {
            let loss = (-net_pnl).max(0.0);
            self.summary.panic_close_fills += 1;
            self.summary.panic_fill_loss_sum_usd += loss;
            scope.panic_loss += loss;
            scope.panic_balance.get_or_insert(account_balance);
        }
    }

    /// A real fill can prove flat even when account liquidation prevents replay.
    /// This completes diagnostic execution accounting, not trading permission.
    pub(super) fn observed_flat(&mut self, key: Key, now: i64) {
        self.advance(now);
        if let Some(scope) = self.scopes.get_mut(&key).filter(|s| s.red) {
            if scope.exit_started.take().is_some() {
                Self::finish_loss(&mut self.stats, scope);
                // A later same-bar fill may restore balance and permit replay.
                // Do not report this already-observed flatten a second time.
                scope.consumed.retain(|(t, _), _| *t == now);
                *scope.consumed.entry((now, "flat")).or_insert(0) += 1;
                scope.watermark = Some(now);
                self.sequence += 1;
                if self.retain_events {
                    self.events.push(Event {
                        sequence: self.sequence,
                        observed_at: now,
                        side: key.0,
                        coin: key.1,
                        kind: "flat",
                        reconstructed_at: Some(now),
                        reason: "liquidating_account_flat",
                    });
                }
            }
        }
    }

    pub(super) fn core_summary(&self, enabled: bool) -> crate::core_metrics::HslSummary {
        let mut stats = self.stats.clone();
        for scope in self.scopes.values() {
            if let Some(balance) = scope.panic_balance {
                Self::record_episode_loss(&mut stats, scope.panic_loss, balance);
            }
        }
        crate::core_metrics::HslSummary {
            enabled,
            trigger_count: self.summary.triggers as u64,
            execution_episode_count: stats.execution_episode_count,
            episode_loss_fraction_sum: stats.panic_loss_ratios.sum,
            episode_loss_fraction_max: stats.panic_loss_ratios.max,
            episode_loss_valid: !stats.episode_loss_invalid,
            red_duration_ms: self.summary.red_duration_ms,
            observed_duration_ms: self.summary.observed_duration_ms,
            observation_valid: true,
        }
    }
}

impl<const ORGANILLO: bool> super::BacktestEngine<'_, ORGANILLO> {
    /// Reporting survives result-array draining and never supplies replay input.
    pub fn hsl_report_metadata(&self) -> Result<Option<serde_json::Value>, String> {
        let Some(config) = &self.backtest_params.equity_hard_stop_loss.hsl else {
            return Ok(None);
        };
        let metrics = &self.hsl_report.summary;
        if ![
            metrics.red_minutes,
            metrics.observed_minutes,
            metrics.panic_fill_loss_sum_usd,
            metrics.signal_raw_max_fraction,
            metrics.signal_ema_max_fraction,
        ]
        .iter()
        .all(|x| x.is_finite())
        {
            return Err("non-finite HSL reporting metric".into());
        }
        let summary = serde_json::to_value(&self.hsl_report.summary).map_err(|e| e.to_string())?;
        let scopes = self
            .hsl_report
            .scopes
            .keys()
            .map(|&(side, coin)| {
                Ok(serde_json::json!({
                    "side": side, "coin": coin, "policy": self.hsl_policy(side, coin)?,
                }))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Some(serde_json::json!({
            "schema_version": 1, "engine": "hsl", "mode": config.mode,
            "detailed": self.hsl_report.detailed, "scopes": scopes,
            "coins": self.backtest_params.coins, "summary": summary,
            "samples": [], "events": [],
        })))
    }

    /// Borrow the diagnostic trace without a second per-field JSON allocation.
    pub fn hsl_samples(&self) -> &[Sample] {
        &self.hsl_report.samples
    }

    pub fn hsl_events(&self) -> &[Event] {
        &self.hsl_report.events
    }

    #[cfg(test)]
    pub fn hsl_report_value(&self) -> Result<Option<serde_json::Value>, String> {
        let Some(mut value) = self.hsl_report_metadata()? else {
            return Ok(None);
        };
        value["samples"] = serde_json::to_value(self.hsl_samples()).map_err(|e| e.to_string())?;
        value["events"] = serde_json::to_value(self.hsl_events()).map_err(|e| e.to_string())?;
        Ok(Some(value))
    }

    pub(super) fn hsl_report_key(&self, side: usize, coin: usize) -> Key {
        match self
            .backtest_params
            .equity_hard_stop_loss
            .hsl
            .as_ref()
            .unwrap()
            .mode
            .as_str()
        {
            "portfolio" => (None, None),
            "pside" => (Some(side), None),
            "coin" => (Some(side), Some(coin)),
            _ => unreachable!("validated HSL mode"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hsl_intent::Decision;

    fn output(now: i64, action: Action, events: Vec<LifecycleEvent>) -> Output {
        Output {
            decision: Some(Decision {
                timestamp: now,
                action,
                red_at: (action != Action::Normal).then_some(60000),
                flat_at: (action == Action::Halted).then_some(now),
                reason: "fixture",
                signal_raw_fraction: 0.2,
                signal_ema_fraction: 0.1,
                numeric_range_approximation: false,
            }),
            events,
            reasons: Default::default(),
            observations: 1,
            episodes: 1,
        }
    }

    fn zero_cooldown_events(now: i64) -> Vec<LifecycleEvent> {
        ["red", "flat", "restart"]
            .into_iter()
            .map(|kind| LifecycleEvent {
                timestamp: now,
                kind,
                red_at: now,
                flat_at: (kind != "red").then_some(now),
                reason: "fixture",
                signal_raw_fraction: Some(0.2),
                signal_ema_fraction: Some(0.1),
            })
            .collect()
    }

    #[test]
    fn same_timestamp_stop_occurrences_are_preserved_but_replays_not_counted_twice() {
        let key = (None, None);
        let mut report = Report::default();
        let mut events = zero_cooldown_events(60000);
        for _ in 0..3 {
            report.observe(
                key,
                60000,
                "scope_flat",
                &output(60000, Action::Normal, events.clone()),
            );
        }
        assert_eq!((report.summary.triggers, report.summary.restarts), (1, 1));
        events.extend(zero_cooldown_events(60000));
        for _ in 0..3 {
            report.observe(
                key,
                60000,
                "scope_flat",
                &output(60000, Action::Normal, events.clone()),
            );
        }
        assert_eq!((report.summary.triggers, report.summary.restarts), (2, 2));
        report.observe(
            key,
            120000,
            "bar_close",
            &output(120000, Action::Normal, events.clone()),
        );
        report.observe(
            key,
            180000,
            "bar_close",
            &output(180000, Action::Normal, events),
        );
        assert_eq!((report.summary.triggers, report.summary.restarts), (2, 2));
        assert_eq!(report.summary.red_minutes, 0.0);
        assert_eq!(
            report.summary.triggers_long + report.summary.triggers_short,
            0
        );
    }

    #[test]
    fn reporting_ignores_newly_revealed_historical_red_when_current_green() {
        let key = (Some(LONG), Some(0));
        let mut report = Report::new(false, true);
        report.observe(key, 0, "bar_close", &output(0, Action::Normal, vec![]));
        report.observe(
            key,
            60_000,
            "bar_close",
            &output(60_000, Action::Panic, vec![]),
        );
        report.observe(
            key,
            60_000,
            "scope_flat",
            &output(60_000, Action::Normal, vec![]),
        );
        report.observe(
            key,
            120_000,
            "bar_close",
            &output(120_000, Action::Normal, vec![]),
        );
        let historical = LifecycleEvent {
            timestamp: 120_000,
            kind: "red",
            red_at: 120_000,
            flat_at: None,
            reason: "drawdown",
            signal_raw_fraction: Some(0.2),
            signal_ema_fraction: Some(0.1),
        };
        // Reconstruction can reveal a new hypothetical breach at the previous
        // watermark. Today's GREEN permission never observed that stop.
        report.observe(
            key,
            180_000,
            "bar_close",
            &output(180_000, Action::Normal, vec![historical.clone()]),
        );
        assert_eq!((report.summary.triggers, report.summary.restarts), (1, 1));
        assert_eq!(report.summary.red_minutes, 0.0);
        assert_eq!(report.events.len(), 2);
        // A genuinely current RED still starts an observed stop, even when its
        // reconstructed onset predates this capture.
        report.observe(
            key,
            240_000,
            "bar_close",
            &output(240_000, Action::Panic, vec![historical]),
        );
        assert_eq!((report.summary.triggers, report.summary.restarts), (2, 1));
        assert_eq!(report.events.last().unwrap().observed_at, 240_000);
    }

    #[test]
    fn reporting_counts_observed_permissions_not_old_hypothetical_history() {
        let key = (Some(LONG), Some(0));
        let mut report = Report::default();
        report.observe(key, 0, "bar_close", &output(0, Action::Normal, vec![]));
        report.observe(
            key,
            60000,
            "bar_close",
            &output(60000, Action::Panic, zero_cooldown_events(-60000)),
        );
        report.observe(
            key,
            120000,
            "bar_close",
            &output(120000, Action::Panic, vec![]),
        );
        report.panic_fill(key, -25.0, 1000.0);
        report.panic_fill(key, 10.0, 1000.0);
        report.observe(
            key,
            180000,
            "scope_flat",
            &output(180000, Action::Halted, vec![]),
        );
        report.observe(
            key,
            240000,
            "bar_close",
            &output(240000, Action::Normal, vec![]),
        );
        report.panic_fill(key, -500.0, 1000.0); // An unrelated manual panic is not this HSL stop.
        assert_eq!((report.summary.triggers, report.summary.restarts), (1, 1));
        assert_eq!(
            (report.summary.triggers_long, report.summary.restarts_long),
            (1, 1)
        );
        assert_eq!(report.summary.panic_close_fills, 2);
        assert_eq!(report.summary.panic_fill_loss_sum_usd, 25.0);
        assert_eq!(report.summary.red_minutes, 3.0);
        assert_eq!(report.summary.observed_minutes, 4.0);
        let rendered = serde_json::to_value(&report.summary).unwrap();
        assert!(rendered
            .as_object()
            .unwrap()
            .keys()
            .all(|k| !k.contains("orange") && !k.contains("yellow")));
    }
    #[test]
    fn metrics_only_keeps_identical_summary_without_per_bar_artifacts() {
        let mut detailed = Report::new(true, true);
        let mut compact = Report::new(false, false);
        for report in [&mut detailed, &mut compact] {
            report.observe(
                (None, None),
                0,
                "bar_close",
                &output(0, Action::Normal, vec![]),
            );
            report.observe(
                (None, None),
                60000,
                "bar_close",
                &output(60000, Action::Panic, vec![]),
            );
            report.panic_fill((None, None), -12.0, 1000.0);
            report.observe(
                (None, None),
                120000,
                "scope_flat",
                &output(120000, Action::Halted, vec![]),
            );
            report.observe(
                (None, None),
                180000,
                "bar_close",
                &output(180000, Action::Normal, vec![]),
            );
        }
        assert_eq!(
            serde_json::to_value(&detailed.summary).unwrap(),
            serde_json::to_value(&compact.summary).unwrap()
        );
        assert!(!detailed.samples.is_empty());
        assert!(!detailed.events.is_empty());
        assert!(compact.samples.is_empty());
        assert!(compact.events.is_empty());
    }
    #[test]
    fn gpu_lifecycle_reference_covers_green_open_and_renewed_exposure() {
        let expected = [
            (2, 1, 9.0, vec![4]),
            (2, 1, 3.0, vec![]),
            (1, 1, 1.0, vec![3]),
            (2, 1, 5.0, vec![4]),
            (1, 0, 2.0, vec![]),
            (2, 1, 5.0, vec![4, 5]),
        ];
        for (mode, expected) in expected.into_iter().enumerate() {
            let key = (Some(LONG), Some(0));
            let mut report = Report::new(true, true);
            let mut observe = |minute: i64, action, kinds: &[&'static str]| {
                let now = minute * 60_000;
                let events = kinds
                    .iter()
                    .map(|kind| LifecycleEvent {
                        timestamp: now,
                        kind,
                        red_at: 120_000,
                        flat_at: (*kind == "flat").then_some(now),
                        reason: "fixture",
                        signal_raw_fraction: Some(0.2),
                        signal_ema_fraction: Some(0.1),
                    })
                    .collect();
                report.observe(key, now, "fixture", &output(now, action, events));
            };
            observe(0, Action::Normal, &[]);
            observe(1, Action::Normal, &[]);
            observe(2, Action::Panic, &[]);
            let last = match mode {
                0 => {
                    observe(3, Action::Panic, &[]);
                    observe(4, Action::Halted, &["flat"]);
                    observe(5, Action::Halted, &[]);
                    observe(9, Action::Normal, &[]);
                    observe(10, Action::Normal, &[]);
                    observe(11, Action::Panic, &[]);
                    observe(11, Action::Panic, &[]);
                    13
                }
                1 => {
                    observe(3, Action::Normal, &[]);
                    observe(4, Action::Panic, &[]);
                    6
                }
                2 => {
                    observe(3, Action::Normal, &["flat", "restart"]);
                    3
                }
                3 => {
                    observe(3, Action::Panic, &[]);
                    observe(4, Action::Halted, &["flat"]);
                    observe(5, Action::Panic, &["restart"]);
                    7
                }
                5 => {
                    observe(4, Action::Halted, &["flat"]);
                    observe(5, Action::Halted, &["restart", "red", "flat"]);
                    7
                }
                _ => 4,
            };
            report.advance(last * 60_000);
            let flat_minutes = report
                .events
                .iter()
                .filter(|e| e.kind == "flat")
                .map(|e| e.observed_at / 60_000)
                .collect::<Vec<_>>();
            assert_eq!(
                (
                    report.summary.triggers,
                    report.summary.restarts,
                    report.summary.red_minutes,
                    flat_minutes
                ),
                expected,
                "mode {mode}"
            );
            let core = report.core_summary(true);
            assert_eq!(core.trigger_count, expected.0 as u64);
            assert_eq!(core.red_duration_ms, (expected.2 * 60_000.0) as u64);
            assert_eq!(core.observed_duration_ms, last as u64 * 60_000);
        }
    }

    #[test]
    fn lifecycle_metrics_include_open_halts_partial_exits_and_retriggers() {
        let key = (Some(LONG), Some(0));
        let mut report = Report::new(false, false);
        let event = |timestamp, kind, red_at| LifecycleEvent {
            timestamp,
            kind,
            red_at,
            flat_at: None,
            reason: "fixture",
            signal_raw_fraction: Some(0.2),
            signal_ema_fraction: Some(0.1),
        };
        report.observe(key, 0, "bar_close", &output(0, Action::Normal, vec![]));
        report.observe(
            key,
            60_000,
            "bar_close",
            &output(60_000, Action::Panic, vec![event(60_000, "red", 60_000)]),
        );
        report.panic_fill(key, -25.0, 1000.0);
        report.panic_fill(key, 10.0, 900.0);
        report.advance(120_000);
        for _ in 0..2 {
            let core = report.core_summary(true);
            assert_eq!(core.execution_episode_count, 1);
            assert_eq!(core.episode_loss_fraction_sum, 0.025);
            assert_eq!(core.red_duration_ms, 60_000);
        }
        assert_eq!(report.stats.execution_episode_count, 0);
        report.observe(
            key,
            180_000,
            "scope_flat",
            &output(
                180_000,
                Action::Halted,
                vec![event(180_000, "flat", 60_000)],
            ),
        );
        report.observe(
            key,
            300_000,
            "bar_close",
            &output(
                300_000,
                Action::Normal,
                vec![event(300_000, "restart", 60_000)],
            ),
        );
        report.observe(
            key,
            360_000,
            "bar_close",
            &output(360_000, Action::Panic, vec![event(360_000, "red", 360_000)]),
        );
        report.panic_fill(key, -40.0, 800.0);
        report.advance(420_000);
        for _ in 0..2 {
            // Snapshot metrics must not consume unfinished episodes.
            let core = report.core_summary(true);
            assert_eq!((report.summary.triggers, report.summary.restarts), (2, 1));
            assert_eq!(core.trigger_count, 2);
            assert_eq!(core.red_duration_ms, 5 * 60_000);
            assert_eq!(core.observed_duration_ms, 7 * 60_000);
            assert_eq!(core.execution_episode_count, 2);
            assert!((core.episode_loss_fraction_sum - 0.075).abs() < 1e-12);
            assert_eq!(core.episode_loss_fraction_max, 0.05);
            assert_eq!(report.summary.panic_fill_loss_sum_usd, 65.0);
            assert_eq!(report.summary.signal_raw_max_fraction, 0.2);
            assert_eq!(report.summary.signal_ema_max_fraction, 0.1);
        }
        assert!(report.samples.is_empty() && report.events.is_empty());
    }

    #[test]
    fn repanic_preserves_flat_event_clocks_and_continuous_red_union() {
        let key = (None, None);
        let mut report = Report::new(true, true);
        let mut events = zero_cooldown_events(60_000);
        events.pop(); // RED then flat, held in cooldown.
        report.observe(
            key,
            60_000,
            "scope_flat",
            &output(60_000, Action::Halted, events),
        );
        let mut events = zero_cooldown_events(180_000);
        events.truncate(1);
        report.observe(
            key,
            180_000,
            "bar_close",
            &output(180_000, Action::Panic, events),
        );
        let mut events = zero_cooldown_events(240_000);
        events.remove(0);
        report.observe(
            key,
            240_000,
            "scope_flat",
            &output(240_000, Action::Normal, events),
        );
        assert_eq!((report.summary.triggers, report.summary.restarts), (2, 1));
        assert_eq!(report.core_summary(true).red_duration_ms, 3 * 60_000);
        assert_eq!(
            report.summary.triggers_long + report.summary.triggers_short,
            0
        );
        assert_eq!(
            report
                .events
                .iter()
                .filter(|e| e.kind == "flat")
                .map(|e| e.observed_at)
                .collect::<Vec<_>>(),
            vec![60_000, 240_000]
        );
    }
    #[test]
    fn event_only_report_preserves_zero_cooldown_and_liquidation_sequences() {
        let key = (None, None);
        let mut detailed = Report::new(true, true);
        let mut compact = Report::new(false, true);
        for report in [&mut detailed, &mut compact] {
            report.observe(
                key,
                60_000,
                "scope_flat",
                &output(60_000, Action::Normal, zero_cooldown_events(60_000)),
            );
            report.observe(
                key,
                120_000,
                "bar_close",
                &output(120_000, Action::Panic, vec![]),
            );
            report.panic_fill(key, -20.0, 1000.0);
            report.observed_flat(key, 180_000);
            report.observed_flat(key, 180_000);
            let mut events = zero_cooldown_events(180_000);
            events.remove(0);
            report.observe(
                key,
                180_000,
                "scope_flat",
                &output(180_000, Action::Normal, events),
            );
        }
        assert!(!detailed.samples.is_empty());
        assert!(compact.samples.is_empty());
        assert_eq!(compact.events.len(), 6);
        assert_eq!(
            serde_json::to_value(&detailed.events).unwrap(),
            serde_json::to_value(&compact.events).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&detailed.summary).unwrap(),
            serde_json::to_value(&compact.summary).unwrap()
        );
    }

    #[test]
    fn fill_proven_flat_is_not_recounted_by_later_replay() {
        let mut report = Report::new(true, true);
        let key = (None, None);
        let mut events = zero_cooldown_events(60_000);
        events.truncate(1);
        report.observe(
            key,
            60_000,
            "bar_close",
            &output(60_000, Action::Panic, events),
        );
        report.panic_fill(key, -20.0, 1000.0);
        report.observed_flat(key, 120_000);
        report.observed_flat(key, 120_000);
        let mut events = zero_cooldown_events(120_000);
        events.remove(0);
        for _ in 0..2 {
            report.observe(
                key,
                120_000,
                "scope_flat",
                &output(120_000, Action::Normal, events.clone()),
            );
        }
        assert_eq!(report.events.iter().filter(|e| e.kind == "flat").count(), 1);
        assert_eq!(report.summary.restarts, 1);
        assert_eq!(
            report
                .events
                .iter()
                .find(|e| e.kind == "flat")
                .unwrap()
                .observed_at,
            120_000
        );
        let core = report.core_summary(true);
        assert_eq!(core.execution_episode_count, 1);
        assert_eq!(core.episode_loss_fraction_sum, 0.02);
    }
    #[test]
    fn core_episode_loss_anchors_once_to_balance_and_keeps_censored_tails() {
        let key = (Some(LONG), Some(0));
        let mut report = Report::default();
        report.advance(0);
        report.observe(key, 0, "pre_fill", &output(0, Action::Panic, vec![]));
        report.panic_fill(key, -100.0, 1000.0);
        report.panic_fill(key, -100.0, 900.0);
        report.panic_fill(key, -100.0, 800.0);
        report.advance(60_000);
        let summary = report.core_summary(true);
        assert_eq!(summary.execution_episode_count, 1);
        assert!((summary.episode_loss_fraction_max - 0.3).abs() < 1e-12);
        assert!((summary.episode_loss_fraction_sum - 0.3).abs() < 1e-12);
        assert_eq!(report.core_summary(true).execution_episode_count, 1);
        report.observed_flat(key, 60_000);
        assert_eq!(report.core_summary(true).execution_episode_count, 1);
        report.observe(
            key,
            120_000,
            "green",
            &output(120_000, Action::Normal, vec![]),
        );
        assert_eq!(report.core_summary(true).red_duration_ms, 120_000);
        assert_eq!(report.core_summary(true).observed_duration_ms, 120_000);
    }
    #[test]
    fn invalid_episode_balance_is_not_floored_or_dropped() {
        let key = (Some(LONG), Some(0));
        let mut report = Report::default();
        report.observe(key, 0, "pre_fill", &output(0, Action::Panic, vec![]));
        report.panic_fill(key, -10.0, 0.0);
        report.observed_flat(key, 60_000);
        let summary = report.core_summary(true);
        assert!(!summary.episode_loss_valid);
        assert_eq!(summary.execution_episode_count, 1);
    }
}
