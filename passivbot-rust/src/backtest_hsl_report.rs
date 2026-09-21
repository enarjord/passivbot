//! Observational revised-HSL reporting. No field here is read by trading code.
use super::*;
use crate::hsl_revised_controller::{Action, LifecycleEvent};
use crate::hsl_revised_evaluator::Output;
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
    pub panic_close_fills: u32,
    pub panic_close_loss: f64,
    pub worst_raw: f64,
    pub worst_ema: f64,
}

#[derive(Debug, Serialize)]
pub struct Sample {
    pub timestamp: i64,
    pub side: Option<usize>,
    pub coin: Option<usize>,
    pub phase: &'static str,
    pub action: Action,
    pub raw: Option<f64>,
    pub ema: Option<f64>,
    pub red_at: Option<i64>,
    pub flat_at: Option<i64>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct Event {
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
    halt_started: Option<i64>,
    exit_started: Option<i64>,
    restarted_without_retrigger: bool,
    panic_loss: f64,
    panic_equity: Option<f64>,
}

#[derive(Clone, Default)]
struct Distribution {
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
}
impl Distribution {
    fn push(&mut self, value: f64) {
        self.min = if self.count == 0 {
            value
        } else {
            self.min.min(value)
        };
        self.max = self.max.max(value);
        self.sum += value;
        self.count += 1;
    }
    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
}

#[derive(Clone, Default)]
struct LifecycleStats {
    durations: Distribution,
    flatten_minutes: Distribution,
    trigger_scores: Distribution,
    panic_loss_ratios: Distribution,
    panic_loss_max: f64,
    retriggers: u32,
}

#[derive(Default)]
pub(super) struct Report {
    scopes: BTreeMap<Key, Scope>,
    pub summary: Summary,
    pub samples: Vec<Sample>,
    pub events: Vec<Event>,
    timestamp: Option<i64>,
    detailed: bool,
    stats: LifecycleStats,
    // One bar-close maximum per signal category: global, long, short.
    pub signal_emas: [Vec<f64>; 3],
}

impl Report {
    pub(super) fn new(detailed: bool) -> Self {
        Self {
            detailed,
            ..Self::default()
        }
    }

    pub(super) fn advance(&mut self, now: i64) {
        if let Some(previous) = self.timestamp {
            let minutes = (now - previous).max(0) as f64 / 60_000.0;
            self.summary.observed_minutes += minutes;
            if self.scopes.values().any(|s| s.red) {
                self.summary.red_minutes += minutes;
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
            if event.timestamp < watermark {
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
            if let Some(d) = decision {
                self.stats.trigger_scores.push(d.raw.min(d.ema));
            }
        } else if action == Action::Normal && scope.red {
            self.restart(key, now, None, "current_permission", &mut scope);
        }
        if let Some(d) = decision {
            self.summary.worst_raw = self.summary.worst_raw.max(d.raw);
            self.summary.worst_ema = self.summary.worst_ema.max(d.ema);
        }
        self.samples.push(Sample {
            timestamp: now,
            side: key.0,
            coin: key.1,
            phase,
            action,
            raw: decision.map(|d| d.raw),
            ema: decision.map(|d| d.ema),
            red_at: decision.and_then(|d| d.red_at),
            flat_at: decision.and_then(|d| d.flat_at),
            reasons: output.reasons.iter().cloned().collect(),
        });
        self.scopes.insert(key, scope);
        if !self.detailed {
            self.samples.clear();
            self.events.clear();
        }
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
        if scope.restarted_without_retrigger {
            self.stats.retriggers += 1;
            scope.restarted_without_retrigger = false;
        }
        scope.halt_started.get_or_insert(now);
        scope.exit_started = Some(now);
        scope.red = true;
        self.events.push(Event {
            observed_at: now,
            side: key.0,
            coin: key.1,
            kind: "red",
            reconstructed_at: reconstructed,
            reason,
        });
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
        if let Some(start) = scope.halt_started.take() {
            self.stats
                .durations
                .push((now - start).max(0) as f64 / 60_000.0);
        }
        Self::finish_loss(&mut self.stats, scope);
        scope.exit_started = None;
        scope.restarted_without_retrigger = true;
        scope.red = false;
        self.events.push(Event {
            observed_at: now,
            side: key.0,
            coin: key.1,
            kind: "restart",
            reconstructed_at: reconstructed,
            reason,
        });
    }

    fn event(&mut self, key: Key, now: i64, event: &LifecycleEvent, scope: &mut Scope) {
        if let Some(raw) = event.raw {
            self.summary.worst_raw = self.summary.worst_raw.max(raw);
        }
        if let Some(ema) = event.ema {
            self.summary.worst_ema = self.summary.worst_ema.max(ema);
        }
        match event.kind {
            "red" => {
                self.trigger(key, now, Some(event.timestamp), event.reason, scope);
                if let (Some(raw), Some(ema)) = (event.raw, event.ema) {
                    self.stats.trigger_scores.push(raw.min(ema));
                }
            }
            "flat" => {
                if !scope.red {
                    self.trigger(key, now, Some(event.red_at), "reconstructed_red", scope);
                }
                if let Some(start) = scope.exit_started.take() {
                    self.stats
                        .flatten_minutes
                        .push((now - start).max(0) as f64 / 60_000.0);
                }
                Self::finish_loss(&mut self.stats, scope);
                self.events.push(Event {
                    observed_at: now,
                    side: key.0,
                    coin: key.1,
                    kind: "flat",
                    reconstructed_at: Some(event.timestamp),
                    reason: event.reason,
                });
            }
            "restart" => self.restart(key, now, Some(event.timestamp), event.reason, scope),
            _ => unreachable!("Rust lifecycle producer emitted unknown kind"),
        }
    }

    fn finish_loss(stats: &mut LifecycleStats, scope: &mut Scope) {
        if let Some(equity) = scope.panic_equity.take() {
            stats.panic_loss_ratios.push(scope.panic_loss / equity);
            scope.panic_loss = 0.0;
        }
    }

    pub(super) fn panic_fill(&mut self, key: Key, net_pnl: f64, account_equity: f64) {
        if let Some(scope) = self.scopes.get_mut(&key).filter(|s| s.red) {
            let loss = (-net_pnl).max(0.0);
            self.summary.panic_close_fills += 1;
            self.summary.panic_close_loss += loss;
            self.stats.panic_loss_max = self.stats.panic_loss_max.max(loss);
            scope.panic_loss += loss;
            scope
                .panic_equity
                .get_or_insert(account_equity.max(f64::EPSILON));
        }
    }

    /// A real fill can prove flat even when account liquidation prevents replay.
    /// This completes diagnostic execution accounting, not trading permission.
    pub(super) fn observed_flat(&mut self, key: Key, now: i64) {
        self.advance(now);
        if let Some(scope) = self.scopes.get_mut(&key).filter(|s| s.red) {
            if let Some(start) = scope.exit_started.take() {
                self.stats
                    .flatten_minutes
                    .push((now - start).max(0) as f64 / 60_000.0);
                Self::finish_loss(&mut self.stats, scope);
                // A later same-bar fill may restore balance and permit replay.
                // Do not report this already-observed flatten a second time.
                scope.consumed.retain(|(t, _), _| *t == now);
                *scope.consumed.entry((now, "flat")).or_insert(0) += 1;
                scope.watermark = Some(now);
                if self.detailed {
                    self.events.push(Event {
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

    pub(super) fn record_bar_signals(&mut self, now: i64, emas: [f64; 3]) {
        self.advance(now);
        for (samples, value) in self.signal_emas.iter_mut().zip(emas) {
            samples.push(value);
        }
    }

    /// Include censored open halts/exits without mutating observation state.
    pub(super) fn metrics(&self, starting_balance: f64, minutes: f64) -> HardStopMetrics {
        let mut stats = self.stats.clone();
        for scope in self.scopes.values() {
            if let (Some(start), Some(end)) = (scope.halt_started, self.timestamp) {
                stats.durations.push((end - start).max(0) as f64 / 60_000.0);
            }
            if let (Some(start), Some(end)) = (scope.exit_started, self.timestamp) {
                stats
                    .flatten_minutes
                    .push((end - start).max(0) as f64 / 60_000.0);
            }
            if let Some(equity) = scope.panic_equity {
                stats.panic_loss_ratios.push(scope.panic_loss / equity);
            }
        }
        let annual = if minutes > 0.0 {
            365.25 * 1440.0 / minutes
        } else {
            0.0
        };
        let s = &self.summary;
        HardStopMetrics {
            triggers: s.triggers,
            triggers_per_year: s.triggers as f64 * annual,
            triggers_long: s.triggers_long,
            triggers_short: s.triggers_short,
            restarts: s.restarts,
            restarts_per_year: s.restarts as f64 * annual,
            restarts_long: s.restarts_long,
            restarts_short: s.restarts_short,
            restarts_per_year_long: s.restarts_long as f64 * annual,
            restarts_per_year_short: s.restarts_short as f64 * annual,
            time_in_red_pct: if s.observed_minutes > 0.0 {
                s.red_minutes / s.observed_minutes
            } else {
                0.0
            },
            duration_minutes_mean: stats.durations.mean(),
            duration_minutes_max: stats.durations.max,
            trigger_drawdown_mean: stats.trigger_scores.mean(),
            panic_close_loss_sum: s.panic_close_loss,
            panic_close_loss_max: stats.panic_loss_max,
            panic_close_loss_drawdown_pct_min: stats.panic_loss_ratios.min,
            panic_close_loss_drawdown_pct_mean: stats.panic_loss_ratios.mean(),
            panic_close_loss_drawdown_pct_max: stats.panic_loss_ratios.max,
            halt_to_restart_equity_loss_pct: s.panic_close_loss
                / starting_balance.max(f64::EPSILON),
            flatten_time_minutes_mean: stats.flatten_minutes.mean(),
            post_restart_retrigger_pct: if s.restarts > 0 {
                stats.retriggers as f64 / s.restarts as f64
            } else {
                0.0
            },
            ..HardStopMetrics::default()
        }
    }
}

impl Backtest<'_> {
    /// Reporting survives result-array draining and never supplies replay input.
    pub fn revised_hsl_report_value(&self) -> Result<Option<serde_json::Value>, String> {
        let Some(config) = &self.backtest_params.equity_hard_stop_loss.revised else {
            return Ok(None);
        };
        let metrics = &self.revised_hsl_report.summary;
        if ![
            metrics.red_minutes,
            metrics.observed_minutes,
            metrics.panic_close_loss,
            metrics.worst_raw,
            metrics.worst_ema,
        ]
        .iter()
        .all(|x| x.is_finite())
        {
            return Err("non-finite revised HSL reporting metric".into());
        }
        let summary =
            serde_json::to_value(&self.revised_hsl_report.summary).map_err(|e| e.to_string())?;
        let samples =
            serde_json::to_value(&self.revised_hsl_report.samples).map_err(|e| e.to_string())?;
        let events =
            serde_json::to_value(&self.revised_hsl_report.events).map_err(|e| e.to_string())?;
        Ok(Some(serde_json::json!({
            "schema_version": 1, "engine": "revised", "mode": config.mode,
            "coins": self.backtest_params.coins, "summary": summary,
            "samples": samples, "events": events,
        })))
    }

    pub(super) fn revised_report_key(&self, side: usize, coin: usize) -> Key {
        match self
            .backtest_params
            .equity_hard_stop_loss
            .revised
            .as_ref()
            .unwrap()
            .mode
            .as_str()
        {
            "unified" => (None, None),
            "pside" => (Some(side), None),
            "coin" => (Some(side), Some(coin)),
            _ => unreachable!("validated revised HSL mode"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hsl_revised_controller::Decision;

    fn output(now: i64, action: Action, events: Vec<LifecycleEvent>) -> Output {
        Output {
            decision: Some(Decision {
                timestamp: now,
                action,
                red_at: (action != Action::Normal).then_some(60000),
                flat_at: (action == Action::Halted).then_some(now),
                reason: "fixture",
                raw: 0.2,
                ema: 0.1,
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
                raw: Some(0.2),
                ema: Some(0.1),
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
        assert_eq!(report.summary.panic_close_loss, 25.0);
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
        let mut detailed = Report::new(true);
        let mut compact = Report::new(false);
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
    fn lifecycle_metrics_include_open_halts_partial_exits_and_retriggers() {
        let key = (Some(LONG), Some(0));
        let mut report = Report::new(false);
        let event = |timestamp, kind, red_at| LifecycleEvent {
            timestamp,
            kind,
            red_at,
            flat_at: None,
            reason: "fixture",
            raw: Some(0.2),
            ema: Some(0.1),
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
        report.record_bar_signals(120_000, [0.1, 0.1, 0.0]);
        for _ in 0..2 {
            assert_eq!(report.metrics(1000.0, 2.0).flatten_time_minutes_mean, 1.0);
        }
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
        report.record_bar_signals(420_000, [0.1, 0.1, 0.0]);
        for _ in 0..2 {
            // Snapshot metrics must not consume unfinished episodes.
            let m = report.metrics(1000.0, 7.0);
            assert_eq!((m.triggers, m.restarts), (2, 1));
            assert!((m.time_in_red_pct - 5.0 / 7.0).abs() < 1e-12);
            assert_eq!(m.duration_minutes_mean, 2.5);
            assert_eq!(m.duration_minutes_max, 4.0);
            assert_eq!(m.flatten_time_minutes_mean, 1.5); // Completed two minutes + open one minute.
            assert_eq!(m.trigger_drawdown_mean, 0.1);
            assert_eq!(m.panic_close_loss_sum, 65.0);
            assert_eq!(m.panic_close_loss_max, 40.0);
            assert_eq!(m.panic_close_loss_drawdown_pct_min, 0.025);
            assert!((m.panic_close_loss_drawdown_pct_mean - 0.0375).abs() < 1e-12);
            assert_eq!(m.panic_close_loss_drawdown_pct_max, 0.05);
            assert_eq!(m.halt_to_restart_equity_loss_pct, 0.065);
            assert_eq!(m.post_restart_retrigger_pct, 1.0);
            assert_eq!(m.triggers_per_year, 2.0 * 365.25 * 1440.0 / 7.0);
        }
        assert!(report.samples.is_empty() && report.events.is_empty());
    }

    #[test]
    fn repanic_flat_latency_uses_new_exit_but_keeps_continuous_halt_duration() {
        let key = (None, None);
        let mut report = Report::new(false);
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
        let m = report.metrics(1000.0, 3.0);
        assert_eq!((m.triggers, m.restarts), (2, 1));
        assert_eq!(m.duration_minutes_mean, 3.0);
        assert_eq!(m.flatten_time_minutes_mean, 0.5); // Two exits: zero and one minute.
        assert_eq!(m.triggers_long + m.triggers_short, 0);
    }
    #[test]
    fn fill_proven_flat_is_not_recounted_by_later_replay() {
        let mut report = Report::new(true);
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
        let m = report.metrics(1000.0, 2.0);
        assert_eq!(m.flatten_time_minutes_mean, 1.0);
        assert_eq!(m.restarts, 1);
        assert_eq!(m.panic_close_loss_drawdown_pct_mean, 0.02);
    }
}
