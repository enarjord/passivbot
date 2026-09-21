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
}

#[derive(Default)]
pub(super) struct Report {
    scopes: BTreeMap<Key, Scope>,
    pub summary: Summary,
    pub samples: Vec<Sample>,
    pub events: Vec<Event>,
    timestamp: Option<i64>,
    detailed: bool,
}

impl Report {
    pub(super) fn new(detailed: bool) -> Self {
        Self {
            detailed,
            ..Self::default()
        }
    }

    fn advance(&mut self, now: i64) {
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
            "red" => self.trigger(key, now, Some(event.timestamp), event.reason, scope),
            "flat" => {
                if !scope.red {
                    self.trigger(key, now, Some(event.red_at), "reconstructed_red", scope);
                }
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

    pub(super) fn panic_fill(&mut self, key: Key, net_pnl: f64) {
        if self.scopes.get(&key).is_some_and(|s| s.red) {
            self.summary.panic_close_fills += 1;
            self.summary.panic_close_loss += (-net_pnl).max(0.0);
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
        report.panic_fill(key, -25.0);
        report.panic_fill(key, 10.0);
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
        report.panic_fill(key, -500.0); // An unrelated manual panic is not this HSL stop.
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
            report.panic_fill((None, None), -12.0);
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
}
